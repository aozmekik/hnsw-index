# ADR 0001: Atomic fixed-capacity adjacency lists with per-node writer locks

- Status: accepted
- Date: 2026-10-18

## Context

HNSW search is dominated by two memory access patterns: reading a node's
neighbor list and reading the vectors of those neighbors. The index must
support

1. many concurrent searches on one index (`HnswIndex: Send + Sync`, search
   through `&self`), and
2. parallel batch construction with rayon, where every insertion both reads
   the graph (searching for neighbors) and writes it (adding back-links to
   existing nodes, shrinking their lists).

The read path must stay as fast as a single-threaded implementation, and
everything must be expressible in safe Rust outside the SIMD kernels.

## Options considered

1. **`RwLock<Vec<u32>>` per node and layer.** Simple, but every neighbor-list
   read during search takes a lock (an atomic read-modify-write on a shared
   cache line), and a `Vec` per list costs an allocation and a pointer chase.
2. **Build into a mutable structure, then freeze into an immutable CSR layout.**
   Fastest reads, but every later `insert` needs a rebuild or a second,
   mutable overlay.
3. **hnswlib's approach: plain memory, a mutex per node for writers, racy
   unsynchronized reads.** Fast, but the concurrent reads are data races, which
   are undefined behavior in Rust (and, strictly, in C++).
4. **Fixed-capacity blocks of `AtomicU32`, a mutex per node for writers,
   lock-free atomic reads.** Each node owns one block per layer: a count word
   followed by `M0` (layer 0) or `M` (upper layers) id slots. Writers hold the
   node's lock, store ids, then publish the count with `Release`; readers load
   the count with `Acquire` and the ids with `Relaxed`.

## Decision

Option 4. Layer-0 blocks of all nodes live in one contiguous `Vec<AtomicU32>`;
upper-layer blocks are a boxed slice per node (empty, so unallocated, for the
roughly 15 of 16 nodes that only live on layer 0 with `M = 16`). Node ids are
dense `u32`s, which halves the size of every adjacency slot compared with
`usize`; `MAX_NODES = 2^32 - 2`.

Mutation of the public API takes `&mut self`, so searches never run
concurrently with insertions from the caller's point of view. Concurrency
between readers and writers only exists inside `insert_batch`, where
insertions read lists that other insertions are rewriting.

Writer locking follows one rule: **a thread holds at most one node lock at a
time**, and the promotion lock (taken by a node that becomes the new entry
point) is only ever acquired before any node lock. Reads never lock.

## Consequences

- Neighbor ids are read with `Relaxed` loads, which compile to ordinary
  loads on x86-64 and AArch64. The count is read with an `Acquire` load: an
  ordinary `MOV` on x86-64, but `LDAPR` (an ordered load) on AArch64, once
  per neighbor list read. That is the whole read-side price of thread
  safety; it was not measured separately. There is no `unsafe` in the graph
  code.
- A reader racing with a rewrite may see a mix of old and new neighbor ids.
  Every id it can see is a valid node id (all storage for a batch is allocated
  before linking starts), so the worst case is a slightly different search
  path during construction, never an out-of-bounds access.
- Fixed capacity wastes memory for nodes with fewer than `M0` neighbors; with
  `M = 16`, layer 0 costs `(32 + 1) × 4 = 132` bytes per node regardless of
  degree. In exchange there is no per-list allocation and the layer-0 block of
  node `i` is found by multiplication.
- Parallel builds are not bit-for-bit reproducible (insertion interleaving
  changes the graph); sequential `insert` loops are, on the same kernel
  backend. Different backends round distances differently (NEON and AVX2
  fuse multiply and add, the portable kernels do not), so the same
  sequential build can produce a different graph on another CPU. Node levels
  are a hash of `(seed, id)` rather than a stateful RNG draw, so levels at
  least are identical across thread counts and machines.
- An earlier variant held the inserted node's lock for its whole insertion, as
  hnswlib does, to stop concurrent back-links from being overwritten. The
  property tests deadlocked with it: a node carried over as an entry point from
  the layer above is visible on the current layer before it is connected there,
  so two insertions can select each other and wait on each other's lock. The
  shipped code instead merges any back-links already present when a new node
  writes its own list, which closes the same race under the one-lock rule.
