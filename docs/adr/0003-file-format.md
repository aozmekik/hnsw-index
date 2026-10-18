# ADR 0003: Versioned, checksummed binary format with a validating loader

- Status: accepted
- Date: 2026-10-18

## Context

Indexes are expensive to build, so they need to be saved and loaded. Index
files get copied between machines, truncated by full disks and, in a service,
may come from places the loading process does not control. The loader must
therefore treat its input as untrusted: any byte sequence must produce either
a valid index or an error, never a panic or an out-of-bounds access, and the
memory it allocates must stay within a limit the caller controls.

That last point is harder than it looks. Neighbour lists have a fixed
capacity per node (ADR 0001), so the index a file describes can be far larger
than the file: a node whose lists are all empty costs a few bytes on disk but
`(M0 + 1) × 4` bytes of layer-0 storage, plus `(M + 1) × 4` bytes for every
upper layer it claims. With the largest accepted parameters (`M = 512`,
`M0 = 2048`) and one-dimensional vectors, that is roughly 500 bytes of memory
per byte of input.

## Options considered

1. **serde + bincode.** Little code, but adds dependencies, ties the format to
   the in-memory struct layout, and a derived deserializer does not check
   graph invariants (ids in range, capacities, links only between nodes that
   exist on the layer).
2. **A memory-mappable layout** loaded without copying. Fastest load, but the
   data cannot be validated without reading it all anyway, and the mutable,
   atomic adjacency storage (ADR 0001) is not a plain byte layout.
3. **A small hand-written format**: fixed little-endian header, columnar
   sections, CRC-32 trailer, and a loader that checks every invariant.

## Decision

Option 3, documented field by field in `src/hnsw/io.rs`:

- magic `HNSWIDX\0` and a `u32` version (currently 1), checked first so a
  wrong file type or a future version gets a precise error;
- a CRC-32 (IEEE, as in zlib and PNG) over everything before the trailer,
  checked before any parsing. CRC-32 detects all burst errors up to 32 bits;
  it is an integrity check, not authentication (an attacker can recompute it,
  which is why validation does not rely on it);
- header fields re-validated with the same rules as `HnswParams`;
- allocation in three steps, before any graph storage exists: every column
  is located in the input as a borrowed slice (a node count larger than the
  input is `Truncated` without allocating); the adjacency section is walked
  once without allocating, so every neighbour list the file makes the loader
  allocate must be present; and the heap size of the described index,
  computed from the header and level column exactly as `memory_usage()` will
  report it, must not exceed a limit. `from_bytes` allows 64 times the input
  size or 64 MiB, whichever is larger; `from_bytes_with_limit` takes an
  explicit limit;
- the level column checked for statistical plausibility under the stored
  `mL`;
- every vector accepted by the same check as `insert` (finite components,
  bounded squared norm), cosine vectors of unit length,
  tombstone flags 0 or 1, live keys unique, neighbour counts within capacity,
  neighbour ids in range, unique within their list, not self-loops and
  present on the layer they are linked on, and the entry point on the top
  level;
- no trailing bytes.

The whole input is read into memory before parsing (`read_from` calls
`read_to_end`), which keeps parsing a bounds-checked slice walk. Callers
reading from an untrusted stream cap its length with `Read::take`.

## Consequences

- Honest indexes need far less than the default limit. One-dimensional
  vectors are the worst case (the heuristic keeps about four neighbours per
  node, so files are small), and a 20,000-node one-dimensional index with
  `M0 = 256` and the default `mL` needs 32 times its file size (unit test
  `default_memory_limit_admits_low_dimensional_indexes_with_m0_256`, which
  asserts less than 48). A larger `M0`, or a much larger `mL`, combined with
  very low-dimensional vectors can exceed 64 times; such indexes are loaded
  with `from_bytes_with_limit`, and the error reports the size needed.
- A rejected file costs no more than reading it: `tests/hostile_files.rs`
  builds a 1.9 MB file that describes about 945 MB of graph storage and
  checks that loading it fails with `MemoryLimitExceeded` while the heap
  grows by less than 64 KiB, and the same for a truncated variant.
- An earlier version of the loader bounded only the per-node columns by the
  input size and allocated graph storage before checking that the file
  contained the lists, so a crafted file could make it allocate hundreds of
  times its own size before returning an error. Review found it; the
  regression test above pins the fix.

- Property tests flip random bytes (and recompute the CRC so the structural
  checks are what gets exercised) and truncate files at random offsets; none
  may panic, and any index that loads must be searchable.
- Loading needs the file size in memory on top of the index; streaming parsing
  is possible later without changing the format.
- Saving a just-loaded index reproduces the input byte for byte, which the
  tests use as a round-trip oracle.
- Changes to the layout require a version bump; the loader rejects versions it
  does not know.
