# ADR 0002: Deletion by tombstones, reclamation by rebuilding

- Status: accepted
- Date: 2026-10-18

## Context

The paper does not define deletion. Removing a node from an HNSW graph
physically means repairing the neighbor lists of every node that pointed to
it, on every layer, without disconnecting parts of the graph. Callers still
expect `remove(key)` to take effect immediately.

## Options considered

1. **Tombstones only.** Mark the node deleted; searches route through it but
   never return it. Memory is never reclaimed.
2. **In-place repair.** On delete, reconnect each in-neighbor of the deleted
   node using its other neighbors (as in FreshDiskANN, or hnswlib's
   `replace_deleted` mode that reuses the slot for the next insert). Needs
   reverse edges or a full scan to find in-neighbors, and repair quality is
   hard to test.
3. **Tombstones plus an explicit rebuild.** As option 1, plus
   `compact()`, which rebuilds the graph from the live vectors.

## Decision

Option 3. `remove` flips a flag and drops the key from the key map in O(1).
Searches treat tombstoned nodes like nodes rejected by a filter: they are
expanded (so connectivity is preserved) but never enter the result set. A key
that was removed can be inserted again; it gets a new node. `compact()`
re-inserts the live vectors with `insert_batch` into a fresh graph and keeps
keys and parameters.

## Consequences

- Deletion is immediate and cannot degrade the graph structure.
- Tombstones keep costing memory and search time (they are expanded but
  never returned) until `compact()` runs. The more of the graph is
  tombstoned, the more nodes a search must expand to collect `ef` live
  results, so search time grows with the tombstone share; `deleted_len()`
  next to `len()` tells callers when compaction is worth its cost.
- `compact()` costs about as much as building a new index of the live size,
  and until it returns the old and the new index exist side by side (the
  live vectors are copied once, straight into the new index). It also
  re-samples node levels and is not reproducible bit-for-bit, like any
  parallel build.
- In-place repair (option 2) stays on the roadmap; tombstones are the
  prerequisite for it anyway.
