# ADR 0003: Versioned, checksummed binary format with a validating loader

- Status: accepted
- Date: 2026-10-18

## Context

Indexes are expensive to build, so they need to be saved and loaded. Index
files get copied between machines, truncated by full disks and, in a service,
may come from places the loading process does not control. The loader must
therefore treat its input as untrusted: any byte sequence must produce either
a valid index or an error, never a panic, out-of-bounds access or an
allocation far larger than the input.

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
- the node count checked against the remaining input before any per-node
  allocation, and the level column checked for statistical plausibility under
  the stored `mL` before adjacency storage (whose size depends on levels) is
  allocated;
- every float finite, cosine vectors of unit length, tombstone flags 0 or 1,
  live keys unique, neighbour counts within capacity, neighbour ids in range,
  not self-loops and present on the layer they are linked on, and the entry
  point on the top level;
- no trailing bytes.

The whole input is read into memory before parsing (`read_from` calls
`read_to_end`), which keeps parsing a bounds-checked slice walk.

## Consequences

- Property tests flip random bytes (and recompute the CRC so the structural
  checks are what gets exercised) and truncate files at random offsets; none
  may panic, and any index that loads must be searchable.
- Loading needs the file size in memory on top of the index; streaming parsing
  is possible later without changing the format.
- Saving a just-loaded index reproduces the input byte for byte, which the
  tests use as a round-trip oracle.
- Changes to the layout require a version bump; the loader rejects versions it
  does not know.
