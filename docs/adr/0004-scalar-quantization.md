# ADR 0004: Symmetric int8 quantization for traversal, exact re-ranking

- Status: accepted
- Date: 2026-10-18

## Context

Search time is dominated by distance computations on vectors scattered in
memory. For 100k × 128-d `f32` vectors the data is about 51 MB, larger than
the CPU caches, so many distance computations wait on memory. Smaller codes
mean fewer cache misses and less arithmetic.

## Options considered

1. **Product quantization (PQ).** Highest compression, but needs k-means
   training, lookup tables per query, and a substantially larger code base.
2. **Per-dimension scale and offset with asymmetric distances** (`f32` query
   against `u8` codes). Accurate, but every term needs a widening conversion
   and a per-dimension weight in the inner loop.
3. **Symmetric int8 with one global scale.** Query and data are quantized the
   same way; distances are exact integer arithmetic on `i8` codes, rescaled
   by `scale²`.

## Decision

Option 3. `code = round((x − offset) / scale)` clamped to `−127..=127`, with
`scale = max |x − offset| / 127` over the stored vectors. For L2 the offset is
the per-dimension mean (translation does not change Euclidean distances); for
inner product and cosine it is zero, because centering changes how
`⟨q − μ, x − μ⟩` ranks candidates. The codes are only used to navigate the
graph. The graph itself is built with `f32` distances, and the `ef` final
candidates are re-ranked with exact `f32` distances, so reported distances are
always exact. The `f32` vectors are kept for that.

## Consequences

- Quantization adds `dim` bytes per vector (about 25% for `f32` vectors)
  rather than saving memory; its benefit is speed. Measured numbers are in
  `docs/benchmarks.md`.
- One global scale is a poor fit when dimensions have very different ranges.
  The synthetic benchmarks have equal variance per dimension, which favors
  this scheme; real embeddings should be measured before relying on it.
- Vectors inserted after `enable_quantization()` are encoded with the existing
  scale and clamped if out of range; calling it again retrains.
- The quantizer parameters (offsets and scale) are saved in the index file;
  codes are recomputed on load.
