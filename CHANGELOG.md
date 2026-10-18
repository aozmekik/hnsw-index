# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before
1.0, minor versions may contain breaking changes.

## [Unreleased]

## [0.1.0] - 2026-10-18

### Added

- `HnswIndex`: HNSW after Malkov & Yashunin (2018) with level sampling
  (`mL = 1/ln M`), greedy upper-layer descent, ef-bounded layer-0 search, and
  simple or heuristic neighbour selection (`extendCandidates`,
  `keepPrunedConnections`); parameters `M`, `M0`, `efConstruction`.
- Squared L2, inner product and cosine metrics; scalar, auto-vectorised, NEON
  and AVX2+FMA kernels with runtime detection.
- Filtered search, tombstone deletion, `compact()`, parallel
  `insert_batch` (rayon), `reserve`, concurrent `&self` search,
  `search_batch`.
- Optional int8 scalar quantization with exact re-ranking.
- Versioned, CRC-32 checksummed file format with a validating loader that
  checks the memory a file would need against a limit before allocating it
  (`from_bytes_with_limit`, `Error::MemoryLimitExceeded`).
- `BruteForceIndex` for exact ground truth, `eval::recall_at_k`, seeded
  synthetic datasets.
- `hnsw-bench` binary, Criterion benches, examples, property tests and a
  recall regression suite.

[Unreleased]: https://github.com/aozmekik/hnsw-index/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/aozmekik/hnsw-index/releases/tag/v0.1.0
