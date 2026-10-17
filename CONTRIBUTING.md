# Contributing

Thanks for taking a look. Issues and pull requests are welcome; for larger
changes, please open an issue first to discuss the design.

## Toolchain

Stable Rust, edition 2024. The minimum supported Rust version is 1.88
(`rust-version` in `Cargo.toml`); CI checks it.

## Build, test, lint

```sh
cargo build --all-targets
cargo test --all-targets          # unit, integration and property tests
cargo test --doc                  # rustdoc examples
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
```

The `test` profile is optimised (`opt-level = 3`, debug assertions kept),
because the recall tests build 10k-vector indexes.

Lints are configured in `Cargo.toml` (`[lints]`): Clippy's `pedantic` group is
on, `unsafe` is denied outside the SIMD kernels and the benchmark allocator,
and every `unsafe` block needs a `// SAFETY:` comment.

## Examples and benchmarks

```sh
cargo run --release --example quickstart        # also: filtered_search, persistence, quantization
cargo run --release --bin hnsw-bench -- --help
cargo run --release --bin hnsw-bench -- --quick # smoke run, about a minute
cargo bench --bench kernels                     # Criterion: distance kernels per backend
cargo bench --bench search                      # Criterion: single-query latency
```

The full benchmark in the README is

```sh
cargo run --release --bin hnsw-bench -- --threads 8 --out docs/bench
```

When you change construction or search code, run the recall tests with
output to see the numbers they assert on:

```sh
cargo test --release --test recall -- --nocapture
```

`scripts/compare_hnswlib.py` measures hnswlib on exactly the same synthetic
data (instructions in the script); keep recall within noise of it.

## SIMD kernels

New kernels go in `src/kernels/`. Every backend must be reachable through
`Kernels::new(backend)`, gated by `cfg(target_arch)` and runtime feature
detection, and is then covered automatically by `tests/kernels.rs`, which
compares all available backends with the scalar reference. Only x86_64 CI
runners execute the AVX2 path and only arm64 runners the NEON path.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/) (`feat:`,
`fix:`, `docs:`, `test:`, `ci:`, `chore:`, ...). Update `CHANGELOG.md` under
`[Unreleased]` for user-visible changes.
