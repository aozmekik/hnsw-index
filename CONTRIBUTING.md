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

The `test` profile is optimized (`opt-level = 3`, debug assertions kept),
because the recall tests build 10k-vector indexes.

Lints are configured in `Cargo.toml` (`[lints]`): Clippy's `pedantic` group is
on, `unsafe` is denied outside the SIMD kernels and the benchmark allocator,
and every `unsafe` block needs a `// SAFETY:` comment.

## Examples and benchmarks

```sh
cargo run --release --example quickstart        # also: filtered_search, persistence, quantization
cargo run --release --bin hnsw-bench -- --help
cargo run --release --bin hnsw-bench -- --quick # smoke run on 20k vectors
cargo bench --bench kernels                     # Criterion: distance kernels per backend
cargo bench --bench search                      # Criterion: single-query latency
```

The benchmarks in the README and [docs/benchmarks.md](docs/benchmarks.md) are
three complete runs of the same command, summarized by a script that reports
the median and range of every figure:

```sh
cargo build --release --locked
for run in 1 2 3; do
  ./target/release/hnsw-bench --threads 8 --out docs/bench/run-$run
done
python3 scripts/bench_summary.py docs/bench/run-1 docs/bench/run-2 docs/bench/run-3 > docs/bench/summary.md
```

Timings depend on what else the machine is doing; each run records the load
average at its start and end in its `results.md`.

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
