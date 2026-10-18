#!/usr/bin/env python3
"""Measure hnswlib's recall and speed on the exact datasets hnsw-index generates.

This is a cross-check of the Rust implementation against the reference C++
implementation by the paper's first author (https://github.com/nmslib/hnswlib).
It reimplements `hnsw_index::synthetic::Dataset` (SplitMix64 + Box-Muller), so
both libraries see the same vectors, and prints recall@k per (M, ef) together
with single-threaded queries per CPU-second (process CPU time, measured the
same way as the "per CPU-second" column of `hnsw-bench`).

The recall figures are comparable on any machine. The speed figures are only
a fair comparison on x86: hnswlib 0.8.0 has hand-written SIMD distance
functions for SSE/AVX/AVX-512 only, so on arm64 it runs scalar loops.

hnswlib and numpy are not dependencies of this repository; install them in a
throwaway virtual environment:

    uv venv .venv && uv pip install --python .venv/bin/python numpy hnswlib
    .venv/bin/python scripts/compare_hnswlib.py --n 10000 --queries 200 --dim 128

Compare with the `recall@10` lines printed by
`cargo test --release --test recall -- --nocapture`.
"""

import argparse
import importlib.metadata
import math
import time

import hnswlib
import numpy as np

MASK = (1 << 64) - 1


class SplitMix64:
    """Same generator and transforms as src/synthetic.rs."""

    def __init__(self, seed: int) -> None:
        self.state = seed & MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def next_f64(self) -> float:
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))

    def next_below(self, n: int) -> int:
        return (self.next_u64() * n) >> 64

    def next_gaussian(self) -> float:
        u1 = 1.0 - self.next_f64()
        u2 = self.next_f64()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(math.tau * u2)


def gaussian(n: int, queries: int, dim: int, seed: int):
    rng = SplitMix64(seed)

    def sample(count: int) -> np.ndarray:
        values = [rng.next_gaussian() for _ in range(count * dim)]
        return np.array(values, dtype=np.float32).reshape(count, dim)

    return sample(n), sample(queries)


def clustered(n: int, queries: int, dim: int, clusters: int, seed: int):
    spread = 0.35
    rng = SplitMix64(seed)
    centres = [rng.next_gaussian() for _ in range(clusters * dim)]

    def sample(count: int) -> np.ndarray:
        out = []
        for _ in range(count):
            c = rng.next_below(clusters)
            centre = centres[c * dim : (c + 1) * dim]
            out.extend(x + spread * rng.next_gaussian() for x in centre)
        return np.array(out, dtype=np.float32).reshape(count, dim)

    return sample(n), sample(queries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=["gaussian", "clustered"], default="gaussian")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--clusters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--m", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--ef", type=int, nargs="+", default=[16, 32, 64, 100, 128, 256])
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if args.dataset == "gaussian":
        base, queries = gaussian(args.n, args.queries, args.dim, args.seed)
    else:
        base, queries = clustered(args.n, args.queries, args.dim, args.clusters, args.seed)
    print(f"first base components: {base[0, :3]}")

    # Exact ground truth (squared L2), in blocks to bound memory.
    truth = []
    for start in range(0, len(queries), 64):
        q = queries[start : start + 64]
        d = (q * q).sum(1)[:, None] - 2 * q @ base.T + (base * base).sum(1)[None, :]
        truth.extend(np.argsort(d, axis=1)[:, : args.k])

    print(f"hnswlib {importlib.metadata.version('hnswlib')}")
    for m in args.m:
        index = hnswlib.Index(space="l2", dim=args.dim)
        index.init_index(max_elements=args.n, M=m, ef_construction=args.ef_construction, random_seed=100)
        index.set_num_threads(1)
        start = time.process_time()
        index.add_items(base, np.arange(args.n))
        print(f"M={m:<3} single-threaded build: {time.process_time() - start:.2f} CPU seconds")
        for ef in args.ef:
            index.set_ef(ef)
            index.knn_query(queries, k=args.k)  # warm-up
            cpu = []
            for _ in range(args.repeats):
                start = time.process_time()
                labels, _ = index.knn_query(queries, k=args.k)
                cpu.append(time.process_time() - start)
            qps = len(queries) / sorted(cpu)[len(cpu) // 2]
            recall = np.mean([len(set(l) & set(t)) / args.k for l, t in zip(labels, truth)])
            print(f"M={m:<3} ef={ef:<4} recall@{args.k}={recall:.4f} qps_per_cpu_second={qps:.0f}")


if __name__ == "__main__":
    main()
