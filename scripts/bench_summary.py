#!/usr/bin/env python3
"""Summarize several `hnsw-bench --out DIR` runs as Markdown tables.

Each figure is shown as the median over the runs with the range in
parentheses, so run-to-run noise on a busy machine is visible instead of
hidden behind one number. Recall is shown once when it is identical in every
run (sequential builds are deterministic) and as a median and range when it
is not (parallel builds depend on thread scheduling).

Uses only the standard library:

    python3 scripts/bench_summary.py docs/bench/run-1 docs/bench/run-2 docs/bench/run-3
"""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def qps(value: float) -> str:
    """Two significant digits: queries per second are only that precise here."""
    if value >= 10_000:
        return f"{value / 1000:.0f}k"
    if value >= 1_000:
        return f"{value / 1000:.1f}k"
    return f"{value:.0f}"


def seconds(value: float) -> str:
    return f"{value:.1f}"


def spread(values: list[float], fmt) -> str:
    """`median (min-max)`, or a single value if all runs agree."""
    lo, mid, hi = min(values), statistics.median(values), max(values)
    if fmt(lo) == fmt(hi):
        return fmt(mid)
    return f"{fmt(mid)} ({fmt(lo)}–{fmt(hi)})"


def recall(values: list[float]) -> str:
    return spread(values, lambda v: f"{v:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path, help="directories written by hnsw-bench --out")
    args = parser.parse_args()

    builds = defaultdict(lambda: defaultdict(list))
    for run in args.runs:
        for row in read_csv(run / "builds.csv"):
            key = (row["dataset"], row["build"], row["threads"])
            builds[key]["wall"].append(float(row["wall_seconds"]))
            if row["cpu_seconds"] != "n/a":
                builds[key]["cpu"].append(float(row["cpu_seconds"]))
            builds[key]["heap"].append(float(row["heap_bytes_per_vector"]))

    print(f"Summary of {len(args.runs)} runs: median (min–max).\n")
    print("| dataset | build | threads | wall time (s) | CPU time (s) | heap bytes / vector |")
    print("|---|---|---:|---:|---:|---:|")
    for (dataset, build, threads), m in builds.items():
        cpu = spread(m["cpu"], seconds) if m["cpu"] else "n/a"
        heap = spread(m["heap"], lambda v: f"{v:.0f}")
        print(f"| {dataset} | {build} | {threads} | {spread(m['wall'], seconds)} | {cpu} | {heap} |")

    searches = defaultdict(lambda: defaultdict(list))
    header = read_csv(args.runs[0] / "results.csv")[0].keys()
    multi = next(h for h in header if h.startswith("qps_") and h.endswith("_threads_wall"))
    threads = multi.removeprefix("qps_").removesuffix("_threads_wall")
    recall_column = next(h for h in header if h.startswith("recall_at_"))
    k = recall_column.removeprefix("recall_at_")
    for run in args.runs:
        for row in read_csv(run / "results.csv"):
            key = (row["dataset"], row["index"], row["distances"], int(row["ef"]))
            searches[key]["recall"].append(float(row[recall_column]))
            if row["qps_1_thread_cpu"] != "n/a":
                searches[key]["cpu"].append(float(row["qps_1_thread_cpu"]))
            searches[key]["wall1"].append(float(row["qps_1_thread_wall"]))
            searches[key]["multi"].append(float(row[multi]))

    print()
    print(
        f"| dataset | index | distances | ef | recall@{k} | QPS per CPU-second, 1 thread "
        f"| QPS, 1 thread (wall) | QPS, {threads} threads (wall) |"
    )
    print("|---|---|---|---:|---:|---:|---:|---:|")
    for (dataset, index, distances, ef), m in searches.items():
        cpu = spread(m["cpu"], qps) if m["cpu"] else "n/a"
        print(
            f"| {dataset} | {index} | {distances} | {ef} | {recall(m['recall'])} | {cpu} "
            f"| {spread(m['wall1'], qps)} | {spread(m['multi'], qps)} |"
        )


if __name__ == "__main__":
    main()
