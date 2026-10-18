//! `hnsw-bench`: build time, memory and recall/throughput sweeps on seeded
//! synthetic data.
//!
//! Run `hnsw-bench --help` for options. Results are printed as Markdown tables;
//! with `--out DIR` they are also written as CSV (`results.csv` for searches,
//! `builds.csv` for builds) and SVG charts.

#![allow(unsafe_code)] // the counting global allocator in `alloc.rs`

mod alloc;
mod chart;
mod cpu;

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use hnsw_index::eval::recall_at_k;
use hnsw_index::synthetic::Dataset;
use hnsw_index::{
    BruteForceIndex, HnswIndex, HnswParams, Metric, Neighbor, NeighborSelection, SearchOptions,
};
use rayon::prelude::*;

#[global_allocator]
static ALLOCATOR: alloc::Counting = alloc::Counting;

const USAGE: &str = "\
hnsw-bench: HNSW benchmarks on seeded synthetic data

USAGE:
    hnsw-bench [OPTIONS]

OPTIONS:
    --dataset <LIST>          gaussian, clustered or both, comma-separated [default: gaussian,clustered]
    --n <N>                   base vectors [default: 100000]
    --queries <N>             query vectors, drawn after (so disjoint from) the base set [default: 1000]
    --dim <D>                 dimension [default: 128]
    --clusters <C>            mixture components for the clustered dataset [default: 100]
    --k <K>                   neighbors per query [default: 10]
    --m <M>                   HNSW M [default: 16]
    --ef-construction <EF>    HNSW efConstruction [default: 200]
    --ef <LIST>               search ef values, comma-separated [default: 16,32,64,128,256]
    --threads <T>             threads for parallel build and search [default: all cores]
    --seed <S>                dataset seed [default: 42]
    --repeats <R>             timed repetitions per measurement; the median is reported [default: 3]
    --skip-sequential         skip the single-threaded build (the slowest step)
    --skip-simple             skip the simple-selection comparison build
    --out <DIR>               also write results.md, results.csv, builds.csv and SVG charts into DIR
    --quick                   small smoke run: --n 20000 --queries 200 --repeats 1
    -h, --help                print this help
";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Gaussian,
    Clustered,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Self::Gaussian => "gaussian",
            Self::Clustered => "clustered",
        }
    }
}

#[derive(Debug, Clone)]
struct Config {
    datasets: Vec<Kind>,
    n: usize,
    queries: usize,
    dim: usize,
    clusters: usize,
    k: usize,
    m: usize,
    ef_construction: usize,
    efs: Vec<usize>,
    threads: usize,
    seed: u64,
    repeats: usize,
    sequential: bool,
    simple: bool,
    out: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            datasets: vec![Kind::Gaussian, Kind::Clustered],
            n: 100_000,
            queries: 1000,
            dim: 128,
            clusters: 100,
            k: 10,
            m: 16,
            ef_construction: 200,
            efs: vec![16, 32, 64, 128, 256],
            threads: std::thread::available_parallelism().map_or(1, usize::from),
            seed: 42,
            repeats: 3,
            sequential: true,
            simple: true,
            out: None,
        }
    }
}

fn parse_list<T: std::str::FromStr>(value: &str, flag: &str) -> Result<Vec<T>, String> {
    value
        .split(',')
        .map(|v| {
            v.trim()
                .parse()
                .map_err(|_| format!("invalid value {v:?} for {flag}"))
        })
        .collect()
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Option<Config>, String> {
    let mut config = Config::default();
    let mut args = args.into_iter();
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
        let number = |v: String, flag: &str| -> Result<usize, String> {
            v.parse::<usize>()
                .ok()
                .filter(|&n| n > 0)
                .ok_or_else(|| format!("{flag} needs a positive integer, got {v:?}"))
        };
        match flag.as_str() {
            "-h" | "--help" => return Ok(None),
            "--dataset" => {
                config.datasets = value()?
                    .split(',')
                    .map(|d| match d.trim() {
                        "gaussian" => Ok(Kind::Gaussian),
                        "clustered" => Ok(Kind::Clustered),
                        other => Err(format!("unknown dataset {other:?}")),
                    })
                    .collect::<Result<_, _>>()?;
            }
            "--n" => config.n = number(value()?, &flag)?,
            "--queries" => config.queries = number(value()?, &flag)?,
            "--dim" => config.dim = number(value()?, &flag)?,
            "--clusters" => config.clusters = number(value()?, &flag)?,
            "--k" => config.k = number(value()?, &flag)?,
            "--m" => config.m = number(value()?, &flag)?,
            "--ef-construction" => config.ef_construction = number(value()?, &flag)?,
            "--ef" => config.efs = parse_list(&value()?, &flag)?,
            "--threads" => config.threads = number(value()?, &flag)?,
            "--seed" => {
                config.seed = value()?
                    .parse()
                    .map_err(|_| "--seed needs an integer".to_string())?;
            }
            "--repeats" => config.repeats = number(value()?, &flag)?,
            "--skip-sequential" => config.sequential = false,
            "--skip-simple" => config.simple = false,
            "--out" => config.out = Some(PathBuf::from(value()?)),
            "--quick" => {
                config.n = 20_000;
                config.queries = 200;
                config.repeats = 1;
            }
            other => return Err(format!("unknown option {other:?} (see --help)")),
        }
    }
    if config.efs.is_empty() || config.efs.contains(&0) {
        return Err("--ef values must be positive".into());
    }
    // Check M, efConstruction and the dimension the way the index will.
    let params = HnswParams::new(config.m).with_ef_construction(config.ef_construction);
    HnswIndex::new(config.dim, Metric::L2, params).map_err(|e| e.to_string())?;
    Ok(Some(config))
}

/// Wall-clock and process CPU seconds of one measurement.
#[derive(Debug, Clone, Copy)]
struct Timing {
    wall: f64,
    cpu: Option<f64>,
}

/// Runs `f` `repeats` times and returns the result of the first run with the
/// median wall-clock and median CPU time (medians taken independently).
fn timed<R>(repeats: usize, mut f: impl FnMut() -> R) -> (R, Timing) {
    let mut first = None;
    let mut walls = Vec::with_capacity(repeats);
    let mut cpus = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let cpu_start = cpu::cpu_seconds();
        let start = Instant::now();
        let out = f();
        walls.push(start.elapsed().as_secs_f64());
        if let (Some(a), Some(b)) = (cpu_start, cpu::cpu_seconds()) {
            cpus.push(b - a);
        }
        first.get_or_insert(out);
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let timing = Timing {
        wall: median(&mut walls),
        cpu: (cpus.len() == repeats).then(|| median(&mut cpus)),
    };
    (first.expect("repeats > 0"), timing)
}

struct BuildRow {
    dataset: Kind,
    label: String,
    threads: usize,
    seconds: f64,
    cpu_seconds: Option<f64>,
    heap_bytes: usize,
    estimated_bytes: usize,
}

struct SearchRow {
    dataset: Kind,
    index: String,
    distances: &'static str,
    ef: usize,
    recall: f64,
    qps_single: f64,
    /// Single-thread queries per second of process CPU time.
    qps_single_cpu: Option<f64>,
    qps_multi: f64,
}

struct Bench<'a> {
    config: &'a Config,
    pool: rayon::ThreadPool,
    builds: Vec<BuildRow>,
    searches: Vec<SearchRow>,
    brute_qps: Vec<(Kind, f64)>,
    /// Index whose quantized variant was measured; charted with it.
    primary_label: &'static str,
    /// 1-minute load average when the run started.
    load_at_start: Option<f64>,
}

impl Bench<'_> {
    fn params(&self, selection: NeighborSelection) -> HnswParams {
        HnswParams::new(self.config.m)
            .with_ef_construction(self.config.ef_construction)
            .with_selection(selection)
    }

    fn build(
        &mut self,
        kind: Kind,
        data: &Dataset,
        label: &str,
        selection: NeighborSelection,
        parallel: bool,
    ) -> HnswIndex {
        let keys: Vec<u64> = (0..data.base_len() as u64).collect();
        let mut index = HnswIndex::new(data.dim(), Metric::L2, self.params(selection))
            .expect("valid parameters");
        eprintln!("[{}] building: {label}", kind.name());
        let before = alloc::live_bytes();
        let ((), timing) = timed(1, || {
            if parallel {
                self.pool
                    .install(|| index.insert_batch(&keys, data.base()))
                    .expect("valid data");
            } else {
                index.reserve(keys.len());
                for (i, &key) in keys.iter().enumerate() {
                    index.insert(key, data.base_vector(i)).expect("valid data");
                }
            }
        });
        let heap_bytes = alloc::live_bytes().saturating_sub(before);
        self.builds.push(BuildRow {
            dataset: kind,
            label: label.to_string(),
            threads: if parallel { self.config.threads } else { 1 },
            seconds: timing.wall,
            cpu_seconds: timing.cpu,
            heap_bytes,
            estimated_bytes: index.memory_usage(),
        });
        index
    }

    fn sweep(
        &mut self,
        kind: Kind,
        data: &Dataset,
        truth: &[Vec<Neighbor>],
        index: &HnswIndex,
        label: &str,
        quantized: bool,
    ) {
        let (k, dim, repeats) = (self.config.k, data.dim(), self.config.repeats);
        let nq = data.query_len() as f64;
        for &ef in &self.config.efs {
            let options = SearchOptions::new(ef).quantized(quantized);
            let search = |q: &[f32]| index.search_with(q, k, &options).expect("valid query");
            // Warm-up pass, also used for recall.
            let found: Vec<Vec<Neighbor>> = data.queries().chunks_exact(dim).map(search).collect();
            let recall = recall_at_k(truth, &found, k);
            let (_, single) = timed(repeats, || {
                data.queries()
                    .chunks_exact(dim)
                    .map(search)
                    .map(|hits| hits.len())
                    .sum::<usize>()
            });
            let (_, multi) = timed(repeats, || {
                self.pool.install(|| {
                    data.queries()
                        .par_chunks_exact(dim)
                        .map(search)
                        .map(|hits| hits.len())
                        .sum::<usize>()
                })
            });
            let qps_single_cpu = single.cpu.map(|cpu| nq / cpu);
            eprintln!(
                "[{}] {label}, {}: ef={ef} recall={recall:.4} qps1={:.0} qps1/cpu={:.0} qps{}={:.0}",
                kind.name(),
                if quantized { "int8" } else { "f32" },
                nq / single.wall,
                qps_single_cpu.unwrap_or(f64::NAN),
                self.config.threads,
                nq / multi.wall
            );
            self.searches.push(SearchRow {
                dataset: kind,
                index: label.to_string(),
                distances: if quantized {
                    "int8 + f32 re-rank"
                } else {
                    "f32"
                },
                ef,
                recall,
                qps_single: nq / single.wall,
                qps_single_cpu,
                qps_multi: nq / multi.wall,
            });
        }
    }

    fn run_dataset(&mut self, kind: Kind) {
        let c = self.config;
        eprintln!(
            "[{}] generating {} + {} vectors",
            kind.name(),
            c.n,
            c.queries
        );
        let data = match kind {
            Kind::Gaussian => Dataset::gaussian(c.n, c.queries, c.dim, c.seed),
            Kind::Clustered => Dataset::clustered(c.n, c.queries, c.dim, c.clusters, c.seed),
        };
        let keys: Vec<u64> = (0..data.base_len() as u64).collect();
        let mut exact = BruteForceIndex::new(data.dim(), Metric::L2).expect("valid dim");
        exact.insert_batch(&keys, data.base()).expect("valid data");
        let (truth, timing) = timed(1, || {
            self.pool
                .install(|| exact.search_batch(data.queries(), c.k))
                .expect("valid queries")
        });
        drop(exact);
        self.brute_qps
            .push((kind, data.query_len() as f64 / timing.wall));

        let heuristic = NeighborSelection::default();
        let parallel = self.build(kind, &data, "heuristic, parallel", heuristic, true);
        self.sweep(kind, &data, &truth, &parallel, "heuristic, parallel", false);
        let (mut primary, label) = if c.sequential {
            drop(parallel);
            let sequential = self.build(kind, &data, "heuristic, sequential", heuristic, false);
            self.sweep(
                kind,
                &data,
                &truth,
                &sequential,
                "heuristic, sequential",
                false,
            );
            (sequential, "heuristic, sequential")
        } else {
            (parallel, "heuristic, parallel")
        };
        primary.enable_quantization();
        self.sweep(kind, &data, &truth, &primary, label, true);
        drop(primary);
        if c.simple {
            let simple = self.build(
                kind,
                &data,
                "simple, parallel",
                NeighborSelection::Simple,
                true,
            );
            self.sweep(kind, &data, &truth, &simple, "simple, parallel", false);
        }
        self.primary_label = label;
    }
}

fn markdown(config: &Config, bench: &Bench<'_>) -> String {
    let mut out = String::new();
    let t = config.threads;
    let _ = writeln!(
        out,
        "Synthetic data: {} base + {} disjoint queries, dim {}, seed {}; M={}, efConstruction={}, k={}; \
         {} thread(s) for parallel runs; median of {} timed run(s); kernel backend: {}; {} {}; \
         1-minute load average {} at the start and {} at the end of the run. \
         Wall-clock throughput depends on what else runs on the machine; CPU-time columns count only \
         this process's CPU time.\n",
        config.n,
        config.queries,
        config.dim,
        config.seed,
        config.m,
        config.ef_construction,
        config.k,
        t,
        config.repeats,
        hnsw_index::kernels::Kernels::best().backend(),
        std::env::consts::OS,
        std::env::consts::ARCH,
        fmt_opt(bench.load_at_start, 1),
        fmt_opt(cpu::load_average(), 1),
    );
    out.push_str("| dataset | build | threads | wall time (s) | CPU time (s) | heap bytes / vector (measured) | memory_usage() / vector |\n");
    out.push_str("|---|---|---:|---:|---:|---:|---:|\n");
    for b in &bench.builds {
        let _ = writeln!(
            out,
            "| {} | {} | {} | {:.2} | {} | {:.0} | {:.0} |",
            b.dataset.name(),
            b.label,
            b.threads,
            b.seconds,
            fmt_opt(b.cpu_seconds, 2),
            b.heap_bytes as f64 / config.n as f64,
            b.estimated_bytes as f64 / config.n as f64,
        );
    }
    let _ = writeln!(
        out,
        "\n| dataset | index | distances | ef | recall@{} | QPS, 1 thread (wall) | QPS, 1 thread (per CPU-second) | QPS, {t} threads (wall) |",
        config.k
    );
    out.push_str("|---|---|---|---:|---:|---:|---:|---:|\n");
    for s in &bench.searches {
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {:.4} | {:.0} | {} | {:.0} |",
            s.dataset.name(),
            s.index,
            s.distances,
            s.ef,
            s.recall,
            s.qps_single,
            fmt_opt(s.qps_single_cpu, 0),
            s.qps_multi
        );
    }
    out.push_str("\n| dataset | exact brute force QPS |\n|---|---:|\n");
    for (kind, qps) in &bench.brute_qps {
        let _ = writeln!(out, "| {} | {qps:.0} ({t} threads) |", kind.name());
    }
    out
}

fn fmt_opt(value: Option<f64>, decimals: usize) -> String {
    value.map_or_else(|| "n/a".to_string(), |v| format!("{v:.decimals$}"))
}

fn csv(config: &Config, bench: &Bench<'_>) -> String {
    let mut out = format!(
        "dataset,index,distances,ef,recall_at_{},qps_1_thread_wall,qps_1_thread_cpu,qps_{}_threads_wall\n",
        config.k, config.threads
    );
    for s in &bench.searches {
        let _ = writeln!(
            out,
            "{},\"{}\",{},{},{:.4},{:.1},{},{:.1}",
            s.dataset.name(),
            s.index,
            s.distances,
            s.ef,
            s.recall,
            s.qps_single,
            fmt_opt(s.qps_single_cpu, 1),
            s.qps_multi
        );
    }
    out
}

fn builds_csv(config: &Config, bench: &Bench<'_>) -> String {
    let mut out = String::from(
        "dataset,build,threads,wall_seconds,cpu_seconds,heap_bytes_per_vector,memory_usage_per_vector\n",
    );
    for b in &bench.builds {
        let _ = writeln!(
            out,
            "{},\"{}\",{},{:.3},{},{:.1},{:.1}",
            b.dataset.name(),
            b.label,
            b.threads,
            b.seconds,
            fmt_opt(b.cpu_seconds, 3),
            b.heap_bytes as f64 / config.n as f64,
            b.estimated_bytes as f64 / config.n as f64,
        );
    }
    out
}

/// One chart per dataset and thread count: recall vs QPS for the primary
/// index (f32 and int8 traversal) and the simple-selection index.
fn charts(config: &Config, bench: &Bench<'_>) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for &kind in &config.datasets {
        for threads in [1, config.threads] {
            let mut series: Vec<chart::Series> = Vec::new();
            let charted = bench.searches.iter().filter(|s| {
                s.dataset == kind
                    && (s.index == bench.primary_label || s.index.starts_with("simple"))
            });
            for s in charted {
                let traversal = if s.distances == "f32" {
                    "f32 distances"
                } else {
                    "int8 traversal + re-rank"
                };
                let label = format!("{} build\n{traversal}", s.index);
                let qps = if threads == 1 {
                    s.qps_single_cpu.unwrap_or(s.qps_single)
                } else {
                    s.qps_multi
                };
                match series.iter_mut().find(|x| x.label == label) {
                    Some(x) => x.points.push((s.recall, qps)),
                    None => series.push(chart::Series {
                        label,
                        points: vec![(s.recall, qps)],
                    }),
                }
            }
            let plural = if threads == 1 { "" } else { "s" };
            let basis = if threads == 1 && cpu::cpu_seconds().is_some() {
                " (per CPU-second)"
            } else {
                " (wall clock)"
            };
            let title = format!(
                "{} {}k x {}-d (synthetic): recall@{} vs QPS, {threads} thread{plural}{basis}",
                kind.name(),
                config.n / 1000,
                config.dim,
                config.k
            );
            let subtitle = format!(
                "M={}, efConstruction={}, ef in {:?}; {} {}, {} kernels",
                config.m,
                config.ef_construction,
                config.efs,
                std::env::consts::OS,
                std::env::consts::ARCH,
                hnsw_index::kernels::Kernels::best().backend()
            );
            out.push((
                format!("recall-qps-{}-{threads}-thread{plural}.svg", kind.name()),
                chart::render(&title, &subtitle, &format!("recall@{}", config.k), &series),
            ));
        }
    }
    out
}

fn main() -> ExitCode {
    let config = match parse_args(std::env::args().skip(1)) {
        Ok(Some(config)) => config,
        Ok(None) => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.threads)
        .build()
        .expect("thread pool");
    let mut bench = Bench {
        config: &config,
        pool,
        builds: Vec::new(),
        searches: Vec::new(),
        brute_qps: Vec::new(),
        primary_label: "",
        load_at_start: cpu::load_average(),
    };
    for &kind in &config.datasets {
        bench.run_dataset(kind);
    }
    let report = markdown(&config, &bench);
    print!("{report}");
    if let Some(dir) = &config.out {
        let write = |name: &str, contents: &str| {
            let path = dir.join(name);
            std::fs::write(&path, contents)
                .map_err(|e| format!("cannot write {}: {e}", path.display()))
        };
        let result = std::fs::create_dir_all(dir)
            .map_err(|e| format!("cannot create {}: {e}", dir.display()))
            .and_then(|()| write("results.md", &report))
            .and_then(|()| write("results.csv", &csv(&config, &bench)))
            .and_then(|()| write("builds.csv", &builds_csv(&config, &bench)))
            .and_then(|()| {
                charts(&config, &bench)
                    .iter()
                    .try_for_each(|(name, svg)| write(name, svg))
            });
        if let Err(e) = result {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Result<Option<Config>, String> {
        parse_args(list.iter().map(ToString::to_string))
    }

    #[test]
    fn parses_options() {
        let c = args(&[
            "--n",
            "500",
            "--ef",
            "8,16",
            "--dataset",
            "clustered",
            "--skip-simple",
        ])
        .unwrap()
        .unwrap();
        assert_eq!(
            (c.n, c.efs.clone(), c.datasets.clone()),
            (500, vec![8, 16], vec![Kind::Clustered])
        );
        assert!(!c.simple && c.sequential);
        assert!(args(&["--help"]).unwrap().is_none());
    }

    #[test]
    fn rejects_bad_options() {
        assert!(args(&["--n"]).is_err());
        assert!(args(&["--n", "0"]).is_err());
        assert!(args(&["--ef", "8,x"]).is_err());
        assert!(args(&["--dataset", "sift"]).is_err());
        assert!(args(&["--frobnicate"]).is_err());
        // Out-of-range index parameters are usage errors, not panics.
        let err = args(&["--m", "1"]).unwrap_err();
        assert!(err.contains("`m`"), "{err}");
        assert!(args(&["--m", "600"]).is_err());
        assert!(args(&["--ef-construction", "100000"]).is_err());
        assert!(args(&["--dim", "100000"]).is_err());
    }

    #[test]
    fn median_of_timings() {
        let mut calls = 0;
        let (first, _) = timed(3, || {
            calls += 1;
            calls
        });
        assert_eq!((first, calls), (1, 3));
    }
}
