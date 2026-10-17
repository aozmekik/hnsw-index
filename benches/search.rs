//! Single-query search latency on a 10k x 128 synthetic Gaussian index.
//!
//! Run with `cargo bench --bench search`. The index is built once, outside
//! the measurement.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use hnsw_index::synthetic::Dataset;
use hnsw_index::{BruteForceIndex, HnswIndex, HnswParams, Metric, SearchOptions};

fn search(c: &mut Criterion) {
    let data = Dataset::gaussian(10_000, 100, 128, 42);
    let keys: Vec<u64> = (0..data.base_len() as u64).collect();
    let mut index = HnswIndex::new(data.dim(), Metric::L2, HnswParams::default()).unwrap();
    index.insert_batch(&keys, data.base()).unwrap();
    index.enable_quantization();
    let mut exact = BruteForceIndex::new(data.dim(), Metric::L2).unwrap();
    exact.insert_batch(&keys, data.base()).unwrap();

    let mut group = c.benchmark_group("search_10k_128d_k10");
    group
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(3));
    let mut q = 0;
    let mut next_query = || {
        q = (q + 1) % data.query_len();
        data.query(q)
    };
    for ef in [16, 64, 256] {
        group.bench_with_input(BenchmarkId::new("hnsw_f32", ef), &ef, |b, &ef| {
            b.iter(|| index.search(black_box(next_query()), 10, ef).unwrap());
        });
        let options = SearchOptions::new(ef).quantized(true);
        group.bench_with_input(BenchmarkId::new("hnsw_int8_rerank", ef), &ef, |b, _| {
            b.iter(|| {
                index
                    .search_with(black_box(next_query()), 10, &options)
                    .unwrap()
            });
        });
    }
    group.bench_function("brute_force", |b| {
        b.iter(|| exact.search(black_box(next_query()), 10).unwrap());
    });
    group.finish();
}

criterion_group!(benches, search);
criterion_main!(benches);
