//! Recall regression tests on fixed-seed synthetic data.
//!
//! The datasets are synthetic (see `hnsw_index::synthetic`), so these numbers
//! say nothing about real embeddings; they exist to catch regressions in
//! graph construction and search.

use hnsw_index::eval::recall_at_k;
use hnsw_index::synthetic::Dataset;
use hnsw_index::{BruteForceIndex, HnswIndex, HnswParams, Metric, SearchOptions};

const K: usize = 10;
const EF: usize = 100;

fn ground_truth(data: &Dataset, metric: Metric) -> Vec<Vec<hnsw_index::Neighbor>> {
    let keys: Vec<u64> = (0..data.base_len() as u64).collect();
    let mut exact = BruteForceIndex::new(data.dim(), metric).unwrap();
    exact.insert_batch(&keys, data.base()).unwrap();
    exact.search_batch(data.queries(), K).unwrap()
}

/// Sequential insertion: the graph is a pure function of the data and seed,
/// so the measured recall is identical on every run.
fn sequential(data: &Dataset, metric: Metric, m: usize) -> HnswIndex {
    let mut index = HnswIndex::new(data.dim(), metric, HnswParams::new(m)).unwrap();
    for i in 0..data.base_len() {
        index.insert(i as u64, data.base_vector(i)).unwrap();
    }
    index
}

fn recall(index: &HnswIndex, data: &Dataset, truth: &[Vec<hnsw_index::Neighbor>]) -> f64 {
    let found = index.search_batch(data.queries(), K, EF).unwrap();
    recall_at_k(truth, &found, K)
}

fn gaussian_10k() -> Dataset {
    Dataset::gaussian(10_000, 200, 128, 42)
}

/// Isotropic 128-d Gaussian data is a hard case for any graph index. With
/// `M = 32` the paper's algorithm reaches the 0.95 bar at `ef = 100`.
#[test]
fn gaussian_10k_recall_at_10_is_at_least_095_with_m32() {
    let data = gaussian_10k();
    let truth = ground_truth(&data, Metric::L2);
    let index = sequential(&data, Metric::L2, 32);
    let r = recall(&index, &data, &truth);
    eprintln!("gaussian 10k x 128, M=32, sequential build: recall@10 = {r:.4}");
    assert!(r >= 0.95, "recall@10 = {r}");
}

/// With the default `M = 16` the same data stays below 0.95 at `ef = 100`;
/// hnswlib measures 0.889 on this exact dataset (`scripts/compare_hnswlib.py`),
/// so this pins parity rather than an arbitrary target.
#[test]
fn gaussian_10k_m16_matches_reference_implementation() {
    let data = gaussian_10k();
    let truth = ground_truth(&data, Metric::L2);
    let index = sequential(&data, Metric::L2, 16);
    let r = recall(&index, &data, &truth);
    eprintln!("gaussian 10k x 128, M=16, sequential build: recall@10 = {r:.4}");
    assert!(r >= 0.87, "recall@10 = {r}");
}

#[test]
fn clustered_10k_recall_at_10_is_at_least_095() {
    let data = Dataset::clustered(10_000, 200, 128, 50, 42);
    let truth = ground_truth(&data, Metric::L2);
    let index = sequential(&data, Metric::L2, 16);
    let r = recall(&index, &data, &truth);
    eprintln!("clustered 10k x 128, sequential build: recall@10 = {r:.4}");
    assert!(r >= 0.95, "recall@10 = {r}");
}

#[test]
fn cosine_and_inner_product_recall() {
    let data = Dataset::gaussian(5_000, 100, 64, 7);
    for metric in [Metric::Cosine, Metric::InnerProduct] {
        let truth = ground_truth(&data, metric);
        let index = sequential(&data, metric, 16);
        let r = recall(&index, &data, &truth);
        eprintln!("gaussian 5k x 64, {metric}: recall@10 = {r:.4}");
        assert!(r >= 0.9, "{metric}: recall@10 = {r}");
    }
}

/// Parallel insertion is not bit-for-bit reproducible, so this asserts a
/// floor with a wide margin below the sequential result rather than equality.
#[test]
fn parallel_build_matches_sequential_quality() {
    let data = gaussian_10k();
    let truth = ground_truth(&data, Metric::L2);
    let keys: Vec<u64> = (0..data.base_len() as u64).collect();
    let mut index = HnswIndex::new(data.dim(), Metric::L2, HnswParams::new(32)).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    pool.install(|| index.insert_batch(&keys, data.base()))
        .unwrap();
    let r = recall(&index, &data, &truth);
    eprintln!("gaussian 10k x 128, M=32, 4-thread build: recall@10 = {r:.4}");
    assert!(r >= 0.94, "recall@10 = {r}");
}

#[test]
fn quantized_search_with_reranking_keeps_recall() {
    let data = gaussian_10k();
    let truth = ground_truth(&data, Metric::L2);
    let mut index = sequential(&data, Metric::L2, 32);
    index.enable_quantization();
    let options = SearchOptions::new(EF).quantized(true);
    let found: Vec<_> = (0..data.query_len())
        .map(|q| index.search_with(data.query(q), K, &options).unwrap())
        .collect();
    let r = recall_at_k(&truth, &found, K);
    eprintln!("gaussian 10k x 128, M=32, int8 traversal + re-rank: recall@10 = {r:.4}");
    assert!(r >= 0.93, "recall@10 = {r}");
}

#[test]
fn dataset_matches_the_python_reference_generator() {
    // First components of `Dataset::gaussian(10_000, 200, 128, 42)` as
    // produced by scripts/compare_hnswlib.py, which reimplements the generator.
    let data = gaussian_10k();
    let expected = [0.882_248_9_f32, -0.450_849_86, 0.188_352_63];
    for (got, want) in data.base()[..3].iter().zip(expected) {
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
    }
}
