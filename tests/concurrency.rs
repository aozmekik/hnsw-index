//! Concurrent use: shared searches and parallel construction.

use std::sync::Arc;
use std::thread;

use hnsw_index::eval::recall_at_k;
use hnsw_index::synthetic::Dataset;
use hnsw_index::{BruteForceIndex, HnswIndex, HnswParams, Metric};

fn keys(n: usize) -> Vec<u64> {
    (0..n as u64).collect()
}

fn sequential_index(data: &Dataset) -> HnswIndex {
    let mut index = HnswIndex::new(data.dim(), Metric::L2, HnswParams::default()).unwrap();
    for i in 0..data.base_len() {
        index.insert(i as u64, data.base_vector(i)).unwrap();
    }
    index
}

#[test]
fn concurrent_searches_match_sequential_results() {
    let data = Dataset::gaussian(3000, 64, 32, 21);
    let index = sequential_index(&data);
    let expected: Vec<_> = (0..data.query_len())
        .map(|q| index.search(data.query(q), 10, 48).unwrap())
        .collect();
    thread::scope(|s| {
        for t in 0..8 {
            let (index, data, expected) = (&index, &data, &expected);
            s.spawn(move || {
                // Each thread walks the queries in a different order.
                for i in 0..data.query_len() {
                    let q = (i * 7 + t) % data.query_len();
                    let got = index.search(data.query(q), 10, 48).unwrap();
                    assert_eq!(got, expected[q], "thread {t}, query {q}");
                }
            });
        }
    });
}

#[test]
fn index_can_be_shared_through_an_arc() {
    let data = Dataset::gaussian(500, 4, 8, 2);
    let index = Arc::new(sequential_index(&data));
    let handles: Vec<_> = (0..4)
        .map(|t| {
            let index = Arc::clone(&index);
            let query = data.query(t).to_vec();
            thread::spawn(move || index.search(&query, 5, 32).unwrap().len())
        })
        .collect();
    for h in handles {
        assert_eq!(h.join().unwrap(), 5);
    }
}

#[test]
fn search_batch_equals_individual_searches() {
    let data = Dataset::gaussian(2000, 50, 16, 5);
    let index = sequential_index(&data);
    let batch = index.search_batch(data.queries(), 10, 40).unwrap();
    for (q, hits) in batch.iter().enumerate() {
        assert_eq!(hits, &index.search(data.query(q), 10, 40).unwrap());
    }
    assert!(index.search_batch(&data.queries()[..15], 10, 40).is_err());
}

#[test]
fn parallel_builds_are_complete_and_accurate_for_any_thread_count() {
    let data = Dataset::clustered(4000, 100, 32, 16, 13);
    let mut exact = BruteForceIndex::new(data.dim(), Metric::L2).unwrap();
    exact
        .insert_batch(&keys(data.base_len()), data.base())
        .unwrap();
    let truth = exact.search_batch(data.queries(), 10).unwrap();
    for threads in [1, 2, 3, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let mut index = HnswIndex::new(data.dim(), Metric::L2, HnswParams::default()).unwrap();
        // Insert in two batches so the second one links against an existing graph.
        let half = data.base_len() / 2;
        let (first, second) = data.base().split_at(half * data.dim());
        pool.install(|| {
            index.insert_batch(&keys(half), first).unwrap();
            let rest: Vec<u64> = (half as u64..data.base_len() as u64).collect();
            index.insert_batch(&rest, second).unwrap();
        });
        assert_eq!(index.len(), data.base_len());
        assert!((0..data.base_len() as u64).all(|k| index.contains(k)));
        let found = index.search_batch(data.queries(), 10, 64).unwrap();
        let recall = recall_at_k(&truth, &found, 10);
        assert!(recall > 0.95, "{threads} threads: recall {recall}");
    }
}
