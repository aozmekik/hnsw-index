//! Build an index, search it, measure recall, then delete and compact.
//!
//! Run with `cargo run --release --example quickstart`.

use hnsw_index::synthetic::Dataset;
use hnsw_index::{BruteForceIndex, Error, HnswIndex, HnswParams, Metric, eval};

fn main() -> Result<(), Error> {
    // 20k synthetic 64-d vectors (seeded, so every run sees the same data).
    let data = Dataset::clustered(20_000, 100, 64, 50, 7);
    let keys: Vec<u64> = (0..data.base_len() as u64).collect();

    let params = HnswParams::new(16).with_ef_construction(200);
    let mut index = HnswIndex::new(data.dim(), Metric::L2, params)?;
    index.insert_batch(&keys, data.base())?; // parallel, on the rayon pool
    println!("built {index:?}");

    let query = data.query(0);
    let hits = index.search(query, 5, 64)?;
    println!("5 nearest neighbours of query 0:");
    for hit in &hits {
        println!("  key {:>5}  squared L2 {:.3}", hit.key, hit.distance);
    }

    // Recall against exact search over the same vectors.
    let mut exact = BruteForceIndex::new(data.dim(), Metric::L2)?;
    exact.insert_batch(&keys, data.base())?;
    let found = index.search_batch(data.queries(), 10, 64)?;
    let truth = exact.search_batch(data.queries(), 10)?;
    println!(
        "recall@10 at ef=64 on {} queries: {:.3}",
        data.query_len(),
        eval::recall_at_k(&truth, &found, 10)
    );

    // Tombstone the best match: it disappears from results immediately, and
    // `compact` later rebuilds the graph without it.
    index.remove(hits[0].key);
    let after = index.search(query, 5, 64)?;
    assert!(after.iter().all(|h| h.key != hits[0].key));
    println!(
        "removed key {}; best match is now key {}",
        hits[0].key, after[0].key
    );
    index.compact();
    println!(
        "after compact: {} live, {} tombstones",
        index.len(),
        index.deleted_len()
    );
    Ok(())
}
