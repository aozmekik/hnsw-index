//! Search restricted to a subset of keys, e.g. documents of one tenant.
//!
//! Run with `cargo run --release --example filtered_search`.

use std::collections::HashMap;

use hnsw_index::synthetic::Dataset;
use hnsw_index::{BruteForceIndex, Error, HnswIndex, HnswParams, Metric};

fn main() -> Result<(), Error> {
    let data = Dataset::gaussian(10_000, 1, 32, 3);
    let keys: Vec<u64> = (0..data.base_len() as u64).collect();
    // Assign every vector to one of 20 tenants.
    let tenant: HashMap<u64, u32> = keys.iter().map(|&k| (k, (k % 20) as u32)).collect();

    let mut index = HnswIndex::new(data.dim(), Metric::Cosine, HnswParams::default())?;
    index.insert_batch(&keys, data.base())?;

    let wanted = 7;
    let in_tenant = |key: u64| tenant[&key] == wanted;
    let hits = index.search_filtered(data.query(0), 5, 64, in_tenant)?;
    println!("top 5 for tenant {wanted}:");
    for hit in &hits {
        assert_eq!(tenant[&hit.key], wanted);
        println!("  key {:>5}  cosine distance {:.4}", hit.key, hit.distance);
    }

    // The filter is applied during graph traversal, so the HNSW answer can
    // be compared with an exact filtered scan.
    let mut exact = BruteForceIndex::new(data.dim(), Metric::Cosine)?;
    exact.insert_batch(&keys, data.base())?;
    let truth = exact.search_filtered(data.query(0), 5, in_tenant)?;
    let overlap = hits
        .iter()
        .filter(|h| truth.iter().any(|t| t.key == h.key))
        .count();
    println!("{overlap}/5 match the exact filtered result");
    Ok(())
}
