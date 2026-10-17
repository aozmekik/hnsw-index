//! Compare exact and int8-quantized graph traversal on the same index.
//!
//! Run with `cargo run --release --example quantization`. Timings printed by
//! this example are indicative only; see the README for measured numbers.

use std::time::Instant;

use hnsw_index::synthetic::Dataset;
use hnsw_index::{BruteForceIndex, Error, HnswIndex, HnswParams, Metric, SearchOptions, eval};

fn main() -> Result<(), Error> {
    let data = Dataset::gaussian(20_000, 200, 128, 5);
    let keys: Vec<u64> = (0..data.base_len() as u64).collect();
    let mut index = HnswIndex::new(data.dim(), Metric::L2, HnswParams::default())?;
    index.insert_batch(&keys, data.base())?;
    let before = index.memory_usage();
    index.enable_quantization();
    println!(
        "quantization adds {} KiB of codes ({} bytes per vector)",
        (index.memory_usage() - before) / 1024,
        data.dim()
    );

    let mut exact = BruteForceIndex::new(data.dim(), Metric::L2)?;
    exact.insert_batch(&keys, data.base())?;
    let truth = exact.search_batch(data.queries(), 10)?;

    for quantized in [false, true] {
        let options = SearchOptions::new(128).quantized(quantized);
        let start = Instant::now();
        let found = (0..data.query_len())
            .map(|q| index.search_with(data.query(q), 10, &options))
            .collect::<Result<Vec<_>, _>>()?;
        let elapsed = start.elapsed();
        println!(
            "{:<22} recall@10 {:.3}, {:.0} queries/s (1 thread)",
            if quantized {
                "int8 + f32 re-rank:"
            } else {
                "f32:"
            },
            eval::recall_at_k(&truth, &found, 10),
            data.query_len() as f64 / elapsed.as_secs_f64()
        );
    }
    Ok(())
}
