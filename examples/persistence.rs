//! Save an index to disk, load it back and check that it answers identically.
//!
//! Run with `cargo run --release --example persistence`.

use std::fs::File;

use hnsw_index::synthetic::Dataset;
use hnsw_index::{Error, FormatError, HnswIndex, HnswParams, Metric};

fn main() -> Result<(), Error> {
    let data = Dataset::gaussian(5_000, 10, 48, 11);
    let keys: Vec<u64> = (0..data.base_len() as u64).collect();
    let mut index = HnswIndex::new(data.dim(), Metric::InnerProduct, HnswParams::default())?;
    index.insert_batch(&keys, data.base())?;
    index.remove(42);

    let path = std::env::temp_dir().join(format!("hnsw-index-example-{}.hnsw", std::process::id()));
    // `write_to` buffers internally and `read_from` reads the whole file at
    // once, so neither needs a `BufWriter` or `BufReader`.
    index.write_to(File::create(&path)?)?;
    let size = std::fs::metadata(&path)?.len();
    println!("wrote {} ({size} bytes)", path.display());

    let loaded = HnswIndex::read_from(File::open(&path)?)?;
    for q in 0..data.query_len() {
        assert_eq!(
            loaded.search(data.query(q), 10, 64)?,
            index.search(data.query(q), 10, 64)?
        );
    }
    println!(
        "loaded index answers all {} queries identically",
        data.query_len()
    );

    // A corrupted file is rejected with a descriptive error, not a panic.
    let mut bytes = std::fs::read(&path)?;
    bytes[1000] ^= 0xFF;
    match HnswIndex::from_bytes(&bytes) {
        Err(Error::Format(FormatError::ChecksumMismatch { .. })) => {
            println!("flipped byte detected by the checksum");
        }
        other => panic!("corruption went unnoticed: {other:?}"),
    }
    std::fs::remove_file(&path)?;
    Ok(())
}
