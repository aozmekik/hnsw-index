//! A from-scratch implementation of Hierarchical Navigable Small World (HNSW)
//! graphs for approximate nearest neighbour search, following
//!
//! > Yu. A. Malkov and D. A. Yashunin, "Efficient and robust approximate
//! > nearest neighbor search using Hierarchical Navigable Small World graphs",
//! > IEEE TPAMI 42(4), 2018. [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)
//!
//! # Overview
//!
//! * [`HnswIndex`]: the approximate index. Insertion with level sampling
//!   (`mL = 1/ln M`), greedy descent through the upper layers and an
//!   ef-bounded search on layer 0, with either neighbour-selection strategy
//!   of the paper ([`NeighborSelection`]).
//! * [`BruteForceIndex`]: exact search, for ground truth.
//! * [`Metric`]: squared L2, inner product and cosine, computed by SIMD
//!   kernels chosen at runtime ([`kernels`]).
//! * Filtered search, tombstone deletion with [`HnswIndex::compact`],
//!   parallel batch insertion with rayon, concurrent `&self` search,
//!   optional 8-bit scalar quantization with exact re-ranking, and a
//!   versioned, checksummed file format that validates untrusted input.
//!
//! # Example
//!
//! ```
//! use hnsw_index::{BruteForceIndex, HnswIndex, HnswParams, Metric, eval, synthetic::Dataset};
//!
//! let data = Dataset::gaussian(2_000, 20, 16, 7);
//! let keys: Vec<u64> = (0..data.base_len() as u64).collect();
//!
//! let mut index = HnswIndex::new(data.dim(), Metric::L2, HnswParams::default())?;
//! index.insert_batch(&keys, data.base())?;
//!
//! let mut exact = BruteForceIndex::new(data.dim(), Metric::L2)?;
//! exact.insert_batch(&keys, data.base())?;
//!
//! let found = index.search_batch(data.queries(), 10, 64)?;
//! let truth = exact.search_batch(data.queries(), 10)?;
//! assert!(eval::recall_at_k(&truth, &found, 10) > 0.9);
//! # Ok::<(), hnsw_index::Error>(())
//! ```

mod brute;
mod crc32;
mod error;
pub mod eval;
mod hnsw;
pub mod kernels;
mod metric;
mod neighbor;
mod params;
mod quantization;
pub mod synthetic;
mod vector;

pub use brute::BruteForceIndex;
pub use error::{Error, FormatError};
pub use hnsw::{GraphStats, HnswIndex, SearchOptions};
pub use metric::Metric;
pub use neighbor::Neighbor;
pub use params::{HnswParams, NeighborSelection};

/// Largest supported vector dimension.
pub const MAX_DIM: usize = 1 << 16;

/// Largest number of nodes (live plus tombstoned) an index can hold. Node ids
/// are `u32`, and `u32::MAX` is reserved as a sentinel.
pub const MAX_NODES: usize = u32::MAX as usize - 1;
