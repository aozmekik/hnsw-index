//! Exact brute-force search, used as ground truth.

use std::collections::{BinaryHeap, HashSet};

use rayon::prelude::*;

use crate::kernels::Kernels;
use crate::vector::{prepare_batch, prepare_into, prepare_query};
use crate::{Error, MAX_DIM, Metric, Neighbor};

/// An exact index that compares the query with every stored vector.
///
/// Search costs `O(n·d)` per query, but results are exact, which makes this
/// the ground truth for measuring the recall of [`HnswIndex`](crate::HnswIndex).
/// It uses the same validation, normalisation and distance kernels.
///
/// ```
/// use hnsw_index::{BruteForceIndex, Metric};
///
/// let mut exact = BruteForceIndex::new(2, Metric::L2)?;
/// exact.insert(10, &[0.0, 0.0])?;
/// exact.insert(20, &[3.0, 4.0])?;
/// let hits = exact.search(&[2.5, 4.0], 1)?;
/// assert_eq!(hits[0].key, 20);
/// assert_eq!(hits[0].distance, 0.25);
/// # Ok::<(), hnsw_index::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct BruteForceIndex {
    dim: usize,
    metric: Metric,
    kernels: Kernels,
    keys: Vec<u64>,
    vectors: Vec<f32>,
    key_set: HashSet<u64>,
}

/// Max-heap entry ordered like [`Neighbor::cmp_by_distance`].
#[derive(Clone, Copy)]
struct Ranked(Neighbor);

impl PartialEq for Ranked {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}
impl Eq for Ranked {}
impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Ranked {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp_by_distance(&other.0)
    }
}

impl BruteForceIndex {
    /// An empty index for `dim`-dimensional vectors.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidDimension`] if `dim` is zero or above [`MAX_DIM`].
    pub fn new(dim: usize, metric: Metric) -> Result<Self, Error> {
        if dim == 0 || dim > MAX_DIM {
            return Err(Error::InvalidDimension(dim));
        }
        Ok(Self {
            dim,
            metric,
            kernels: Kernels::best(),
            keys: Vec::new(),
            vectors: Vec::new(),
            key_set: HashSet::new(),
        })
    }

    /// Adds one vector.
    ///
    /// # Errors
    ///
    /// [`Error::DuplicateKey`], or any vector validation error
    /// ([`Error::DimensionMismatch`], [`Error::NonFiniteVector`],
    /// [`Error::ZeroVector`]). The index is unchanged on error.
    pub fn insert(&mut self, key: u64, vector: &[f32]) -> Result<(), Error> {
        if self.key_set.contains(&key) {
            return Err(Error::DuplicateKey(key));
        }
        prepare_into(self.metric, self.dim, vector, &mut self.vectors)?;
        self.keys.push(key);
        self.key_set.insert(key);
        Ok(())
    }

    /// Adds `keys.len()` vectors given flat and row-major in `vectors`.
    ///
    /// # Errors
    ///
    /// As for [`insert`](Self::insert); a batch with an error is rejected as
    /// a whole and the index is unchanged.
    pub fn insert_batch(&mut self, keys: &[u64], vectors: &[f32]) -> Result<(), Error> {
        let prepared = prepare_batch(self.metric, self.dim, keys.len(), vectors)?;
        let mut seen = HashSet::with_capacity(keys.len());
        for &key in keys {
            if self.key_set.contains(&key) || !seen.insert(key) {
                return Err(Error::DuplicateKey(key));
            }
        }
        self.vectors.extend_from_slice(&prepared);
        self.keys.extend_from_slice(keys);
        self.key_set.extend(seen);
        Ok(())
    }

    /// The exact `k` nearest neighbours, closest first. Returns every vector
    /// if `k` exceeds [`len`](Self::len).
    ///
    /// # Errors
    ///
    /// Query validation errors, as for [`insert`](Self::insert).
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<Neighbor>, Error> {
        self.search_filtered(query, k, |_| true)
    }

    /// The exact `k` nearest neighbours among keys accepted by `filter`.
    ///
    /// # Errors
    ///
    /// Query validation errors, as for [`insert`](Self::insert).
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: impl Fn(u64) -> bool,
    ) -> Result<Vec<Neighbor>, Error> {
        let query = prepare_query(self.metric, self.dim, query)?;
        if k == 0 {
            return Ok(Vec::new());
        }
        // At most `len` results can come back, whatever `k` asks for.
        let k = k.min(self.len());
        let mut heap = BinaryHeap::with_capacity(k + 1);
        for (&key, v) in self.keys.iter().zip(self.vectors.chunks_exact(self.dim)) {
            if !filter(key) {
                continue;
            }
            let candidate = Ranked(Neighbor {
                key,
                distance: self.metric.distance(&self.kernels, &query, v),
            });
            if heap.len() < k {
                heap.push(candidate);
            } else if heap.peek().is_some_and(|worst| candidate < *worst) {
                heap.pop();
                heap.push(candidate);
            }
        }
        Ok(heap.into_sorted_vec().into_iter().map(|r| r.0).collect())
    }

    /// Searches every query of a flat, row-major batch in parallel (rayon).
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] if `queries.len()` is not a multiple of
    /// the dimension, or the first query validation error.
    pub fn search_batch(&self, queries: &[f32], k: usize) -> Result<Vec<Vec<Neighbor>>, Error> {
        if !queries.len().is_multiple_of(self.dim) {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                actual: queries.len(),
            });
        }
        queries
            .par_chunks_exact(self.dim)
            .map(|q| self.search(q, k))
            .collect()
    }

    /// Number of stored vectors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Dimension of the stored vectors.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The metric used for ranking.
    #[must_use]
    pub fn metric(&self) -> Metric {
        self.metric
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index() -> BruteForceIndex {
        let mut idx = BruteForceIndex::new(1, Metric::L2).unwrap();
        idx.insert_batch(&[5, 3, 9, 1], &[5.0, 3.0, 9.0, 1.0])
            .unwrap();
        idx
    }

    #[test]
    fn returns_sorted_exact_neighbours() {
        let hits = index().search(&[4.0], 3).unwrap();
        let keys: Vec<u64> = hits.iter().map(|n| n.key).collect();
        // Keys 3 and 5 tie at distance 1; ties break by key.
        assert_eq!(keys, vec![3, 5, 1]);
        assert_eq!(hits[0].distance, 1.0);
        assert_eq!(index().search(&[4.0], 0).unwrap(), []);
        assert_eq!(index().search(&[4.0], 100).unwrap().len(), 4);
    }

    #[test]
    fn huge_k_neither_overflows_nor_over_allocates() {
        let idx = index();
        for k in [usize::MAX, 1 << 60] {
            assert_eq!(idx.search(&[4.0], k).unwrap().len(), 4);
            assert_eq!(
                idx.search_filtered(&[4.0], k, |key| key > 4).unwrap().len(),
                2
            );
        }
    }

    #[test]
    fn ties_break_by_key() {
        let mut idx = BruteForceIndex::new(1, Metric::L2).unwrap();
        idx.insert_batch(&[7, 2, 4], &[1.0, -1.0, 1.0]).unwrap();
        let keys: Vec<u64> = idx
            .search(&[0.0], 3)
            .unwrap()
            .iter()
            .map(|n| n.key)
            .collect();
        assert_eq!(keys, vec![2, 4, 7]);
    }

    #[test]
    fn filter_and_batch() {
        let idx = index();
        let hits = idx.search_filtered(&[4.0], 2, |k| k > 4).unwrap();
        assert_eq!(hits.iter().map(|n| n.key).collect::<Vec<_>>(), vec![5, 9]);
        let batch = idx.search_batch(&[0.0, 10.0], 1).unwrap();
        assert_eq!((batch[0][0].key, batch[1][0].key), (1, 9));
        assert!(idx.search_batch(&[0.0, 1.0], 1).is_ok());
    }

    #[test]
    fn rejects_duplicates_atomically() {
        let mut idx = index();
        assert!(matches!(idx.insert(5, &[0.0]), Err(Error::DuplicateKey(5))));
        assert!(matches!(
            idx.insert_batch(&[100, 100], &[0.0, 1.0]),
            Err(Error::DuplicateKey(100))
        ));
        assert_eq!(idx.len(), 4);
        assert!(matches!(
            BruteForceIndex::new(0, Metric::L2),
            Err(Error::InvalidDimension(0))
        ));
    }
}
