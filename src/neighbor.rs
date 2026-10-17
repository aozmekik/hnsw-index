//! Search results.

use std::cmp::Ordering;

/// One search result: a key and its distance to the query.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor {
    /// The key the vector was inserted with.
    pub key: u64,
    /// Distance to the query under the index [`Metric`](crate::Metric).
    pub distance: f32,
}

impl Neighbor {
    /// Total order used for every result list: by distance, then by key so
    /// that ties are broken deterministically.
    pub(crate) fn cmp_by_distance(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then(self.key.cmp(&other.key))
    }
}
