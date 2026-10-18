//! Evaluation helpers.

use crate::Neighbor;

/// Mean recall@k of `found` against exact `truth`, in `[0, 1]`.
///
/// For each query, counts how many of the first `k` true neighbors appear
/// among the first `k` found results, and divides by the number of true
/// neighbors considered (`k`, or fewer when the index holds fewer than `k`
/// vectors). Returns 1.0 when there is nothing to find.
///
/// # Panics
///
/// Panics if `truth` and `found` describe different numbers of queries.
///
/// ```
/// use hnsw_index::{Neighbor, eval::recall_at_k};
///
/// let n = |key| Neighbor { key, distance: 0.0 };
/// let truth = vec![vec![n(1), n(2)]];
/// let found = vec![vec![n(2), n(7)]];
/// assert_eq!(recall_at_k(&truth, &found, 2), 0.5);
/// ```
#[must_use]
pub fn recall_at_k(truth: &[Vec<Neighbor>], found: &[Vec<Neighbor>], k: usize) -> f64 {
    assert_eq!(
        truth.len(),
        found.len(),
        "truth and found must have one entry per query"
    );
    let mut hits = 0_usize;
    let mut expected = 0_usize;
    for (t, f) in truth.iter().zip(found) {
        let t = &t[..k.min(t.len())];
        let f = &f[..k.min(f.len())];
        expected += t.len();
        hits += f
            .iter()
            .filter(|n| t.iter().any(|m| m.key == n.key))
            .count();
    }
    if expected == 0 {
        1.0
    } else {
        hits as f64 / expected as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(keys: &[u64]) -> Vec<Neighbor> {
        keys.iter()
            .map(|&key| Neighbor { key, distance: 0.0 })
            .collect()
    }

    #[test]
    fn counts_overlap_of_top_k() {
        let truth = vec![keys(&[1, 2, 3]), keys(&[4, 5, 6])];
        let found = vec![keys(&[3, 2, 1]), keys(&[4, 9, 8])];
        assert!((recall_at_k(&truth, &found, 3) - 4.0 / 6.0).abs() < 1e-12);
        assert_eq!(recall_at_k(&truth, &found, 1), 0.5);
    }

    #[test]
    fn short_truth_and_empty_input() {
        assert_eq!(recall_at_k(&[keys(&[1])], &[keys(&[1])], 10), 1.0);
        assert_eq!(recall_at_k(&[], &[], 10), 1.0);
        assert_eq!(recall_at_k(&[keys(&[1, 2])], &[keys(&[])], 2), 0.0);
    }
}
