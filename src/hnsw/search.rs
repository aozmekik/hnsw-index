//! Graph traversal: greedy descent and the ef-bounded layer search.

use std::cmp::{Ordering, Reverse};

use super::HnswIndex;
use super::scratch::Scratch;

/// A node id paired with its distance to the current query.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Candidate {
    pub(crate) distance: f32,
    pub(crate) id: u32,
}

impl Candidate {
    #[inline]
    pub(crate) fn new(distance: f32, id: u32) -> Self {
        Self { distance, id }
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    /// Distance first (`total_cmp`, so even a NaN cannot break heap
    /// invariants), then id for determinism.
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then(self.id.cmp(&other.id))
    }
}

impl HnswIndex {
    /// Greedy walk on one layer: move to the closest neighbor until no
    /// neighbor is closer. This is SEARCH-LAYER with `ef = 1`, without heaps.
    pub(super) fn greedy_closest(
        &self,
        start: Candidate,
        layer: usize,
        distance: &impl Fn(u32) -> f32,
    ) -> Candidate {
        let mut best = start;
        loop {
            let from = best.id;
            self.links.for_each(from, layer, |n| {
                let d = distance(n);
                if d < best.distance {
                    best = Candidate::new(d, n);
                }
            });
            if best.id == from {
                return best;
            }
        }
    }

    /// SEARCH-LAYER (Algorithm 2 of the paper), extended with an admission
    /// predicate.
    ///
    /// Every reachable node may be expanded, but only nodes for which
    /// `admit(id)` holds enter the result set. While fewer than `ef` admissible
    /// nodes have been found there is no distance bound, so the search keeps
    /// expanding; with a very selective predicate it can visit the whole
    /// connected graph, which is slow but never wrong. Tombstoned nodes are
    /// handled the same way: they guide the search but are never returned.
    ///
    /// Returns up to `ef` admissible candidates sorted by distance.
    pub(super) fn search_layer(
        &self,
        scratch: &mut Scratch,
        entry_points: &[Candidate],
        ef: usize,
        layer: usize,
        distance: &impl Fn(u32) -> f32,
        admit: &impl Fn(u32) -> bool,
    ) -> Vec<Candidate> {
        let Scratch {
            visited,
            candidates,
            results,
        } = scratch;
        visited.reset(self.node_count());
        candidates.clear();
        results.clear();

        for &ep in entry_points {
            if !visited.insert(ep.id) {
                continue;
            }
            candidates.push(Reverse(ep));
            if admit(ep.id) {
                results.push(ep);
                if results.len() > ef {
                    results.pop();
                }
            }
        }

        while let Some(Reverse(current)) = candidates.pop() {
            if results.len() >= ef
                && results
                    .peek()
                    .is_some_and(|w| current.distance > w.distance)
            {
                break;
            }
            self.links.for_each(current.id, layer, |n| {
                if !visited.insert(n) {
                    return;
                }
                let d = distance(n);
                let improves = results.len() < ef || results.peek().is_some_and(|w| d < w.distance);
                if improves {
                    let c = Candidate::new(d, n);
                    candidates.push(Reverse(c));
                    if admit(n) {
                        results.push(c);
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            });
        }

        let mut out: Vec<Candidate> = results.drain().collect();
        out.sort_unstable();
        out
    }

    /// Full search: greedy descent through the upper layers, then an
    /// ef-bounded search on layer 0. Returns up to `ef` candidates, sorted.
    pub(super) fn search_graph(
        &self,
        ef: usize,
        distance: &impl Fn(u32) -> f32,
        admit: &impl Fn(u32) -> bool,
    ) -> Vec<Candidate> {
        let Some((entry, top)) = self.entry_point() else {
            return Vec::new();
        };
        let mut ep = Candidate::new(distance(entry), entry);
        for layer in (1..=top).rev() {
            ep = self.greedy_closest(ep, layer, distance);
        }
        self.scratch
            .with(|scratch| self.search_layer(scratch, &[ep], ef, 0, distance, admit))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BinaryHeap;

    #[test]
    fn candidates_order_by_distance_then_id() {
        let mut heap = BinaryHeap::new();
        heap.push(Candidate::new(1.0, 5));
        heap.push(Candidate::new(1.0, 2));
        heap.push(Candidate::new(0.5, 9));
        heap.push(Candidate::new(f32::NAN, 1));
        let order: Vec<u32> = heap.into_sorted_vec().iter().map(|c| c.id).collect();
        assert_eq!(order, vec![9, 2, 5, 1]);
    }
}
