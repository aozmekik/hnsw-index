//! Insertion: INSERT (Algorithm 1) and neighbor selection (Algorithms 3–4).

use std::collections::HashSet;
use std::sync::atomic::Ordering;

use super::search::Candidate;
use super::{HnswIndex, MAX_LEVEL, pack_entry, unpack_entry};
use crate::NeighborSelection;
use crate::synthetic::mix64;

/// The level of node `id`: `⌊−ln(U) · mL⌋` with `U` uniform in `(0, 1]`,
/// capped at [`MAX_LEVEL`].
///
/// `U` is derived by hashing `(seed, id)` rather than drawn from a stateful
/// generator, so a node's level does not depend on insertion order or on how
/// work is split across threads, and nothing needs to be serialized to keep
/// sampling reproducible.
#[allow(clippy::cast_sign_loss)] // `level` is in `0.0..31.0` when cast
pub(super) fn sample_level(seed: u64, id: u32, level_multiplier: f64) -> u8 {
    let h = mix64(seed ^ mix64(u64::from(id).wrapping_add(0x9E37_79B9_7F4A_7C15)));
    let u = ((h >> 11) + 1) as f64 * (1.0 / (1_u64 << 53) as f64);
    let level = (-u.ln() * level_multiplier).floor();
    if level >= f64::from(MAX_LEVEL) {
        MAX_LEVEL
    } else {
        level as u8
    }
}

impl HnswIndex {
    /// Links an already-stored node into the graph (INSERT, Algorithm 1).
    ///
    /// Takes `&self` so that many nodes can be linked in parallel: neighbor
    /// lists are atomics guarded by per-node locks (see `graph.rs`), and the
    /// entry point is an atomic word updated under `promotion`.
    ///
    /// Deadlock freedom: a thread holds at most one node lock at a time (in
    /// `install_own_neighbors`, `connect` and `write_shrunk`, none of which
    /// takes another lock), and `promotion` is only ever acquired before any
    /// node lock. Reads never lock.
    pub(super) fn link(&self, id: u32) {
        let level = usize::from(self.levels[id as usize]);
        let query = self.vector(id);
        let distance = |other: u32| {
            self.metric
                .distance(&self.kernels, query, self.vector(other))
        };

        // A node that will become the new top of the graph holds the promotion
        // lock for its whole insertion, so that two such nodes cannot both
        // install themselves from a stale view (the approach hnswlib uses for
        // its global lock). That happens about log_M(n) times per build.
        let mut entry = self.entry.load(Ordering::Acquire);
        let mut promotion = None;
        if unpack_entry(entry).is_none_or(|(_, top)| level > top) {
            let guard = self.promotion.lock();
            entry = self.entry.load(Ordering::Acquire);
            match unpack_entry(entry) {
                None => {
                    self.entry.store(pack_entry(id, level), Ordering::Release);
                    return;
                }
                Some((_, top)) if level > top => promotion = Some(guard),
                Some(_) => drop(guard),
            }
        }
        let (entry_id, top) = unpack_entry(entry).expect("entry point exists");

        let mut ep = Candidate::new(distance(entry_id), entry_id);
        for layer in (level + 1..=top).rev() {
            ep = self.greedy_closest(ep, layer, &distance);
        }

        let ef = self.params.ef_construction().max(self.params.m());
        let mut entry_points = vec![ep];
        self.scratch.with(|scratch| {
            for layer in (0..=level.min(top)).rev() {
                let found =
                    self.search_layer(scratch, &entry_points, ef, layer, &distance, &|n| n != id);
                let selected = self.select_neighbors(id, query, &found, self.params.m(), layer);
                self.install_own_neighbors(id, &selected, layer);
                for &neighbor in &selected {
                    self.connect(neighbor, id, layer);
                }
                if !found.is_empty() {
                    entry_points = found;
                }
            }
        });

        if promotion.is_some() {
            self.entry.store(pack_entry(id, level), Ordering::Release);
        }
    }

    /// Writes the new node's own neighbor list on `layer`.
    ///
    /// The list is normally still empty, but a concurrent insertion may
    /// already have linked back to this node: nodes carried over as entry
    /// points from the layer above are visible before they are connected on
    /// the current layer. Those back-links are merged in rather than
    /// overwritten, shrinking with the selection strategy if over capacity.
    fn install_own_neighbors(&self, id: u32, selected: &[u32], layer: usize) {
        let _lock = self.locks[id as usize].lock();
        let mut merged = Vec::with_capacity(self.links.capacity(layer) + selected.len());
        self.links.read_into(id, layer, &mut merged);
        if merged.is_empty() {
            self.links.set(id, layer, selected);
            return;
        }
        for &n in selected {
            if !merged.contains(&n) {
                merged.push(n);
            }
        }
        self.write_shrunk(id, &merged, layer);
    }

    /// Adds the edge `node → new` on `layer`, shrinking `node`'s list with the
    /// selection strategy when it is full (Algorithm 1, lines 11–16).
    fn connect(&self, node: u32, new: u32, layer: usize) {
        let _lock = self.locks[node as usize].lock();
        let mut current = Vec::with_capacity(self.links.capacity(layer) + 1);
        self.links.read_into(node, layer, &mut current);
        if current.contains(&new) || self.links.try_push(node, layer, new) {
            return;
        }
        current.push(new);
        self.write_shrunk(node, &current, layer);
    }

    /// Sets `node`'s list on `layer` to `ids`, applying the selection strategy
    /// if `ids` exceeds the layer capacity. The caller holds `node`'s lock.
    fn write_shrunk(&self, node: u32, ids: &[u32], layer: usize) {
        let capacity = self.links.capacity(layer);
        if ids.len() <= capacity {
            self.links.set(node, layer, ids);
            return;
        }
        let base = self.vector(node);
        let mut candidates: Vec<Candidate> = ids
            .iter()
            .map(|&c| Candidate::new(self.metric.distance(&self.kernels, base, self.vector(c)), c))
            .collect();
        candidates.sort_unstable();
        let selected = self.select_neighbors(node, base, &candidates, capacity, layer);
        self.links.set(node, layer, &selected);
    }

    /// Chooses at most `m` neighbors for `base_id` (whose vector is `base`)
    /// from `candidates`, which must be sorted by distance to `base`.
    fn select_neighbors(
        &self,
        base_id: u32,
        base: &[f32],
        candidates: &[Candidate],
        m: usize,
        layer: usize,
    ) -> Vec<u32> {
        match self.params.selection() {
            // SELECT-NEIGHBORS-SIMPLE (Algorithm 3).
            NeighborSelection::Simple => candidates.iter().take(m).map(|c| c.id).collect(),
            // SELECT-NEIGHBORS-HEURISTIC (Algorithm 4).
            NeighborSelection::Heuristic {
                extend_candidates,
                keep_pruned,
            } => {
                let extended;
                let pool = if extend_candidates {
                    extended = self.extend_candidates(base_id, base, candidates, layer);
                    &extended[..]
                } else {
                    candidates
                };
                let mut selected: Vec<Candidate> = Vec::with_capacity(m);
                let mut discarded = Vec::new();
                for &c in pool {
                    if selected.len() >= m {
                        break;
                    }
                    let v = self.vector(c.id);
                    // Keep `c` only if it is closer to the base than to every
                    // neighbor already kept.
                    let diverse = selected.iter().all(|s| {
                        self.metric.distance(&self.kernels, v, self.vector(s.id)) >= c.distance
                    });
                    if diverse {
                        selected.push(c);
                    } else if keep_pruned {
                        discarded.push(c);
                    }
                }
                let room = m - selected.len();
                selected.extend(discarded.into_iter().take(room));
                selected.into_iter().map(|c| c.id).collect()
            }
        }
    }

    /// The paper's `extendCandidates`: `candidates` plus their neighbors on
    /// `layer`, deduplicated, without `base_id`, sorted by distance to `base`.
    fn extend_candidates(
        &self,
        base_id: u32,
        base: &[f32],
        candidates: &[Candidate],
        layer: usize,
    ) -> Vec<Candidate> {
        let mut seen: HashSet<u32> = candidates.iter().map(|c| c.id).collect();
        seen.insert(base_id);
        let mut out = candidates.to_vec();
        for c in candidates {
            self.links.for_each(c.id, layer, |n| {
                if seen.insert(n) {
                    let d = self.metric.distance(&self.kernels, base, self.vector(n));
                    out.push(Candidate::new(d, n));
                }
            });
        }
        out.sort_unstable();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_follow_the_geometric_distribution() {
        let ml = 1.0 / 16_f64.ln();
        let n = 200_000_u32;
        let mut at_least = [0_u32; 4];
        for id in 0..n {
            let level = usize::from(sample_level(42, id, ml));
            for (l, count) in at_least.iter_mut().enumerate() {
                if level >= l {
                    *count += 1;
                }
            }
        }
        // P(level >= l) = 16^-l.
        for (l, &count) in at_least.iter().enumerate().skip(1) {
            let expected = f64::from(n) * 16_f64.powf(-(l as f64));
            let observed = f64::from(count);
            assert!(
                (observed - expected).abs() < 5.0 * expected.sqrt() + 2.0,
                "level {l}: observed {observed}, expected {expected}"
            );
        }
    }

    #[test]
    fn levels_are_deterministic_and_seed_dependent() {
        let ml = 1.0 / 4_f64.ln();
        let a: Vec<u8> = (0..1000).map(|i| sample_level(1, i, ml)).collect();
        let b: Vec<u8> = (0..1000).map(|i| sample_level(1, i, ml)).collect();
        let c: Vec<u8> = (0..1000).map(|i| sample_level(2, i, ml)).collect();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a.iter().all(|&l| l <= MAX_LEVEL));
    }

    #[test]
    fn levels_are_capped() {
        // With mL = 10, P(level >= 31) = exp(-3.1), about 4.5%, so some of
        // 1000 nodes would exceed the cap without it.
        let mut capped = 0;
        for id in 0..1000 {
            let level = sample_level(3, id, 10.0);
            assert!(level <= MAX_LEVEL);
            capped += usize::from(level == MAX_LEVEL);
        }
        assert!((10..100).contains(&capped), "{capped} nodes at the cap");
    }
}
