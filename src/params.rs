//! Construction parameters.

use crate::Error;

/// Largest accepted `M`.
pub(crate) const MAX_M: usize = 512;
/// Largest accepted `M0`.
pub(crate) const MAX_M0: usize = 2048;
/// Largest accepted `efConstruction`.
pub(crate) const MAX_EF_CONSTRUCTION: usize = 1 << 16;
/// Largest accepted level multiplier `mL`.
pub(crate) const MAX_LEVEL_MULTIPLIER: f64 = 10.0;

/// How a node chooses its neighbours from a candidate list.
///
/// Both strategies come from Malkov & Yashunin (2018): `Simple` is
/// SELECT-NEIGHBORS-SIMPLE (Algorithm 3), `Heuristic` is
/// SELECT-NEIGHBORS-HEURISTIC (Algorithm 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NeighborSelection {
    /// Keep the `M` nearest candidates.
    Simple,
    /// Keep a candidate only if it is closer to the base node than to every
    /// neighbour kept so far. This favours neighbours in different directions,
    /// which keeps clustered data connected.
    Heuristic {
        /// Also consider the neighbours of each candidate (the paper's
        /// `extendCandidates`). Costs extra distance computations; the paper
        /// recommends it only for extremely clustered data.
        extend_candidates: bool,
        /// Fill up to `M` with the closest discarded candidates (the paper's
        /// `keepPrunedConnections`).
        keep_pruned: bool,
    },
}

impl Default for NeighborSelection {
    /// The heuristic without extensions, as in most HNSW implementations.
    fn default() -> Self {
        Self::Heuristic {
            extend_candidates: false,
            keep_pruned: false,
        }
    }
}

/// Parameters that shape the HNSW graph, named as in the paper.
///
/// ```
/// use hnsw_index::{HnswParams, NeighborSelection};
///
/// let params = HnswParams::new(32)
///     .with_ef_construction(400)
///     .with_selection(NeighborSelection::Simple)
///     .with_seed(7);
/// assert_eq!(params.m0(), 64);
/// assert!((params.level_multiplier() - 1.0 / 32_f64.ln()).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HnswParams {
    m: usize,
    m0: usize,
    ef_construction: usize,
    level_multiplier: f64,
    selection: NeighborSelection,
    seed: u64,
}

impl HnswParams {
    /// Parameters with `M = m` and the paper's recommended derived values:
    /// `M0 = 2·M`, `mL = 1 / ln(M)`, `efConstruction = 200`, heuristic
    /// selection and seed 0.
    ///
    /// Values are validated when an index is created, so this never fails.
    #[must_use]
    pub fn new(m: usize) -> Self {
        Self {
            m,
            m0: m.saturating_mul(2),
            ef_construction: 200,
            level_multiplier: 1.0 / (m as f64).ln(),
            selection: NeighborSelection::default(),
            seed: 0,
        }
    }

    /// Sets `M0`, the maximum number of neighbours per node on layer 0.
    #[must_use]
    pub fn with_m0(mut self, m0: usize) -> Self {
        self.m0 = m0;
        self
    }

    /// Sets `efConstruction`, the candidate list size used while inserting.
    /// Values below `M` are raised to `M` internally.
    #[must_use]
    pub fn with_ef_construction(mut self, ef_construction: usize) -> Self {
        self.ef_construction = ef_construction;
        self
    }

    /// Overrides the level multiplier `mL` (default `1 / ln(M)`).
    #[must_use]
    pub fn with_level_multiplier(mut self, level_multiplier: f64) -> Self {
        self.level_multiplier = level_multiplier;
        self
    }

    /// Sets the neighbour selection strategy.
    #[must_use]
    pub fn with_selection(mut self, selection: NeighborSelection) -> Self {
        self.selection = selection;
        self
    }

    /// Sets the seed from which node levels are derived.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// `M`: neighbours a new node links to on every layer, and the maximum
    /// number of neighbours per node on layers above 0.
    #[must_use]
    pub fn m(&self) -> usize {
        self.m
    }

    /// `M0` (the paper's `Mmax0`): maximum neighbours per node on layer 0.
    #[must_use]
    pub fn m0(&self) -> usize {
        self.m0
    }

    /// `efConstruction`: candidate list size used while inserting.
    #[must_use]
    pub fn ef_construction(&self) -> usize {
        self.ef_construction
    }

    /// `mL`: a node reaches level `l` or higher with probability `exp(−l / mL)`.
    #[must_use]
    pub fn level_multiplier(&self) -> f64 {
        self.level_multiplier
    }

    /// The neighbour selection strategy.
    #[must_use]
    pub fn selection(&self) -> NeighborSelection {
        self.selection
    }

    /// Seed for level sampling.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Checks every parameter against its accepted range.
    pub(crate) fn validate(&self) -> Result<(), Error> {
        let invalid = |name, reason| Err(Error::InvalidParameter { name, reason });
        if !(2..=MAX_M).contains(&self.m) {
            return invalid("m", "must be between 2 and 512");
        }
        if !(self.m..=MAX_M0).contains(&self.m0) {
            return invalid("m0", "must be between m and 2048");
        }
        if !(1..=MAX_EF_CONSTRUCTION).contains(&self.ef_construction) {
            return invalid("ef_construction", "must be between 1 and 65536");
        }
        if !(self.level_multiplier.is_finite()
            && self.level_multiplier > 0.0
            && self.level_multiplier <= MAX_LEVEL_MULTIPLIER)
        {
            return invalid("level_multiplier", "must be finite and in (0, 10]");
        }
        Ok(())
    }
}

impl Default for HnswParams {
    /// `HnswParams::new(16)`.
    fn default() -> Self {
        Self::new(16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_follow_the_paper() {
        let p = HnswParams::default();
        assert_eq!((p.m(), p.m0(), p.ef_construction()), (16, 32, 200));
        assert!((p.level_multiplier() - 1.0 / 16_f64.ln()).abs() < 1e-15);
        assert!(p.validate().is_ok());
    }

    #[test]
    fn rejects_out_of_range_values() {
        let bad = [
            HnswParams::new(1),
            HnswParams::new(0),
            HnswParams::new(513),
            HnswParams::new(16).with_m0(15),
            HnswParams::new(16).with_m0(4096),
            HnswParams::new(16).with_ef_construction(0),
            HnswParams::new(16).with_level_multiplier(0.0),
            HnswParams::new(16).with_level_multiplier(f64::NAN),
            HnswParams::new(16).with_level_multiplier(11.0),
        ];
        for p in bad {
            assert!(
                matches!(p.validate(), Err(Error::InvalidParameter { .. })),
                "{p:?}"
            );
        }
    }
}
