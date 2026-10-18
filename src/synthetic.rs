//! Seeded synthetic datasets for tests, examples and benchmarks.
//!
//! The generators are deterministic: the same arguments produce the same
//! vectors on a given platform. (They use `f64::ln` and `f64::cos`, whose last
//! bit may differ between libm implementations, so bit-identical output across
//! operating systems is not guaranteed.) Queries are drawn after the base set
//! from the same distribution, so they are never members of the base set.
//!
//! ```
//! use hnsw_index::synthetic::Dataset;
//!
//! let data = Dataset::gaussian(1_000, 10, 32, 42);
//! assert_eq!(data.base().len(), 1_000 * 32);
//! assert_eq!(data.query(9).len(), 32);
//! assert_eq!(data.base(), Dataset::gaussian(1_000, 10, 32, 42).base());
//! ```

use std::f64::consts::TAU;

/// `SplitMix64` (Steele, Lea & Flood, "Fast splittable pseudorandom number
/// generators", OOPSLA 2014), with the constants of Vigna's reference code.
#[derive(Debug, Clone)]
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix64(self.state)
    }

    /// Uniform in `[0, 1)` with 53 bits of precision.
    pub(crate) fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1_u64 << 53) as f64)
    }

    /// Uniform in `0..n` (`n > 0`), by Lemire's multiply-shift reduction.
    pub(crate) fn next_below(&mut self, n: usize) -> usize {
        ((u128::from(self.next_u64()) * n as u128) >> 64) as usize
    }

    /// Standard normal sample via the Box–Muller transform.
    pub(crate) fn next_gaussian(&mut self) -> f64 {
        let u1 = 1.0 - self.next_f64(); // (0, 1], so ln is finite
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }
}

/// The `SplitMix64` output function: a bijective 64-bit mixer.
pub(crate) fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A base set and a disjoint query set of `dim`-dimensional vectors, stored
/// flat and row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    dim: usize,
    base: Vec<f32>,
    queries: Vec<f32>,
}

impl Dataset {
    /// Vectors with i.i.d. standard normal components.
    ///
    /// Isotropic Gaussian data has no low-dimensional structure for a graph
    /// index to exploit, which makes it a hard, pessimistic workload.
    ///
    /// # Panics
    ///
    /// Panics if `dim` is zero.
    #[must_use]
    pub fn gaussian(base: usize, queries: usize, dim: usize, seed: u64) -> Self {
        assert!(dim > 0, "dim must be positive");
        let mut rng = SplitMix64::new(seed);
        let mut sample = |count: usize| -> Vec<f32> {
            (0..count * dim)
                .map(|_| rng.next_gaussian() as f32)
                .collect()
        };
        let base = sample(base);
        let queries = sample(queries);
        Self { dim, base, queries }
    }

    /// A Gaussian mixture: `clusters` centers with standard normal components,
    /// and points scattered around a uniformly chosen center with standard
    /// deviation 0.35 per component.
    ///
    /// # Panics
    ///
    /// Panics if `dim` or `clusters` is zero.
    #[must_use]
    pub fn clustered(base: usize, queries: usize, dim: usize, clusters: usize, seed: u64) -> Self {
        const SPREAD: f64 = 0.35;
        assert!(dim > 0, "dim must be positive");
        assert!(clusters > 0, "clusters must be positive");
        let mut rng = SplitMix64::new(seed);
        let centers: Vec<f64> = (0..clusters * dim).map(|_| rng.next_gaussian()).collect();
        let mut sample = |count: usize| -> Vec<f32> {
            let mut out = Vec::with_capacity(count * dim);
            for _ in 0..count {
                let c = rng.next_below(clusters);
                let center = &centers[c * dim..(c + 1) * dim];
                out.extend(
                    center
                        .iter()
                        .map(|&x| (x + SPREAD * rng.next_gaussian()) as f32),
                );
            }
            out
        };
        let base = sample(base);
        let queries = sample(queries);
        Self { dim, base, queries }
    }

    /// Dimension of every vector.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// All base vectors, flat and row-major.
    #[must_use]
    pub fn base(&self) -> &[f32] {
        &self.base
    }

    /// All query vectors, flat and row-major.
    #[must_use]
    pub fn queries(&self) -> &[f32] {
        &self.queries
    }

    /// Number of base vectors.
    #[must_use]
    pub fn base_len(&self) -> usize {
        self.base.len() / self.dim
    }

    /// Number of query vectors.
    #[must_use]
    pub fn query_len(&self) -> usize {
        self.queries.len() / self.dim
    }

    /// The `i`-th base vector.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.base_len()`.
    #[must_use]
    pub fn base_vector(&self, i: usize) -> &[f32] {
        &self.base[i * self.dim..(i + 1) * self.dim]
    }

    /// The `i`-th query vector.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.query_len()`.
    #[must_use]
    pub fn query(&self, i: usize) -> &[f32] {
        &self.queries[i * self.dim..(i + 1) * self.dim]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_matches_reference_output() {
        // First outputs of Vigna's splitmix64.c seeded with 1234567.
        let mut rng = SplitMix64::new(1_234_567);
        assert_eq!(rng.next_u64(), 6_457_827_717_110_365_317);
        assert_eq!(rng.next_u64(), 3_203_168_211_198_807_973);
    }

    #[test]
    fn gaussian_moments_are_plausible() {
        let mut rng = SplitMix64::new(9);
        let n = 200_000;
        let samples: Vec<f64> = (0..n).map(|_| rng.next_gaussian()).collect();
        let mean = samples.iter().sum::<f64>() / f64::from(n);
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / f64::from(n);
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "variance {var}");
    }

    #[test]
    fn next_below_stays_in_range() {
        let mut rng = SplitMix64::new(3);
        let mut seen = [false; 7];
        for _ in 0..1000 {
            seen[rng.next_below(7)] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn datasets_are_deterministic_and_shaped() {
        let a = Dataset::clustered(100, 5, 8, 4, 1);
        let b = Dataset::clustered(100, 5, 8, 4, 1);
        assert_eq!(a, b);
        assert_eq!((a.base_len(), a.query_len(), a.dim()), (100, 5, 8));
        assert_ne!(a, Dataset::clustered(100, 5, 8, 4, 2));
        let g = Dataset::gaussian(10, 3, 4, 0);
        assert_eq!(g.base_vector(9).len(), 4);
        assert_ne!(g.query(0), g.base_vector(0));
    }
}
