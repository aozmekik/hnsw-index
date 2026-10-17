//! Symmetric 8-bit scalar quantization.
//!
//! Each component is mapped to an `i8` code with one global scale:
//! `code = round((x − offsetᵢ) / scale)`, clamped to `−127..=127`, where
//! `scale = max |x − offsetᵢ| / 127` over the training data. Queries are encoded
//! the same way, and distances are computed exactly in integers and rescaled
//! by `scale²`. A single global scale keeps the code-space distance
//! proportional to the true distance up to rounding, so no per-dimension
//! weights are needed in the inner loop.
//!
//! For L2, `offset` is the per-dimension mean: translation does not change
//! Euclidean distances and centring halves the range to cover. For inner
//! product and cosine the offset is zero, because `⟨q − μ, x − μ⟩` does not
//! rank `x` the same way as `⟨q, x⟩`.
//!
//! The approximate distances are only used to navigate the graph; the final
//! candidates are re-ranked with exact `f32` distances (see
//! [`SearchOptions::quantized`](crate::SearchOptions::quantized)).

use crate::Metric;
use crate::kernels::int8;

/// Trained quantizer parameters.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ScalarQuantizer {
    offset: Vec<f32>,
    scale: f32,
}

impl ScalarQuantizer {
    /// Fits the quantizer to `vectors` (flat, row-major, `dim` columns).
    pub(crate) fn train(metric: Metric, dim: usize, vectors: &[f32]) -> Self {
        let rows = vectors.len() / dim;
        let mut offset = vec![0.0_f32; dim];
        if metric == Metric::L2 && rows > 0 {
            let mut sum = vec![0.0_f64; dim];
            for row in vectors.chunks_exact(dim) {
                for (s, &x) in sum.iter_mut().zip(row) {
                    *s += f64::from(x);
                }
            }
            for (o, s) in offset.iter_mut().zip(&sum) {
                *o = (s / rows as f64) as f32;
            }
        }
        let max_abs = vectors
            .chunks_exact(dim)
            .flat_map(|row| row.iter().zip(&offset).map(|(&x, &o)| (x - o).abs()))
            .fold(0.0_f32, f32::max);
        let scale = if max_abs > 0.0 && max_abs.is_finite() {
            max_abs / 127.0
        } else {
            1.0
        };
        Self { offset, scale }
    }

    /// Rebuilds a quantizer from stored parts; `None` if they are invalid.
    pub(crate) fn from_parts(offset: Vec<f32>, scale: f32) -> Option<Self> {
        let valid = scale.is_finite() && scale > 0.0 && offset.iter().all(|o| o.is_finite());
        valid.then_some(Self { offset, scale })
    }

    pub(crate) fn offset(&self) -> &[f32] {
        &self.offset
    }

    pub(crate) fn scale(&self) -> f32 {
        self.scale
    }

    /// Appends the codes of `vector` to `out`.
    pub(crate) fn encode_into(&self, vector: &[f32], out: &mut Vec<i8>) {
        let inv = 1.0 / self.scale;
        out.extend(
            vector
                .iter()
                .zip(&self.offset)
                .map(|(&x, &o)| ((x - o) * inv).round().clamp(-127.0, 127.0) as i8),
        );
    }

    /// Approximate distance between two code vectors under `metric`.
    #[inline]
    pub(crate) fn distance(&self, metric: Metric, a: &[i8], b: &[i8]) -> f32 {
        let s2 = self.scale * self.scale;
        match metric {
            Metric::L2 => int8::l2_squared(a, b) as f32 * s2,
            Metric::InnerProduct => -(int8::dot(a, b) as f32) * s2,
            Metric::Cosine => 1.0 - int8::dot(a, b) as f32 * s2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::Kernels;
    use crate::synthetic::Dataset;

    #[test]
    fn approximate_distances_track_exact_ones() {
        let data = Dataset::gaussian(500, 50, 32, 11);
        let dim = data.dim();
        let k = Kernels::best();
        for metric in [Metric::L2, Metric::InnerProduct] {
            let sq = ScalarQuantizer::train(metric, dim, data.base());
            let mut codes = Vec::new();
            for i in 0..data.base_len() {
                sq.encode_into(data.base_vector(i), &mut codes);
            }
            let mut q = Vec::new();
            sq.encode_into(data.query(0), &mut q);
            let mut max_rel = 0.0_f32;
            for i in 0..data.base_len() {
                let exact = metric.distance(&k, data.query(0), data.base_vector(i));
                let approx = sq.distance(metric, &q, &codes[i * dim..(i + 1) * dim]);
                let scale = if metric == Metric::L2 { exact } else { 32.0 };
                max_rel = max_rel.max((exact - approx).abs() / scale);
            }
            assert!(max_rel < 0.05, "{metric}: max relative error {max_rel}");
        }
    }

    #[test]
    fn codes_are_clamped_and_offsets_centre_l2() {
        let vectors = [0.0, 10.0, 2.0, 30.0];
        let sq = ScalarQuantizer::train(Metric::L2, 2, &vectors);
        assert_eq!(sq.offset(), &[1.0, 20.0]);
        assert!((sq.scale() - 10.0 / 127.0).abs() < 1e-6);
        let mut codes = Vec::new();
        sq.encode_into(&[1000.0, -1000.0], &mut codes);
        assert_eq!(codes, vec![127, -127]);
        let ip = ScalarQuantizer::train(Metric::InnerProduct, 2, &vectors);
        assert_eq!(ip.offset(), &[0.0, 0.0]);
    }

    #[test]
    fn degenerate_data_and_invalid_parts() {
        let sq = ScalarQuantizer::train(Metric::L2, 3, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(sq.scale(), 1.0);
        assert!(ScalarQuantizer::from_parts(vec![0.0], 0.0).is_none());
        assert!(ScalarQuantizer::from_parts(vec![f32::NAN], 1.0).is_none());
        assert!(ScalarQuantizer::from_parts(vec![0.0], 0.5).is_some());
    }
}
