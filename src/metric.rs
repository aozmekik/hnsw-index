//! Distance metrics.

use std::fmt;

use crate::kernels::Kernels;

/// How the distance between two vectors is measured. Smaller is closer for
/// every metric, so all indexes in this crate rank results in ascending order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Metric {
    /// Squared Euclidean distance `Σ (aᵢ − bᵢ)²`.
    ///
    /// The square root is skipped: it is monotonic, so rankings are identical,
    /// and it would cost a `sqrt` per comparison. Take the square root of
    /// [`Neighbor::distance`](crate::Neighbor::distance) if you need the true
    /// Euclidean distance.
    L2,
    /// Negated inner product `−Σ aᵢ·bᵢ`, for maximum-inner-product search.
    ///
    /// Inner product is not a metric (no triangle inequality), so graph
    /// quality can be lower than for [`Metric::L2`] on the same data.
    InnerProduct,
    /// Cosine distance `1 − cos(a, b)`, in `[0, 2]`.
    ///
    /// Vectors and queries are normalised to unit length when they enter the
    /// index, so each comparison is a single dot product. Zero vectors are
    /// rejected with [`Error::ZeroVector`](crate::Error::ZeroVector).
    Cosine,
}

impl Metric {
    /// Distance between two prepared vectors (unit length for cosine).
    #[inline]
    pub(crate) fn distance(self, kernels: &Kernels, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Self::L2 => kernels.l2_squared(a, b),
            Self::InnerProduct => -kernels.dot(a, b),
            Self::Cosine => 1.0 - kernels.dot(a, b),
        }
    }

    /// Stable numeric tag used by the file format.
    pub(crate) fn to_tag(self) -> u8 {
        match self {
            Self::L2 => 0,
            Self::InnerProduct => 1,
            Self::Cosine => 2,
        }
    }

    /// Inverse of [`Metric::to_tag`].
    pub(crate) fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::L2),
            1 => Some(Self::InnerProduct),
            2 => Some(Self::Cosine),
            _ => None,
        }
    }
}

impl fmt::Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::L2 => "l2",
            Self::InnerProduct => "inner-product",
            Self::Cosine => "cosine",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::Backend;

    #[test]
    fn distances_follow_definitions() {
        let k = Kernels::new(Backend::Scalar).unwrap();
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        assert_eq!(Metric::L2.distance(&k, &a, &b), 2.0);
        assert_eq!(Metric::InnerProduct.distance(&k, &a, &b), -0.0);
        assert_eq!(Metric::Cosine.distance(&k, &a, &b), 1.0);
        assert_eq!(Metric::Cosine.distance(&k, &a, &a), 0.0);
    }

    #[test]
    fn tags_round_trip() {
        for m in [Metric::L2, Metric::InnerProduct, Metric::Cosine] {
            assert_eq!(Metric::from_tag(m.to_tag()), Some(m));
        }
        assert_eq!(Metric::from_tag(3), None);
    }
}
