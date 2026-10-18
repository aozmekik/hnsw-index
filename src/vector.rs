//! Input validation shared by every index type.

use std::borrow::Cow;

use crate::{Error, Metric};

/// Largest accepted squared Euclidean norm of a vector or query.
///
/// For vectors `a` and `b` with `‖a‖², ‖b‖² ≤ f32::MAX / 8`, Cauchy–Schwarz
/// bounds `|a · b|`, and every partial sum of it, by `f32::MAX / 8`, and
/// `‖a − b‖² ≤ 2‖a‖² + 2‖b‖²` by `f32::MAX / 2`. The remaining factor of two
/// covers rounding in any summation order (the relative error of an `f32` sum
/// of at most [`MAX_DIM`](crate::MAX_DIM) terms is below 1%). So every
/// exact distance between accepted vectors is finite, for every metric and
/// kernel backend.
pub(crate) const MAX_SQUARED_NORM: f64 = f32::MAX as f64 / 8.0;

/// Checks length and finiteness and returns the squared norm (in `f64`).
///
/// A vector whose squared norm exceeds [`MAX_SQUARED_NORM`] is rejected even
/// though its components are finite, so that no distance overflows.
pub(crate) fn check(dim: usize, vector: &[f32]) -> Result<f64, Error> {
    if vector.len() != dim {
        return Err(Error::DimensionMismatch {
            expected: dim,
            actual: vector.len(),
        });
    }
    let mut norm_sq = 0.0_f64;
    for &x in vector {
        if !x.is_finite() {
            return Err(Error::NonFiniteVector);
        }
        norm_sq += f64::from(x) * f64::from(x);
    }
    if norm_sq > MAX_SQUARED_NORM {
        return Err(Error::NonFiniteVector);
    }
    Ok(norm_sq)
}

fn normalized(vector: &[f32], norm_sq: f64) -> impl Iterator<Item = f32> + '_ {
    let inv = 1.0 / norm_sq.sqrt();
    vector.iter().map(move |&x| (f64::from(x) * inv) as f32)
}

/// Validates `vector` for an index of `dim` dimensions under `metric` and
/// appends its stored form to `out`: unchanged for L2 and inner product, unit
/// length for cosine. On error, `out` is left unchanged.
pub(crate) fn prepare_into(
    metric: Metric,
    dim: usize,
    vector: &[f32],
    out: &mut Vec<f32>,
) -> Result<(), Error> {
    let norm_sq = check(dim, vector)?;
    if metric == Metric::Cosine {
        if norm_sq == 0.0 {
            return Err(Error::ZeroVector);
        }
        out.extend(normalized(vector, norm_sq));
    } else {
        out.extend_from_slice(vector);
    }
    Ok(())
}

/// Validates and prepares a flat, row-major batch of `count` vectors.
pub(crate) fn prepare_batch(
    metric: Metric,
    dim: usize,
    count: usize,
    vectors: &[f32],
) -> Result<Vec<f32>, Error> {
    let expected = count.checked_mul(dim).ok_or(Error::CapacityExceeded)?;
    if vectors.len() != expected {
        return Err(Error::DimensionMismatch {
            expected,
            actual: vectors.len(),
        });
    }
    let mut out = Vec::with_capacity(expected);
    for v in vectors.chunks_exact(dim) {
        prepare_into(metric, dim, v, &mut out)?;
    }
    Ok(out)
}

/// Validates a query; borrows it unless cosine requires a normalised copy.
pub(crate) fn prepare_query(
    metric: Metric,
    dim: usize,
    query: &[f32],
) -> Result<Cow<'_, [f32]>, Error> {
    let norm_sq = check(dim, query)?;
    if metric == Metric::Cosine {
        if norm_sq == 0.0 {
            return Err(Error::ZeroVector);
        }
        Ok(Cow::Owned(normalized(query, norm_sq).collect()))
    } else {
        Ok(Cow::Borrowed(query))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_vectors_are_normalised() {
        let mut out = Vec::new();
        prepare_into(Metric::Cosine, 2, &[3.0, 4.0], &mut out).unwrap();
        assert!((out[0] - 0.6).abs() < 1e-7 && (out[1] - 0.8).abs() < 1e-7);
        let q = prepare_query(Metric::Cosine, 2, &[0.0, 2.0]).unwrap();
        assert_eq!(&*q, &[0.0, 1.0]);
        assert!(matches!(
            prepare_query(Metric::L2, 2, &[0.0, 2.0]).unwrap(),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn rejects_bad_input_without_side_effects() {
        let mut out = vec![1.0];
        let cases: [(&[f32], Metric); 5] = [
            (&[1.0], Metric::L2),
            (&[f32::NAN, 1.0], Metric::L2),
            (&[f32::INFINITY, 1.0], Metric::InnerProduct),
            (&[f32::MAX, f32::MAX], Metric::L2),
            (&[0.0, 0.0], Metric::Cosine),
        ];
        for (v, metric) in cases {
            assert!(prepare_into(metric, 2, v, &mut out).is_err(), "{v:?}");
            assert!(prepare_query(metric, 2, v).is_err(), "{v:?}");
        }
        assert_eq!(out, vec![1.0]);
    }

    #[test]
    fn distances_between_accepted_vectors_are_finite() {
        use crate::kernels::{Backend, Kernels};
        for dim in [1, 7, 128, 1024] {
            // The largest accepted vectors, pointing in opposite directions:
            // the worst case for both the dot product and the L2 distance.
            let c = ((MAX_SQUARED_NORM / dim as f64).sqrt() * 0.9999) as f32;
            let (a, b) = (vec![c; dim], vec![-c; dim]);
            assert!(check(dim, &a).is_ok() && check(dim, &b).is_ok());
            for backend in Backend::available() {
                let k = Kernels::new(backend).unwrap();
                assert!(k.l2_squared(&a, &b).is_finite(), "{backend}, dim {dim}");
                assert!(k.dot(&a, &b).is_finite(), "{backend}, dim {dim}");
                assert!(k.dot(&a, &a).is_finite(), "{backend}, dim {dim}");
            }
        }
        // Accepted by an `f32::MAX` bound, but two such vectors of opposite
        // sign are at an L2 distance of 6.8e38, which overflows.
        assert!(matches!(check(1, &[1.3e19]), Err(Error::NonFiniteVector)));
    }

    #[test]
    fn zero_vector_is_fine_for_l2() {
        let mut out = Vec::new();
        prepare_into(Metric::L2, 2, &[0.0, 0.0], &mut out).unwrap();
        assert_eq!(out, vec![0.0, 0.0]);
    }

    #[test]
    fn batch_length_must_match() {
        let err = prepare_batch(Metric::L2, 3, 2, &[0.0; 5]).unwrap_err();
        assert!(matches!(
            err,
            Error::DimensionMismatch {
                expected: 6,
                actual: 5
            }
        ));
    }
}
