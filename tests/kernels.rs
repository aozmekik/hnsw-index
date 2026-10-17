//! Every available SIMD backend must agree with the scalar reference.
//!
//! Backends add the same products in different orders (and NEON/AVX2 fuse
//! multiply and add), so results may differ by rounding. The tolerance is
//! relative to the sum of absolute terms, which bounds the rounding error of
//! any summation order: |error| <= ~len * eps * sum(|term|).

use hnsw_index::kernels::{Backend, Kernels};
use proptest::prelude::*;

fn tolerance(terms: impl Iterator<Item = f32>, len: usize) -> f32 {
    let magnitude: f32 = terms.map(f32::abs).sum();
    (len as f32 + 4.0) * f32::EPSILON * magnitude + f32::MIN_POSITIVE
}

fn check(a: &[f32], b: &[f32]) -> Result<(), TestCaseError> {
    let scalar = Kernels::new(Backend::Scalar).unwrap();
    let want_dot = scalar.dot(a, b);
    let want_l2 = scalar.l2_squared(a, b);
    let dot_tol = tolerance(a.iter().zip(b).map(|(x, y)| x * y), a.len());
    let l2_tol = tolerance(a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)), a.len());
    for backend in Backend::available() {
        let k = Kernels::new(backend).unwrap();
        let dot = k.dot(a, b);
        let l2 = k.l2_squared(a, b);
        prop_assert!(
            (dot - want_dot).abs() <= dot_tol,
            "{backend} dot: {dot} vs {want_dot} (tol {dot_tol}, len {})",
            a.len()
        );
        prop_assert!(
            (l2 - want_l2).abs() <= l2_tol,
            "{backend} l2: {l2} vs {want_l2} (tol {l2_tol}, len {})",
            a.len()
        );
        prop_assert!(l2 >= 0.0);
    }
    Ok(())
}

fn pair(max_len: usize) -> impl Strategy<Value = (Vec<f32>, Vec<f32>)> {
    (0..=max_len).prop_flat_map(|len| {
        (
            prop::collection::vec(-100.0_f32..100.0, len),
            prop::collection::vec(-100.0_f32..100.0, len),
        )
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn backends_match_scalar((a, b) in pair(300)) {
        check(&a, &b)?;
    }

    #[test]
    fn backends_match_scalar_on_wide_magnitudes(
        (a, b) in (0..=96_usize).prop_flat_map(|len| (
            prop::collection::vec(prop::num::f32::NORMAL, len),
            prop::collection::vec(prop::num::f32::NORMAL, len),
        )).prop_filter("products must stay finite", |(a, b)| {
            a.iter().zip(b).all(|(x, y)| (x * y).is_finite() && ((x - y) * (x - y)).is_finite())
                && a.iter().zip(b).map(|(x, y)| (x * y).abs()).sum::<f32>().is_finite()
                && a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().is_finite()
        })
    ) {
        check(&a, &b)?;
    }
}

#[test]
fn every_length_up_to_two_full_simd_blocks() {
    // Exercises each tail-loop path (16-, 4/8-wide and scalar remainders).
    for len in 0..=80 {
        let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.37).sin()).collect();
        let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.11).cos()).collect();
        check(&a, &b).unwrap();
    }
}

#[test]
fn exact_on_small_integers() {
    // Small integers and their products are exactly representable, so every
    // summation order gives the same, exact result.
    let a: Vec<f32> = (0..133).map(|i| (i % 7) as f32 - 3.0).collect();
    let b: Vec<f32> = (0..133).map(|i| (i % 5) as f32 - 2.0).collect();
    let want_dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    let want_l2: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
    for backend in Backend::available() {
        let k = Kernels::new(backend).unwrap();
        assert_eq!(k.dot(&a, &b), want_dot, "{backend}");
        assert_eq!(k.l2_squared(&a, &b), want_l2, "{backend}");
    }
}

#[test]
fn reports_which_backends_ran() {
    let names: Vec<&str> = Backend::available().iter().map(|b| b.name()).collect();
    eprintln!("kernel backends tested on this machine: {names:?}");
    assert!(names.len() >= 2);
}
