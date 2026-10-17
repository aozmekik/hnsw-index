//! Reference kernels: one accumulator, strict left-to-right summation.
//!
//! Floating-point addition is not associative, so LLVM may not vectorise these
//! loops. That makes them slow but predictable, which is what a reference for
//! equivalence tests needs.

pub(super) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for (x, y) in a.iter().zip(b) {
        sum += x * y;
    }
    sum
}

pub(super) fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for (x, y) in a.iter().zip(b) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_known_values() {
        assert_eq!(dot(&[1.0, -2.0, 0.5], &[3.0, 4.0, 2.0]), -4.0);
        assert_eq!(l2_squared(&[1.0, -2.0, 0.5], &[3.0, 4.0, 2.0]), 42.25);
        assert_eq!(dot(&[], &[]), 0.0);
    }
}
