//! Integer kernels for scalar-quantized (`i8`) codes.
//!
//! Products are widened to `i32` per lane; with codes in `-127..=127` a lane
//! can absorb 2^31 / 254² ≈ 33 000 terms, and with 16 lanes the
//! [`MAX_DIM`](crate::MAX_DIM) limit keeps every lane far below that. The
//! final reduction happens in `i64`, so the totals cannot overflow either.
//! LLVM turns these loops into widening multiply-accumulate instructions on
//! both NEON and SSE/AVX targets.

const LANES: usize = 16;

/// Dot product of two code vectors.
pub(crate) fn dot(a: &[i8], b: &[i8]) -> i64 {
    debug_assert_eq!(a.len(), b.len());
    let (a_chunks, a_tail) = a.as_chunks::<LANES>();
    let (b_chunks, b_tail) = b.as_chunks::<LANES>();
    let mut acc = [0_i32; LANES];
    for (x, y) in a_chunks.iter().zip(b_chunks) {
        for ((s, &x), &y) in acc.iter_mut().zip(x).zip(y) {
            *s += i32::from(x) * i32::from(y);
        }
    }
    let mut sum: i64 = acc.iter().map(|&s| i64::from(s)).sum();
    for (&x, &y) in a_tail.iter().zip(b_tail) {
        sum += i64::from(i32::from(x) * i32::from(y));
    }
    sum
}

/// Squared Euclidean distance between two code vectors.
pub(crate) fn l2_squared(a: &[i8], b: &[i8]) -> i64 {
    debug_assert_eq!(a.len(), b.len());
    let (a_chunks, a_tail) = a.as_chunks::<LANES>();
    let (b_chunks, b_tail) = b.as_chunks::<LANES>();
    let mut acc = [0_i32; LANES];
    for (x, y) in a_chunks.iter().zip(b_chunks) {
        for ((s, &x), &y) in acc.iter_mut().zip(x).zip(y) {
            let d = i32::from(x) - i32::from(y);
            *s += d * d;
        }
    }
    let mut sum: i64 = acc.iter().map(|&s| i64::from(s)).sum();
    for (&x, &y) in a_tail.iter().zip(b_tail) {
        let d = i32::from(x) - i32::from(y);
        sum += i64::from(d * d);
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dot(a: &[i8], b: &[i8]) -> i64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| i64::from(x) * i64::from(y))
            .sum()
    }

    fn naive_l2(a: &[i8], b: &[i8]) -> i64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (i64::from(x) - i64::from(y)).pow(2))
            .sum()
    }

    #[test]
    fn matches_naive_for_many_lengths() {
        for len in 0..80 {
            let a: Vec<i8> = (0..len).map(|i| ((i * 37) % 255) as i8).collect();
            let b: Vec<i8> = (0..len).map(|i| ((i * 91 + 13) % 255) as i8).collect();
            assert_eq!(dot(&a, &b), naive_dot(&a, &b), "len {len}");
            assert_eq!(l2_squared(&a, &b), naive_l2(&a, &b), "len {len}");
        }
    }

    #[test]
    fn extreme_codes_at_max_dim_do_not_overflow() {
        let a = vec![127_i8; crate::MAX_DIM];
        let b = vec![-127_i8; crate::MAX_DIM];
        let dim = i64::try_from(crate::MAX_DIM).unwrap();
        assert_eq!(dot(&a, &b), -(127 * 127) * dim);
        assert_eq!(l2_squared(&a, &b), 254 * 254 * dim);
    }
}
