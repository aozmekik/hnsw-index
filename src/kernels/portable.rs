//! Portable kernels that auto-vectorize.
//!
//! The loops keep [`LANES`] independent partial sums. Because each lane is a
//! separate dependency chain, LLVM is free to map them onto whatever vector
//! registers the compile target has (two AVX registers, four SSE or NEON
//! registers, ...) without reassociating any single sum, which strict IEEE
//! semantics would forbid.

/// Number of independent accumulators. 16 `f32`s fill four 128-bit or two
/// 256-bit registers, enough to hide floating-point add latency.
const LANES: usize = 16;

pub(super) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let (a_chunks, a_tail) = a.as_chunks::<LANES>();
    let (b_chunks, b_tail) = b.as_chunks::<LANES>();
    let mut acc = [0.0_f32; LANES];
    for (x, y) in a_chunks.iter().zip(b_chunks) {
        for ((s, &x), &y) in acc.iter_mut().zip(x).zip(y) {
            *s += x * y;
        }
    }
    let mut sum = reduce(acc);
    for (x, y) in a_tail.iter().zip(b_tail) {
        sum += x * y;
    }
    sum
}

pub(super) fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    let (a_chunks, a_tail) = a.as_chunks::<LANES>();
    let (b_chunks, b_tail) = b.as_chunks::<LANES>();
    let mut acc = [0.0_f32; LANES];
    for (x, y) in a_chunks.iter().zip(b_chunks) {
        for ((s, &x), &y) in acc.iter_mut().zip(x).zip(y) {
            let d = x - y;
            *s += d * d;
        }
    }
    let mut sum = reduce(acc);
    for (x, y) in a_tail.iter().zip(b_tail) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

/// Pairwise (tree) reduction of the lanes, which is both vectorizable and more
/// accurate than a linear sum.
#[inline]
fn reduce(mut acc: [f32; LANES]) -> f32 {
    let mut width = LANES / 2;
    while width > 0 {
        let (lo, hi) = acc.split_at_mut(width);
        for (l, h) in lo.iter_mut().zip(hi.iter()) {
            *l += *h;
        }
        width /= 2;
    }
    acc[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce_sums_every_lane() {
        let lanes: [f32; LANES] = std::array::from_fn(|i| i as f32);
        assert_eq!(reduce(lanes), (0..LANES).sum::<usize>() as f32);
    }

    #[test]
    fn handles_chunk_and_tail() {
        let a: Vec<f32> = (0..37).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..37).map(|i| 1.0 - i as f32 * 0.25).collect();
        let expected_dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot(&a, &b) - expected_dot).abs() < 1e-3);
        let expected_l2: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
        assert!((l2_squared(&a, &b) - expected_l2).abs() < 1e-3);
    }
}
