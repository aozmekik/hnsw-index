//! AVX2 + FMA kernels for `x86_64`, selected at runtime.
//!
//! Four independent FMA chains of eight lanes each consume 32 floats per
//! iteration; an 8-wide loop and a scalar loop handle the remainder. The final
//! horizontal sum goes through a stack buffer, which costs a few cycles once
//! per call and keeps the shuffle logic out of the unsafe code.

use std::arch::x86_64::{
    __m256, _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    _mm256_sub_ps,
};

/// Sums the eight lanes of `v`.
///
/// # Safety
///
/// The CPU must support AVX.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn horizontal_sum(v: __m256) -> f32 {
    let mut lanes = [0.0_f32; 8];
    // SAFETY: `lanes` is a writable buffer of exactly eight floats, which is
    // what an unaligned 256-bit store writes.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), v) };
    (lanes[0] + lanes[4]) + (lanes[1] + lanes[5]) + ((lanes[2] + lanes[6]) + (lanes[3] + lanes[7]))
}

/// # Safety
///
/// The CPU must support AVX2 and FMA, and `a.len() == b.len()`.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut i = 0;
    // SAFETY: every `_mm256_loadu_ps(p.add(i))` reads the eight floats at
    // indices `i..i + 8`, and each loop condition guarantees `i + 8 <= len` for
    // every load it guards, so all reads stay inside both slices (length `len`).
    // AVX2/FMA support is the caller's obligation.
    let mut sum = unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        while i + 32 <= len {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
            acc1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i + 8)),
                _mm256_loadu_ps(pb.add(i + 8)),
                acc1,
            );
            acc2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i + 16)),
                _mm256_loadu_ps(pb.add(i + 16)),
                acc2,
            );
            acc3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(i + 24)),
                _mm256_loadu_ps(pb.add(i + 24)),
                acc3,
            );
            i += 32;
        }
        while i + 8 <= len {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
            i += 8;
        }
        horizontal_sum(_mm256_add_ps(
            _mm256_add_ps(acc0, acc1),
            _mm256_add_ps(acc2, acc3),
        ))
    };
    for (x, y) in a[i..].iter().zip(&b[i..]) {
        sum += x * y;
    }
    sum
}

/// # Safety
///
/// The CPU must support AVX2 and FMA, and `a.len() == b.len()`.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut i = 0;
    // SAFETY: same bounds argument as in `dot`.
    let mut sum = unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        while i + 32 <= len {
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
            let d1 = _mm256_sub_ps(
                _mm256_loadu_ps(pa.add(i + 8)),
                _mm256_loadu_ps(pb.add(i + 8)),
            );
            let d2 = _mm256_sub_ps(
                _mm256_loadu_ps(pa.add(i + 16)),
                _mm256_loadu_ps(pb.add(i + 16)),
            );
            let d3 = _mm256_sub_ps(
                _mm256_loadu_ps(pa.add(i + 24)),
                _mm256_loadu_ps(pb.add(i + 24)),
            );
            acc0 = _mm256_fmadd_ps(d0, d0, acc0);
            acc1 = _mm256_fmadd_ps(d1, d1, acc1);
            acc2 = _mm256_fmadd_ps(d2, d2, acc2);
            acc3 = _mm256_fmadd_ps(d3, d3, acc3);
            i += 32;
        }
        while i + 8 <= len {
            let d = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
            acc0 = _mm256_fmadd_ps(d, d, acc0);
            i += 8;
        }
        horizontal_sum(_mm256_add_ps(
            _mm256_add_ps(acc0, acc1),
            _mm256_add_ps(acc2, acc3),
        ))
    };
    for (x, y) in a[i..].iter().zip(&b[i..]) {
        let diff = x - y;
        sum += diff * diff;
    }
    sum
}
