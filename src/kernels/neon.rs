//! NEON kernels for `aarch64`.
//!
//! Four independent FMA chains of four lanes each consume 16 floats per
//! iteration; a 4-wide loop and a scalar loop handle the remainder.

use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vsubq_f32};

/// # Safety
///
/// The CPU must support NEON and `a.len() == b.len()`.
#[target_feature(enable = "neon")]
pub(super) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut i = 0;
    // SAFETY: every `vld1q_f32(p.add(i))` reads the four floats at indices
    // `i..i + 4`, and each loop condition guarantees `i + 4 <= len` for every
    // load it guards, so all reads stay inside both slices (length `len`). NEON
    // support is the caller's obligation.
    let mut sum = unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        while i + 16 <= len {
            acc0 = vfmaq_f32(acc0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            acc1 = vfmaq_f32(acc1, vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
            acc2 = vfmaq_f32(acc2, vld1q_f32(pa.add(i + 8)), vld1q_f32(pb.add(i + 8)));
            acc3 = vfmaq_f32(acc3, vld1q_f32(pa.add(i + 12)), vld1q_f32(pb.add(i + 12)));
            i += 16;
        }
        while i + 4 <= len {
            acc0 = vfmaq_f32(acc0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            i += 4;
        }
        vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)))
    };
    for (x, y) in a[i..].iter().zip(&b[i..]) {
        sum += x * y;
    }
    sum
}

/// # Safety
///
/// The CPU must support NEON and `a.len() == b.len()`.
#[target_feature(enable = "neon")]
pub(super) unsafe fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut i = 0;
    // SAFETY: same bounds argument as in `dot`.
    let mut sum = unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        while i + 16 <= len {
            let d0 = vsubq_f32(vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            let d1 = vsubq_f32(vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
            let d2 = vsubq_f32(vld1q_f32(pa.add(i + 8)), vld1q_f32(pb.add(i + 8)));
            let d3 = vsubq_f32(vld1q_f32(pa.add(i + 12)), vld1q_f32(pb.add(i + 12)));
            acc0 = vfmaq_f32(acc0, d0, d0);
            acc1 = vfmaq_f32(acc1, d1, d1);
            acc2 = vfmaq_f32(acc2, d2, d2);
            acc3 = vfmaq_f32(acc3, d3, d3);
            i += 16;
        }
        while i + 4 <= len {
            let d = vsubq_f32(vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            acc0 = vfmaq_f32(acc0, d, d);
            i += 4;
        }
        vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)))
    };
    for (x, y) in a[i..].iter().zip(&b[i..]) {
        let diff = x - y;
        sum += diff * diff;
    }
    sum
}
