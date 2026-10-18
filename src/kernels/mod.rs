//! Distance kernels and runtime backend selection.
//!
//! Every metric in this crate reduces to two primitives over equal-length
//! `f32` slices: the dot product and the squared Euclidean distance. Each
//! primitive has up to four implementations, one per [`Backend`]:
//!
//! | Backend | How it is vectorized | Availability |
//! |---|---|---|
//! | [`Backend::Scalar`] | not at all: one accumulator, strict left-to-right order | everywhere (reference) |
//! | [`Backend::Portable`] | 16 independent accumulators that LLVM auto-vectorizes for the compile target | everywhere |
//! | [`Backend::Neon`] | `std::arch::aarch64` intrinsics, four FMA chains | `aarch64` + runtime check |
//! | [`Backend::Avx2`] | `std::arch::x86_64` AVX2 + FMA intrinsics, four FMA chains | `x86_64` + runtime check |
//!
//! [`Kernels::best`] picks the preferred available backend once per process:
//! a hand-written SIMD backend when the CPU has one, otherwise the portable
//! one. The hand-written kernels are not always faster: on an Apple M1 Pro
//! they measure within 7% of the portable kernels, in either direction (see
//! `docs/benchmarks.md`). They are preferred because their instruction
//! sequence (fused multiply-adds, which also round once instead of twice)
//! does not depend on what the auto-vectorizer makes of a loop for a given
//! compiler and target.
//!
//! Different backends add the same products in a different order, so their
//! results agree only up to floating-point rounding; the test suite checks each
//! backend against [`Backend::Scalar`] with a tolerance proportional to the
//! magnitude of the summed terms.
//!
//! ```
//! use hnsw_index::kernels::{Backend, Kernels};
//!
//! let a = [1.0_f32, 2.0, 3.0];
//! let b = [4.0_f32, 5.0, 6.0];
//! let scalar = Kernels::new(Backend::Scalar).unwrap();
//! assert_eq!(scalar.dot(&a, &b), 32.0);
//! assert_eq!(scalar.l2_squared(&a, &b), 27.0);
//!
//! let best = Kernels::best();
//! assert!((best.dot(&a, &b) - 32.0).abs() < 1e-6);
//! ```

use std::fmt;
use std::sync::OnceLock;

pub(crate) mod int8;
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon;
mod portable;
mod scalar;
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod x86;

/// An implementation strategy for the distance primitives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Backend {
    /// Straightforward single-accumulator loops; the reference implementation.
    Scalar,
    /// Chunked loops with independent accumulators that auto-vectorize.
    Portable,
    /// Hand-written NEON intrinsics (`aarch64` only).
    Neon,
    /// Hand-written AVX2 + FMA intrinsics (`x86_64` only, detected at runtime).
    Avx2,
}

impl Backend {
    /// Every backend this crate knows about, available on this CPU or not.
    pub const ALL: [Self; 4] = [Self::Scalar, Self::Portable, Self::Neon, Self::Avx2];

    /// Whether this backend can run on the current CPU.
    #[must_use]
    pub fn is_available(self) -> bool {
        match self {
            Self::Scalar | Self::Portable => true,
            Self::Neon => neon_detected(),
            Self::Avx2 => avx2_detected(),
        }
    }

    /// The backends that can run on the current CPU, slowest first.
    #[must_use]
    pub fn available() -> Vec<Self> {
        Self::ALL.into_iter().filter(|b| b.is_available()).collect()
    }

    /// A short lowercase name, e.g. `"neon"`.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Portable => "portable",
            Self::Neon => "neon",
            Self::Avx2 => "avx2",
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

fn neon_detected() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

fn avx2_detected() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Signature shared by every `f32` kernel. Kernels are `unsafe fn` because the
/// SIMD variants require CPU features; [`Kernels`] only stores a pointer after
/// checking that its backend is available.
type F32Kernel = unsafe fn(&[f32], &[f32]) -> f32;

/// The distance primitives of one [`Backend`], resolved to function pointers.
///
/// Construct with [`Kernels::new`] (a specific backend) or [`Kernels::best`].
/// Copying is free; an index keeps its own copy so the hot loop never touches
/// global state.
#[derive(Clone, Copy)]
pub struct Kernels {
    backend: Backend,
    dot: F32Kernel,
    l2: F32Kernel,
}

impl Kernels {
    /// The kernels of `backend`, or `None` if it cannot run on this CPU.
    #[must_use]
    pub fn new(backend: Backend) -> Option<Self> {
        if !backend.is_available() {
            return None;
        }
        let (dot, l2): (F32Kernel, F32Kernel) = match backend {
            Backend::Scalar => (scalar::dot, scalar::l2_squared),
            Backend::Portable => (portable::dot, portable::l2_squared),
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => (neon::dot, neon::l2_squared),
            #[cfg(not(target_arch = "aarch64"))]
            Backend::Neon => return None,
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2 => (x86::dot, x86::l2_squared),
            #[cfg(not(target_arch = "x86_64"))]
            Backend::Avx2 => return None,
        };
        Some(Self { backend, dot, l2 })
    }

    /// The preferred available kernels, detected once per process.
    ///
    /// Preference order: AVX2, NEON, portable (see the [module
    /// documentation](self) for why, and for how they compare in speed). The
    /// scalar backend is never chosen automatically; it exists as a
    /// reference.
    #[must_use]
    pub fn best() -> Self {
        static BEST: OnceLock<Kernels> = OnceLock::new();
        *BEST.get_or_init(|| {
            [Backend::Avx2, Backend::Neon]
                .into_iter()
                .find_map(Self::new)
                .unwrap_or(Self {
                    backend: Backend::Portable,
                    dot: portable::dot,
                    l2: portable::l2_squared,
                })
        })
    }

    /// The backend these kernels belong to.
    #[must_use]
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Dot product `Σ aᵢ·bᵢ`.
    ///
    /// # Panics
    ///
    /// Panics if `a` and `b` have different lengths.
    #[inline]
    #[must_use]
    #[allow(unsafe_code)]
    pub fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "dot: slices must have equal length");
        // SAFETY: `Kernels` is only constructed by `Kernels::new`, which checks
        // that the backend's CPU features are present, and the lengths are
        // equal as asserted above. Those are the kernels' only preconditions.
        unsafe { (self.dot)(a, b) }
    }

    /// Squared Euclidean distance `Σ (aᵢ − bᵢ)²`.
    ///
    /// # Panics
    ///
    /// Panics if `a` and `b` have different lengths.
    #[inline]
    #[must_use]
    #[allow(unsafe_code)]
    pub fn l2_squared(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(
            a.len(),
            b.len(),
            "l2_squared: slices must have equal length"
        );
        // SAFETY: see `Kernels::dot`.
        unsafe { (self.l2)(a, b) }
    }
}

impl fmt::Debug for Kernels {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Kernels")
            .field("backend", &self.backend)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_and_portable_are_always_available() {
        assert!(Backend::Scalar.is_available());
        assert!(Backend::Portable.is_available());
        let available = Backend::available();
        assert!(available.contains(&Backend::Scalar));
        assert!(available.contains(&Backend::Portable));
    }

    #[test]
    fn unavailable_backends_yield_none() {
        for backend in Backend::ALL {
            assert_eq!(Kernels::new(backend).is_some(), backend.is_available());
        }
    }

    #[test]
    fn best_is_never_scalar() {
        assert_ne!(Kernels::best().backend(), Backend::Scalar);
    }

    #[test]
    fn simd_backend_is_detected_on_common_targets() {
        if cfg!(target_arch = "aarch64") {
            assert!(Backend::Neon.is_available());
            assert_eq!(Kernels::best().backend(), Backend::Neon);
        }
        assert!(!(Backend::Neon.is_available() && Backend::Avx2.is_available()));
    }

    #[test]
    fn empty_slices_have_zero_distance() {
        for backend in Backend::available() {
            let k = Kernels::new(backend).unwrap();
            assert_eq!(k.dot(&[], &[]), 0.0, "{backend}");
            assert_eq!(k.l2_squared(&[], &[]), 0.0, "{backend}");
        }
    }

    #[test]
    #[should_panic(expected = "equal length")]
    fn mismatched_lengths_panic() {
        let _ = Kernels::best().dot(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    fn debug_shows_backend() {
        let k = Kernels::new(Backend::Scalar).unwrap();
        assert_eq!(format!("{k:?}"), "Kernels { backend: Scalar, .. }");
        assert_eq!(Backend::Avx2.to_string(), "avx2");
    }
}
