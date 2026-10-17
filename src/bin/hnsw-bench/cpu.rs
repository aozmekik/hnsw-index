//! Process CPU time (user + system) via `getrusage(2)`.
//!
//! Wall-clock throughput drops when other processes compete for cores. CPU
//! time only counts the time this process actually ran, so "queries per CPU
//! second" of a single-threaded loop is much less sensitive to a busy machine
//! (cache interference from other processes still applies). Only available on
//! 64-bit Unix; elsewhere the bench reports wall-clock numbers only.

#[cfg(all(unix, target_pointer_width = "64"))]
mod imp {
    /// `struct timeval`: `suseconds_t` is 32-bit on Apple platforms (followed
    /// by padding) and 64-bit on 64-bit Linux and the BSDs.
    #[repr(C)]
    #[derive(Default)]
    struct Timeval {
        sec: i64,
        #[cfg(target_vendor = "apple")]
        usec: i32,
        #[cfg(target_vendor = "apple")]
        _pad: i32,
        #[cfg(not(target_vendor = "apple"))]
        usec: i64,
    }

    /// `struct rusage`: two `timeval`s followed by 14 `long`s. The tail is
    /// over-sized on purpose, so the kernel can never write past the buffer.
    #[repr(C)]
    #[derive(Default)]
    struct Rusage {
        utime: Timeval,
        stime: Timeval,
        rest: [i64; 32],
    }

    unsafe extern "C" {
        fn getrusage(who: i32, usage: *mut Rusage) -> i32;
    }

    const RUSAGE_SELF: i32 = 0;

    // `usec` is `i32` on Apple targets and `i64` elsewhere; `as f64` covers both.
    #[allow(clippy::cast_lossless)]
    pub(crate) fn cpu_seconds() -> Option<f64> {
        let mut usage = Rusage::default();
        // SAFETY: `usage` is a valid, writable buffer at least as large as
        // `struct rusage` on this target, and `RUSAGE_SELF` is a valid `who`.
        let rc = unsafe { getrusage(RUSAGE_SELF, &raw mut usage) };
        let seconds = |t: &Timeval| t.sec as f64 + t.usec as f64 * 1e-6;
        (rc == 0).then(|| seconds(&usage.utime) + seconds(&usage.stime))
    }
}

#[cfg(all(unix, target_pointer_width = "64"))]
pub(crate) use imp::cpu_seconds;

/// CPU time is not measured on this platform.
#[cfg(not(all(unix, target_pointer_width = "64")))]
pub(crate) fn cpu_seconds() -> Option<f64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_time_advances_with_work() {
        let Some(before) = cpu_seconds() else {
            return;
        };
        let mut x = 0_u64;
        for i in 0..20_000_000_u64 {
            x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(i);
        }
        std::hint::black_box(x);
        let after = cpu_seconds().unwrap();
        assert!(after > before, "{before} -> {after}");
    }
}
