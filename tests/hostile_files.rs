//! Crafted index files must not make the loader allocate much memory.
//!
//! Graph storage has a fixed capacity per node, so a small file can describe
//! a huge index. These files are built by hand from the documented format
//! (`src/hnsw/io.rs`) with a valid checksum, and the peak heap growth during
//! loading is measured with a counting allocator.
//!
//! This file is its own test binary with a single test, so the allocator sees
//! no allocations from concurrently running tests.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use hnsw_index::synthetic::Dataset;
use hnsw_index::{Error, FormatError, HnswIndex, HnswParams, Metric};

struct PeakCounting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(size: usize) {
    let live = LIVE.fetch_add(size, Ordering::Relaxed) + size;
    PEAK.fetch_max(live, Ordering::Relaxed);
}

// SAFETY: every method forwards to `System` with the caller's arguments, so
// `System`'s guarantees carry over; the counters have no effect on allocation.
#[allow(unsafe_code)]
unsafe impl GlobalAlloc for PeakCounting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded unchanged; the caller upholds `alloc`'s contract.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            grew(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded unchanged; the caller upholds the contract.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            grew(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded unchanged; the caller upholds the contract.
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: forwarded unchanged; the caller upholds the contract.
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if !new.is_null() {
            grew(new_size);
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        }
        new
    }
}

#[global_allocator]
static ALLOCATOR: PeakCounting = PeakCounting;

/// Runs `f` and returns its result with the peak heap growth it caused.
fn peak_growth<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let out = f();
    (out, PEAK.load(Ordering::Relaxed) - before)
}

/// Independent bitwise CRC-32 (IEEE).
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0_u32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// A file with `n` one-dimensional nodes, all on `level`, under the largest
/// accepted `M`, `M0` and `mL`. Every neighbor list is empty. With
/// `layers_present` below `level + 1`, only that many count words are
/// written per node, so the adjacency section is truncated.
fn crafted(n: u32, level: u8, layers_present: usize) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"HNSWIDX\0");
    b.extend_from_slice(&1_u32.to_le_bytes()); // version
    b.extend_from_slice(&[0, 1, 0, 0]); // L2, heuristic, no flags, not quantized
    for v in [1_u32, 512, 2048, 200] {
        // dim, M, M0, efConstruction
        b.extend_from_slice(&v.to_le_bytes());
    }
    b.extend_from_slice(&10.0_f64.to_le_bytes()); // mL
    b.extend_from_slice(&0_u64.to_le_bytes()); // seed
    b.extend_from_slice(&n.to_le_bytes());
    b.extend_from_slice(&0_u32.to_le_bytes()); // entry point
    for key in 0..u64::from(n) {
        b.extend_from_slice(&key.to_le_bytes());
    }
    b.extend(std::iter::repeat_n(level, n as usize));
    b.extend(std::iter::repeat_n(0_u8, n as usize)); // tombstones
    for i in 0..n {
        b.extend_from_slice(&(i as f32).to_le_bytes());
    }
    for _ in 0..n {
        for _ in 0..layers_present {
            b.extend_from_slice(&0_u32.to_le_bytes());
        }
    }
    let crc = crc32(&b);
    b.extend_from_slice(&crc.to_le_bytes());
    b
}

#[test]
fn crafted_files_are_rejected_without_large_allocations() {
    // 20,000 nodes on level 19 each: plausible under mL = 10, and about
    // 945 MB of fixed-capacity graph storage described by a 1.9 MB file.
    let complete = crafted(20_000, 19, 20);
    assert_eq!(complete.len(), 1_880_060);
    let (result, peak) = peak_growth(|| HnswIndex::from_bytes(&complete));
    match result {
        Err(Error::MemoryLimitExceeded { required, limit }) => {
            eprintln!(
                "complete crafted file: {} B describing {required} B",
                complete.len()
            );
            assert!(required > 900_000_000, "required {required}");
            assert_eq!(limit, 64 * complete.len());
        }
        other => panic!("expected MemoryLimitExceeded, got {other:?}"),
    }
    assert!(peak < 64 * 1024, "peak heap growth {peak} B");

    // The same file cut after the layer-0 count words: the missing lists are
    // noticed before any graph storage is allocated.
    let truncated = crafted(20_000, 19, 1);
    assert_eq!(truncated.len(), 360_060);
    let limit = usize::MAX;
    let (result, peak) = peak_growth(|| HnswIndex::from_bytes_with_limit(&truncated, limit));
    assert!(
        matches!(result, Err(Error::Format(FormatError::Truncated))),
        "{result:?}"
    );
    assert!(peak < 64 * 1024, "peak heap growth {peak} B");

    // A file that fits the limit loads, and loading allocates about what the
    // loaded index holds: no column is copied twice.
    let data = Dataset::gaussian(5_000, 1, 32, 9);
    let keys: Vec<u64> = (0..data.base_len() as u64).collect();
    let mut index = HnswIndex::new(data.dim(), Metric::L2, HnswParams::default()).unwrap();
    index.insert_batch(&keys, data.base()).unwrap();
    let bytes = index.to_bytes();
    drop(index);
    let (loaded, peak) = peak_growth(|| HnswIndex::from_bytes(&bytes).unwrap());
    let usage = loaded.memory_usage();
    eprintln!(
        "file {} B, memory_usage {usage} B, peak growth while loading {peak} B",
        bytes.len()
    );
    assert!(
        peak <= usage + usage / 20 + 64 * 1024,
        "peak {peak}, usage {usage}"
    );
}
