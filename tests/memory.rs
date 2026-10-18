//! Checks `HnswIndex::memory_usage` against bytes actually allocated, and the
//! peak memory of `compact`.
//!
//! This file is its own test binary with a single test, so the counting
//! allocator sees no allocations from concurrently running tests.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use hnsw_index::synthetic::Dataset;
use hnsw_index::{HnswIndex, HnswParams, Metric};

struct Counting;

static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

fn grew(size: usize) {
    let live = LIVE_BYTES.fetch_add(size, Ordering::Relaxed) + size;
    PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
}

// SAFETY: every method forwards to `System` with the caller's arguments, so
// `System`'s guarantees carry over; the counter has no effect on allocation.
#[allow(unsafe_code)]
unsafe impl GlobalAlloc for Counting {
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
        LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: forwarded unchanged; the caller upholds the contract.
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if !new.is_null() {
            grew(new_size);
            LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        }
        new
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

#[test]
fn memory_usage_matches_allocations_and_compaction_copies_vectors_once() {
    let data = Dataset::gaussian(20_000, 10, 64, 3);
    let keys: Vec<u64> = (0..data.base_len() as u64).collect();
    let before = LIVE_BYTES.load(Ordering::Relaxed);
    let mut index = HnswIndex::new(data.dim(), Metric::L2, HnswParams::default()).unwrap();
    for (i, &key) in keys.iter().enumerate() {
        index.insert(key, data.base_vector(i)).unwrap();
    }
    let _ = index.search(data.query(0), 10, 64).unwrap();
    let measured = LIVE_BYTES.load(Ordering::Relaxed) - before;
    let estimated = index.memory_usage();
    let ratio = estimated as f64 / measured as f64;
    eprintln!("measured {measured} B, estimated {estimated} B, ratio {ratio:.3}");
    assert!((0.9..=1.1).contains(&ratio), "ratio {ratio}");

    // While compacting, the old index, the new one and the list of live keys
    // and ids exist side by side; nothing else of size proportional to the
    // index may be allocated (an extra copy of the live vectors would add a
    // third of the new index here).
    for key in (0..keys.len() as u64).filter(|k| k % 3 == 0) {
        index.remove(key);
    }
    let live = index.len();
    let before = LIVE_BYTES.load(Ordering::Relaxed);
    PEAK_BYTES.store(before, Ordering::Relaxed);
    index.compact();
    let peak = PEAK_BYTES.load(Ordering::Relaxed) - before;
    let fresh = index.memory_usage();
    let bound = fresh + live * (8 + 4) + fresh / 20;
    eprintln!("compaction: peak growth {peak} B, new index {fresh} B, bound {bound} B");
    assert!(peak <= bound, "peak {peak} > bound {bound}");
}
