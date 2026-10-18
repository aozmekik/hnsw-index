//! Adjacency storage.
//!
//! Each node owns one fixed-capacity block of `u32` words per layer it lives
//! on: word 0 is the neighbour count, the rest are neighbour ids. Layer-0
//! blocks for all nodes live in one contiguous allocation (most traversal
//! happens there); a node with level `l ≥ 1` additionally owns a boxed slice
//! of `l` upper-layer blocks.
//!
//! All words are `AtomicU32`. During a parallel build, threads read lists
//! that other threads are rewriting; atomics make that well-defined without
//! taking a lock on the read path. Writers serialise per node through the
//! index's node locks and publish a new list by storing the ids first and the
//! count last (`Release`); readers load the count first (`Acquire`). A reader
//! racing with a rewrite may see a mix of old and new ids, all of which are
//! valid node ids, so the worst outcome is a slightly worse search path, never
//! an out-of-bounds access.
//!
//! On the read path, neighbour ids are `Relaxed` loads, which compile to plain
//! loads on `x86_64` and `aarch64`. The count is an `Acquire` load: a plain
//! `MOV` on `x86_64`, but an `LDAPR` on `aarch64`, an ordered load, once per
//! list read. Its cost was not measured separately.

use std::sync::atomic::{AtomicU32, Ordering};

#[derive(Debug)]
pub(crate) struct Links {
    /// `m0 + 1` words per node.
    layer0: Vec<AtomicU32>,
    /// Per node: `level · (m + 1)` words (empty for level-0 nodes).
    upper: Vec<Box<[AtomicU32]>>,
    m: usize,
    m0: usize,
}

fn zeroed(words: usize) -> impl Iterator<Item = AtomicU32> {
    (0..words).map(|_| AtomicU32::new(0))
}

impl Links {
    pub(crate) fn new(m: usize, m0: usize) -> Self {
        Self {
            layer0: Vec::new(),
            upper: Vec::new(),
            m,
            m0,
        }
    }

    /// Reserves space for `additional` more nodes on layer 0.
    pub(crate) fn reserve(&mut self, additional: usize) {
        self.layer0.reserve(additional * (self.m0 + 1));
        self.upper.reserve(additional);
    }

    /// Appends storage for a node that lives on layers `0..=level`.
    pub(crate) fn push_node(&mut self, level: usize) {
        self.layer0.extend(zeroed(self.m0 + 1));
        self.upper.push(zeroed(level * (self.m + 1)).collect());
    }

    /// Neighbour capacity of `layer`.
    #[inline]
    pub(crate) fn capacity(&self, layer: usize) -> usize {
        if layer == 0 { self.m0 } else { self.m }
    }

    /// The `(count, slots)` block of `id` on `layer`.
    ///
    /// Panics if `layer` is above the node's level; callers only follow links
    /// on layers where both endpoints exist, which insertion and
    /// deserialisation guarantee.
    #[inline]
    fn block(&self, id: u32, layer: usize) -> (&AtomicU32, &[AtomicU32]) {
        let id = id as usize;
        let block = if layer == 0 {
            let stride = self.m0 + 1;
            &self.layer0[id * stride..(id + 1) * stride]
        } else {
            let stride = self.m + 1;
            &self.upper[id][(layer - 1) * stride..layer * stride]
        };
        let (count, slots) = block.split_first().expect("blocks are never empty");
        (count, slots)
    }

    /// Calls `f` for each neighbour of `id` on `layer`.
    #[inline]
    pub(crate) fn for_each(&self, id: u32, layer: usize, mut f: impl FnMut(u32)) {
        let (count, slots) = self.block(id, layer);
        let n = (count.load(Ordering::Acquire) as usize).min(slots.len());
        for slot in &slots[..n] {
            f(slot.load(Ordering::Relaxed));
        }
    }

    /// Copies the neighbours of `id` on `layer` into `out` (cleared first).
    pub(crate) fn read_into(&self, id: u32, layer: usize, out: &mut Vec<u32>) {
        out.clear();
        self.for_each(id, layer, |n| out.push(n));
    }

    /// Number of neighbours of `id` on `layer`.
    pub(crate) fn degree(&self, id: u32, layer: usize) -> usize {
        let (count, slots) = self.block(id, layer);
        (count.load(Ordering::Acquire) as usize).min(slots.len())
    }

    /// Replaces the neighbour list of `id` on `layer`.
    ///
    /// The caller must hold the node's lock (or have exclusive access), and
    /// `neighbors` must fit the layer's capacity.
    pub(crate) fn set(&self, id: u32, layer: usize, neighbors: &[u32]) {
        let (count, slots) = self.block(id, layer);
        assert!(
            neighbors.len() <= slots.len(),
            "neighbour list over capacity"
        );
        for (slot, &n) in slots.iter().zip(neighbors) {
            slot.store(n, Ordering::Relaxed);
        }
        count.store(neighbors.len() as u32, Ordering::Release);
    }

    /// Appends `neighbor` to the list of `id` on `layer`, if there is room.
    /// Returns `false` when the list is full. Same locking rule as [`Links::set`].
    pub(crate) fn try_push(&self, id: u32, layer: usize, neighbor: u32) -> bool {
        let (count, slots) = self.block(id, layer);
        let n = count.load(Ordering::Relaxed) as usize;
        if n >= slots.len() {
            return false;
        }
        slots[n].store(neighbor, Ordering::Relaxed);
        count.store(n as u32 + 1, Ordering::Release);
        true
    }

    /// Heap bytes held by the adjacency lists.
    pub(crate) fn heap_bytes(&self) -> usize {
        let word = size_of::<AtomicU32>();
        self.layer0.capacity() * word
            + self.upper.capacity() * size_of::<Box<[AtomicU32]>>()
            + self.upper.iter().map(|b| b.len() * word).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_push_and_read() {
        let mut links = Links::new(2, 3);
        links.push_node(0);
        links.push_node(2);
        assert_eq!((links.capacity(0), links.capacity(1)), (3, 2));
        let mut out = Vec::new();
        links.read_into(1, 2, &mut out);
        assert_eq!(out, []);

        links.set(1, 0, &[0, 0, 0]);
        assert!(!links.try_push(1, 0, 0));
        links.set(1, 2, &[0]);
        assert!(links.try_push(1, 2, 0));
        assert!(!links.try_push(1, 2, 0));
        links.read_into(1, 2, &mut out);
        assert_eq!(out, vec![0, 0]);
        assert_eq!(links.degree(1, 0), 3);
        assert_eq!(links.degree(0, 0), 0);
    }

    #[test]
    #[should_panic(expected = "over capacity")]
    fn set_rejects_oversized_lists() {
        let mut links = Links::new(2, 4);
        links.push_node(1);
        links.set(0, 1, &[1, 2, 3]);
    }
}
