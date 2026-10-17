//! Reusable per-search buffers.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use parking_lot::Mutex;

use super::search::Candidate;

/// A set of node ids with O(1) reset: a node is visited iff its mark equals
/// the current epoch, so clearing is a counter increment. Marks are `u16`, so
/// the array is fully rewritten only once every 65 535 resets.
#[derive(Debug, Default)]
pub(crate) struct VisitedSet {
    marks: Vec<u16>,
    epoch: u16,
}

impl VisitedSet {
    /// Empties the set and makes room for ids below `nodes`.
    pub(crate) fn reset(&mut self, nodes: usize) {
        if self.marks.len() < nodes {
            self.marks.resize(nodes, 0);
        }
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.marks.fill(0);
            self.epoch = 1;
        }
    }

    /// Marks `id`; returns `true` if it was not marked before.
    #[inline]
    pub(crate) fn insert(&mut self, id: u32) -> bool {
        let mark = &mut self.marks[id as usize];
        if *mark == self.epoch {
            false
        } else {
            *mark = self.epoch;
            true
        }
    }

    fn heap_bytes(&self) -> usize {
        self.marks.capacity() * size_of::<u16>()
    }
}

/// Buffers for one layer search.
#[derive(Debug, Default)]
pub(crate) struct Scratch {
    pub(crate) visited: VisitedSet,
    /// Min-heap of nodes still to expand.
    pub(crate) candidates: BinaryHeap<Reverse<Candidate>>,
    /// Max-heap of the best `ef` admissible nodes found so far.
    pub(crate) results: BinaryHeap<Candidate>,
}

/// A pool of [`Scratch`] buffers shared by concurrent searches and inserts.
/// The lock is held only to pop or push a buffer, never during a search.
#[derive(Debug, Default)]
pub(crate) struct ScratchPool {
    free: Mutex<Vec<Scratch>>,
}

impl ScratchPool {
    /// Runs `f` with a pooled buffer, creating one if none is free.
    pub(crate) fn with<R>(&self, f: impl FnOnce(&mut Scratch) -> R) -> R {
        let mut scratch = self.free.lock().pop().unwrap_or_default();
        let out = f(&mut scratch);
        self.free.lock().push(scratch);
        out
    }

    /// Heap bytes held by pooled buffers.
    pub(crate) fn heap_bytes(&self) -> usize {
        let free = self.free.lock();
        free.capacity() * size_of::<Scratch>()
            + free
                .iter()
                .map(|s| {
                    s.visited.heap_bytes()
                        + (s.candidates.capacity() + s.results.capacity()) * size_of::<Candidate>()
                })
                .sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_clears_and_survives_epoch_wrap() {
        let mut set = VisitedSet::default();
        set.reset(4);
        assert!(set.insert(3));
        assert!(!set.insert(3));
        for _ in 0..70_000 {
            set.reset(4);
            assert!(set.insert(1));
        }
        set.reset(8);
        assert!(set.insert(7) && set.insert(3) && set.insert(1));
    }

    #[test]
    fn pool_reuses_buffers() {
        let pool = ScratchPool::default();
        pool.with(|s| s.visited.reset(100));
        let bytes = pool.heap_bytes();
        assert!(bytes >= 200);
        pool.with(|s| assert!(s.visited.marks.len() >= 100));
    }
}
