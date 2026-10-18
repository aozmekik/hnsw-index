//! The HNSW index.

mod build;
mod graph;
mod io;
mod scratch;
mod search;

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;
use rayon::prelude::*;

use self::graph::Links;
use self::scratch::ScratchPool;
use crate::kernels::{Backend, Kernels};
use crate::quantization::ScalarQuantizer;
use crate::vector::{prepare_batch, prepare_into, prepare_query};
use crate::{Error, HnswParams, MAX_DIM, MAX_NODES, Metric, Neighbor};

/// Highest level a node can be assigned. With the default `mL` a level above
/// 8 needs more than 2³² nodes, so the cap only matters for unusual `mL`.
pub(crate) const MAX_LEVEL: u8 = 31;

/// Sentinel for "no entry point" (empty graph).
const NO_ENTRY: u64 = u64::MAX;

fn pack_entry(id: u32, level: usize) -> u64 {
    ((level as u64) << 32) | u64::from(id)
}

fn unpack_entry(word: u64) -> Option<(u32, usize)> {
    (word != NO_ENTRY).then_some((word as u32, (word >> 32) as usize))
}

/// Codes for quantized traversal; see [`crate::quantization`].
#[derive(Debug)]
struct Quantized {
    quantizer: ScalarQuantizer,
    codes: Vec<i8>,
}

/// Options for [`HnswIndex::search_with`].
///
/// ```
/// use hnsw_index::SearchOptions;
///
/// let even = |key: u64| key % 2 == 0;
/// let options = SearchOptions::new(64).filter(&even).quantized(true);
/// assert_eq!(options.ef(), 64);
/// ```
#[derive(Clone, Copy)]
pub struct SearchOptions<'a> {
    ef: usize,
    filter: Option<&'a (dyn Fn(u64) -> bool + Sync)>,
    quantized: bool,
}

impl<'a> SearchOptions<'a> {
    /// Options with candidate list size `ef`, no filter and exact distances.
    #[must_use]
    pub fn new(ef: usize) -> Self {
        Self {
            ef,
            filter: None,
            quantized: false,
        }
    }

    /// Only return keys for which `filter` returns `true`.
    ///
    /// The filter must be `Sync` so that one `SearchOptions` value can be
    /// shared by searches running on several threads.
    #[must_use]
    pub fn filter(mut self, filter: &'a (dyn Fn(u64) -> bool + Sync)) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Traverse the graph with 8-bit quantized distances, then re-rank the
    /// `ef` candidates with exact distances. Requires
    /// [`HnswIndex::enable_quantization`].
    #[must_use]
    pub fn quantized(mut self, quantized: bool) -> Self {
        self.quantized = quantized;
        self
    }

    /// The candidate list size.
    #[must_use]
    pub fn ef(&self) -> usize {
        self.ef
    }
}

impl fmt::Debug for SearchOptions<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SearchOptions")
            .field("ef", &self.ef)
            .field("filter", &self.filter.map(|_| "<fn>"))
            .field("quantized", &self.quantized)
            .finish()
    }
}

/// Summary statistics of the graph structure.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphStats {
    /// Nodes stored, tombstones included.
    pub nodes: usize,
    /// Nodes not deleted.
    pub live: usize,
    /// `nodes_per_layer[l]` is the number of nodes present on layer `l`.
    pub nodes_per_layer: Vec<usize>,
    /// Mean number of neighbours per node on layer 0.
    pub mean_degree_layer0: f64,
}

/// A Hierarchical Navigable Small World graph over fixed-dimension `f32`
/// vectors, after Malkov & Yashunin, "Efficient and robust approximate nearest
/// neighbor search using Hierarchical Navigable Small World graphs"
/// (IEEE TPAMI 2018, arXiv:1603.09320).
///
/// Vectors are identified by caller-chosen `u64` keys. Search takes `&self`
/// and the type is `Send + Sync`, so any number of threads can search one
/// index concurrently; mutation takes `&mut self`.
///
/// ```
/// use hnsw_index::{HnswIndex, HnswParams, Metric};
///
/// let mut index = HnswIndex::new(2, Metric::L2, HnswParams::default())?;
/// index.insert(1, &[0.0, 0.0])?;
/// index.insert(2, &[1.0, 1.0])?;
/// index.insert(3, &[5.0, 5.0])?;
///
/// let hits = index.search(&[0.9, 0.8], 2, 16)?;
/// assert_eq!(hits.iter().map(|n| n.key).collect::<Vec<_>>(), vec![2, 1]);
///
/// index.remove(2);
/// assert_eq!(index.search(&[0.9, 0.8], 1, 16)?[0].key, 1);
/// # Ok::<(), hnsw_index::Error>(())
/// ```
pub struct HnswIndex {
    dim: usize,
    metric: Metric,
    params: HnswParams,
    kernels: Kernels,
    /// Stored vectors, row-major, indexed by node id.
    vectors: Vec<f32>,
    /// Key of each node id.
    keys: Vec<u64>,
    /// Top layer of each node id.
    levels: Vec<u8>,
    /// Tombstones, indexed by node id.
    deleted: Vec<bool>,
    links: Links,
    /// One lock per node, serialising writers of its neighbour lists.
    locks: Vec<Mutex<()>>,
    /// Packed `(level << 32) | id` of the entry point, or [`NO_ENTRY`].
    entry: AtomicU64,
    /// Held while a node above the current top level is being inserted.
    promotion: Mutex<()>,
    /// Live keys only.
    key_to_id: HashMap<u64, u32>,
    scratch: ScratchPool,
    quantized: Option<Quantized>,
}

impl HnswIndex {
    /// An empty index for `dim`-dimensional vectors.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidDimension`] if `dim` is zero or above [`MAX_DIM`];
    /// [`Error::InvalidParameter`] if `params` is out of range.
    pub fn new(dim: usize, metric: Metric, params: HnswParams) -> Result<Self, Error> {
        if dim == 0 || dim > MAX_DIM {
            return Err(Error::InvalidDimension(dim));
        }
        params.validate()?;
        Ok(Self::empty(dim, metric, params))
    }

    /// Unvalidated constructor shared with deserialisation.
    fn empty(dim: usize, metric: Metric, params: HnswParams) -> Self {
        Self {
            dim,
            metric,
            links: Links::new(params.m(), params.m0()),
            params,
            kernels: Kernels::best(),
            vectors: Vec::new(),
            keys: Vec::new(),
            levels: Vec::new(),
            deleted: Vec::new(),
            locks: Vec::new(),
            entry: AtomicU64::new(NO_ENTRY),
            promotion: Mutex::new(()),
            key_to_id: HashMap::new(),
            scratch: ScratchPool::default(),
            quantized: None,
        }
    }

    /// Dimension of the stored vectors.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The metric used for ranking.
    #[must_use]
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// The construction parameters.
    #[must_use]
    pub fn params(&self) -> &HnswParams {
        &self.params
    }

    /// The distance kernel backend in use (detected at construction).
    #[must_use]
    pub fn backend(&self) -> Backend {
        self.kernels.backend()
    }

    /// Number of live (non-deleted) vectors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.key_to_id.len()
    }

    /// Whether there are no live vectors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.key_to_id.is_empty()
    }

    /// Number of tombstoned nodes still in the graph; [`compact`](Self::compact)
    /// removes them.
    #[must_use]
    pub fn deleted_len(&self) -> usize {
        self.node_count() - self.len()
    }

    /// Whether `key` is present and not deleted.
    #[must_use]
    pub fn contains(&self, key: u64) -> bool {
        self.key_to_id.contains_key(&key)
    }

    /// The stored vector for `key`, if live. For [`Metric::Cosine`] this is
    /// the normalised vector.
    #[must_use]
    pub fn get(&self, key: u64) -> Option<&[f32]> {
        self.key_to_id.get(&key).map(|&id| self.vector(id))
    }

    /// Reserves capacity for at least `additional` more vectors, so that a
    /// sequence of [`insert`](Self::insert) calls does not reallocate.
    /// ([`insert_batch`](Self::insert_batch) reserves exactly what it needs.)
    pub fn reserve(&mut self, additional: usize) {
        self.reserve_nodes(additional);
        self.key_to_id.reserve(additional);
        if let Some(q) = &mut self.quantized {
            q.codes.reserve(additional * self.dim);
        }
    }

    /// Inserts one vector and links it into the graph.
    ///
    /// Inserting a key that was previously [`remove`](Self::remove)d is
    /// allowed and stores a new node; the old one stays tombstoned until
    /// [`compact`](Self::compact).
    ///
    /// # Errors
    ///
    /// [`Error::DuplicateKey`] if `key` is live; [`Error::CapacityExceeded`];
    /// or a vector validation error ([`Error::DimensionMismatch`],
    /// [`Error::NonFiniteVector`], [`Error::ZeroVector`]). The index is
    /// unchanged on error.
    pub fn insert(&mut self, key: u64, vector: &[f32]) -> Result<(), Error> {
        if self.contains(key) {
            return Err(Error::DuplicateKey(key));
        }
        if self.node_count() >= MAX_NODES {
            return Err(Error::CapacityExceeded);
        }
        let mut prepared = Vec::with_capacity(self.dim);
        prepare_into(self.metric, self.dim, vector, &mut prepared)?;
        let id = self.append_nodes(&[key], &prepared);
        self.link(id);
        Ok(())
    }

    /// Inserts `keys.len()` vectors given flat and row-major in `vectors`,
    /// linking them in parallel on the current rayon thread pool.
    ///
    /// Use [`rayon::ThreadPool::install`] to control the thread count. The
    /// resulting graph depends on thread scheduling, so unlike repeated
    /// [`insert`](Self::insert) calls it is not bit-for-bit reproducible; its
    /// search quality is equivalent (see the README benchmarks).
    ///
    /// # Errors
    ///
    /// As for [`insert`](Self::insert), plus [`Error::DuplicateKey`] for keys
    /// repeated within the batch. The whole batch is validated before
    /// anything is inserted, so on error the index is unchanged.
    pub fn insert_batch(&mut self, keys: &[u64], vectors: &[f32]) -> Result<(), Error> {
        let prepared = prepare_batch(self.metric, self.dim, keys.len(), vectors)?;
        let mut seen = HashSet::with_capacity(keys.len());
        for &key in keys {
            if self.contains(key) || !seen.insert(key) {
                return Err(Error::DuplicateKey(key));
            }
        }
        if MAX_NODES - self.node_count() < keys.len() {
            return Err(Error::CapacityExceeded);
        }
        self.insert_prepared_parallel(keys, &prepared);
        Ok(())
    }

    /// Appends already-validated vectors and links them in parallel.
    fn insert_prepared_parallel(&mut self, keys: &[u64], prepared: &[f32]) {
        if keys.is_empty() {
            return;
        }
        let first = self.append_nodes(keys, prepared);
        let end = first + keys.len() as u32;
        let mut next = first;
        if self.entry_point().is_none() {
            self.link(first);
            next += 1;
        }
        let this = &*self;
        (next..end).into_par_iter().for_each(|id| this.link(id));
    }

    /// Marks `key` as deleted. Returns `false` if it was not live.
    ///
    /// The node stays in the graph as a tombstone: searches still route
    /// through it, so connectivity is preserved, but it is never returned.
    /// Tombstones cost memory and some search time until
    /// [`compact`](Self::compact) is called.
    pub fn remove(&mut self, key: u64) -> bool {
        match self.key_to_id.remove(&key) {
            Some(id) => {
                self.deleted[id as usize] = true;
                true
            }
            None => false,
        }
    }

    /// Rebuilds the graph from the live vectors only, dropping tombstones.
    ///
    /// Keys and parameters are kept; node levels are re-sampled and vectors
    /// are re-linked in parallel as by [`insert_batch`](Self::insert_batch).
    /// If quantization is enabled, the quantizer is retrained on the live
    /// vectors. Costs about as much as building a new index and temporarily
    /// needs memory for both copies. Does nothing if there are no tombstones.
    pub fn compact(&mut self) {
        if self.deleted_len() == 0 {
            return;
        }
        let live: Vec<u32> = (0..self.node_count() as u32)
            .filter(|&id| !self.deleted[id as usize])
            .collect();
        let keys: Vec<u64> = live.iter().map(|&id| self.keys[id as usize]).collect();
        let mut vectors = Vec::with_capacity(live.len() * self.dim);
        for &id in &live {
            vectors.extend_from_slice(self.vector(id));
        }
        let mut fresh = Self::empty(self.dim, self.metric, self.params.clone());
        fresh.insert_prepared_parallel(&keys, &vectors);
        if self.quantized.is_some() {
            fresh.enable_quantization();
        }
        *self = fresh;
    }

    /// The `k` approximate nearest neighbours of `query`, closest first.
    ///
    /// `ef` is the size of the candidate list on layer 0 (raised to `k` if
    /// smaller): larger values raise recall and cost time. Returns fewer than
    /// `k` results only if fewer than `k` live vectors are reachable.
    ///
    /// # Errors
    ///
    /// Query validation errors ([`Error::DimensionMismatch`],
    /// [`Error::NonFiniteVector`], [`Error::ZeroVector`]).
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<Neighbor>, Error> {
        self.search_with(query, k, &SearchOptions::new(ef))
    }

    /// Like [`search`](Self::search), but only returns keys for which
    /// `filter` returns `true`.
    ///
    /// The filter is applied during traversal (rejected nodes still guide the
    /// search), not after it, so up to `k` matching results are returned
    /// even when most keys are rejected. Very selective filters make the
    /// search visit a large part of the graph; for those, an exact scan of
    /// the matching subset (for example with a
    /// [`BruteForceIndex`](crate::BruteForceIndex)) can be faster.
    ///
    /// # Errors
    ///
    /// As for [`search`](Self::search).
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter: impl Fn(u64) -> bool + Sync,
    ) -> Result<Vec<Neighbor>, Error> {
        self.search_with(query, k, &SearchOptions::new(ef).filter(&filter))
    }

    /// Search with explicit [`SearchOptions`].
    ///
    /// # Errors
    ///
    /// As for [`search`](Self::search), plus [`Error::QuantizationDisabled`]
    /// if a quantized search is requested on an index without quantization.
    pub fn search_with(
        &self,
        query: &[f32],
        k: usize,
        options: &SearchOptions<'_>,
    ) -> Result<Vec<Neighbor>, Error> {
        let query = prepare_query(self.metric, self.dim, query)?;
        let quantized = match (options.quantized, &self.quantized) {
            (false, _) => None,
            (true, Some(q)) => Some(q),
            (true, None) => return Err(Error::QuantizationDisabled),
        };
        if k == 0 || self.is_empty() {
            return Ok(Vec::new());
        }
        let ef = options.ef.max(k);
        let admit = |id: u32| {
            !self.deleted[id as usize] && options.filter.is_none_or(|f| f(self.keys[id as usize]))
        };
        let exact = |id: u32| self.metric.distance(&self.kernels, &query, self.vector(id));

        let found = match quantized {
            None => self.search_graph(ef, &exact, &admit),
            Some(q) => {
                let mut code = Vec::with_capacity(self.dim);
                q.quantizer.encode_into(&query, &mut code);
                let dim = self.dim;
                let approx = |id: u32| {
                    let start = id as usize * dim;
                    q.quantizer
                        .distance(self.metric, &code, &q.codes[start..start + dim])
                };
                let mut found = self.search_graph(ef, &approx, &admit);
                for c in &mut found {
                    c.distance = exact(c.id);
                }
                found
            }
        };
        let mut out: Vec<Neighbor> = found
            .into_iter()
            .map(|c| Neighbor {
                key: self.keys[c.id as usize],
                distance: c.distance,
            })
            .collect();
        out.sort_unstable_by(Neighbor::cmp_by_distance);
        out.truncate(k);
        Ok(out)
    }

    /// Searches every query of a flat, row-major batch in parallel (rayon).
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] if `queries.len()` is not a multiple of
    /// the dimension, or the first query validation error.
    pub fn search_batch(
        &self,
        queries: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<Vec<Neighbor>>, Error> {
        if !queries.len().is_multiple_of(self.dim) {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                actual: queries.len(),
            });
        }
        queries
            .par_chunks_exact(self.dim)
            .map(|q| self.search(q, k, ef))
            .collect()
    }

    /// Trains an 8-bit scalar quantizer on the stored vectors and encodes
    /// them, enabling [`SearchOptions::quantized`]. Calling it again retrains.
    ///
    /// Codes take `dim` bytes per vector on top of the `f32` vectors, which
    /// are kept for re-ranking. Vectors inserted later are encoded with the
    /// existing quantizer; values outside its range are clamped.
    pub fn enable_quantization(&mut self) {
        let quantizer = ScalarQuantizer::train(self.metric, self.dim, &self.vectors);
        self.install_quantizer(quantizer);
    }

    /// Drops the quantizer and its codes.
    pub fn disable_quantization(&mut self) {
        self.quantized = None;
    }

    /// Whether quantized search is available.
    #[must_use]
    pub fn is_quantized(&self) -> bool {
        self.quantized.is_some()
    }

    /// Structure statistics (layer sizes, mean degree).
    #[must_use]
    pub fn stats(&self) -> GraphStats {
        let top = self.levels.iter().copied().max().map_or(0, usize::from);
        let mut nodes_per_layer = vec![0; if self.levels.is_empty() { 0 } else { top + 1 }];
        for &level in &self.levels {
            for count in &mut nodes_per_layer[..=usize::from(level)] {
                *count += 1;
            }
        }
        let degree_sum: usize = (0..self.node_count() as u32)
            .map(|id| self.links.degree(id, 0))
            .sum();
        GraphStats {
            nodes: self.node_count(),
            live: self.len(),
            nodes_per_layer,
            mean_degree_layer0: if self.node_count() == 0 {
                0.0
            } else {
                degree_sum as f64 / self.node_count() as f64
            },
        }
    }

    /// Approximate heap memory held by the index, in bytes.
    ///
    /// Computed from container capacities; allocator overhead and the
    /// hash map's exact bucket layout are estimated.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.vectors.capacity() * size_of::<f32>()
            + self.keys.capacity() * size_of::<u64>()
            + self.levels.capacity()
            + self.deleted.capacity()
            + self.locks.capacity() * size_of::<Mutex<()>>()
            + self.links.heap_bytes()
            + key_map_heap_bytes(self.key_to_id.capacity())
            + self.scratch.heap_bytes()
            + self.quantized.as_ref().map_or(0, |q| {
                q.codes.capacity() + size_of_val(q.quantizer.offset())
            })
    }

    /// Serialises the index (format described in `docs/adr/0003`).
    ///
    /// Output is buffered internally, so `writer` need not be a `BufWriter`.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] if writing fails.
    pub fn write_to(&self, writer: impl Write) -> Result<(), Error> {
        io::write(self, writer).map_err(Error::Io)
    }

    /// Reads an index written by [`write_to`](Self::write_to).
    ///
    /// The whole stream is read into memory first (with
    /// [`Read::read_to_end`]) and then parsed as by
    /// [`from_bytes`](Self::from_bytes), so wrapping `reader` in a
    /// `BufReader` gains nothing. For a stream from an untrusted source, cap
    /// its length with [`Read::take`]; nothing else limits how much this reads.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] if reading fails, otherwise as for
    /// [`from_bytes`](Self::from_bytes).
    pub fn read_from(mut reader: impl Read) -> Result<Self, Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        Self::from_bytes(&bytes)
    }

    /// Serialises the index into a byte vector.
    ///
    /// # Panics
    ///
    /// Never in practice: writing to a `Vec` cannot fail.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        io::write(self, &mut out).expect("writing to a Vec cannot fail");
        out
    }

    /// Parses and validates a serialised index.
    ///
    /// Checks the magic bytes, version and CRC-32, then every structural
    /// invariant (ids in range, capacities, vectors that
    /// [`insert`](Self::insert) would accept, unique keys, ...). Corrupt or
    /// malicious input yields an error, never a panic.
    ///
    /// The loaded index may hold at most 64 times `bytes.len()`, or 64 MiB if
    /// that is larger, as reported by [`memory_usage`](Self::memory_usage).
    /// Graph storage has a fixed capacity per node, so a small file can
    /// describe a large index; the limit is checked before that storage is
    /// allocated, so a hostile file gets an error instead of the memory.
    /// Honestly built indexes need much less than the default; only a large
    /// `M0` (above 256) or level multiplier combined with very
    /// low-dimensional vectors may need
    /// [`from_bytes_with_limit`](Self::from_bytes_with_limit).
    ///
    /// # Errors
    ///
    /// [`Error::Format`] describing the first violation found, or
    /// [`Error::MemoryLimitExceeded`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        io::read(bytes, io::default_memory_limit(bytes.len()))
    }

    /// Like [`from_bytes`](Self::from_bytes), with an explicit limit on the
    /// heap memory the loaded index may hold.
    ///
    /// The heap size of the index the file describes is computed from its
    /// header and node levels, after checking that the file is complete and
    /// before any graph storage is allocated; it is the figure
    /// [`memory_usage`](Self::memory_usage) reports once the index is loaded.
    /// Pass `usize::MAX` for no limit beyond the input itself.
    ///
    /// ```
    /// use hnsw_index::{Error, HnswIndex, HnswParams, Metric};
    ///
    /// let mut index = HnswIndex::new(2, Metric::L2, HnswParams::default())?;
    /// index.insert(1, &[0.5, 0.5])?;
    /// let bytes = index.to_bytes();
    ///
    /// assert!(HnswIndex::from_bytes_with_limit(&bytes, 1 << 20).is_ok());
    /// assert!(matches!(
    ///     HnswIndex::from_bytes_with_limit(&bytes, 100),
    ///     Err(Error::MemoryLimitExceeded { limit: 100, .. })
    /// ));
    /// # Ok::<(), Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// As for [`from_bytes`](Self::from_bytes); [`Error::MemoryLimitExceeded`]
    /// if the index would need more than `max_memory` bytes.
    pub fn from_bytes_with_limit(bytes: &[u8], max_memory: usize) -> Result<Self, Error> {
        io::read(bytes, max_memory)
    }

    // ---- internals shared by the submodules ----

    /// Nodes stored, tombstones included.
    fn node_count(&self) -> usize {
        self.keys.len()
    }

    #[inline]
    fn vector(&self, id: u32) -> &[f32] {
        let start = id as usize * self.dim;
        &self.vectors[start..start + self.dim]
    }

    fn entry_point(&self) -> Option<(u32, usize)> {
        unpack_entry(self.entry.load(Ordering::Acquire))
    }

    fn reserve_nodes(&mut self, additional: usize) {
        self.keys.reserve(additional);
        self.levels.reserve(additional);
        self.deleted.reserve(additional);
        self.locks.reserve(additional);
        self.vectors.reserve(additional * self.dim);
        self.links.reserve(additional);
    }

    /// Adjacency storage and lock for one more node (level only; the caller
    /// pushes key, vector and flags).
    fn push_node_storage(&mut self, level: u8) {
        self.links.push_node(usize::from(level));
        self.locks.push(Mutex::new(()));
    }

    /// Stores prepared vectors as new, unlinked nodes and returns the first id.
    fn append_nodes(&mut self, keys: &[u64], prepared: &[f32]) -> u32 {
        debug_assert_eq!(prepared.len(), keys.len() * self.dim);
        let first = self.node_count() as u32;
        self.reserve_nodes(keys.len());
        for (i, &key) in keys.iter().enumerate() {
            let id = first + i as u32;
            let level = build::sample_level(self.params.seed(), id, self.params.level_multiplier());
            self.push_node_storage(level);
            self.keys.push(key);
            self.levels.push(level);
            self.deleted.push(false);
            self.key_to_id.insert(key, id);
        }
        self.vectors.extend_from_slice(prepared);
        if let Some(q) = &mut self.quantized {
            for v in prepared.chunks_exact(self.dim) {
                q.quantizer.encode_into(v, &mut q.codes);
            }
        }
        first
    }

    fn install_quantizer(&mut self, quantizer: ScalarQuantizer) {
        let mut codes = Vec::with_capacity(self.vectors.len());
        for v in self.vectors.chunks_exact(self.dim) {
            quantizer.encode_into(v, &mut codes);
        }
        self.quantized = Some(Quantized { quantizer, codes });
    }
}

/// Heap bytes of a `HashMap<u64, u32>` with room for `capacity` entries,
/// following hashbrown's layout: a power-of-two number of buckets at most 7/8
/// full, each holding an entry and one control byte.
fn key_map_heap_bytes(capacity: usize) -> usize {
    let buckets = (capacity.saturating_mul(8) / 7)
        .checked_next_power_of_two()
        .unwrap_or(usize::MAX);
    buckets.saturating_mul(size_of::<(u64, u32)>() + 1)
}

impl fmt::Debug for HnswIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HnswIndex")
            .field("dim", &self.dim)
            .field("metric", &self.metric)
            .field("params", &self.params)
            .field("backend", &self.backend())
            .field("len", &self.len())
            .field("deleted", &self.deleted_len())
            .field("quantized", &self.is_quantized())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests;
