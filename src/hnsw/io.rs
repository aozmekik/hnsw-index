//! Versioned, checksummed binary format.
//!
//! All integers and floats are little-endian. Layout of version 1:
//!
//! ```text
//! offset  size  field
//!      0     8  magic  b"HNSWIDX\0"
//!      8     4  version (u32) = 1
//!     12     1  metric (0 = L2, 1 = inner product, 2 = cosine)
//!     13     1  selection (0 = simple, 1 = heuristic)
//!     14     1  selection flags (bit 0 = extend candidates, bit 1 = keep pruned)
//!     15     1  quantized (0 or 1)
//!     16     4  dim (u32)
//!     20     4  m (u32)
//!     24     4  m0 (u32)
//!     28     4  ef_construction (u32)
//!     32     8  level multiplier (f64)
//!     40     8  seed (u64)
//!     48     4  node count n, tombstones included (u32)
//!     52     4  entry point id (u32, 0xFFFF_FFFF if n = 0)
//!     56  8·n   keys (u64 each)
//!          n    levels (u8 each)
//!          n    tombstone flags (u8 each, 0 or 1)
//!     4·n·dim   vectors (f32 each, as stored: unit length for cosine)
//!          …    per node, per layer 0..=level: count (u32), then count ids (u32)
//!     4·dim+4   if quantized: offsets (f32 each), then scale (f32)
//!          4    CRC-32 (IEEE) of every preceding byte
//! ```
//!
//! Loading treats the input as untrusted. The magic, version and checksum are
//! checked first; then every field is validated before it is used: enum tags,
//! parameter ranges, every vector passing the same checks as
//! [`HnswIndex::insert`] (finite, squared norm in range, unit length for
//! cosine), unique live keys, neighbor counts within capacity, and neighbor
//! ids in range, unique within their list, not self-referencing, and present
//! on the layer they are linked on, and an entry point on the top level. Any
//! violation is a [`FormatError`], never a panic.
//!
//! Allocation is bounded in three steps, all before the graph storage is
//! allocated:
//!
//! 1. Every column is located in the input (as a borrowed slice) before
//!    anything is parsed out of it, so a node count larger than the input
//!    fails as [`FormatError::Truncated`] without allocating.
//! 2. The adjacency section is walked once without allocating: every
//!    neighbor list must be present, with a count within its capacity. A
//!    file therefore pays at least four bytes for every list it makes the
//!    loader allocate.
//! 3. Adjacency storage is fixed-capacity per layer (see `graph.rs`), so even
//!    a complete file can describe an index much larger than itself: up to
//!    `(M0 + 1) * 4` bytes of storage per four-byte count word. The heap size
//!    of the described index is computed from the header and the level column
//!    and compared with the caller's limit ([`Error::MemoryLimitExceeded`]).
//!
//! The level column must also be statistically plausible for the stored `mL`,
//! which rejects files that put every node on dozens of layers early.

use std::collections::HashMap;
use std::io::{self, BufWriter, Write};
use std::sync::atomic::{AtomicU32, Ordering};

use parking_lot::Mutex;

use super::{HnswIndex, MAX_LEVEL, NO_ENTRY, key_map_heap_bytes, pack_entry, unpack_entry};
use crate::crc32::{self, Crc32};
use crate::quantization::ScalarQuantizer;
use crate::vector;
use crate::{Error, FormatError, HnswParams, MAX_DIM, MAX_NODES, Metric, NeighborSelection};

const MAGIC: [u8; 8] = *b"HNSWIDX\0";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 56;
const TRAILER_LEN: usize = 4;
/// Slack, in levels, on top of twice the expected level sum. Covers small
/// indexes, where a single node can legitimately sit on a high level.
const LEVEL_SUM_SLACK: f64 = 1024.0;
/// Tolerance on `|‖v‖² − 1|` for stored cosine vectors.
const UNIT_NORM_TOLERANCE: f64 = 1e-3;
/// [`HnswIndex::from_bytes`] lets the loaded index hold up to this many times
/// the input size...
const DEFAULT_MEMORY_FACTOR: usize = 64;
/// ...or this many bytes, whichever is larger.
const DEFAULT_MEMORY_FLOOR: usize = 64 << 20;

/// `Write` adapter that checksums everything written through it.
struct CrcWriter<W> {
    inner: W,
    crc: Crc32,
}

impl<W: Write> Write for CrcWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.crc.update(&buf[..n]);
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

pub(super) fn write(index: &HnswIndex, writer: impl Write) -> io::Result<()> {
    let mut w = CrcWriter {
        inner: BufWriter::with_capacity(1 << 16, writer),
        crc: Crc32::new(),
    };
    let params = &index.params;
    let (selection, flags) = match params.selection() {
        NeighborSelection::Simple => (0_u8, 0_u8),
        NeighborSelection::Heuristic {
            extend_candidates,
            keep_pruned,
        } => (
            1,
            u8::from(extend_candidates) | (u8::from(keep_pruned) << 1),
        ),
    };
    let entry = index.entry.load(Ordering::Acquire);
    let entry_id = unpack_entry(entry).map_or(u32::MAX, |(id, _)| id);

    w.write_all(&MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&[
        index.metric.to_tag(),
        selection,
        flags,
        u8::from(index.quantized.is_some()),
    ])?;
    for v in [index.dim, params.m(), params.m0(), params.ef_construction()] {
        w.write_all(&(v as u32).to_le_bytes())?;
    }
    w.write_all(&params.level_multiplier().to_le_bytes())?;
    w.write_all(&params.seed().to_le_bytes())?;
    w.write_all(&(index.node_count() as u32).to_le_bytes())?;
    w.write_all(&entry_id.to_le_bytes())?;

    for key in &index.keys {
        w.write_all(&key.to_le_bytes())?;
    }
    w.write_all(&index.levels)?;
    for &deleted in &index.deleted {
        w.write_all(&[u8::from(deleted)])?;
    }
    for x in &index.vectors {
        w.write_all(&x.to_le_bytes())?;
    }
    let mut neighbors = Vec::new();
    for id in 0..index.node_count() as u32 {
        for layer in 0..=usize::from(index.levels[id as usize]) {
            index.links.read_into(id, layer, &mut neighbors);
            w.write_all(&(neighbors.len() as u32).to_le_bytes())?;
            for n in &neighbors {
                w.write_all(&n.to_le_bytes())?;
            }
        }
    }
    if let Some(q) = &index.quantized {
        for x in q.quantizer.offset() {
            w.write_all(&x.to_le_bytes())?;
        }
        w.write_all(&q.quantizer.scale().to_le_bytes())?;
    }

    let crc = w.crc.finish();
    let mut inner = w.inner;
    inner.write_all(&crc.to_le_bytes())?;
    inner.flush()
}

/// Bounds-checked little-endian reader over the checksummed body.
#[derive(Clone)]
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], FormatError> {
        let end = self.pos.checked_add(n).ok_or(FormatError::Truncated)?;
        let bytes = self.buf.get(self.pos..end).ok_or(FormatError::Truncated)?;
        self.pos = end;
        Ok(bytes)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], FormatError> {
        let bytes = self.take(N)?;
        Ok(bytes.try_into().expect("take returns exactly N bytes"))
    }

    fn u8(&mut self) -> Result<u8, FormatError> {
        Ok(self.array::<1>()?[0])
    }

    fn u32(&mut self) -> Result<u32, FormatError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, FormatError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn f32(&mut self) -> Result<f32, FormatError> {
        Ok(f32::from_le_bytes(self.array()?))
    }

    fn f64(&mut self) -> Result<f64, FormatError> {
        Ok(f64::from_le_bytes(self.array()?))
    }

    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// Reads `count` finite `f32`s.
    fn finite_f32s(&mut self, count: usize, what: &'static str) -> Result<Vec<f32>, FormatError> {
        let bytes = self.take(count.checked_mul(4).ok_or(FormatError::Truncated)?)?;
        let out: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        if out.iter().all(|x| x.is_finite()) {
            Ok(out)
        } else {
            Err(FormatError::Invalid(what))
        }
    }
}

fn invalid(what: &'static str) -> Error {
    Error::Format(FormatError::Invalid(what))
}

pub(super) fn read(bytes: &[u8], max_memory: usize) -> Result<HnswIndex, Error> {
    let body = verify_envelope(bytes)?;
    let mut r = Reader {
        buf: body,
        pos: MAGIC.len() + 4,
    };
    let header = read_header(&mut r)?;
    let columns = read_columns(&mut r, &header)?;
    // Nothing proportional to the graph has been allocated yet. Before it is,
    // make sure the file really contains that graph and that it fits.
    scan_links(r.clone(), columns.levels, &header.params)?;
    let required = required_memory(&header, columns.level_sum);
    if required > max_memory {
        return Err(Error::MemoryLimitExceeded {
            required,
            limit: max_memory,
        });
    }
    let mut index = read_nodes(&header, &columns)?;
    read_links(&mut r, &mut index)?;
    if header.quantized {
        let offset = r.finite_f32s(header.dim, "quantizer offset")?;
        let scale = r.f32()?;
        let quantizer =
            ScalarQuantizer::from_parts(offset, scale).ok_or_else(|| invalid("quantizer"))?;
        index.install_quantizer(quantizer);
    }
    if r.remaining() != 0 {
        return Err(FormatError::TrailingBytes.into());
    }
    finish(index, header.entry_id)
}

/// The default `max_memory` of [`HnswIndex::from_bytes`] for an input of
/// `input_len` bytes.
pub(super) fn default_memory_limit(input_len: usize) -> usize {
    input_len
        .saturating_mul(DEFAULT_MEMORY_FACTOR)
        .max(DEFAULT_MEMORY_FLOOR)
}

/// Checks magic, version and checksum; returns the checksummed body.
fn verify_envelope(bytes: &[u8]) -> Result<&[u8], Error> {
    if bytes.len() < MAGIC.len() + 4 {
        let prefix = &bytes[..bytes.len().min(MAGIC.len())];
        return Err(if MAGIC.starts_with(prefix) {
            FormatError::Truncated
        } else {
            FormatError::BadMagic
        }
        .into());
    }
    if bytes[..MAGIC.len()] != MAGIC {
        return Err(FormatError::BadMagic.into());
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes"));
    if version != VERSION {
        return Err(FormatError::UnsupportedVersion(version).into());
    }
    if bytes.len() < HEADER_LEN + TRAILER_LEN {
        return Err(FormatError::Truncated.into());
    }
    let (body, trailer) = bytes.split_at(bytes.len() - TRAILER_LEN);
    let stored = u32::from_le_bytes(trailer.try_into().expect("4 bytes"));
    let computed = crc32::checksum(body);
    if stored != computed {
        return Err(FormatError::ChecksumMismatch { stored, computed }.into());
    }
    Ok(body)
}

/// The fixed-size header fields after magic and version.
struct Header {
    metric: Metric,
    params: HnswParams,
    dim: usize,
    quantized: bool,
    node_count: usize,
    entry_id: u32,
}

fn read_header(r: &mut Reader<'_>) -> Result<Header, Error> {
    let metric = Metric::from_tag(r.u8()?).ok_or_else(|| invalid("metric"))?;
    let selection = match (r.u8()?, r.u8()?) {
        (0, 0) => NeighborSelection::Simple,
        (1, flags) if flags <= 0b11 => NeighborSelection::Heuristic {
            extend_candidates: flags & 1 != 0,
            keep_pruned: flags & 2 != 0,
        },
        _ => return Err(invalid("neighbor selection")),
    };
    let quantized = match r.u8()? {
        0 => false,
        1 => true,
        _ => return Err(invalid("quantized flag")),
    };
    let dim = r.u32()? as usize;
    if dim == 0 || dim > MAX_DIM {
        return Err(invalid("dimension"));
    }
    let m = r.u32()? as usize;
    let m0 = r.u32()? as usize;
    let ef_construction = r.u32()? as usize;
    let level_multiplier = r.f64()?;
    let seed = r.u64()?;
    let params = HnswParams::new(m)
        .with_m0(m0)
        .with_ef_construction(ef_construction)
        .with_level_multiplier(level_multiplier)
        .with_selection(selection)
        .with_seed(seed);
    params.validate().map_err(|_| invalid("parameters"))?;
    let node_count = r.u32()? as usize;
    if node_count > MAX_NODES {
        return Err(invalid("node count"));
    }
    let entry_id = r.u32()?;
    Ok(Header {
        metric,
        params,
        dim,
        quantized,
        node_count,
        entry_id,
    })
}

/// The per-node columns, located in the input but not yet parsed.
struct Columns<'a> {
    keys: &'a [u8],
    levels: &'a [u8],
    flags: &'a [u8],
    vectors: &'a [u8],
    /// Sum of all node levels: the number of upper-layer neighbor lists.
    level_sum: usize,
}

/// Locates the key, level, tombstone and vector columns and validates the
/// level and tombstone bytes. Allocates nothing.
fn read_columns<'a>(r: &mut Reader<'a>, h: &Header) -> Result<Columns<'a>, Error> {
    let n = h.node_count;
    let bytes = |width: usize| n.checked_mul(width).ok_or(FormatError::Truncated);
    let keys = r.take(bytes(size_of::<u64>())?)?;
    let levels = r.take(n)?;
    let flags = r.take(n)?;
    let vectors = r.take(bytes(size_of::<f32>() * h.dim)?)?;
    if levels.iter().any(|&l| l > MAX_LEVEL) {
        return Err(invalid("node level"));
    }
    if flags.iter().any(|&f| f > 1) {
        return Err(invalid("tombstone flag"));
    }
    // Levels follow P(level >= l) = exp(-l / mL), so their expected sum is
    // n / (exp(1 / mL) - 1). Allow twice that plus slack.
    let level_sum: usize = levels.iter().map(|&l| usize::from(l)).sum();
    let expected = n as f64 / (1.0 / h.params.level_multiplier()).exp_m1();
    if level_sum as f64 > 2.0 * expected + LEVEL_SUM_SLACK {
        return Err(invalid("implausible level distribution"));
    }
    Ok(Columns {
        keys,
        levels,
        flags,
        vectors,
        level_sum,
    })
}

/// Walks the adjacency section without allocating: every neighbor list of
/// every node must be present, with a count within the layer's capacity.
/// Contents are validated later by [`read_links`].
fn scan_links(mut r: Reader<'_>, levels: &[u8], params: &HnswParams) -> Result<(), Error> {
    for &level in levels {
        for layer in 0..=usize::from(level) {
            let count = r.u32()? as usize;
            let capacity = if layer == 0 { params.m0() } else { params.m() };
            if count > capacity {
                return Err(invalid("neighbor count exceeds capacity"));
            }
            r.take(count * size_of::<u32>())?;
        }
    }
    Ok(())
}

/// Heap bytes of the index the file describes, as
/// [`HnswIndex::memory_usage`] will report them once it is loaded. Saturates
/// instead of overflowing.
fn required_memory(h: &Header, level_sum: usize) -> usize {
    let word = size_of::<AtomicU32>();
    let code = if h.quantized { h.dim } else { 0 };
    let per_node = size_of::<f32>() * h.dim
        + size_of::<u64>()
        + size_of::<u8>()
        + size_of::<bool>()
        + size_of::<Mutex<()>>()
        + size_of::<Box<[AtomicU32]>>()
        + (h.params.m0() + 1) * word
        + code;
    let quantizer = if h.quantized {
        size_of::<f32>() * h.dim
    } else {
        0
    };
    h.node_count
        .saturating_mul(per_node)
        .saturating_add(level_sum.saturating_mul((h.params.m() + 1) * word))
        .saturating_add(key_map_heap_bytes(h.node_count))
        .saturating_add(quantizer)
}

/// Parses the per-node columns and allocates (empty) adjacency storage.
fn read_nodes(h: &Header, c: &Columns<'_>) -> Result<HnswIndex, Error> {
    let keys: Vec<u64> = c
        .keys
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| u64::from_le_bytes(*b))
        .collect();
    let vectors: Vec<f32> = c
        .vectors
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    for v in vectors.chunks_exact(h.dim) {
        // The checks `insert` applies, so that a loaded index only holds
        // vectors `insert` would have accepted.
        let norm_sq = vector::check(h.dim, v).map_err(|_| invalid("vector out of range"))?;
        if h.metric == Metric::Cosine && (norm_sq - 1.0).abs() > UNIT_NORM_TOLERANCE {
            return Err(invalid("cosine vector is not unit length"));
        }
    }

    // The columns are moved in whole; only graph storage and locks are
    // reserved, so nothing is allocated twice.
    let mut index = HnswIndex::empty(h.dim, h.metric, h.params.clone());
    index.links.reserve(h.node_count);
    index.locks.reserve(h.node_count);
    for &level in c.levels {
        index.push_node_storage(level);
    }
    index.keys = keys;
    index.levels = c.levels.to_vec();
    index.deleted = c.flags.iter().map(|&f| f == 1).collect();
    index.vectors = vectors;
    Ok(index)
}

fn read_links(r: &mut Reader<'_>, index: &mut HnswIndex) -> Result<(), Error> {
    let n = index.node_count();
    let (mut neighbors, mut sorted) = (Vec::new(), Vec::new());
    for id in 0..n {
        for layer in 0..=usize::from(index.levels[id]) {
            let count = r.u32()? as usize;
            if count > index.links.capacity(layer) {
                return Err(invalid("neighbor count exceeds capacity"));
            }
            neighbors.clear();
            for _ in 0..count {
                let nb = r.u32()? as usize;
                let valid = nb < n && nb != id && usize::from(index.levels[nb]) >= layer;
                if !valid {
                    return Err(invalid("neighbor id"));
                }
                neighbors.push(nb as u32);
            }
            sorted.clone_from(&neighbors);
            sorted.sort_unstable();
            if sorted.windows(2).any(|w| w[0] == w[1]) {
                return Err(invalid("duplicate neighbor id"));
            }
            index.links.set(id as u32, layer, &neighbors);
        }
    }
    Ok(())
}

/// Rebuilds the key map and installs the entry point.
fn finish(mut index: HnswIndex, entry_id: u32) -> Result<HnswIndex, Error> {
    let n = index.node_count();
    let mut key_to_id = HashMap::with_capacity(n);
    for (id, &key) in index.keys.iter().enumerate() {
        if !index.deleted[id] && key_to_id.insert(key, id as u32).is_some() {
            return Err(invalid("duplicate live key"));
        }
    }
    index.key_to_id = key_to_id;
    let entry = if n == 0 {
        if entry_id != u32::MAX {
            return Err(invalid("entry point of an empty index"));
        }
        NO_ENTRY
    } else {
        let top = index.levels.iter().copied().max().unwrap_or(0);
        if entry_id as usize >= n || index.levels[entry_id as usize] != top {
            return Err(invalid("entry point"));
        }
        pack_entry(entry_id, usize::from(top))
    };
    index.entry.store(entry, Ordering::Release);
    Ok(index)
}
