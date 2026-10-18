//! Unit tests with access to the graph internals.

use std::collections::{HashSet, VecDeque};

use super::*;
use crate::synthetic::Dataset;
use crate::{BruteForceIndex, FormatError, NeighborSelection};

fn keys(n: usize) -> Vec<u64> {
    (0..n as u64).collect()
}

fn build(data: &Dataset, metric: Metric, params: HnswParams, parallel: bool) -> HnswIndex {
    let mut index = HnswIndex::new(data.dim(), metric, params).unwrap();
    if parallel {
        index
            .insert_batch(&keys(data.base_len()), data.base())
            .unwrap();
    } else {
        for i in 0..data.base_len() {
            index.insert(i as u64, data.base_vector(i)).unwrap();
        }
    }
    index
}

/// Structural invariants every graph must satisfy.
fn assert_invariants(index: &HnswIndex) {
    let n = index.node_count();
    let mut neighbors = Vec::new();
    for id in 0..n as u32 {
        let level = usize::from(index.levels[id as usize]);
        for layer in 0..=level {
            index.links.read_into(id, layer, &mut neighbors);
            assert!(neighbors.len() <= index.links.capacity(layer));
            for &nb in &neighbors {
                assert!((nb as usize) < n, "dangling id");
                assert_ne!(nb, id, "self loop");
                assert!(usize::from(index.levels[nb as usize]) >= layer);
            }
            let unique: HashSet<u32> = neighbors.iter().copied().collect();
            assert_eq!(unique.len(), neighbors.len(), "duplicate neighbor");
        }
    }
    if n > 0 {
        let (entry, top) = index.entry_point().unwrap();
        assert_eq!(usize::from(index.levels[entry as usize]), top);
        assert_eq!(
            index.levels.iter().copied().max().map(usize::from),
            Some(top)
        );
    }
}

/// Nodes reachable on layer 0 from some node of the upper layers (where the
/// greedy descent can hand over), or from the entry point.
fn reachable(index: &HnswIndex) -> usize {
    let Some((entry, _)) = index.entry_point() else {
        return 0;
    };
    let mut seen = vec![false; index.node_count()];
    let mut queue: VecDeque<u32> = (0..index.node_count() as u32)
        .filter(|&id| id == entry || index.levels[id as usize] > 0)
        .collect();
    for &id in &queue {
        seen[id as usize] = true;
    }
    let mut count = queue.len();
    while let Some(id) = queue.pop_front() {
        index.links.for_each(id, 0, |nb| {
            if !seen[nb as usize] {
                seen[nb as usize] = true;
                count += 1;
                queue.push_back(nb);
            }
        });
    }
    count
}

#[test]
fn index_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<HnswIndex>();
    assert_send_sync::<SearchOptions<'static>>();
}

#[test]
fn empty_index_returns_nothing() {
    let index = HnswIndex::new(3, Metric::L2, HnswParams::default()).unwrap();
    assert!(index.is_empty());
    assert_eq!(index.search(&[0.0; 3], 5, 10).unwrap(), []);
    assert_eq!(index.stats().nodes_per_layer, Vec::<usize>::new());
    assert_invariants(&index);
}

#[test]
fn constructor_validates_dimension_and_params() {
    assert!(matches!(
        HnswIndex::new(0, Metric::L2, HnswParams::default()),
        Err(Error::InvalidDimension(0))
    ));
    assert!(matches!(
        HnswIndex::new(MAX_DIM + 1, Metric::L2, HnswParams::default()),
        Err(Error::InvalidDimension(_))
    ));
    assert!(matches!(
        HnswIndex::new(4, Metric::L2, HnswParams::new(1)),
        Err(Error::InvalidParameter { name: "m", .. })
    ));
}

#[test]
fn complete_graph_with_simple_selection_is_exact() {
    // With n <= M every node links to every other node, so the search is
    // exhaustive and must agree with brute force exactly.
    let data = Dataset::gaussian(16, 20, 8, 3);
    let params = HnswParams::new(16).with_selection(NeighborSelection::Simple);
    let index = build(&data, Metric::L2, params, false);
    let mut exact = BruteForceIndex::new(8, Metric::L2).unwrap();
    exact.insert_batch(&keys(16), data.base()).unwrap();
    for q in 0..data.query_len() {
        let got = index.search(data.query(q), 5, 5).unwrap();
        let want = exact.search(data.query(q), 5).unwrap();
        assert_eq!(got, want);
    }
}

#[test]
fn sequential_build_is_deterministic() {
    let data = Dataset::gaussian(500, 10, 8, 1);
    let a = build(&data, Metric::L2, HnswParams::default(), false);
    let b = build(&data, Metric::L2, HnswParams::default(), false);
    assert_eq!(a.to_bytes(), b.to_bytes());
}

#[test]
fn graphs_satisfy_invariants_and_stay_connected() {
    let data = Dataset::clustered(3000, 1, 16, 20, 5);
    for selection in [
        NeighborSelection::Simple,
        NeighborSelection::default(),
        NeighborSelection::Heuristic {
            extend_candidates: true,
            keep_pruned: true,
        },
    ] {
        for parallel in [false, true] {
            let params = HnswParams::new(8).with_selection(selection);
            let index = build(&data, Metric::L2, params, parallel);
            assert_invariants(&index);
            // Plain HNSW does not guarantee that every node keeps an incoming
            // edge; the heuristic keeps clustered data almost fully connected.
            if selection != NeighborSelection::Simple {
                let reached = reachable(&index);
                assert!(
                    reached >= 2970,
                    "{selection:?}, parallel={parallel}: {reached}"
                );
            }
            let stats = index.stats();
            assert_eq!(stats.nodes_per_layer[0], 3000);
            assert!(stats.mean_degree_layer0 > 4.0 && stats.mean_degree_layer0 <= 16.0);
        }
    }
}

#[test]
fn insert_rejects_invalid_input_without_side_effects() {
    let mut index = HnswIndex::new(2, Metric::Cosine, HnswParams::default()).unwrap();
    index.insert(1, &[1.0, 0.0]).unwrap();
    assert!(matches!(
        index.insert(1, &[0.0, 1.0]),
        Err(Error::DuplicateKey(1))
    ));
    assert!(matches!(
        index.insert(2, &[1.0]),
        Err(Error::DimensionMismatch {
            expected: 2,
            actual: 1
        })
    ));
    assert!(matches!(
        index.insert(2, &[f32::NAN, 1.0]),
        Err(Error::NonFiniteVector)
    ));
    assert!(matches!(
        index.insert(2, &[0.0, 0.0]),
        Err(Error::ZeroVector)
    ));
    // One bad vector or a repeated key rejects the whole batch.
    assert!(index.insert_batch(&[2, 3], &[1.0, 1.0, 0.0, 0.0]).is_err());
    assert!(matches!(
        index.insert_batch(&[2, 2], &[1.0, 1.0, 1.0, 2.0]),
        Err(Error::DuplicateKey(2))
    ));
    assert!(matches!(
        index.insert_batch(&[2, 3], &[1.0, 1.0, 1.0]),
        Err(Error::DimensionMismatch { .. })
    ));
    assert_eq!((index.len(), index.node_count()), (1, 1));
    assert!(matches!(
        index.search(&[0.0, 0.0], 1, 10),
        Err(Error::ZeroVector)
    ));
}

#[test]
fn k_and_ef_edge_cases() {
    let data = Dataset::gaussian(100, 1, 4, 2);
    let index = build(&data, Metric::L2, HnswParams::default(), false);
    let q = data.query(0);
    assert_eq!(index.search(q, 0, 10).unwrap(), []);
    // ef below k is raised to k.
    assert_eq!(index.search(q, 10, 1).unwrap().len(), 10);
    // k above the number of vectors returns everything.
    assert_eq!(index.search(q, 500, 500).unwrap().len(), 100);
}

#[test]
fn removed_keys_are_never_returned_and_can_be_reinserted() {
    let data = Dataset::gaussian(300, 5, 8, 9);
    let mut index = build(&data, Metric::L2, HnswParams::default(), false);
    let (entry, _) = index.entry_point().unwrap();
    let entry_key = index.keys[entry as usize];
    // Delete every even key, and the entry point's key whatever its parity,
    // so searches have to start from a tombstone.
    for key in (0..300).step_by(2) {
        assert!(index.remove(key));
    }
    if index.contains(entry_key) {
        assert!(index.remove(entry_key));
    }
    assert!(index.deleted[entry as usize]);
    assert!(!index.remove(0));
    assert!(!index.contains(0));
    for q in 0..data.query_len() {
        let hits = index.search(data.query(q), 20, 50).unwrap();
        assert_eq!(hits.len(), 20);
        for hit in hits {
            assert!(hit.key % 2 == 1 && hit.key != entry_key && index.contains(hit.key));
        }
    }
    // Re-inserting a deleted key creates a fresh node.
    index.insert(0, data.base_vector(0)).unwrap();
    assert_eq!(index.search(data.base_vector(0), 1, 10).unwrap()[0].key, 0);
    assert_eq!(index.get(0), Some(data.base_vector(0)));
}

#[test]
fn removing_everything_yields_empty_results() {
    let data = Dataset::gaussian(50, 1, 4, 4);
    let mut index = build(&data, Metric::L2, HnswParams::default(), false);
    for key in 0..50 {
        index.remove(key);
    }
    assert!(index.is_empty());
    assert_eq!(index.deleted_len(), 50);
    assert_eq!(index.search(data.query(0), 5, 10).unwrap(), []);
    index.compact();
    assert_eq!((index.node_count(), index.deleted_len()), (0, 0));
    assert!(index.entry_point().is_none());
    index.insert(7, data.base_vector(0)).unwrap();
    assert_eq!(index.search(data.query(0), 5, 10).unwrap()[0].key, 7);
}

#[test]
fn compaction_drops_tombstones_and_keeps_live_keys() {
    let data = Dataset::gaussian(1000, 20, 16, 6);
    let mut index = build(&data, Metric::L2, HnswParams::default(), false);
    index.enable_quantization();
    for key in (0..1000).filter(|k| k % 3 == 0) {
        index.remove(key);
    }
    let live = index.len();
    index.compact();
    assert_eq!(
        (index.len(), index.deleted_len(), index.node_count()),
        (live, 0, live)
    );
    assert!(index.is_quantized());
    assert_invariants(&index);
    assert!(reachable(&index) * 100 >= live * 99);
    let mut exact = BruteForceIndex::new(16, Metric::L2).unwrap();
    for key in (0..1000_u64).filter(|k| k % 3 != 0) {
        exact.insert(key, data.base_vector(key as usize)).unwrap();
    }
    let found = index.search_batch(data.queries(), 10, 100).unwrap();
    let truth = exact.search_batch(data.queries(), 10).unwrap();
    assert!(crate::eval::recall_at_k(&truth, &found, 10) > 0.95);
}

#[test]
fn filter_is_applied_during_traversal() {
    let data = Dataset::gaussian(2000, 10, 8, 8);
    let index = build(&data, Metric::L2, HnswParams::default(), false);
    let mut exact = BruteForceIndex::new(8, Metric::L2).unwrap();
    exact.insert_batch(&keys(2000), data.base()).unwrap();
    for q in 0..data.query_len() {
        let query = data.query(q);
        let hits = index
            .search_filtered(query, 10, 64, |k| k % 10 == 3)
            .unwrap();
        assert_eq!(hits.len(), 10);
        assert!(hits.iter().all(|h| h.key % 10 == 3));
        // A filter matching a single key still finds it.
        let only = index.search_filtered(query, 3, 10, |k| k == 1234).unwrap();
        assert_eq!(only.iter().map(|h| h.key).collect::<Vec<_>>(), vec![1234]);
        assert_eq!(index.search_filtered(query, 3, 10, |_| false).unwrap(), []);
        let want = exact.search_filtered(query, 1, |k| k == 1234).unwrap();
        assert_eq!(only[0], want[0]);
    }
}

#[test]
fn filters_need_not_be_sync() {
    use std::cell::Cell;
    use std::rc::Rc;

    let data = Dataset::gaussian(500, 1, 8, 3);
    let index = build(&data, Metric::L2, HnswParams::default(), false);
    // `Rc` and `Cell` are neither `Send` nor `Sync`.
    let allowed: Rc<HashSet<u64>> = Rc::new((0..500).filter(|k| k % 7 == 0).collect());
    let calls = Cell::new(0_usize);
    let hits = index
        .search_filtered(data.query(0), 5, 32, |key| {
            calls.set(calls.get() + 1);
            allowed.contains(&key)
        })
        .unwrap();
    assert_eq!(hits.len(), 5);
    assert!(hits.iter().all(|h| allowed.contains(&h.key)));
    assert!(calls.get() >= 5);
}

#[test]
fn quantized_search_requires_quantizer_and_reranks_exactly() {
    let data = Dataset::gaussian(2000, 20, 32, 12);
    let mut index = build(&data, Metric::L2, HnswParams::default(), false);
    let options = SearchOptions::new(64).quantized(true);
    assert!(matches!(
        index.search_with(data.query(0), 10, &options),
        Err(Error::QuantizationDisabled)
    ));
    index.enable_quantization();
    let mut exact = BruteForceIndex::new(32, Metric::L2).unwrap();
    exact.insert_batch(&keys(2000), data.base()).unwrap();
    let mut found = Vec::new();
    for q in 0..data.query_len() {
        let hits = index.search_with(data.query(q), 10, &options).unwrap();
        // Reported distances are exact, not approximations.
        for h in &hits {
            let d = Metric::L2.distance(&index.kernels, data.query(q), index.get(h.key).unwrap());
            assert_eq!(h.distance, d);
        }
        found.push(hits);
    }
    let truth = exact.search_batch(data.queries(), 10).unwrap();
    assert!(crate::eval::recall_at_k(&truth, &found, 10) > 0.9);
    // Vectors inserted after training are encoded too.
    index.insert(5000, data.query(0)).unwrap();
    assert_eq!(
        index.search_with(data.query(0), 1, &options).unwrap()[0].key,
        5000
    );
    index.disable_quantization();
    assert!(!index.is_quantized());
}

#[test]
fn inner_product_and_cosine_rank_correctly() {
    let mut ip = HnswIndex::new(2, Metric::InnerProduct, HnswParams::default()).unwrap();
    ip.insert_batch(&[1, 2, 3], &[1.0, 0.0, 3.0, 0.0, 0.0, 1.0])
        .unwrap();
    let hits = ip.search(&[1.0, 0.0], 3, 10).unwrap();
    assert_eq!(
        hits.iter().map(|h| h.key).collect::<Vec<_>>(),
        vec![2, 1, 3]
    );
    assert_eq!(hits[0].distance, -3.0);

    let mut cos = HnswIndex::new(2, Metric::Cosine, HnswParams::default()).unwrap();
    cos.insert_batch(&[1, 2], &[10.0, 0.0, 1.0, 1.0]).unwrap();
    let hits = cos.search(&[0.1, 0.0], 2, 10).unwrap();
    assert_eq!(hits[0].key, 1);
    assert!(hits[0].distance.abs() < 1e-6);
    assert!((hits[1].distance - (1.0 - std::f32::consts::FRAC_1_SQRT_2)).abs() < 1e-6);
}

#[test]
fn reserve_avoids_reallocation_during_inserts() {
    let data = Dataset::gaussian(300, 1, 8, 1);
    let mut index = HnswIndex::new(8, Metric::L2, HnswParams::default()).unwrap();
    index.reserve(300);
    let (vectors, keys) = (index.vectors.capacity(), index.keys.capacity());
    for i in 0..300 {
        index.insert(i as u64, data.base_vector(i)).unwrap();
    }
    assert_eq!(
        (index.vectors.capacity(), index.keys.capacity()),
        (vectors, keys)
    );
}

#[test]
fn memory_usage_accounts_for_vectors_and_links() {
    let data = Dataset::gaussian(1000, 1, 32, 1);
    let index = build(&data, Metric::L2, HnswParams::default(), false);
    let floor = 1000 * (32 * 4 + 33 * 4 + 8);
    let usage = index.memory_usage();
    assert!(usage > floor && usage < 2 * floor, "{usage}");
}

#[test]
fn round_trip_preserves_everything() {
    let data = Dataset::clustered(500, 10, 8, 5, 2);
    let params = HnswParams::new(6).with_selection(NeighborSelection::Heuristic {
        extend_candidates: true,
        keep_pruned: false,
    });
    let mut index = build(&data, Metric::Cosine, params, true);
    for key in [3, 50, 499] {
        index.remove(key);
    }
    index.enable_quantization();
    let bytes = index.to_bytes();
    let loaded = HnswIndex::from_bytes(&bytes).unwrap();
    assert_eq!(loaded.to_bytes(), bytes);
    assert_eq!(loaded.params(), index.params());
    assert_eq!((loaded.len(), loaded.deleted_len()), (497, 3));
    assert_invariants(&loaded);
    let options = SearchOptions::new(32).quantized(true);
    for q in 0..data.query_len() {
        let query = data.query(q);
        assert_eq!(
            loaded.search(query, 5, 32).unwrap(),
            index.search(query, 5, 32).unwrap()
        );
        assert_eq!(
            loaded.search_with(query, 5, &options).unwrap(),
            index.search_with(query, 5, &options).unwrap()
        );
    }
    let mut via_io = Vec::new();
    index.write_to(&mut via_io).unwrap();
    assert_eq!(via_io, bytes);
    let empty = HnswIndex::new(3, Metric::L2, HnswParams::default()).unwrap();
    let reloaded = HnswIndex::read_from(empty.to_bytes().as_slice()).unwrap();
    assert!(reloaded.is_empty());
}

#[test]
fn loader_reports_envelope_errors() {
    let data = Dataset::gaussian(20, 1, 4, 1);
    let bytes = build(&data, Metric::L2, HnswParams::default(), false).to_bytes();
    let err = |b: &[u8]| match HnswIndex::from_bytes(b) {
        Err(Error::Format(e)) => e,
        other => panic!("expected a format error, got {other:?}"),
    };
    assert_eq!(err(b""), FormatError::Truncated);
    assert_eq!(err(b"HNSW"), FormatError::Truncated);
    assert_eq!(err(b"not an index file"), FormatError::BadMagic);
    let mut v2 = bytes.clone();
    v2[8] = 2;
    assert_eq!(err(&v2), FormatError::UnsupportedVersion(2));
    let mut flipped = bytes.clone();
    flipped[100] ^= 0x10;
    assert!(matches!(
        err(&flipped),
        FormatError::ChecksumMismatch { .. }
    ));
    for len in [12, 40, bytes.len() / 2, bytes.len() - 1] {
        assert!(
            HnswIndex::from_bytes(&bytes[..len]).is_err(),
            "prefix {len}"
        );
    }
}

/// Recomputes the trailer so a mutation reaches structural validation.
fn with_valid_crc(mut bytes: Vec<u8>) -> Vec<u8> {
    let body_len = bytes.len() - 4;
    let crc = crate::crc32::checksum(&bytes[..body_len]);
    bytes[body_len..].copy_from_slice(&crc.to_le_bytes());
    bytes
}

#[test]
fn loader_rejects_structural_violations() {
    let data = Dataset::gaussian(30, 1, 4, 1);
    let bytes = build(&data, Metric::L2, HnswParams::default(), false).to_bytes();
    let invalid = |offset: usize, value: &[u8]| {
        let mut b = bytes.clone();
        b[offset..offset + value.len()].copy_from_slice(value);
        match HnswIndex::from_bytes(&with_valid_crc(b)) {
            Err(Error::Format(e)) => e,
            other => panic!("offset {offset}: expected a format error, got {other:?}"),
        }
    };
    assert_eq!(invalid(12, &[9]), FormatError::Invalid("metric"));
    assert_eq!(
        invalid(13, &[0, 1]),
        FormatError::Invalid("neighbor selection")
    );
    assert_eq!(invalid(15, &[2]), FormatError::Invalid("quantized flag"));
    assert_eq!(
        invalid(16, &0_u32.to_le_bytes()),
        FormatError::Invalid("dimension")
    );
    assert_eq!(
        invalid(20, &1_u32.to_le_bytes()),
        FormatError::Invalid("parameters")
    );
    assert_eq!(
        invalid(32, &f64::NAN.to_le_bytes()),
        FormatError::Invalid("parameters")
    );
    assert_eq!(
        invalid(48, &u32::MAX.to_le_bytes()),
        FormatError::Invalid("node count")
    );
    assert_eq!(
        invalid(48, &1_000_000_u32.to_le_bytes()),
        FormatError::Truncated
    );
    assert_eq!(
        invalid(52, &30_u32.to_le_bytes()),
        FormatError::Invalid("entry point")
    );
    // Two live nodes with the same key.
    assert_eq!(
        invalid(56 + 8, &0_u64.to_le_bytes()),
        FormatError::Invalid("duplicate live key")
    );
    let levels = 56 + 30 * 8;
    assert_eq!(invalid(levels, &[200]), FormatError::Invalid("node level"));
    assert_eq!(
        invalid(levels + 30, &[7]),
        FormatError::Invalid("tombstone flag")
    );
    let vectors = levels + 60;
    assert_eq!(
        invalid(vectors, &f32::INFINITY.to_le_bytes()),
        FormatError::Invalid("vector out of range")
    );
    // Finite components, but a norm `insert` rejects: distances would overflow.
    assert_eq!(
        invalid(vectors, &3e38_f32.to_le_bytes()),
        FormatError::Invalid("vector out of range")
    );
    let links = vectors + 30 * 4 * 4;
    assert_eq!(
        invalid(links, &1000_u32.to_le_bytes()),
        FormatError::Invalid("neighbor count exceeds capacity")
    );
    assert_eq!(
        invalid(links + 4, &999_u32.to_le_bytes()),
        FormatError::Invalid("neighbor id")
    );
    let count = u32::from_le_bytes(bytes[links..links + 4].try_into().unwrap());
    assert!(count >= 2, "node 0 needs two neighbors for this case");
    let first_neighbor = bytes[links + 4..links + 8].to_vec();
    assert_eq!(
        invalid(links + 8, &first_neighbor),
        FormatError::Invalid("duplicate neighbor id")
    );
    let mut longer = bytes[..bytes.len() - 4].to_vec();
    longer.extend_from_slice(&[0, 0, 0, 0, 0]);
    assert!(matches!(
        HnswIndex::from_bytes(&with_valid_crc(longer)),
        Err(Error::Format(FormatError::TrailingBytes))
    ));
}

#[test]
fn loader_rejects_implausible_levels_before_allocating() {
    // A file claiming that every node lives on 31 layers would make the
    // loader reserve 31 * (M + 1) words per node. It must be rejected before
    // the adjacency storage is allocated.
    let data = Dataset::gaussian(2000, 1, 1, 1);
    let index = build(&data, Metric::L2, HnswParams::new(64), false);
    let mut bytes = index.to_bytes();
    let levels = 56 + 2000 * 8;
    for b in &mut bytes[levels..levels + 2000] {
        *b = MAX_LEVEL;
    }
    assert!(matches!(
        HnswIndex::from_bytes(&with_valid_crc(bytes)),
        Err(Error::Format(FormatError::Invalid(
            "implausible level distribution"
        )))
    ));
}

#[test]
fn loader_rejects_vectors_outside_the_cosine_unit_sphere() {
    let mut index = HnswIndex::new(2, Metric::Cosine, HnswParams::default()).unwrap();
    index.insert_batch(&[1, 2], &[1.0, 0.0, 0.0, 1.0]).unwrap();
    let mut bytes = index.to_bytes();
    let vectors = 56 + 2 * 8 + 2 + 2;
    bytes[vectors..vectors + 4].copy_from_slice(&2.0_f32.to_le_bytes());
    assert!(matches!(
        HnswIndex::from_bytes(&with_valid_crc(bytes)),
        Err(Error::Format(FormatError::Invalid(
            "cosine vector is not unit length"
        )))
    ));
}

#[test]
fn memory_limit_is_the_loaded_memory_usage() {
    for (metric, quantize) in [(Metric::L2, false), (Metric::Cosine, true)] {
        let data = Dataset::gaussian(3000, 1, 24, 4);
        let mut index = build(&data, metric, HnswParams::new(12), true);
        if quantize {
            index.enable_quantization();
        }
        let bytes = index.to_bytes();
        let Err(Error::MemoryLimitExceeded { required, limit: 0 }) =
            HnswIndex::from_bytes_with_limit(&bytes, 0)
        else {
            panic!("a zero limit must be exceeded");
        };
        assert!(matches!(
            HnswIndex::from_bytes_with_limit(&bytes, required - 1),
            Err(Error::MemoryLimitExceeded { .. })
        ));
        let loaded = HnswIndex::from_bytes_with_limit(&bytes, required).unwrap();
        // The estimate is made before loading; it must describe the result.
        let usage = loaded.memory_usage();
        assert!(
            usage.abs_diff(required) * 1000 <= required,
            "required {required}, memory_usage {usage}"
        );
    }
}

#[test]
fn default_memory_limit_admits_low_dimensional_indexes_with_m0_256() {
    // One-dimensional vectors are the worst case for the ratio of memory to
    // file size: the heuristic keeps about four neighbors per node, so the
    // file is small while every node still owns M0 + 1 slots.
    let data = Dataset::gaussian(20_000, 1, 1, 5);
    let index = build(&data, Metric::L2, HnswParams::new(128), true);
    let bytes = index.to_bytes();
    let Err(Error::MemoryLimitExceeded { required, .. }) =
        HnswIndex::from_bytes_with_limit(&bytes, 0)
    else {
        panic!("a zero limit must be exceeded");
    };
    let ratio = required as f64 / bytes.len() as f64;
    eprintln!("1-d, M0 = 256: memory / file size = {ratio:.1}");
    assert!(ratio < 48.0, "memory / file size = {ratio}");
    assert!(io::default_memory_limit(bytes.len()) >= required);
    assert!(HnswIndex::from_bytes(&bytes).is_ok());
}
