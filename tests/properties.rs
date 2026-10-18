//! Property-based tests of the public API.

use std::collections::HashSet;

use hnsw_index::kernels::Kernels;
use hnsw_index::{Error, HnswIndex, HnswParams, Metric, NeighborSelection, SearchOptions};
use proptest::prelude::*;

/// A small random index description: vectors, deletions and parameters.
#[derive(Debug, Clone)]
struct Case {
    dim: usize,
    vectors: Vec<Vec<f32>>,
    deleted: Vec<bool>,
    metric: Metric,
    params: HnswParams,
    parallel: bool,
    quantize: bool,
}

fn metric() -> impl Strategy<Value = Metric> {
    prop_oneof![
        Just(Metric::L2),
        Just(Metric::InnerProduct),
        Just(Metric::Cosine)
    ]
}

fn selection() -> impl Strategy<Value = NeighborSelection> {
    prop_oneof![
        Just(NeighborSelection::Simple),
        (any::<bool>(), any::<bool>()).prop_map(|(extend_candidates, keep_pruned)| {
            NeighborSelection::Heuristic {
                extend_candidates,
                keep_pruned,
            }
        }),
    ]
}

fn case() -> impl Strategy<Value = Case> {
    (1..8_usize, 0..120_usize).prop_flat_map(|(dim, n)| {
        (
            // Cosine rejects zero vectors, so keep one component away from 0.
            prop::collection::vec(
                prop::collection::vec(-10.0_f32..10.0, dim).prop_map(|mut v| {
                    v[0] += 10.5_f32.copysign(v[0]);
                    v
                }),
                n,
            ),
            prop::collection::vec(prop::bool::weighted(0.3), n),
            metric(),
            (2..8_usize, 1..40_usize, selection(), any::<u64>()),
            any::<bool>(),
            any::<bool>(),
        )
            .prop_map(
                move |(vectors, deleted, metric, (m, efc, sel, seed), parallel, quantize)| Case {
                    dim,
                    vectors,
                    deleted,
                    metric,
                    params: HnswParams::new(m)
                        .with_ef_construction(efc)
                        .with_selection(sel)
                        .with_seed(seed),
                    parallel,
                    quantize,
                },
            )
    })
}

fn build(case: &Case) -> HnswIndex {
    let mut index = HnswIndex::new(case.dim, case.metric, case.params.clone()).unwrap();
    // Keys are spread out so that they differ from node ids.
    let keys: Vec<u64> = (0..case.vectors.len() as u64).map(|i| i * 7 + 3).collect();
    if case.parallel {
        let flat: Vec<f32> = case.vectors.concat();
        index.insert_batch(&keys, &flat).unwrap();
    } else {
        for (key, v) in keys.iter().zip(&case.vectors) {
            index.insert(*key, v).unwrap();
        }
    }
    for (key, &deleted) in keys.iter().zip(&case.deleted) {
        if deleted {
            assert!(index.remove(*key));
        }
    }
    if case.quantize {
        index.enable_quantization();
    }
    index
}

fn live_keys(case: &Case) -> HashSet<u64> {
    (0..case.vectors.len() as u64)
        .filter(|&i| !case.deleted[i as usize])
        .map(|i| i * 7 + 3)
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn results_are_sorted_bounded_unique_and_live(
        case in case(),
        query in prop::collection::vec(-20.0_f32..20.0, 8),
        k in 0..20_usize,
        ef in 1..64_usize,
    ) {
        let index = build(&case);
        let mut query = query[..case.dim].to_vec();
        query[0] += 1.0_f32.copysign(query[0]);
        let live = live_keys(&case);
        let kernels = Kernels::best();
        for quantized in [false, case.quantize] {
            let options = SearchOptions::new(ef).quantized(quantized);
            let hits = index.search_with(&query, k, &options).unwrap();
            prop_assert!(hits.len() <= k);
            prop_assert!(hits.len() <= live.len());
            prop_assert!(hits.windows(2).all(|w| w[0].distance <= w[1].distance));
            let keys: HashSet<u64> = hits.iter().map(|h| h.key).collect();
            prop_assert_eq!(keys.len(), hits.len(), "duplicate keys");
            for hit in &hits {
                prop_assert!(live.contains(&hit.key), "returned deleted or unknown key {}", hit.key);
                // Reported distances are exact, also after quantized traversal.
                let stored = index.get(hit.key).unwrap();
                let q = if case.metric == Metric::Cosine {
                    let norm = query.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
                    query.iter().map(|&x| (f64::from(x) / norm) as f32).collect()
                } else {
                    query.clone()
                };
                let want = match case.metric {
                    Metric::L2 => kernels.l2_squared(&q, stored),
                    Metric::InnerProduct => -kernels.dot(&q, stored),
                    Metric::Cosine => 1.0 - kernels.dot(&q, stored),
                    other => unreachable!("no reference distance for {other}"),
                };
                // Bit-identical: the index must report the exact distance.
                prop_assert_eq!(hit.distance.to_bits(), want.to_bits());
            }
        }
    }

    #[test]
    fn filtered_search_only_returns_accepted_keys(case in case(), modulus in 1..5_u64) {
        let index = build(&case);
        let query = vec![1.0_f32; case.dim];
        let accept = |key: u64| key.is_multiple_of(modulus);
        let hits = index.search_filtered(&query, 10, 32, accept).unwrap();
        prop_assert!(hits.iter().all(|h| accept(h.key)));
        let live_accepted = live_keys(&case).into_iter().filter(|&k| accept(k)).count();
        prop_assert!(hits.len() <= live_accepted.min(10));
    }

    #[test]
    fn serialization_round_trips(case in case()) {
        let index = build(&case);
        let bytes = index.to_bytes();
        let loaded = HnswIndex::from_bytes(&bytes).unwrap();
        prop_assert_eq!(loaded.to_bytes(), bytes);
        prop_assert_eq!(loaded.len(), index.len());
        prop_assert_eq!(loaded.deleted_len(), index.deleted_len());
        prop_assert_eq!(loaded.params(), index.params());
        let query = vec![0.5_f32; case.dim];
        prop_assert_eq!(loaded.search(&query, 10, 32).unwrap(), index.search(&query, 10, 32).unwrap());
    }

    #[test]
    fn compaction_keeps_exactly_the_live_keys(case in case()) {
        let mut index = build(&case);
        index.compact();
        prop_assert_eq!(index.deleted_len(), 0);
        let live = live_keys(&case);
        prop_assert_eq!(index.len(), live.len());
        for key in &live {
            prop_assert!(index.contains(*key));
        }
        let query = vec![0.5_f32; case.dim];
        let hits = index.search(&query, live.len(), live.len().max(1)).unwrap();
        prop_assert!(hits.iter().all(|h| live.contains(&h.key)));
    }

    #[test]
    fn truncated_files_are_errors(case in case(), cut in any::<prop::sample::Index>()) {
        let bytes = build(&case).to_bytes();
        let len = cut.index(bytes.len());
        prop_assert!(HnswIndex::from_bytes(&bytes[..len]).is_err());
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1024))]

    /// Mutated files must be rejected or load into a usable index; never panic.
    /// The checksum is recomputed after mutating, so the structural validation
    /// (not just the CRC) is what gets exercised.
    #[test]
    fn mutated_files_never_panic(
        case in case(),
        flips in prop::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 1..8),
        links_only in any::<bool>(),
    ) {
        let index = build(&case);
        let mut bytes = index.to_bytes();
        let body = bytes.len() - 4;
        // Optionally aim at the adjacency section, which starts after the
        // header and the key, level, tombstone and vector columns.
        let links = 56 + case.vectors.len() * (8 + 1 + 1 + 4 * case.dim);
        let (start, span) = if links_only && links < body { (links, body - links) } else { (0, body) };
        for (at, value) in flips {
            bytes[start + at.index(span)] = value;
        }
        let crc = crc32(&bytes[..body]);
        bytes[body..].copy_from_slice(&crc.to_le_bytes());
        if let Ok(loaded) = HnswIndex::from_bytes(&bytes) {
            let query = vec![0.5_f32; loaded.dim()];
            match loaded.search(&query, 5, 16) {
                Ok(hits) => prop_assert!(hits.len() <= 5),
                Err(e) => prop_assert!(matches!(e, Error::ZeroVector), "{e}"),
            }
            let _ = loaded.to_bytes();
        }
    }
}

/// Independent bitwise CRC-32 (IEEE), so the tests do not trust the
/// implementation under test.
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
