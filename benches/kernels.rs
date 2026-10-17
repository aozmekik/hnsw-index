//! Distance kernel throughput per backend.
//!
//! Run with `cargo bench --bench kernels`.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use hnsw_index::kernels::{Backend, Kernels};

fn vectors(dim: usize) -> (Vec<f32>, Vec<f32>) {
    let a = (0..dim).map(|i| (i as f32 * 0.31).sin()).collect();
    let b = (0..dim).map(|i| (i as f32 * 0.17).cos()).collect();
    (a, b)
}

fn kernels(c: &mut Criterion) {
    for (name, op) in [
        ("dot", Kernels::dot as fn(&Kernels, &[f32], &[f32]) -> f32),
        ("l2_squared", Kernels::l2_squared),
    ] {
        let mut group = c.benchmark_group(name);
        group
            .warm_up_time(Duration::from_millis(500))
            .measurement_time(Duration::from_secs(2));
        for dim in [128, 768] {
            let (a, b) = vectors(dim);
            group.throughput(Throughput::Elements(dim as u64));
            for backend in Backend::available() {
                let k = Kernels::new(backend).expect("available");
                group.bench_with_input(BenchmarkId::new(backend.name(), dim), &dim, |bench, _| {
                    bench.iter(|| op(&k, black_box(&a), black_box(&b)));
                });
            }
        }
        group.finish();
    }
}

criterion_group!(benches, kernels);
criterion_main!(benches);
