//! Benchmarks for usearch HNSW implementation
//!
//! Run with: cargo bench --bench usearch_benchmark
//!
//! This enables direct comparison with the arroy benchmarks

#[cfg(feature = "usearch")]
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
#[cfg(feature = "usearch")]
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

/// Generate realistic test vectors (384D, similar to actual embeddings)
fn generate_test_vectors(count: usize, dimensions: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| {
            (0..dimensions)
                .map(|j| {
                    let base = (i as f32) * 0.001;
                    let offset = (j as f32) * 0.01;
                    ((base + offset).sin() + 1.0) * 0.5
                })
                .collect()
        })
        .collect()
}

/// Generate query vectors (similar to test vectors but with slight variations)
fn generate_query_vectors(count: usize, dataset_size: usize, dimensions: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| {
            let base_idx = (i * 7) % dataset_size;
            (0..dimensions)
                .map(|j| {
                    let base = (base_idx as f32) * 0.001;
                    let offset = (j as f32) * 0.01;
                    let noise = (rand::random::<f32>() - 0.5) * 0.1;
                    (((base + offset).sin() + 1.0) * 0.5 + noise).clamp(0.0, 1.0)
                })
                .collect()
        })
        .collect()
}

/// Benchmark usearch index building
#[cfg(feature = "usearch")]
fn bench_usearch_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("usearch_build");

    let sizes = [1000, 5000, 10000, 50000];

    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.sample_size(10);

        group.bench_with_input(BenchmarkId::new("size", size), &size, |b, &size| {
            b.iter(|| {
                let dimensions = black_box(384);
                let mut options = IndexOptions::default();
                options.dimensions = dimensions;
                options.metric = MetricKind::Cos;
                options.quantization = ScalarKind::F32;
                let index = Index::new(&options).expect("Failed to create index");
                index.reserve(size).expect("Failed to reserve capacity");

                let vectors = generate_test_vectors(size, dimensions);

                for (i, vector) in vectors.iter().enumerate() {
                    index
                        .add(black_box(i as u64), black_box(vector.as_slice()))
                        .expect("Failed to add vector");
                }
            });
        });
    }

    group.finish();
}

/// Benchmark usearch search performance
#[cfg(feature = "usearch")]
fn bench_usearch_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("usearch_search");

    let sizes = [1000, 5000, 10000, 50000];
    let k_values = [10, 50, 100];
    let dimensions = 384;

    for &size in &sizes {
        let vectors = generate_test_vectors(size, dimensions);

        let mut options = IndexOptions::default();
        options.dimensions = dimensions;
        options.metric = MetricKind::Cos;
        options.quantization = ScalarKind::F32;
        let index = Index::new(&options).expect("Failed to create index");
        index.reserve(size).expect("Failed to reserve capacity");

        for (i, vector) in vectors.iter().enumerate() {
            index
                .add(i as u64, vector.as_slice())
                .expect("Failed to add vector");
        }

        // Benchmark search with different k values
        for &k in &k_values {
            let query_count = 100;
            let queries = generate_query_vectors(query_count, size, dimensions);

            group.bench_with_input(
                BenchmarkId::from_parameter(format!("size_{}_k_{}", size, k)),
                &(size, k),
                |b, _| {
                    b.iter(|| {
                        for query in &queries {
                            let _ = black_box(index.search(black_box(query.as_slice()), k));
                        }
                    });
                },
            );
        }

        drop(index);
    }

    group.finish();
}

/// Benchmark query throughput (queries per second)
#[cfg(feature = "usearch")]
fn bench_usearch_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("usearch_throughput");

    let sizes = [1000, 5000, 10000, 50000];
    let query_counts = [100, 500, 1000];
    let dimensions = 384;
    let k = 10;

    for &size in &sizes {
        let mut options = IndexOptions::default();
        options.dimensions = dimensions;
        options.metric = MetricKind::Cos;
        options.quantization = ScalarKind::F32;
        let index = Index::new(&options).expect("Failed to create index");
        index.reserve(size).expect("Failed to reserve capacity");

        let vectors = generate_test_vectors(size, dimensions);
        for (i, vector) in vectors.iter().enumerate() {
            index
                .add(i as u64, vector.as_slice())
                .expect("Failed to add vector");
        }

        for &query_count in &query_counts {
            group.throughput(Throughput::Elements(query_count as u64));
            group.sample_size(50);

            let queries = generate_query_vectors(query_count, size, dimensions);

            group.bench_with_input(
                BenchmarkId::from_parameter(format!("size_{}_queries_{}", size, query_count)),
                &query_count,
                |b, _| {
                    b.iter(|| {
                        for query in &queries {
                            let _ = black_box(index.search(black_box(query.as_slice()), k));
                        }
                    });
                },
            );
        }

        drop(index);
    }

    group.finish();
}

/// Benchmark search latency for single queries
#[cfg(feature = "usearch")]
fn bench_usearch_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("usearch_latency");
    group.sample_size(100);

    let sizes = [1000, 5000, 10000, 50000];
    let dimensions = 384;
    let k = 10;

    for &size in &sizes {
        let mut options = IndexOptions::default();
        options.dimensions = dimensions;
        options.metric = MetricKind::Cos;
        options.quantization = ScalarKind::F32;
        let index = Index::new(&options).expect("Failed to create index");
        index.reserve(size).expect("Failed to reserve capacity");

        let vectors = generate_test_vectors(size, dimensions);
        for (i, vector) in vectors.iter().enumerate() {
            index
                .add(i as u64, vector.as_slice())
                .expect("Failed to add vector");
        }

        let queries = generate_query_vectors(100, size, dimensions);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            let mut query_idx = 0;
            b.iter(|| {
                let query = &queries[query_idx];
                query_idx = (query_idx + 1) % queries.len();
                let _ = black_box(index.search(black_box(query.as_slice()), k));
            });
        });

        drop(index);
    }

    group.finish();
}

/// Benchmark serialization/deserialization
#[cfg(feature = "usearch")]
fn bench_usearch_persistence(c: &mut Criterion) {
    let mut group = c.benchmark_group("usearch_persistence");

    let sizes = [1000, 5000, 10000];
    let dimensions = 384;

    for &size in &sizes {
        let mut options = IndexOptions::default();
        options.dimensions = dimensions;
        options.metric = MetricKind::Cos;
        options.quantization = ScalarKind::F32;

        let index = Index::new(&options).expect("Failed to create index");
        index.reserve(size).expect("Failed to reserve capacity");

        let vectors = generate_test_vectors(size, dimensions);
        for (i, vector) in vectors.iter().enumerate() {
            index
                .add(i as u64, vector.as_slice())
                .expect("Failed to add vector");
        }

        // Serialize to buffer
        let mut buffer = vec![0u8; index.serialized_length()];
        index
            .save_to_buffer(&mut buffer)
            .expect("Failed to serialize");

        // Create deserialization index outside the loop to avoid measuring creation overhead
        let deser_index = Index::new(&options).expect("Failed to create index");

        // Benchmark deserialization only
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                deser_index
                    .load_from_buffer(black_box(&buffer))
                    .expect("Failed to load");
            });
        });

        drop(index);

        // Print size info
        println!(
            "usearch - Size: {}, Serialized: {} MB, Per Vector: {:.1} KB",
            size,
            buffer.len() / (1024 * 1024),
            buffer.len() as f64 / (size as f64 * 1024.0)
        );
    }

    group.finish();
}

#[cfg(feature = "usearch")]
criterion_group!(
    benches,
    bench_usearch_build,
    bench_usearch_search,
    bench_usearch_throughput,
    bench_usearch_latency,
    bench_usearch_persistence,
);

#[cfg(feature = "usearch")]
criterion_main!(benches);

#[cfg(not(feature = "usearch"))]
fn main() {
    println!("USearch benchmarks require the 'usearch' feature.");
    println!("Run with: cargo bench --bench usearch_benchmark --features usearch");
}
