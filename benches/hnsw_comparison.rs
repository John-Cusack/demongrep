//! Benchmarks comparing usearch (HNSW) vs arroy for approximate nearest neighbor search
//!
//! Run with: cargo bench --bench hnsw_comparison
//!
//! This benchmark tests:
//! - Index building speed
//! - Single query latency
//! - Query throughput (queries per second)
//! - Different dataset sizes
//! - Different k (number of results) values

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use demongrep::chunker::{Chunk, ChunkKind};
use demongrep::embed::EmbeddedChunk;
use demongrep::vectordb::VectorStore;
use tempfile::tempdir;

/// Generate realistic test vectors (384D, similar to actual embeddings)
fn generate_test_vectors(count: usize, dimensions: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| {
            (0..dimensions)
                .map(|j| {
                    // Generate somewhat meaningful patterns
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
            let base_idx = (i * 7) % dataset_size; // Skip through dataset
            (0..dimensions)
                .map(|j| {
                    // Similar to dataset vectors but with noise
                    let base = (base_idx as f32) * 0.001;
                    let offset = (j as f32) * 0.01;
                    let noise = (rand::random::<f32>() - 0.5) * 0.1; // Small noise
                    (((base + offset).sin() + 1.0) * 0.5 + noise).clamp(0.0, 1.0)
                })
                .collect()
        })
        .collect()
}

/// Test context for storing pre-built indexes
struct BenchmarkContext {
    arroy_store: Option<VectorStore>,
    // We'll add usearch index here if we integrate it
}

/// Benchmark arroy index building
fn bench_arroy_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("arroy_build");

    let sizes = [1000, 5000, 10000, 50000];
    let dimensions = 384;

    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.sample_size(10);

        group.bench_with_input(BenchmarkId::new("size", size), &size, |b, &size| {
            b.iter(|| {
                let temp_dir = tempdir().unwrap();
                let db_path = temp_dir.path().join(format!("arroy_{}.db", size));

                let mut store = VectorStore::new(&db_path, dimensions).unwrap();

                let vectors = generate_test_vectors(size, dimensions);
                let chunks: Vec<EmbeddedChunk> = vectors
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        EmbeddedChunk::new(
                            Chunk::new(
                                format!("fn test_{}() {{}}", i),
                                i,
                                i + 1,
                                ChunkKind::Function,
                                format!("test_{}.rs", i / 10),
                            ),
                            v.clone(),
                        )
                    })
                    .collect();

                store.insert_chunks(chunks).unwrap();
                store.build_index().unwrap();
            });
        });
    }

    group.finish();
}

/// Benchmark arroy search performance
fn bench_arroy_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("arroy_search");

    let sizes = [1000, 5000, 10000, 50000];
    let k_values = [10, 50, 100];
    let dimensions = 384;

    for &size in &sizes {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join(format!("arroy_search_{}.db", size));

        // Build index
        let vectors = generate_test_vectors(size, dimensions);
        let chunks: Vec<EmbeddedChunk> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                EmbeddedChunk::new(
                    Chunk::new(
                        format!("fn test_{}() {{}}", i),
                        i,
                        i + 1,
                        ChunkKind::Function,
                        format!("test_{}.rs", i / 10),
                    ),
                    v.clone(),
                )
            })
            .collect();

        let mut store = VectorStore::new(&db_path, dimensions).unwrap();
        store.insert_chunks(chunks).unwrap();
        store.build_index().unwrap();

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
                            let _ = black_box(store.search(black_box(query), k));
                        }
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark query throughput (queries per second)
fn bench_arroy_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("arroy_throughput");

    let sizes = [1000, 5000, 10000, 50000];
    let query_counts = [100, 500, 1000];
    let dimensions = 384;
    let k = 10;

    for &size in &sizes {
        // Build index
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir
            .path()
            .join(format!("arroy_throughput_{}.db", size));

        let vectors = generate_test_vectors(size, dimensions);
        let chunks: Vec<EmbeddedChunk> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                EmbeddedChunk::new(
                    Chunk::new(
                        format!("fn test_{}() {{}}", i),
                        i,
                        i + 1,
                        ChunkKind::Function,
                        format!("test_{}.rs", i / 10),
                    ),
                    v.clone(),
                )
            })
            .collect();

        let mut store = VectorStore::new(&db_path, dimensions).unwrap();
        store.insert_chunks(chunks).unwrap();
        store.build_index().unwrap();

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
                            let _ = black_box(store.search(black_box(query), k));
                        }
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark search latency for single queries (cold caches)
fn bench_arroy_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("arroy_latency");
    group.sample_size(100); // High sample size for accurate latency

    let sizes = [1000, 5000, 10000, 50000];
    let dimensions = 384;
    let k = 10;

    for &size in &sizes {
        // Build index
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join(format!("arroy_latency_{}.db", size));

        let vectors = generate_test_vectors(size, dimensions);
        let chunks: Vec<EmbeddedChunk> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                EmbeddedChunk::new(
                    Chunk::new(
                        format!("fn test_{}() {{}}", i),
                        i,
                        i + 1,
                        ChunkKind::Function,
                        format!("test_{}.rs", i / 10),
                    ),
                    v.clone(),
                )
            })
            .collect();

        let mut store = VectorStore::new(&db_path, dimensions).unwrap();
        store.insert_chunks(chunks).unwrap();
        store.build_index().unwrap();

        let queries = generate_query_vectors(100, size, dimensions);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            let mut query_idx = 0;
            b.iter(|| {
                let query = &queries[query_idx];
                query_idx = (query_idx + 1) % queries.len();
                let _ = black_box(store.search(black_box(query), k));
            });
        });
    }

    group.finish();
}

/// Benchmark persistence: save and reload
fn bench_arroy_persistence(c: &mut Criterion) {
    let mut group = c.benchmark_group("arroy_persistence");

    let sizes = [1000, 5000, 10000];
    let dimensions = 384;

    for &size in &sizes {
        // Create and populate database
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join(format!("arroy_persist_{}.db", size));

        let vectors = generate_test_vectors(size, dimensions);
        let chunks: Vec<EmbeddedChunk> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                EmbeddedChunk::new(
                    Chunk::new(
                        format!("fn test_{}() {{}}", i),
                        i,
                        i + 1,
                        ChunkKind::Function,
                        format!("test_{}.rs", i / 10),
                    ),
                    v.clone(),
                )
            })
            .collect();

        // Insert and build index
        {
            let mut store = VectorStore::new(&db_path, dimensions).unwrap();
            store.insert_chunks(chunks).unwrap();
            store.build_index().unwrap();
        }

        // Benchmark reload time
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let _store = VectorStore::new(black_box(&db_path), dimensions).unwrap();
            });
        });
    }

    group.finish();
}

/// Memory usage estimation
fn bench_memory_usage(_c: &mut Criterion) {
    // Criterion doesn't directly measure memory, but we can show stats
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("memory_test.db");

    let sizes = [1000, 5000, 10000, 50000];
    let dimensions = 384;

    for size in sizes {
        let vectors = generate_test_vectors(size, dimensions);
        let chunks: Vec<EmbeddedChunk> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                EmbeddedChunk::new(
                    Chunk::new(
                        format!("fn test_{}() {{}}", i),
                        i,
                        i + 1,
                        ChunkKind::Function,
                        format!("test_{}.rs", i / 10),
                    ),
                    v.clone(),
                )
            })
            .collect();

        let mut store = VectorStore::new(&db_path, dimensions).unwrap();
        store.insert_chunks(chunks).unwrap();
        store.build_index().unwrap();

        let stats = store.stats().unwrap();
        let db_size = store.db_size().unwrap();

        println!(
            "Arroy - Size: {}, Chunks: {}, DB Size: {} MB, Per Chunk: {:.1} KB",
            size,
            stats.total_chunks,
            db_size / (1024 * 1024),
            db_size as f64 / (stats.total_chunks as f64 * 1024.0)
        );
    }
}

criterion_group!(
    benches,
    bench_arroy_build,
    bench_arroy_search,
    bench_arroy_throughput,
    bench_arroy_latency,
    bench_arroy_persistence,
    // bench_memory_usage is commented out as it's informational, not a benchmark
);

criterion_main!(benches);
