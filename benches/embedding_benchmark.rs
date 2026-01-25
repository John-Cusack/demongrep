//! Benchmarks for embedding performance across different providers and batch sizes
//!
//! Run with: cargo bench --bench embedding_benchmark
//! Run with CUDA: cargo bench --features cuda --bench embedding_benchmark

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use demongrep::embed::{ExecutionProviderType, FastEmbedder, ModelType};

/// Generate test texts of varying complexity
fn generate_test_texts(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| {
            format!(
                r#"fn process_data_{}(input: &[u8]) -> Result<Vec<u8>, Error> {{
    let mut output = Vec::with_capacity(input.len());
    for byte in input.iter() {{
        output.push(byte.wrapping_add(1));
    }}
    Ok(output)
}}"#,
                i
            )
        })
        .collect()
}

/// Benchmark different batch sizes on CPU
fn bench_cpu_batch_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_batch_sizes");

    // Test batch sizes
    let batch_sizes = [32, 64, 128, 256, 512];
    let text_count = 500;
    let texts = generate_test_texts(text_count);

    group.throughput(Throughput::Elements(text_count as u64));
    group.sample_size(10); // Fewer samples since embedding is slow

    // Create embedder once (model loading is slow)
    let mut embedder = match FastEmbedder::with_model_and_provider(
        ModelType::AllMiniLML6V2Q,
        ExecutionProviderType::Cpu,
        None,
    ) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to create embedder: {}", e);
            return;
        }
    };

    for batch_size in batch_sizes {
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            &batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    embedder
                        .embed_batch_chunked(black_box(texts.clone()), batch_size)
                        .expect("Embedding failed")
                });
            },
        );
    }

    group.finish();
}

/// Benchmark optimal vs suboptimal batch sizes
fn bench_optimal_vs_suboptimal(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimal_vs_suboptimal");

    let text_count = 200;
    let texts = generate_test_texts(text_count);

    group.throughput(Throughput::Elements(text_count as u64));
    group.sample_size(10);

    let mut embedder = match FastEmbedder::with_model_and_provider(
        ModelType::AllMiniLML6V2Q,
        ExecutionProviderType::Cpu,
        None,
    ) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to create embedder: {}", e);
            return;
        }
    };

    // Suboptimal batch size (too small)
    group.bench_function("suboptimal_batch_32", |b| {
        b.iter(|| {
            embedder
                .embed_batch_chunked(black_box(texts.clone()), 32)
                .expect("Embedding failed")
        });
    });

    // Suboptimal batch size (medium)
    group.bench_function("suboptimal_batch_64", |b| {
        b.iter(|| {
            embedder
                .embed_batch_chunked(black_box(texts.clone()), 64)
                .expect("Embedding failed")
        });
    });

    // Optimal for CPU (256 for 384-dim models)
    group.bench_function("optimal_batch_256", |b| {
        b.iter(|| {
            embedder
                .embed_batch_chunked(black_box(texts.clone()), 256)
                .expect("Embedding failed")
        });
    });

    group.finish();
}

/// Benchmark embedding with auto provider detection
fn bench_auto_provider(c: &mut Criterion) {
    let mut group = c.benchmark_group("auto_provider");

    let text_count = 100;
    let texts = generate_test_texts(text_count);

    group.throughput(Throughput::Elements(text_count as u64));
    group.sample_size(10);

    let mut embedder = match FastEmbedder::with_model_and_provider(
        ModelType::AllMiniLML6V2Q,
        ExecutionProviderType::Auto,
        None,
    ) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to create embedder: {}", e);
            return;
        }
    };

    let provider_name = embedder.provider().name();
    let optimal_batch = embedder.provider().optimal_batch_size(384);

    group.bench_function(format!("auto_{}_batch_{}", provider_name, optimal_batch), |b| {
        b.iter(|| {
            embedder
                .embed_batch(black_box(texts.clone()))
                .expect("Embedding failed")
        });
    });

    group.finish();
}

/// Benchmark CUDA if available
#[cfg(feature = "cuda")]
fn bench_cuda(c: &mut Criterion) {
    use demongrep::embed::is_cuda_available;

    if !is_cuda_available() {
        eprintln!("CUDA not available, skipping CUDA benchmarks");
        return;
    }

    let mut group = c.benchmark_group("cuda");

    let text_count = 1000;
    let texts = generate_test_texts(text_count);

    group.throughput(Throughput::Elements(text_count as u64));
    group.sample_size(10);

    let mut embedder = match FastEmbedder::with_model_and_provider(
        ModelType::AllMiniLML6V2Q,
        ExecutionProviderType::Cuda { device_id: 0 },
        Some(0),
    ) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to create CUDA embedder: {}", e);
            return;
        }
    };

    // Test different batch sizes
    let batch_sizes = [256, 512, 1024, 2048];

    for batch_size in batch_sizes {
        group.bench_with_input(
            BenchmarkId::new("cuda_batch", batch_size),
            &batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    embedder
                        .embed_batch_chunked(black_box(texts.clone()), batch_size)
                        .expect("Embedding failed")
                });
            },
        );
    }

    group.finish();
}

/// Benchmark CPU vs CUDA comparison
#[cfg(feature = "cuda")]
fn bench_cpu_vs_cuda(c: &mut Criterion) {
    use demongrep::embed::is_cuda_available;

    if !is_cuda_available() {
        eprintln!("CUDA not available, skipping CPU vs CUDA benchmark");
        return;
    }

    let mut group = c.benchmark_group("cpu_vs_cuda");

    let text_count = 500;
    let texts = generate_test_texts(text_count);

    group.throughput(Throughput::Elements(text_count as u64));
    group.sample_size(10);

    // CPU with optimal batch size
    {
        let mut cpu_embedder = FastEmbedder::with_model_and_provider(
            ModelType::AllMiniLML6V2Q,
            ExecutionProviderType::Cpu,
            None,
        )
        .expect("Failed to create CPU embedder");

        let cpu_optimal = ExecutionProviderType::Cpu.optimal_batch_size(384);

        group.bench_function(format!("cpu_optimal_{}", cpu_optimal), |b| {
            b.iter(|| {
                cpu_embedder
                    .embed_batch_chunked(black_box(texts.clone()), cpu_optimal)
                    .expect("Embedding failed")
            });
        });
    }

    // CUDA with optimal batch size
    {
        let mut cuda_embedder = FastEmbedder::with_model_and_provider(
            ModelType::AllMiniLML6V2Q,
            ExecutionProviderType::Cuda { device_id: 0 },
            Some(0),
        )
        .expect("Failed to create CUDA embedder");

        let cuda_optimal = ExecutionProviderType::Cuda { device_id: 0 }.optimal_batch_size(384);

        group.bench_function(format!("cuda_optimal_{}", cuda_optimal), |b| {
            b.iter(|| {
                cuda_embedder
                    .embed_batch_chunked(black_box(texts.clone()), cuda_optimal)
                    .expect("Embedding failed")
            });
        });

        // CUDA with CPU's batch size - demonstrates the bug we're fixing
        let cpu_batch = ExecutionProviderType::Cpu.optimal_batch_size(384);

        group.bench_function(format!("cuda_suboptimal_{}", cpu_batch), |b| {
            b.iter(|| {
                cuda_embedder
                    .embed_batch_chunked(black_box(texts.clone()), cpu_batch)
                    .expect("Embedding failed")
            });
        });
    }

    group.finish();
}

// Define benchmark groups
#[cfg(not(feature = "cuda"))]
criterion_group!(
    benches,
    bench_cpu_batch_sizes,
    bench_optimal_vs_suboptimal,
    bench_auto_provider,
);

#[cfg(feature = "cuda")]
criterion_group!(
    benches,
    bench_cpu_batch_sizes,
    bench_optimal_vs_suboptimal,
    bench_auto_provider,
    bench_cuda,
    bench_cpu_vs_cuda,
);

criterion_main!(benches);
