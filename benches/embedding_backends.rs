use criterion::{black_box, criterion_group, criterion_main, Criterion};
use demongrep::embed::{ExecutionProviderType, FastEmbedder, ModelType};

fn benchmark_embedding_backend(c: &mut Criterion) {
    let model_type = ModelType::default();

    // Benchmark CPU
    let cpu_group = c.benchmark_group("CPU");
    cpu_group.bench_function("cpu_embedding", |b| {
        b.iter(|| {
            let mut embedder = black_box(
                FastEmbedder::with_model_and_provider(model_type, ExecutionProviderType::Cpu, None)
                    .unwrap(),
            );

            let _ = black_box(embedder.embed_one("Test text for CPU embedding").unwrap());
        })
    });
    cpu_group.finish();

    // Benchmark CUDA if available and feature is enabled
    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            let cuda_group = c.benchmark_group("CUDA");
            cuda_group.bench_function("cuda_embedding", |b| {
                b.iter(|| {
                    let mut embedder = black_box(
                        FastEmbedder::with_model_and_provider(
                            model_type,
                            ExecutionProviderType::Cuda { device_id: 0 },
                            Some(0),
                        )
                        .unwrap(),
                    );

                    let _ = black_box(embedder.embed_one("Test text for CUDA embedding").unwrap());
                })
            });
            cuda_group.finish();
        }
    }

    // Benchmark Auto (should use best available)
    let auto_group = c.benchmark_group("Auto");
    auto_group.bench_function("auto_embedding", |b| {
        b.iter(|| {
            let mut embedder = black_box(
                FastEmbedder::with_model_and_provider(
                    model_type,
                    ExecutionProviderType::Auto,
                    None,
                )
                .unwrap(),
            );

            let _ = black_box(embedder.embed_one("Test text for auto embedding").unwrap());
        })
    });
    auto_group.finish();
}

criterion_group!(benches, benchmark_embedding_backend);
criterion_main!(benches);
