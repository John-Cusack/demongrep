//! Comprehensive benchmark for finding optimal batch sizes
//!
//! Tests the interaction between:
//! - Batch size (32, 64, 128, 256, 512, 1024, 2048)
//! - Model dimensions (384, 768, 1024)
//! - Chunk length distribution (short, medium, long, mixed/realistic)
//!
//! Run with: cargo bench --bench batch_size_optimization
//! Run with CUDA: cargo bench --features cuda --bench batch_size_optimization
//!
//! Results will help determine empirically optimal batch sizes for each
//! provider/model/chunk-length combination.

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use demongrep::embed::{ExecutionProviderType, FastEmbedder, ModelType};
use rand::prelude::*;
use std::time::Duration;

/// Chunk length profiles representing different code patterns
#[derive(Debug, Clone, Copy)]
enum ChunkProfile {
    /// Short chunks: 50-150 chars (single-line functions, imports)
    Short,
    /// Medium chunks: 150-400 chars (typical functions)
    Medium,
    /// Long chunks: 400-800 chars (complex functions, classes)
    Long,
    /// Mixed/realistic distribution matching real codebases
    Realistic,
}

impl ChunkProfile {
    fn name(&self) -> &'static str {
        match self {
            Self::Short => "short",
            Self::Medium => "medium",
            Self::Long => "long",
            Self::Realistic => "realistic",
        }
    }

    /// Generate a chunk of appropriate length
    fn generate_chunk(&self, rng: &mut impl Rng, index: usize) -> String {
        let target_len = match self {
            Self::Short => rng.gen_range(50..150),
            Self::Medium => rng.gen_range(150..400),
            Self::Long => rng.gen_range(400..800),
            Self::Realistic => {
                // Realistic distribution: 40% short, 40% medium, 20% long
                let roll: f32 = rng.gen();
                if roll < 0.4 {
                    rng.gen_range(50..150)
                } else if roll < 0.8 {
                    rng.gen_range(150..400)
                } else {
                    rng.gen_range(400..800)
                }
            }
        };

        generate_code_chunk(index, target_len)
    }
}

/// Generate a realistic-looking code chunk of approximately target_len characters
fn generate_code_chunk(index: usize, target_len: usize) -> String {
    let templates = [
        // Short function template
        |i: usize| format!(
            r#"fn process_{}(input: &str) -> Result<String, Error> {{
    Ok(input.to_uppercase())
}}"#, i
        ),
        // Medium function template
        |i: usize| format!(
            r#"/// Process data with validation and transformation
pub fn handle_request_{}(request: Request) -> Result<Response, ApiError> {{
    let validated = validate_input(&request.body)?;
    let transformed = transform_data(validated)?;
    let result = execute_operation(transformed)?;
    Ok(Response::new(result))
}}"#, i
        ),
        // Longer struct + impl template
        |i: usize| format!(
            r#"/// Configuration for the {} processor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorConfig_{} {{
    pub max_retries: usize,
    pub timeout_ms: u64,
    pub buffer_size: usize,
    pub enable_logging: bool,
}}

impl ProcessorConfig_{} {{
    pub fn new() -> Self {{
        Self {{
            max_retries: 3,
            timeout_ms: 5000,
            buffer_size: 1024,
            enable_logging: true,
        }}
    }}

    pub fn with_timeout(mut self, timeout: u64) -> Self {{
        self.timeout_ms = timeout;
        self
    }}
}}"#, i, i, i
        ),
        // Complex function with error handling
        |i: usize| format!(
            r#"async fn fetch_and_process_{}(
    client: &HttpClient,
    url: &str,
    options: &RequestOptions,
) -> Result<ProcessedData, FetchError> {{
    let response = client
        .get(url)
        .timeout(options.timeout)
        .send()
        .await
        .map_err(|e| FetchError::Network(e.to_string()))?;

    if !response.status().is_success() {{
        return Err(FetchError::HttpStatus(response.status().as_u16()));
    }}

    let body = response
        .text()
        .await
        .map_err(|e| FetchError::ReadBody(e.to_string()))?;

    let parsed: RawData = serde_json::from_str(&body)
        .map_err(|e| FetchError::Parse(e.to_string()))?;

    Ok(ProcessedData::from(parsed))
}}"#, i
        ),
    ];

    // Start with a template and adjust to target length
    let template_idx = index % templates.len();
    let mut chunk = templates[template_idx](index);

    // Pad or trim to approximate target length
    if chunk.len() < target_len {
        // Add comments to reach target length
        let padding_needed = target_len - chunk.len();
        let comment = format!(
            "\n// Additional context for chunk {}: This function handles {} operations\n// with proper error handling and validation of inputs.",
            index, index
        );
        let repeats = (padding_needed / comment.len()).max(1);
        for _ in 0..repeats {
            if chunk.len() >= target_len {
                break;
            }
            chunk.push_str(&comment);
        }
    }

    // Truncate if too long (keep it valid-ish by ending at a line)
    if chunk.len() > target_len + 100 {
        if let Some(pos) = chunk[..target_len].rfind('\n') {
            chunk.truncate(pos);
        }
    }

    chunk
}

/// Generate test corpus with specific chunk profile
fn generate_corpus(count: usize, profile: ChunkProfile, seed: u64) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|i| profile.generate_chunk(&mut rng, i))
        .collect()
}

/// Calculate corpus statistics
fn corpus_stats(corpus: &[String]) -> (usize, usize, usize, f64) {
    let lengths: Vec<usize> = corpus.iter().map(|s| s.len()).collect();
    let min = *lengths.iter().min().unwrap_or(&0);
    let max = *lengths.iter().max().unwrap_or(&0);
    let total: usize = lengths.iter().sum();
    let avg = total as f64 / lengths.len() as f64;
    (min, max, total, avg)
}

/// Benchmark batch sizes for a specific model and chunk profile
fn bench_batch_sizes_for_config(
    c: &mut Criterion,
    model_type: ModelType,
    provider: ExecutionProviderType,
    profile: ChunkProfile,
) {
    let group_name = format!(
        "batch_{}_{}_{}",
        model_type.short_name(),
        provider.name().to_lowercase(),
        profile.name()
    );
    let mut group = c.benchmark_group(&group_name);

    // Test corpus size - large enough to see batch effects
    let corpus_size = 500;
    let corpus = generate_corpus(corpus_size, profile, 42);

    let (min_len, max_len, total_len, avg_len) = corpus_stats(&corpus);
    eprintln!(
        "\n{}: {} chunks, len range [{}-{}], avg {:.0}, total {} chars",
        group_name, corpus_size, min_len, max_len, avg_len, total_len
    );

    group.throughput(Throughput::Elements(corpus_size as u64));
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));

    // Create embedder once (model loading is slow)
    let mut embedder = match FastEmbedder::with_model_and_provider(model_type, provider, None) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to create embedder for {}: {}", group_name, e);
            return;
        }
    };

    // Test different batch sizes
    let batch_sizes = [32, 64, 128, 256, 512, 1024, 2048];

    for batch_size in batch_sizes {
        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    embedder
                        .embed_batch_chunked(black_box(corpus.clone()), batch_size)
                        .expect("Embedding failed")
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// CPU Benchmarks - 384-dim model (AllMiniLML6V2Q)
// ============================================================================

fn bench_cpu_384_short(c: &mut Criterion) {
    bench_batch_sizes_for_config(
        c,
        ModelType::AllMiniLML6V2Q,
        ExecutionProviderType::Cpu,
        ChunkProfile::Short,
    );
}

fn bench_cpu_384_medium(c: &mut Criterion) {
    bench_batch_sizes_for_config(
        c,
        ModelType::AllMiniLML6V2Q,
        ExecutionProviderType::Cpu,
        ChunkProfile::Medium,
    );
}

fn bench_cpu_384_long(c: &mut Criterion) {
    bench_batch_sizes_for_config(
        c,
        ModelType::AllMiniLML6V2Q,
        ExecutionProviderType::Cpu,
        ChunkProfile::Long,
    );
}

fn bench_cpu_384_realistic(c: &mut Criterion) {
    bench_batch_sizes_for_config(
        c,
        ModelType::AllMiniLML6V2Q,
        ExecutionProviderType::Cpu,
        ChunkProfile::Realistic,
    );
}

// ============================================================================
// CPU Benchmarks - 768-dim model (BGEBaseENV15)
// ============================================================================

fn bench_cpu_768_short(c: &mut Criterion) {
    bench_batch_sizes_for_config(
        c,
        ModelType::BGEBaseENV15,
        ExecutionProviderType::Cpu,
        ChunkProfile::Short,
    );
}

fn bench_cpu_768_realistic(c: &mut Criterion) {
    bench_batch_sizes_for_config(
        c,
        ModelType::BGEBaseENV15,
        ExecutionProviderType::Cpu,
        ChunkProfile::Realistic,
    );
}

// ============================================================================
// CPU Benchmarks - 1024-dim model (BGELargeENV15)
// ============================================================================

fn bench_cpu_1024_realistic(c: &mut Criterion) {
    bench_batch_sizes_for_config(
        c,
        ModelType::BGELargeENV15,
        ExecutionProviderType::Cpu,
        ChunkProfile::Realistic,
    );
}

// ============================================================================
// CUDA Benchmarks (conditional)
// ============================================================================

#[cfg(feature = "cuda")]
fn bench_cuda_384_short(c: &mut Criterion) {
    use demongrep::embed::is_cuda_available;
    if !is_cuda_available() {
        eprintln!("CUDA not available, skipping bench_cuda_384_short");
        return;
    }
    bench_batch_sizes_for_config(
        c,
        ModelType::AllMiniLML6V2Q,
        ExecutionProviderType::Cuda { device_id: 0 },
        ChunkProfile::Short,
    );
}

#[cfg(feature = "cuda")]
fn bench_cuda_384_medium(c: &mut Criterion) {
    use demongrep::embed::is_cuda_available;
    if !is_cuda_available() {
        eprintln!("CUDA not available, skipping bench_cuda_384_medium");
        return;
    }
    bench_batch_sizes_for_config(
        c,
        ModelType::AllMiniLML6V2Q,
        ExecutionProviderType::Cuda { device_id: 0 },
        ChunkProfile::Medium,
    );
}

#[cfg(feature = "cuda")]
fn bench_cuda_384_long(c: &mut Criterion) {
    use demongrep::embed::is_cuda_available;
    if !is_cuda_available() {
        eprintln!("CUDA not available, skipping bench_cuda_384_long");
        return;
    }
    bench_batch_sizes_for_config(
        c,
        ModelType::AllMiniLML6V2Q,
        ExecutionProviderType::Cuda { device_id: 0 },
        ChunkProfile::Long,
    );
}

#[cfg(feature = "cuda")]
fn bench_cuda_384_realistic(c: &mut Criterion) {
    use demongrep::embed::is_cuda_available;
    if !is_cuda_available() {
        eprintln!("CUDA not available, skipping bench_cuda_384_realistic");
        return;
    }
    bench_batch_sizes_for_config(
        c,
        ModelType::AllMiniLML6V2Q,
        ExecutionProviderType::Cuda { device_id: 0 },
        ChunkProfile::Realistic,
    );
}

#[cfg(feature = "cuda")]
fn bench_cuda_768_realistic(c: &mut Criterion) {
    use demongrep::embed::is_cuda_available;
    if !is_cuda_available() {
        eprintln!("CUDA not available, skipping bench_cuda_768_realistic");
        return;
    }
    bench_batch_sizes_for_config(
        c,
        ModelType::BGEBaseENV15,
        ExecutionProviderType::Cuda { device_id: 0 },
        ChunkProfile::Realistic,
    );
}

#[cfg(feature = "cuda")]
fn bench_cuda_1024_realistic(c: &mut Criterion) {
    use demongrep::embed::is_cuda_available;
    if !is_cuda_available() {
        eprintln!("CUDA not available, skipping bench_cuda_1024_realistic");
        return;
    }
    bench_batch_sizes_for_config(
        c,
        ModelType::BGELargeENV15,
        ExecutionProviderType::Cuda { device_id: 0 },
        ChunkProfile::Realistic,
    );
}

// ============================================================================
// Benchmark Groups
// ============================================================================

#[cfg(not(feature = "cuda"))]
criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(10));
    targets =
        // 384-dim model with different chunk profiles
        bench_cpu_384_short,
        bench_cpu_384_medium,
        bench_cpu_384_long,
        bench_cpu_384_realistic,
        // 768-dim model
        bench_cpu_768_short,
        bench_cpu_768_realistic,
        // 1024-dim model
        bench_cpu_1024_realistic,
);

#[cfg(feature = "cuda")]
criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(10));
    targets =
        // CPU benchmarks
        bench_cpu_384_short,
        bench_cpu_384_medium,
        bench_cpu_384_long,
        bench_cpu_384_realistic,
        bench_cpu_768_short,
        bench_cpu_768_realistic,
        bench_cpu_1024_realistic,
        // CUDA benchmarks
        bench_cuda_384_short,
        bench_cuda_384_medium,
        bench_cuda_384_long,
        bench_cuda_384_realistic,
        bench_cuda_768_realistic,
        bench_cuda_1024_realistic,
);

criterion_main!(benches);
