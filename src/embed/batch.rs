use super::backend::Embedder;
use super::embedder::{ExecutionProviderType, FastEmbedder};
#[cfg(feature = "ollama")]
use super::ollama::OllamaEmbedder;
use crate::chunker::Chunk;
use anyhow::Result;
use std::sync::{Arc, Mutex};

// TODO: Better calculation of optimal token budget based on:
//   - Actual GPU memory bandwidth (not just capacity)
//   - Model architecture (layers, attention heads)
//   - Chunk length distribution in the corpus
//   - Empirical profiling per GPU model
//
// Current values are empirically determined on RTX A2000 8GB.
// Benchmarks showed 10K tokens optimal (70.4 c/s vs 53.2 c/s at 4K, 48.5 c/s at 177K).

/// Optimal token budget - empirically determined sweet spot
/// Balances batch overhead (too small) vs memory bandwidth limits (too large)
const DEFAULT_TOKEN_BUDGET: usize = 10_000;

/// Minimum budget to ensure progress
const MIN_TOKEN_BUDGET: usize = 1_000;

/// Maximum budget cap (larger batches show diminishing/negative returns)
const MAX_TOKEN_BUDGET: usize = 50_000;

/// Statistics for embedding operations
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct EmbeddingStats {
    pub total_chunks: usize,
    pub embedded_chunks: usize,
    pub cached_chunks: usize,
    #[allow(dead_code)]
    pub failed_chunks: usize,
    pub total_time_ms: u128,
}

impl EmbeddingStats {
    #[allow(dead_code)]
    pub fn cache_hit_rate(&self) -> f32 {
        if self.total_chunks == 0 {
            return 0.0;
        }
        self.cached_chunks as f32 / self.total_chunks as f32
    }

    #[allow(dead_code)]
    pub fn success_rate(&self) -> f32 {
        if self.total_chunks == 0 {
            return 0.0;
        }
        self.embedded_chunks as f32 / self.total_chunks as f32
    }

    #[allow(dead_code)]
    pub fn chunks_per_second(&self) -> f32 {
        if self.total_time_ms == 0 {
            return 0.0;
        }
        (self.embedded_chunks as f32 / self.total_time_ms as f32) * 1000.0
    }
}

/// Chunk with its embedding
#[derive(Debug, Clone)]
pub struct EmbeddedChunk {
    pub chunk: Chunk,
    pub embedding: Vec<f32>,
}

impl EmbeddedChunk {
    pub fn new(chunk: Chunk, embedding: Vec<f32>) -> Self {
        Self { chunk, embedding }
    }

    /// Calculate cosine similarity with another embedded chunk
    pub fn similarity(&self, other: &EmbeddedChunk) -> f32 {
        cosine_similarity(&self.embedding, &other.embedding)
    }

    /// Calculate cosine similarity with a query embedding
    pub fn similarity_to(&self, query_embedding: &[f32]) -> f32 {
        cosine_similarity(&self.embedding, query_embedding)
    }
}

/// Embedding backend implementation wrapper
///
/// This enum allows BatchEmbedder to work with different embedding backends
/// while maintaining type safety and avoiding trait object overhead.
pub enum EmbedderImpl {
    /// FastEmbed/ONNX backend (requires mutable access via Mutex)
    FastEmbed(Arc<Mutex<FastEmbedder>>),
    /// Ollama backend (thread-safe, uses &self)
    #[cfg(feature = "ollama")]
    Ollama(Arc<OllamaEmbedder>),
}

impl EmbedderImpl {
    /// Embed a batch of texts
    fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::FastEmbed(embedder) => {
                embedder.lock().unwrap().embed_batch(texts)
            }
            #[cfg(feature = "ollama")]
            Self::Ollama(embedder) => {
                Embedder::embed_batch(embedder.as_ref(), texts)
            }
        }
    }

    /// Embed a single text
    fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        match self {
            Self::FastEmbed(embedder) => {
                embedder.lock().unwrap().embed_one(text)
            }
            #[cfg(feature = "ollama")]
            Self::Ollama(embedder) => {
                Embedder::embed_one(embedder.as_ref(), text)
            }
        }
    }

    /// Get embedding dimensions
    fn dimensions(&self) -> usize {
        match self {
            Self::FastEmbed(embedder) => {
                embedder.lock().unwrap().dimensions()
            }
            #[cfg(feature = "ollama")]
            Self::Ollama(embedder) => {
                embedder.dimensions()
            }
        }
    }

    /// Get model name
    fn model_name(&self) -> String {
        match self {
            Self::FastEmbed(embedder) => {
                embedder.lock().unwrap().model_name().to_string()
            }
            #[cfg(feature = "ollama")]
            Self::Ollama(embedder) => {
                embedder.model_name().to_string()
            }
        }
    }

    /// Get model short name
    fn model_short_name(&self) -> String {
        match self {
            Self::FastEmbed(embedder) => {
                embedder.lock().unwrap().model_type().short_name().to_string()
            }
            #[cfg(feature = "ollama")]
            Self::Ollama(embedder) => {
                embedder.model_short_name().to_string()
            }
        }
    }

    /// Get backend name
    fn backend_name(&self) -> &'static str {
        match self {
            Self::FastEmbed(_) => "fastembed",
            #[cfg(feature = "ollama")]
            Self::Ollama(_) => "ollama",
        }
    }
}

/// Batch processor for embedding chunks efficiently
pub struct BatchEmbedder {
    pub embedder: Arc<Mutex<FastEmbedder>>,
    embedder_impl: EmbedderImpl,
    token_budget: usize,
    provider: ExecutionProviderType,
    model_dims: usize,
}

impl BatchEmbedder {
    /// Create a new batch embedder with automatic token budget based on provider and model
    pub fn new(embedder: Arc<Mutex<FastEmbedder>>) -> Self {
        let (provider, model_dims) = {
            let e = embedder.lock().unwrap();
            (e.provider(), e.dimensions())
        };

        // Allow override via environment variable for testing
        let token_budget = if let Ok(val) = std::env::var("DEMONGREP_TOKEN_BUDGET") {
            val.parse().unwrap_or_else(|_| Self::calculate_token_budget(&provider, model_dims))
        } else {
            Self::calculate_token_budget(&provider, model_dims)
        };

        Self {
            embedder: embedder.clone(),
            embedder_impl: EmbedderImpl::FastEmbed(embedder),
            token_budget,
            provider,
            model_dims,
        }
    }

    /// Create with custom batch size (converted to token budget)
    /// For backwards compatibility - batch_size is converted to approximate token budget
    pub fn with_batch_size(embedder: Arc<Mutex<FastEmbedder>>, batch_size: usize) -> Self {
        let (provider, model_dims) = {
            let e = embedder.lock().unwrap();
            (e.provider(), e.dimensions())
        };
        // Convert batch_size to token budget: assume average 500 chars per chunk
        // So batch_size=32 → 32 * 500 / 4 = 4000 tokens
        let token_budget = batch_size * 125; // 500 chars / 4 chars per token
        Self {
            embedder: embedder.clone(),
            embedder_impl: EmbedderImpl::FastEmbed(embedder),
            token_budget: token_budget.max(1000), // Minimum 1000 tokens
            provider,
            model_dims,
        }
    }

    /// Create with explicit token budget
    pub fn with_token_budget(embedder: Arc<Mutex<FastEmbedder>>, token_budget: usize) -> Self {
        let (provider, model_dims) = {
            let e = embedder.lock().unwrap();
            (e.provider(), e.dimensions())
        };
        Self {
            embedder: embedder.clone(),
            embedder_impl: EmbedderImpl::FastEmbed(embedder),
            token_budget,
            provider,
            model_dims,
        }
    }

    /// Create with dynamic embedder (for Ollama backend)
    #[cfg(feature = "ollama")]
    pub fn new_dyn(embedder: Arc<OllamaEmbedder>, model_dims: usize) -> Self {
        let token_budget = Self::calculate_token_budget(&ExecutionProviderType::Cpu, model_dims);

        // Create a dummy FastEmbedder Arc for backwards compatibility
        // This will only be used if someone accesses .embedder directly (legacy code)
        // New code should use embedder_impl
        let dummy_embedder = Arc::new(Mutex::new(
            FastEmbedder::with_model_and_provider(
                super::embedder::ModelType::default(),
                ExecutionProviderType::Cpu,
                None,
            )
            .expect("Failed to create placeholder embedder"),
        ));

        Self {
            embedder: dummy_embedder,
            embedder_impl: EmbedderImpl::Ollama(embedder),
            token_budget,
            provider: ExecutionProviderType::Cpu,
            model_dims,
        }
    }

    /// Create with dynamic embedder and custom batch size
    #[cfg(feature = "ollama")]
    pub fn with_batch_size_dyn(
        embedder: Arc<OllamaEmbedder>,
        batch_size: usize,
        model_dims: usize,
    ) -> Self {
        let token_budget = batch_size * 125;

        // Create a dummy FastEmbedder Arc for backwards compatibility
        let dummy_embedder = Arc::new(Mutex::new(
            FastEmbedder::with_model_and_provider(
                super::embedder::ModelType::default(),
                ExecutionProviderType::Cpu,
                None,
            )
            .expect("Failed to create placeholder embedder"),
        ));

        Self {
            embedder: dummy_embedder,
            embedder_impl: EmbedderImpl::Ollama(embedder),
            token_budget: token_budget.max(1000),
            provider: ExecutionProviderType::Cpu,
            model_dims,
        }
    }

    /// Calculate token budget based on model dimensions
    ///
    /// Uses empirically-determined optimal budget, scaled for model size.
    /// Larger models need smaller batches to stay within memory bandwidth limits.
    fn calculate_token_budget(_provider: &ExecutionProviderType, model_dims: usize) -> usize {
        // Scale budget inversely with model dimensions
        // 384 dims (base) → 10K tokens
        // 768 dims → 5K tokens
        // 1024 dims → ~3.75K tokens
        let scale = 384.0 / model_dims as f64;
        let scaled_budget = (DEFAULT_TOKEN_BUDGET as f64 * scale) as usize;

        scaled_budget.clamp(MIN_TOKEN_BUDGET, MAX_TOKEN_BUDGET)
    }

    /// Estimate tokens from text length (rough: 4 chars per token)
    #[inline]
    fn estimate_tokens(text_len: usize) -> usize {
        (text_len + 3) / 4 // Round up
    }

    /// Embed a batch of chunks using token-budget adaptive batching
    ///
    /// Chunks are sorted by length and grouped into batches that fit within
    /// the token budget. This minimizes padding waste while maximizing throughput.
    pub fn embed_chunks(&mut self, chunks: Vec<Chunk>) -> Result<Vec<EmbeddedChunk>> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        let total = chunks.len();
        let start = std::time::Instant::now();

        // Prepare all texts and track original indices
        let mut indexed_texts: Vec<(usize, String, Chunk)> = chunks
            .into_iter()
            .enumerate()
            .map(|(idx, chunk)| {
                let text = self.prepare_text(&chunk);
                (idx, text, chunk)
            })
            .collect();

        // Sort by text length to minimize padding within batches
        indexed_texts.sort_by_key(|(_, text, _)| text.len());

        // Create batches using token budget
        let batches = self.create_token_budget_batches(&indexed_texts);
        let num_batches = batches.len();

        println!(
            "📊 Embedding {} chunks in {} batches (budget: {} tokens, {}, {} dims)",
            total, num_batches, self.token_budget, self.provider.name(), self.model_dims
        );

        let mut results: Vec<(usize, EmbeddedChunk)> = Vec::with_capacity(total);

        for (batch_idx, batch_indices) in batches.iter().enumerate() {
            let batch: Vec<_> = batch_indices.iter()
                .map(|&i| &indexed_texts[i])
                .collect();

            let batch_size = batch.len();
            let max_len = batch.last().map(|(_, t, _)| t.len()).unwrap_or(0);
            let tokens_used = Self::estimate_tokens(max_len) * batch_size;

            println!(
                "   Batch {}/{}: {} chunks, max_len={}, ~{} tokens",
                batch_idx + 1,
                num_batches,
                batch_size,
                max_len,
                tokens_used,
            );

            // Extract texts for this batch
            let texts: Vec<String> = batch.iter().map(|(_, text, _)| text.clone()).collect();

            // Generate embeddings using the appropriate backend
            let embeddings = self.embedder_impl.embed_batch(texts)?;

            // Combine with original indices
            for ((orig_idx, _, chunk), embedding) in batch.into_iter().zip(embeddings.into_iter()) {
                results.push((*orig_idx, EmbeddedChunk::new(chunk.clone(), embedding)));
            }
        }

        // Restore original order
        results.sort_by_key(|(idx, _)| *idx);
        let embedded_chunks: Vec<EmbeddedChunk> = results.into_iter().map(|(_, ec)| ec).collect();

        let elapsed = start.elapsed();
        println!(
            "✅ Embedded {} chunks in {:.2}s ({:.1} chunks/sec)",
            total,
            elapsed.as_secs_f32(),
            total as f32 / elapsed.as_secs_f32()
        );

        Ok(embedded_chunks)
    }

    /// Create batches using token budget
    ///
    /// Sliding window through sorted chunks, cutting a new batch when
    /// adding the next chunk would exceed the token budget.
    fn create_token_budget_batches(&self, sorted_texts: &[(usize, String, Chunk)]) -> Vec<Vec<usize>> {
        let mut batches: Vec<Vec<usize>> = Vec::new();
        let mut current_batch: Vec<usize> = Vec::new();

        for (i, (_, text, _)) in sorted_texts.iter().enumerate() {
            let chunk_tokens = Self::estimate_tokens(text.len());

            if current_batch.is_empty() {
                // Start new batch
                current_batch.push(i);
            } else {
                // This chunk is the longest so far (sorted order)
                // Check if adding it would exceed token budget
                let new_batch_size = current_batch.len() + 1;
                let tokens_if_added = chunk_tokens * new_batch_size;

                if tokens_if_added <= self.token_budget {
                    // Add to current batch
                    current_batch.push(i);
                } else {
                    // Start new batch
                    batches.push(std::mem::take(&mut current_batch));
                    current_batch.push(i);
                }
            }
        }

        // Don't forget the last batch
        if !current_batch.is_empty() {
            batches.push(current_batch);
        }

        batches
    }

    /// Embed a single chunk
    pub fn embed_chunk(&mut self, chunk: Chunk) -> Result<EmbeddedChunk> {
        let text = self.prepare_text(&chunk);
        let embedding = self.embedder_impl.embed_one(&text)?;
        Ok(EmbeddedChunk::new(chunk, embedding))
    }

    /// Embed a single text string (for query embedding)
    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        self.embedder_impl.embed_one(text)
    }

    /// Prepare chunk text for embedding
    ///
    /// Combines different chunk metadata for better embeddings:
    /// - Context breadcrumbs
    /// - Signature (if available)
    /// - Docstring (if available)
    /// - Content
    fn prepare_text(&self, chunk: &Chunk) -> String {
        let mut parts = Vec::new();

        // Add context breadcrumbs (e.g., "File: main.rs > Class: Server")
        if !chunk.context.is_empty() {
            let context = chunk.context.join(" > ");
            parts.push(format!("Context: {}", context));
        }

        // Add signature if available (e.g., "fn process(data: Vec<T>) -> Result<T>")
        if let Some(sig) = &chunk.signature {
            parts.push(format!("Signature: {}", sig));
        }

        // Add docstring if available
        if let Some(doc) = &chunk.docstring {
            // Clean up docstring
            let cleaned = clean_docstring(doc);
            if !cleaned.is_empty() {
                parts.push(format!("Documentation: {}", cleaned));
            }
        }

        // Add main content
        parts.push(format!("Code:\n{}", chunk.content));

        parts.join("\n")
    }

    /// Get embedding dimensions
    pub fn dimensions(&self) -> usize {
        self.embedder_impl.dimensions()
    }

    /// Get embedder info (model name and dimensions)
    pub fn embedder_info(&self) -> (String, usize) {
        (self.embedder_impl.model_name(), self.embedder_impl.dimensions())
    }

    /// Get model short name
    pub fn model_short_name(&self) -> String {
        self.embedder_impl.model_short_name()
    }

    /// Get backend name
    pub fn backend_name(&self) -> &'static str {
        self.embedder_impl.backend_name()
    }
}

/// Clean docstring by removing comment markers
fn clean_docstring(doc: &str) -> String {
    // First handle triple-quoted strings and JSDoc as special cases
    let cleaned = if let Some(stripped) = doc
        .strip_prefix("\"\"\"")
        .and_then(|s| s.strip_suffix("\"\"\""))
    {
        stripped
    } else if let Some(stripped) = doc.strip_prefix("'''").and_then(|s| s.strip_suffix("'''")) {
        stripped
    } else if let Some(stripped) = doc.strip_prefix("/**").and_then(|s| s.strip_suffix("*/")) {
        stripped
    } else {
        doc
    };

    cleaned
        .lines()
        .map(|line| {
            let trimmed = line.trim();
            // Remove common comment markers
            trimmed
                .strip_prefix("///")
                .or_else(|| trimmed.strip_prefix("//!"))
                .or_else(|| trimmed.strip_prefix("//"))
                .or_else(|| trimmed.strip_prefix("*"))
                .or_else(|| trimmed.strip_prefix("\""))
                .unwrap_or(trimmed)
                .trim()
        })
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Calculate cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }

    dot / (mag_a * mag_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::ChunkKind;

    #[test]
    fn test_embedding_stats() {
        let stats = EmbeddingStats {
            total_chunks: 100,
            embedded_chunks: 80,
            cached_chunks: 20,
            failed_chunks: 0,
            total_time_ms: 1000,
        };

        assert_eq!(stats.cache_hit_rate(), 0.2);
        assert_eq!(stats.success_rate(), 0.8);
        assert_eq!(stats.chunks_per_second(), 80.0);
    }

    #[test]
    fn test_clean_docstring() {
        let rust_doc = "/// This is a doc comment\n/// with multiple lines";
        assert_eq!(
            clean_docstring(rust_doc),
            "This is a doc comment with multiple lines"
        );

        let python_doc = "\"\"\"This is a Python docstring\"\"\"";
        assert_eq!(clean_docstring(python_doc), "This is a Python docstring");

        let jsdoc = "/**\n * JSDoc comment\n * with multiple lines\n */";
        assert_eq!(clean_docstring(jsdoc), "JSDoc comment with multiple lines");
    }

    #[test]
    fn test_prepare_text() {
        let embedder = Arc::new(Mutex::new(FastEmbedder::new().unwrap_or_else(|_| {
            // For tests, create a mock if real embedder fails
            panic!("Cannot create embedder in test");
        })));

        let batch = BatchEmbedder::new(embedder);

        let mut chunk = Chunk::new(
            "fn test() { println!(\"test\"); }".to_string(),
            0,
            1,
            ChunkKind::Function,
            "test.rs".to_string(),
        );
        chunk.context = vec!["File: test.rs".to_string(), "Function: test".to_string()];
        chunk.signature = Some("fn test()".to_string());
        chunk.docstring = Some("/// Test function".to_string());

        let text = batch.prepare_text(&chunk);

        assert!(text.contains("Context: File: test.rs > Function: test"));
        assert!(text.contains("Signature: fn test()"));
        assert!(text.contains("Documentation: Test function"));
        assert!(text.contains("Code:"));
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![1.0, 0.0, 0.0];
        let d = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&c, &d) - 0.0).abs() < 0.001);

        let e = vec![1.0, 1.0, 0.0];
        let f = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&e, &f);
        assert!(sim > 0.7 && sim < 0.72); // Should be ~1/sqrt(2)
    }

    #[test]
    #[ignore] // Requires model
    fn test_batch_embedder() {
        let embedder = Arc::new(Mutex::new(FastEmbedder::new().unwrap()));
        let mut batch = BatchEmbedder::new(embedder);

        let chunks = vec![
            Chunk::new(
                "fn main() {}".to_string(),
                0,
                1,
                ChunkKind::Function,
                "test.rs".to_string(),
            ),
            Chunk::new(
                "struct Point { x: i32, y: i32 }".to_string(),
                2,
                3,
                ChunkKind::Struct,
                "test.rs".to_string(),
            ),
        ];

        let embedded = batch.embed_chunks(chunks).unwrap();
        assert_eq!(embedded.len(), 2);

        for emb_chunk in &embedded {
            assert_eq!(emb_chunk.embedding.len(), 384);
        }
    }
}
