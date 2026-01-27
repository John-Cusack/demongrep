mod backend;
mod batch;
mod cache;
mod embedder;
#[cfg(feature = "ollama")]
mod ollama;
pub mod tuning;

pub use backend::{EmbeddingBackend, Embedder};
pub use batch::{BatchEmbedder, EmbeddedChunk};
pub use cache::{CacheStats, CachedBatchEmbedder};
pub use embedder::{detect_best_provider, ExecutionProviderType, FastEmbedder, ModelType};
#[cfg(feature = "ollama")]
pub use ollama::OllamaEmbedder;
#[cfg(feature = "ollama")]
pub use ollama::known_dimensions as ollama_dimensions;

#[cfg(feature = "cuda")]
pub use embedder::is_cuda_available;

#[cfg(feature = "tensorrt")]
pub use embedder::is_tensorrt_available;

#[cfg(feature = "coreml")]
pub use embedder::is_coreml_available;

#[cfg(feature = "directml")]
pub use embedder::is_directml_available;

use anyhow::Result;
use std::sync::{Arc, Mutex};

use crate::config::Config;

/// High-level embedding service that combines all features
pub struct EmbeddingService {
    cached_embedder: CachedBatchEmbedder,
    model_type: ModelType,
    backend: EmbeddingBackend,
}

impl EmbeddingService {
    /// Create a new embedding service with default model
    pub fn new() -> Result<Self> {
        Self::with_model(ModelType::default())
    }

    /// Create a new embedding service with specified model
    pub fn with_model(model_type: ModelType) -> Result<Self> {
        Self::with_model_and_provider(model_type, ExecutionProviderType::Auto, None, None)
    }

    /// Create a new embedding service with specified model and execution provider
    pub fn with_model_and_provider(
        model_type: ModelType,
        provider: ExecutionProviderType,
        device_id: Option<i32>,
        batch_size: Option<usize>,
    ) -> Result<Self> {
        let embedder = FastEmbedder::with_model_and_provider(model_type, provider, device_id)?;
        let arc_embedder = Arc::new(Mutex::new(embedder));

        // Use explicit batch_size if provided (from config), otherwise use dynamic token budget
        let batch_embedder = match batch_size {
            Some(size) => BatchEmbedder::with_batch_size(arc_embedder, size),
            None => BatchEmbedder::new(arc_embedder), // Uses dynamic budget based on VRAM/RAM and model
        };
        let cached_embedder = CachedBatchEmbedder::new(batch_embedder);

        Ok(Self {
            cached_embedder,
            model_type,
            backend: EmbeddingBackend::FastEmbed,
        })
    }

    /// Create embedding service with specified backend
    ///
    /// This is the preferred factory method that supports both FastEmbed and Ollama backends.
    ///
    /// # Arguments
    /// * `backend` - The embedding backend to use
    /// * `config` - Full configuration (for Ollama settings)
    /// * `model_type` - Optional FastEmbed model type (ignored for Ollama)
    /// * `provider` - Execution provider for FastEmbed (ignored for Ollama)
    /// * `device_id` - GPU device ID for FastEmbed (ignored for Ollama)
    /// * `batch_size` - Optional batch size override
    pub fn with_backend(
        backend: EmbeddingBackend,
        #[allow(unused_variables)] config: &Config,
        model_type: Option<ModelType>,
        provider: ExecutionProviderType,
        device_id: Option<i32>,
        batch_size: Option<usize>,
    ) -> Result<Self> {
        match backend {
            EmbeddingBackend::FastEmbed => {
                let model = model_type.unwrap_or_default();
                Self::with_model_and_provider(model, provider, device_id, batch_size)
            }
            #[cfg(feature = "ollama")]
            EmbeddingBackend::Ollama => {
                use crate::info_print;

                info_print!("📦 Loading Ollama embedding backend");
                info_print!("   Model: {}", config.embedding.ollama.model);
                info_print!("   URL: {}", config.embedding.ollama.url);

                let ollama_embedder = OllamaEmbedder::new(config.embedding.ollama.clone())?;
                let dimensions = ollama_embedder.dimensions();

                // Wrap in Arc for BatchEmbedder compatibility
                let arc_embedder = Arc::new(ollama_embedder);

                // Create batch embedder (token budgeting still applies)
                let batch_embedder = match batch_size {
                    Some(size) => BatchEmbedder::with_batch_size_dyn(arc_embedder, size, dimensions),
                    None => BatchEmbedder::new_dyn(arc_embedder, dimensions),
                };
                let cached_embedder = CachedBatchEmbedder::new(batch_embedder);

                info_print!("✅ Ollama backend loaded successfully!");

                Ok(Self {
                    cached_embedder,
                    model_type: model_type.unwrap_or_default(), // Placeholder for Ollama
                    backend: EmbeddingBackend::Ollama,
                })
            }
        }
    }

    /// Embed a batch of chunks with caching
    pub fn embed_chunks(
        &mut self,
        chunks: Vec<crate::chunker::Chunk>,
    ) -> Result<Vec<EmbeddedChunk>> {
        self.cached_embedder.embed_chunks(chunks)
    }

    /// Embed a single chunk with caching
    pub fn embed_chunk(&mut self, chunk: crate::chunker::Chunk) -> Result<EmbeddedChunk> {
        self.cached_embedder.embed_chunk(chunk)
    }

    /// Embed query text
    pub fn embed_query(&mut self, query: &str) -> Result<Vec<f32>> {
        // Use the batch embedder's embed_one which uses the correct backend
        self.cached_embedder.batch_embedder.embed_one(query)
    }

    /// Get embedding dimensions
    pub fn dimensions(&self) -> usize {
        self.cached_embedder.dimensions()
    }

    /// Get model information
    pub fn model_name(&self) -> String {
        match self.backend {
            EmbeddingBackend::FastEmbed => self.model_type.name().to_string(),
            #[cfg(feature = "ollama")]
            EmbeddingBackend::Ollama => {
                self.cached_embedder.batch_embedder.model_name()
            }
        }
    }

    /// Get model type
    pub fn model_type(&self) -> ModelType {
        self.model_type
    }

    /// Get model short name (for storage)
    pub fn model_short_name(&self) -> String {
        // For Ollama backend, get from the batch embedder which has the actual model info
        match self.backend {
            EmbeddingBackend::FastEmbed => self.model_type.short_name().to_string(),
            #[cfg(feature = "ollama")]
            EmbeddingBackend::Ollama => {
                // Return the configured Ollama model name
                // The actual name is stored in the embedder, but we can use the batch embedder's info
                self.cached_embedder.batch_embedder.model_short_name()
            }
        }
    }

    /// Get the embedding backend being used
    pub fn backend(&self) -> EmbeddingBackend {
        self.backend
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        self.cached_embedder.cache_stats()
    }

    /// Clear the cache
    pub fn clear_cache(&self) {
        self.cached_embedder.clear_cache();
    }

    /// Search for most similar chunks to a query
    pub fn search<'a>(
        &self,
        query_embedding: &[f32],
        embedded_chunks: &'a [EmbeddedChunk],
        limit: usize,
    ) -> Vec<(&'a EmbeddedChunk, f32)> {
        let mut results: Vec<_> = embedded_chunks
            .iter()
            .map(|chunk| {
                let similarity = chunk.similarity_to(query_embedding);
                (chunk, similarity)
            })
            .collect();

        // Sort by similarity (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top N
        results.into_iter().take(limit).collect()
    }
}

impl Default for EmbeddingService {
    fn default() -> Self {
        Self::new().expect("Failed to create default embedding service")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::{Chunk, ChunkKind};

    #[test]
    fn test_model_type_default() {
        let model = ModelType::default();
        assert_eq!(model.dimensions(), 384);
    }

    #[test]
    #[ignore] // Requires model download
    fn test_embedding_service_creation() {
        let service = EmbeddingService::new();
        assert!(service.is_ok());

        let service = service.unwrap();
        assert_eq!(service.dimensions(), 384);
    }

    #[test]
    #[ignore] // Requires model
    fn test_embed_query() {
        let mut service = EmbeddingService::new().unwrap();
        let query_embedding = service.embed_query("find authentication code").unwrap();

        assert_eq!(query_embedding.len(), 384);
    }

    #[test]
    #[ignore] // Requires model
    fn test_embedding_service_with_batch_size() {
        let service = EmbeddingService::with_model_and_provider(
            ModelType::AllMiniLML6V2Q,
            ExecutionProviderType::Cpu,
            None,
            Some(128), // explicit batch size
        );
        assert!(service.is_ok());
    }

    #[test]
    #[ignore] // Requires model
    fn test_embed_chunks_with_cache() {
        let mut service = EmbeddingService::new().unwrap();

        let chunks = vec![
            Chunk::new(
                "fn authenticate(user: &str) -> bool { true }".to_string(),
                0,
                1,
                ChunkKind::Function,
                "auth.rs".to_string(),
            ),
            Chunk::new(
                "fn hash_password(pwd: &str) -> String { pwd.to_string() }".to_string(),
                2,
                3,
                ChunkKind::Function,
                "auth.rs".to_string(),
            ),
        ];

        // First embedding - no cache
        let embedded1 = service.embed_chunks(chunks.clone()).unwrap();
        assert_eq!(embedded1.len(), 2);

        let stats1 = service.cache_stats();
        assert_eq!(stats1.size, 2);

        // Second embedding - should hit cache
        let embedded2 = service.embed_chunks(chunks.clone()).unwrap();
        assert_eq!(embedded2.len(), 2);

        let stats2 = service.cache_stats();
        assert!(stats2.hit_rate() > 0.0);
    }

    #[test]
    #[ignore] // Requires model
    fn test_search() {
        let mut service = EmbeddingService::new().unwrap();

        let chunks = vec![
            Chunk::new(
                "fn authenticate(user: &str) -> bool { check_credentials(user) }".to_string(),
                0,
                1,
                ChunkKind::Function,
                "auth.rs".to_string(),
            ),
            Chunk::new(
                "fn calculate_fibonacci(n: u32) -> u32 { if n <= 1 { n } else { calculate_fibonacci(n-1) + calculate_fibonacci(n-2) } }".to_string(),
                0,
                1,
                ChunkKind::Function,
                "math.rs".to_string(),
            ),
            Chunk::new(
                "fn hash_password(password: &str) -> String { sha256(password) }".to_string(),
                2,
                3,
                ChunkKind::Function,
                "auth.rs".to_string(),
            ),
        ];

        let embedded = service.embed_chunks(chunks).unwrap();

        // Search for authentication related code
        let query = "authentication and password hashing";
        let query_embedding = service.embed_query(query).unwrap();

        let results = service.search(&query_embedding, &embedded, 2);

        assert_eq!(results.len(), 2);
        // First two results should be auth-related (higher similarity)
        assert!(results[0].0.chunk.path.contains("auth"));
        assert!(results[0].1 > results[1].1); // Scores should be descending
    }
}
