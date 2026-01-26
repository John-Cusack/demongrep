//! Ollama embedding backend
//!
//! This module provides embedding capabilities via a running Ollama server.
//! It uses `ureq` for simple blocking HTTP requests and `rayon` for parallel
//! batch processing.
//!
//! # Requirements
//!
//! - Ollama must be running (`ollama serve`)
//! - An embedding model must be pulled (`ollama pull nomic-embed-text`)
//!
//! # Example
//!
//! ```no_run
//! use demongrep::embed::OllamaEmbedder;
//! use demongrep::config::OllamaConfig;
//!
//! let config = OllamaConfig::default();
//! let embedder = OllamaEmbedder::new(config).expect("Failed to create embedder");
//! let embedding = embedder.embed_one("Hello, world!").unwrap();
//! ```

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::time::Duration;

use super::backend::Embedder;
use crate::config::OllamaConfig;

/// Get known dimensions for common Ollama embedding models
///
/// Returns `Some(dimensions)` for models with known output sizes,
/// `None` for unknown models (which will require probing).
pub fn known_dimensions(model: &str) -> Option<usize> {
    // Strip version tags like ":latest" for matching
    let base_model = model.split(':').next().unwrap_or(model);

    match base_model {
        "nomic-embed-text" => Some(768),
        "mxbai-embed-large" => Some(1024),
        "all-minilm" => Some(384),
        "snowflake-arctic-embed" => Some(1024),
        "bge-m3" => Some(1024),
        "bge-large" => Some(1024),
        "paraphrase-multilingual" => Some(768),
        _ => None,
    }
}

/// Ollama-based embedder using HTTP API
///
/// This embedder connects to a running Ollama server to generate embeddings.
/// It provides GPU acceleration on supported systems (Metal on macOS, CUDA on Linux/Windows)
/// without requiring complex ONNX runtime setup.
pub struct OllamaEmbedder {
    agent: ureq::Agent,
    config: OllamaConfig,
    dimensions: usize,
}

impl OllamaEmbedder {
    /// Create a new Ollama embedder
    ///
    /// This will:
    /// 1. Validate that Ollama is running
    /// 2. Check that the specified model is available
    /// 3. Determine embedding dimensions
    pub fn new(config: OllamaConfig) -> Result<Self> {
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(config.timeout))
            .build();

        // Validate Ollama is running
        let health_url = format!("{}/api/tags", config.url);
        let resp = agent.get(&health_url).call().map_err(|e| {
            anyhow::anyhow!(
                "Cannot connect to Ollama at {}. Is it running? Start with: ollama serve\nError: {}",
                config.url,
                e
            )
        })?;

        // Check model is available
        let models: serde_json::Value = resp
            .into_json()
            .context("Failed to parse Ollama response")?;

        let model_exists = models["models"]
            .as_array()
            .map(|arr| {
                arr.iter().any(|m| {
                    m["name"]
                        .as_str()
                        .map(|n| n.starts_with(&config.model) || n == config.model)
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false);

        if !model_exists {
            // List available models for helpful error
            let available: Vec<&str> = models["models"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|m| m["name"].as_str())
                        .collect()
                })
                .unwrap_or_default();

            anyhow::bail!(
                "Model '{}' not found in Ollama. Install with: ollama pull {}\nAvailable models: {:?}",
                config.model,
                config.model,
                available
            );
        }

        // Get dimensions (known or probe)
        let dimensions = match known_dimensions(&config.model) {
            Some(d) => d,
            None => Self::probe_dimensions(&agent, &config)?,
        };

        println!(
            "   Ollama embedder: {} ({} dimensions) at {}",
            config.model, dimensions, config.url
        );

        Ok(Self {
            agent,
            config,
            dimensions,
        })
    }

    /// Probe model dimensions by generating a test embedding
    fn probe_dimensions(agent: &ureq::Agent, config: &OllamaConfig) -> Result<usize> {
        let url = format!("{}/api/embeddings", config.url);
        let resp: serde_json::Value = agent
            .post(&url)
            .send_json(serde_json::json!({
                "model": config.model,
                "prompt": "dimension probe"
            }))
            .context("Failed to probe model dimensions")?
            .into_json()
            .context("Failed to parse dimension probe response")?;

        resp["embedding"]
            .as_array()
            .map(|arr| arr.len())
            .ok_or_else(|| anyhow::anyhow!("Invalid embedding response from Ollama"))
    }

    /// Internal method to embed a single text
    fn embed_one_internal(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!("{}/api/embeddings", self.config.url);
        let resp: serde_json::Value = self
            .agent
            .post(&url)
            .send_json(serde_json::json!({
                "model": self.config.model,
                "prompt": text
            }))
            .context("Ollama embedding request failed")?
            .into_json()
            .context("Failed to parse Ollama response")?;

        // Check for error response
        if let Some(error) = resp.get("error") {
            anyhow::bail!("Ollama error: {}", error);
        }

        resp["embedding"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("No embedding in Ollama response"))?
            .iter()
            .map(|v| {
                v.as_f64()
                    .map(|f| f as f32)
                    .ok_or_else(|| anyhow::anyhow!("Invalid float in embedding"))
            })
            .collect()
    }
}

impl Embedder for OllamaEmbedder {
    fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // For small batches, just process sequentially
        if texts.len() <= 2 {
            return texts
                .iter()
                .map(|text| self.embed_one_internal(text))
                .collect();
        }

        // Use rayon for parallel HTTP calls, limited by parallelism config
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.config.parallelism)
            .build()
            .context("Failed to create thread pool")?;

        pool.install(|| {
            texts
                .par_iter()
                .map(|text| self.embed_one_internal(text))
                .collect()
        })
    }

    fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_one_internal(text)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }

    fn model_short_name(&self) -> &str {
        // Extract base model name (e.g., "nomic-embed-text" from "nomic-embed-text:latest")
        self.config.model.split(':').next().unwrap_or(&self.config.model)
    }

    fn backend_name(&self) -> &str {
        "ollama"
    }
}

// OllamaEmbedder is Send + Sync because ureq::Agent is thread-safe
unsafe impl Send for OllamaEmbedder {}
unsafe impl Sync for OllamaEmbedder {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_known_dimensions() {
        assert_eq!(known_dimensions("nomic-embed-text"), Some(768));
        assert_eq!(known_dimensions("nomic-embed-text:latest"), Some(768));
        assert_eq!(known_dimensions("mxbai-embed-large"), Some(1024));
        assert_eq!(known_dimensions("all-minilm"), Some(384));
        assert_eq!(known_dimensions("unknown-model"), None);
    }

    #[test]
    fn test_model_short_name() {
        // We can't create a real embedder without Ollama running,
        // but we can test the short name extraction logic
        let full = "nomic-embed-text:latest";
        let short = full.split(':').next().unwrap_or(full);
        assert_eq!(short, "nomic-embed-text");
    }

    #[test]
    #[ignore] // Requires running Ollama server
    fn test_ollama_embedder_creation() {
        let config = OllamaConfig::default();
        let embedder = OllamaEmbedder::new(config);
        assert!(embedder.is_ok());
    }

    #[test]
    #[ignore] // Requires running Ollama server
    fn test_ollama_embed_one() {
        let config = OllamaConfig::default();
        let embedder = OllamaEmbedder::new(config).unwrap();
        let embedding = embedder.embed_one("Hello, world!").unwrap();
        assert_eq!(embedding.len(), embedder.dimensions());
    }

    #[test]
    #[ignore] // Requires running Ollama server
    fn test_ollama_embed_batch() {
        let config = OllamaConfig::default();
        let embedder = OllamaEmbedder::new(config).unwrap();

        let texts = vec![
            "Hello".to_string(),
            "World".to_string(),
            "Test".to_string(),
        ];

        let embeddings = embedder.embed_batch(texts).unwrap();
        assert_eq!(embeddings.len(), 3);
        for emb in embeddings {
            assert_eq!(emb.len(), embedder.dimensions());
        }
    }
}
