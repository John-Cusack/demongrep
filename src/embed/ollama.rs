//! Ollama embedding backend
//!
//! This module provides embedding capabilities via a running Ollama server.
//! It uses `ureq` for simple blocking HTTP requests and `rayon` for parallel
//! batch processing.
//!
//! # Requirements
//!
//! - Ollama must be running (`ollama serve`)
//! - An embedding model must be pulled (`ollama pull all-minilm`)
//!
//! # Example
//!
//! ```no_run
//! use demongrep::embed::{OllamaEmbedder, Embedder};
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
            // Model not found - try to pull it automatically
            eprintln!(
                "📥 Model '{}' not found locally. Pulling from Ollama library...",
                config.model
            );

            Self::pull_model(&agent, &config)?;

            eprintln!("✅ Model '{}' pulled successfully!", config.model);
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

    /// Pull a model from the Ollama library
    fn pull_model(agent: &ureq::Agent, config: &OllamaConfig) -> Result<()> {
        use std::io::{BufRead, BufReader};

        let url = format!("{}/api/pull", config.url);

        // Use a longer timeout for pulling (models can be large)
        let pull_agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(600)) // 10 minute timeout for large models
            .build();

        let resp = pull_agent
            .post(&url)
            .send_json(serde_json::json!({
                "name": config.model,
                "stream": true
            }))
            .map_err(|e| anyhow::anyhow!("Failed to start model pull: {}", e))?;

        // Read streaming response to show progress
        let reader = BufReader::new(resp.into_reader());
        let mut last_status = String::new();

        for line in reader.lines() {
            let line = line.context("Failed to read pull response")?;
            if line.is_empty() {
                continue;
            }

            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
                // Check for error
                if let Some(error) = json.get("error") {
                    anyhow::bail!("Failed to pull model: {}", error);
                }

                // Show progress
                if let Some(status) = json.get("status").and_then(|s| s.as_str()) {
                    if status != last_status {
                        if status.contains("pulling") {
                            eprint!("\r   {} ", status);
                        } else if status == "success" {
                            eprintln!();
                        } else {
                            eprintln!("   {}", status);
                        }
                        last_status = status.to_string();
                    }

                    // Show download progress if available
                    if let (Some(completed), Some(total)) = (
                        json.get("completed").and_then(|v| v.as_u64()),
                        json.get("total").and_then(|v| v.as_u64()),
                    ) {
                        if total > 0 {
                            let pct = (completed as f64 / total as f64 * 100.0) as u32;
                            let mb_done = completed / 1_000_000;
                            let mb_total = total / 1_000_000;
                            eprint!("\r   {} [{:>3}%] {}/{}MB", status, pct, mb_done, mb_total);
                        }
                    }
                }
            }
        }

        // Verify the model was actually pulled by checking the list again
        let health_url = format!("{}/api/tags", config.url);
        let resp = agent.get(&health_url).call().context("Failed to verify model pull")?;
        let models: serde_json::Value = resp.into_json().context("Failed to parse model list")?;

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
            anyhow::bail!(
                "Model '{}' could not be pulled. It may not exist in the Ollama library.\n\
                 Check available models at: https://ollama.ai/library",
                config.model
            );
        }

        Ok(())
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

    #[test]
    #[ignore] // Requires running Ollama server - tests auto-pull
    fn test_auto_pull_model() {
        // Use a small model for testing auto-pull
        // First, remove it if it exists (to test pulling)
        let _ = std::process::Command::new("ollama")
            .args(["rm", "all-minilm"])
            .output();

        // Now try to create embedder - should auto-pull
        let config = OllamaConfig {
            model: "all-minilm".to_string(),
            ..Default::default()
        };

        let result = OllamaEmbedder::new(config);
        assert!(result.is_ok(), "Auto-pull should succeed: {:?}", result.err());

        let embedder = result.unwrap();
        assert_eq!(embedder.dimensions(), 384);
    }

    #[test]
    #[ignore] // Requires running Ollama server
    fn test_auto_pull_invalid_model_fails() {
        // Try to pull a model that doesn't exist
        let config = OllamaConfig {
            model: "this-model-does-not-exist-12345".to_string(),
            ..Default::default()
        };

        let result = OllamaEmbedder::new(config);
        assert!(result.is_err(), "Should fail for non-existent model");

        let err = result.err().unwrap().to_string();
        assert!(
            err.contains("could not be pulled") || err.contains("not exist"),
            "Error should mention pull failure: {}",
            err
        );
    }

    #[test]
    #[ignore] // Requires running Ollama server
    fn test_auto_pull_then_embed() {
        // Remove model first to ensure we test the full flow
        let _ = std::process::Command::new("ollama")
            .args(["rm", "all-minilm"])
            .output();

        let config = OllamaConfig {
            model: "all-minilm".to_string(),
            ..Default::default()
        };

        // Create embedder (should auto-pull)
        let embedder = OllamaEmbedder::new(config).expect("Should auto-pull and create embedder");

        // Now test embedding works
        let embedding = embedder.embed_one("test embedding after auto-pull").unwrap();
        assert_eq!(embedding.len(), 384);
        assert!(embedding.iter().any(|&x| x != 0.0), "Embedding should have non-zero values");
    }

    #[test]
    fn test_ollama_config_default() {
        let config = OllamaConfig::default();
        assert_eq!(config.url, "http://localhost:11434");
        assert_eq!(config.model, "all-minilm");
        assert_eq!(config.timeout, 30);
        assert_eq!(config.parallelism, 8);
    }

    #[test]
    #[ignore] // Requires running Ollama server
    fn test_already_installed_model_no_pull() {
        // First ensure model is installed
        let _ = std::process::Command::new("ollama")
            .args(["pull", "nomic-embed-text"])
            .output();

        let config = OllamaConfig::default();

        // Should create without needing to pull
        let start = std::time::Instant::now();
        let result = OllamaEmbedder::new(config);
        let elapsed = start.elapsed();

        assert!(result.is_ok());
        // If model is already installed, creation should be fast (< 5 seconds)
        // Pull would take much longer
        assert!(
            elapsed.as_secs() < 5,
            "Creating embedder with installed model took too long ({:?}), suggests unnecessary pull",
            elapsed
        );
    }
}
