//! Ollama embedding backend
//!
//! This module provides embedding capabilities via a running Ollama server.
//! It uses `ureq` for simple blocking HTTP requests and `rayon` for parallel
//! batch processing.
//!
//! # Requirements
//!
//! - Ollama must be running (`ollama serve`)
//! - An embedding model must be available (auto-pulled if missing)
//!
//! # Default Model
//!
//! The default model is `unclemusclez/jina-embeddings-v2-base-code`:
//! - 768 dimensions (cross-compatible with FastEmbed's `jina-code`)
//! - 8192 token context (handles full functions/classes)
//! - Trained on 150M+ code Q&A pairs from GitHub
//!
//! # Example
//!
//! ```no_run
//! use demongrep::embed::{OllamaEmbedder, Embedder};
//! use demongrep::config::OllamaConfig;
//!
//! let config = OllamaConfig::default();
//! let embedder = OllamaEmbedder::new(config).expect("Failed to create embedder");
//! let embedding = embedder.embed_one("def hello(): print('world')").unwrap();
//! ```

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::time::Duration;

use super::backend::Embedder;
use crate::config::OllamaConfig;

/// Default chars-per-token estimate for initial truncation attempt.
/// This is optimistic (assumes ~3 chars per token like English prose).
const CHARS_PER_TOKEN_OPTIMISTIC: usize = 3;

/// Conservative chars-per-token estimate for retry after context overflow.
/// This is pessimistic (assumes ~1.5 chars per token like dense code).
const CHARS_PER_TOKEN_CONSERVATIVE: usize = 3; // Actually 1.5, but we use 3/2

/// Get known dimensions for common Ollama embedding models
///
/// Returns `Some(dimensions)` for models with known output sizes,
/// `None` for unknown models (which will require probing).
pub fn known_dimensions(model: &str) -> Option<usize> {
    // Strip version tags like ":latest" for matching
    let base_model = model.split(':').next().unwrap_or(model);

    match base_model {
        // Jina code model - best for code search (8k context, trained on code)
        "jina-embeddings-v2-base-code" | "unclemusclez/jina-embeddings-v2-base-code" => Some(768),
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

/// Fallback context lengths for models when we can't query Ollama
fn fallback_context_tokens(model: &str) -> usize {
    let base_model = model.split(':').next().unwrap_or(model);

    match base_model {
        "all-minilm" => 256,
        "snowflake-arctic-embed" => 512,
        "mxbai-embed-large" => 512,
        "bge-large" => 512,
        "paraphrase-multilingual" => 512,
        "nomic-embed-text" => 2048,
        // Long context models (8k)
        "jina-embeddings-v2-base-code" | "unclemusclez/jina-embeddings-v2-base-code" => 8192,
        "bge-m3" => 8192,
        _ => 512, // Conservative default
    }
}

/// Truncate text to fit within estimated token limit
///
/// Returns the truncated string slice at a valid UTF-8 boundary.
fn truncate_to_char_limit(text: &str, max_chars: usize) -> &str {
    if text.len() <= max_chars {
        return text;
    }

    // Find valid UTF-8 boundary
    let mut end = max_chars.min(text.len());
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// Ollama-based embedder using HTTP API
///
/// This embedder connects to a running Ollama server to generate embeddings.
/// It provides GPU acceleration on supported systems (Metal on macOS, CUDA on Linux/Windows)
/// without requiring complex ONNX runtime setup.
///
/// Context length is queried from Ollama at startup. If text exceeds the context,
/// the embedder will automatically truncate and retry with progressively more
/// aggressive truncation.
pub struct OllamaEmbedder {
    agent: ureq::Agent,
    config: OllamaConfig,
    dimensions: usize,
    /// Context length in tokens, queried from Ollama
    context_length: usize,
}

impl OllamaEmbedder {
    /// Create a new Ollama embedder
    ///
    /// This will:
    /// 1. Validate that Ollama is running
    /// 2. Check that the specified model is available (auto-pull if not)
    /// 3. Query the model's context length from Ollama
    /// 4. Determine embedding dimensions
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

        // Query context length from Ollama
        let context_length = Self::query_context_length(&agent, &config)
            .unwrap_or_else(|_| {
                let fallback = fallback_context_tokens(&config.model);
                eprintln!(
                    "   Warning: Could not query context length, using fallback: {} tokens",
                    fallback
                );
                fallback
            });

        // Get dimensions (known or probe)
        let dimensions = match known_dimensions(&config.model) {
            Some(d) => d,
            None => Self::probe_dimensions(&agent, &config)?,
        };

        eprintln!(
            "   Ollama embedder: {} ({} dimensions, {} token context) at {}",
            config.model, dimensions, context_length, config.url
        );

        Ok(Self {
            agent,
            config,
            dimensions,
            context_length,
        })
    }

    /// Query the model's context length from Ollama via /api/show
    fn query_context_length(agent: &ureq::Agent, config: &OllamaConfig) -> Result<usize> {
        let url = format!("{}/api/show", config.url);
        let resp: serde_json::Value = agent
            .post(&url)
            .send_json(serde_json::json!({
                "name": config.model
            }))
            .context("Failed to query model info")?
            .into_json()
            .context("Failed to parse model info response")?;

        // Try to get context length from model_info
        // The structure is: { "model_info": { "general.context_length": N } }
        // or for some models: { "model_info": { "<arch>.context_length": N } }
        if let Some(model_info) = resp.get("model_info").and_then(|v| v.as_object()) {
            // Look for any key ending in "context_length"
            for (key, value) in model_info {
                if key.ends_with("context_length") {
                    if let Some(ctx) = value.as_u64() {
                        return Ok(ctx as usize);
                    }
                }
            }
        }

        // Fallback: check parameters
        if let Some(params) = resp.get("parameters").and_then(|v| v.as_str()) {
            // Parameters is a string like "num_ctx 2048\n..."
            for line in params.lines() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 && parts[0] == "num_ctx" {
                    if let Ok(ctx) = parts[1].parse::<usize>() {
                        return Ok(ctx);
                    }
                }
            }
        }

        anyhow::bail!("Could not find context length in model info")
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
        let url = format!("{}/api/embed", config.url);
        let resp: serde_json::Value = agent
            .post(&url)
            .send_json(serde_json::json!({
                "model": config.model,
                "input": "dimension probe",
                "truncate": true
            }))
            .context("Failed to probe model dimensions")?
            .into_json()
            .context("Failed to parse dimension probe response")?;

        // /api/embed returns { "embeddings": [[...]] }
        resp["embeddings"]
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|emb| emb.as_array())
            .map(|arr| arr.len())
            .ok_or_else(|| anyhow::anyhow!("Invalid embedding response from Ollama"))
    }

    /// Embed text with automatic retry on context overflow
    ///
    /// Strategy:
    /// 1. First attempt: use optimistic estimate (3 chars/token)
    /// 2. On 400 error: retry with conservative estimate (1.5 chars/token)
    /// 3. On second 400 error: retry with very conservative (1 char/token)
    fn embed_with_retry(&self, text: &str) -> Result<Vec<f32>> {
        // First attempt: optimistic truncation (3 chars per token)
        let max_chars_optimistic = self.context_length * CHARS_PER_TOKEN_OPTIMISTIC;
        let truncated = truncate_to_char_limit(text, max_chars_optimistic);

        match self.try_embed(truncated) {
            Ok(embedding) => return Ok(embedding),
            Err(e) if Self::is_context_overflow_error(&e) => {
                // Continue to retry
            }
            Err(e) => return Err(e),
        }

        // Second attempt: conservative truncation (1.5 chars per token)
        let max_chars_conservative = (self.context_length * CHARS_PER_TOKEN_CONSERVATIVE) / 2;
        let truncated = truncate_to_char_limit(text, max_chars_conservative);

        match self.try_embed(truncated) {
            Ok(embedding) => return Ok(embedding),
            Err(e) if Self::is_context_overflow_error(&e) => {
                // Continue to final retry
            }
            Err(e) => return Err(e),
        }

        // Final attempt: very conservative (1 char per token)
        let max_chars_final = self.context_length;
        let truncated = truncate_to_char_limit(text, max_chars_final);

        self.try_embed(truncated)
    }

    /// Check if error is a context overflow error
    fn is_context_overflow_error(e: &anyhow::Error) -> bool {
        let msg = e.to_string().to_lowercase();
        msg.contains("context length") || msg.contains("input length exceeds")
    }

    /// Try to embed text, returning detailed error on failure
    fn try_embed(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!("{}/api/embed", self.config.url);

        let result = self.agent.post(&url).send_json(serde_json::json!({
            "model": self.config.model,
            "input": text,
            "truncate": true
        }));

        let resp = match result {
            Ok(r) => r,
            Err(ureq::Error::Status(code, response)) => {
                let body = response.into_string().unwrap_or_default();
                anyhow::bail!(
                    "Ollama API error ({}): {} [input: {} chars]",
                    code,
                    body.trim(),
                    text.len()
                );
            }
            Err(e) => {
                return Err(anyhow::anyhow!("Ollama request failed: {}", e));
            }
        };

        let resp: serde_json::Value = resp
            .into_json()
            .context("Failed to parse Ollama response")?;

        if let Some(error) = resp.get("error") {
            anyhow::bail!("Ollama error: {}", error);
        }

        resp["embeddings"]
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|emb| emb.as_array())
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
                .map(|text| self.embed_with_retry(text))
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
                .map(|text| self.embed_with_retry(text))
                .collect()
        })
    }

    fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_with_retry(text)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }

    fn model_short_name(&self) -> &str {
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
    fn test_fallback_context_tokens() {
        assert_eq!(fallback_context_tokens("all-minilm"), 256);
        assert_eq!(fallback_context_tokens("nomic-embed-text"), 2048);
        assert_eq!(fallback_context_tokens("bge-m3"), 8192);
        assert_eq!(fallback_context_tokens("unknown"), 512);
    }

    #[test]
    fn test_truncate_to_char_limit() {
        assert_eq!(truncate_to_char_limit("hello", 10), "hello");
        assert_eq!(truncate_to_char_limit("hello world", 5), "hello");
        assert_eq!(truncate_to_char_limit("hello", 0), "");
        // UTF-8 boundary test
        assert_eq!(truncate_to_char_limit("héllo", 2), "h"); // 'é' is 2 bytes
    }

    #[test]
    fn test_model_short_name() {
        let full = "nomic-embed-text:latest";
        let short = full.split(':').next().unwrap_or(full);
        assert_eq!(short, "nomic-embed-text");
    }

    #[test]
    fn test_ollama_config_default() {
        let config = OllamaConfig::default();
        assert_eq!(config.url, "http://localhost:11434");
        assert_eq!(config.model, "unclemusclez/jina-embeddings-v2-base-code");
        assert_eq!(config.timeout, 30);
        assert_eq!(config.parallelism, 8);
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
    #[ignore] // Requires running Ollama server
    fn test_context_length_queried() {
        let config = OllamaConfig::default();
        let embedder = OllamaEmbedder::new(config).unwrap();
        // jina-embeddings-v2-base-code should have 8192 context
        assert!(embedder.context_length >= 8192);
    }

    #[test]
    #[ignore] // Requires running Ollama server
    fn test_long_text_truncation() {
        let config = OllamaConfig::default();
        let embedder = OllamaEmbedder::new(config).unwrap();

        // Create very long text that will need truncation
        let long_text = "x".repeat(50000);
        let result = embedder.embed_one(&long_text);

        // Should succeed with truncation
        assert!(result.is_ok(), "Long text should be truncated and embedded");
    }
}
