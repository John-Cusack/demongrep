//! Backend-agnostic embedding trait
//!
//! This module provides the `Embedder` trait that abstracts over different
//! embedding backends (FastEmbed/ONNX or Ollama). This allows the rest of
//! the codebase to work with embeddings without knowing the underlying provider.

use anyhow::Result;

/// Backend-agnostic embedding trait
///
/// Implemented by both FastEmbedder (ONNX-based) and OllamaEmbedder (HTTP-based).
/// The trait uses `&self` to allow for stateless/clonable embedders.
pub trait Embedder: Send + Sync {
    /// Embed a batch of texts
    fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>>;

    /// Embed a single text
    fn embed_one(&self, text: &str) -> Result<Vec<f32>>;

    /// Get embedding dimensions
    fn dimensions(&self) -> usize;

    /// Get full model name
    fn model_name(&self) -> &str;

    /// Get short model name (for storage)
    fn model_short_name(&self) -> &str;

    /// Get backend name ("fastembed" or "ollama")
    fn backend_name(&self) -> &str;
}

/// Available embedding backends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmbeddingBackend {
    /// FastEmbed/ONNX backend (default) - runs locally via ONNX Runtime
    #[default]
    FastEmbed,
    /// Ollama backend - requires running Ollama server
    #[cfg(feature = "ollama")]
    Ollama,
}

impl EmbeddingBackend {
    /// Parse backend from string
    ///
    /// # Examples
    /// ```
    /// use demongrep::embed::EmbeddingBackend;
    ///
    /// let backend = EmbeddingBackend::from_str("fastembed").unwrap();
    /// assert_eq!(backend, EmbeddingBackend::FastEmbed);
    /// ```
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "fastembed" | "onnx" | "local" => Ok(Self::FastEmbed),
            "ollama" => {
                #[cfg(feature = "ollama")]
                return Ok(Self::Ollama);
                #[cfg(not(feature = "ollama"))]
                anyhow::bail!(
                    "Ollama backend not available. Rebuild with: cargo build --features ollama"
                );
            }
            _ => anyhow::bail!("Unknown backend '{}'. Available: fastembed, ollama", s),
        }
    }

    /// Get the backend name as a string
    pub fn name(&self) -> &'static str {
        match self {
            Self::FastEmbed => "fastembed",
            #[cfg(feature = "ollama")]
            Self::Ollama => "ollama",
        }
    }
}

impl std::fmt::Display for EmbeddingBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_from_str() {
        assert_eq!(
            EmbeddingBackend::from_str("fastembed").unwrap(),
            EmbeddingBackend::FastEmbed
        );
        assert_eq!(
            EmbeddingBackend::from_str("onnx").unwrap(),
            EmbeddingBackend::FastEmbed
        );
        assert_eq!(
            EmbeddingBackend::from_str("local").unwrap(),
            EmbeddingBackend::FastEmbed
        );
        assert_eq!(
            EmbeddingBackend::from_str("FASTEMBED").unwrap(),
            EmbeddingBackend::FastEmbed
        );
    }

    #[test]
    fn test_backend_unknown() {
        assert!(EmbeddingBackend::from_str("unknown").is_err());
    }

    #[test]
    fn test_backend_default() {
        assert_eq!(EmbeddingBackend::default(), EmbeddingBackend::FastEmbed);
    }

    #[test]
    fn test_backend_name() {
        assert_eq!(EmbeddingBackend::FastEmbed.name(), "fastembed");
    }

    #[cfg(feature = "ollama")]
    #[test]
    fn test_ollama_backend() {
        assert_eq!(
            EmbeddingBackend::from_str("ollama").unwrap(),
            EmbeddingBackend::Ollama
        );
        assert_eq!(EmbeddingBackend::Ollama.name(), "ollama");
    }
}
