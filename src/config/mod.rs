//! Configuration file support for demongrep
//!
//! Configuration is loaded from `~/.demongrep/config.toml` if it exists.
//! CLI flags always override config file values.
//!
//! Example config file:
//! ```toml
//! [embedding]
//! model = "bge-small"
//! provider = "cpu"
//! device_id = 0
//! batch_size = 256
//!
//! [indexing]
//! max_chunk_lines = 75
//! max_chunk_chars = 2000
//! overlap_lines = 10
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Global configuration for demongrep
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Config {
    /// Embedding configuration
    pub embedding: EmbeddingConfig,

    /// Indexing configuration
    pub indexing: IndexingConfig,
}

/// Embedding model and provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    /// Model short name (e.g., "bge-small", "minilm-l6-q", "jina-code")
    /// See `demongrep --help` for full list
    pub model: Option<String>,

    /// Execution provider: "cpu", "auto", "cuda", "tensorrt", "coreml", "directml"
    pub provider: String,

    /// GPU device ID (for CUDA/TensorRT/DirectML)
    pub device_id: i32,

    /// Batch size for embedding (None = auto-detect based on provider)
    pub batch_size: Option<usize>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: None, // Use library default
            provider: "cpu".to_string(),
            device_id: 0,
            batch_size: None, // Auto-detect
        }
    }
}

/// Indexing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IndexingConfig {
    /// Maximum chunk size in lines
    pub max_chunk_lines: usize,

    /// Maximum chunk size in characters
    pub max_chunk_chars: usize,

    /// Overlap between chunks in lines
    pub overlap_lines: usize,
}

impl Default for IndexingConfig {
    fn default() -> Self {
        Self {
            max_chunk_lines: 75,
            max_chunk_chars: 2000,
            overlap_lines: 10,
        }
    }
}

impl Config {
    /// Get the config file path
    pub fn config_path() -> Option<PathBuf> {
        dirs::home_dir().map(|h| h.join(".demongrep").join("config.toml"))
    }

    /// Load configuration from the default location
    /// Returns default config if file doesn't exist
    pub fn load() -> Result<Self> {
        let path = match Self::config_path() {
            Some(p) => p,
            None => return Ok(Self::default()),
        };

        if !path.exists() {
            return Ok(Self::default());
        }

        Self::load_from(&path)
    }

    /// Load configuration from a specific path
    pub fn load_from(path: &PathBuf) -> Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&contents)?;
        Ok(config)
    }

    /// Save configuration to the default location
    pub fn save(&self) -> Result<()> {
        let path = Self::config_path()
            .ok_or_else(|| anyhow::anyhow!("Could not determine home directory"))?;

        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let contents = toml::to_string_pretty(self)?;
        std::fs::write(&path, contents)?;
        Ok(())
    }

    /// Create a default config file if one doesn't exist
    /// Returns true if a new file was created
    pub fn create_default_if_missing() -> Result<bool> {
        let path = match Self::config_path() {
            Some(p) => p,
            None => return Ok(false),
        };

        if path.exists() {
            return Ok(false);
        }

        let config = Self::default();
        config.save()?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.embedding.provider, "cpu");
        assert_eq!(config.embedding.device_id, 0);
        assert!(config.embedding.model.is_none());
        assert!(config.embedding.batch_size.is_none());
    }

    #[test]
    fn test_load_config_from_toml() {
        let toml_content = r#"
[embedding]
model = "jina-code"
provider = "cuda"
device_id = 1
batch_size = 512

[indexing]
max_chunk_lines = 100
max_chunk_chars = 3000
overlap_lines = 15
"#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(toml_content.as_bytes()).unwrap();
        let path = temp_file.path().to_path_buf();

        let config = Config::load_from(&path).unwrap();

        assert_eq!(config.embedding.model, Some("jina-code".to_string()));
        assert_eq!(config.embedding.provider, "cuda");
        assert_eq!(config.embedding.device_id, 1);
        assert_eq!(config.embedding.batch_size, Some(512));
        assert_eq!(config.indexing.max_chunk_lines, 100);
        assert_eq!(config.indexing.max_chunk_chars, 3000);
        assert_eq!(config.indexing.overlap_lines, 15);
    }

    #[test]
    fn test_partial_config() {
        // Test that missing fields use defaults
        let toml_content = r#"
[embedding]
provider = "auto"
"#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(toml_content.as_bytes()).unwrap();
        let path = temp_file.path().to_path_buf();

        let config = Config::load_from(&path).unwrap();

        assert_eq!(config.embedding.provider, "auto");
        // These should be defaults
        assert_eq!(config.embedding.device_id, 0);
        assert!(config.embedding.model.is_none());
        assert_eq!(config.indexing.max_chunk_lines, 75);
    }

    #[test]
    fn test_empty_config() {
        let toml_content = "";

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(toml_content.as_bytes()).unwrap();
        let path = temp_file.path().to_path_buf();

        let config = Config::load_from(&path).unwrap();

        // All defaults
        assert_eq!(config.embedding.provider, "cpu");
        assert_eq!(config.indexing.max_chunk_lines, 75);
    }

    #[test]
    fn test_serialize_config() {
        let config = Config::default();
        let toml_str = toml::to_string_pretty(&config).unwrap();

        assert!(toml_str.contains("[embedding]"));
        assert!(toml_str.contains("provider = \"cpu\""));
    }
}
