use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use crate::config::Config;
use crate::embed::{EmbeddingBackend, ExecutionProviderType, ModelType};

/// Check if Ollama is available at the given URL
#[cfg(feature = "ollama")]
fn is_ollama_available(url: &str) -> bool {
    let health_url = format!("{}/api/tags", url);
    // Quick check with short timeout
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(2))
        .build();
    agent.get(&health_url).call().is_ok()
}

/// Fast, local semantic code search powered by Rust
#[derive(Parser, Debug)]
#[command(name = "demongrep")]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Suppress informational output (only show results/errors)
    #[arg(short, long, global = true)]
    pub quiet: bool,

    /// Override default store name
    #[arg(long, global = true)]
    pub store: Option<String>,

    /// Embedding model to use (e.g., bge-small, minilm-l6-q, jina-code)
    /// Available: minilm-l6, minilm-l6-q, minilm-l12, minilm-l12-q, paraphrase-minilm,
    ///            bge-small, bge-small-q, bge-base, nomic-v1, nomic-v1.5, nomic-v1.5-q,
    ///            jina-code, e5-multilingual, mxbai-large, modernbert-large
    #[arg(long, global = true)]
    pub model: Option<String>,

    /// Batch size for embedding (default determined by provider)
    #[arg(long, global = true)]
    pub batch_size: Option<usize>,

    /// Embedding backend: fastembed (default) or ollama
    /// Use ollama for GPU acceleration via Ollama server
    #[arg(long, global = true)]
    pub backend: Option<String>,

    /// Ollama model name (when using ollama backend)
    /// Examples: unclemusclez/jina-embeddings-v2-base-code (default), nomic-embed-text, all-minilm
    #[arg(long, global = true)]
    pub ollama_model: Option<String>,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Search the codebase using natural language
    Search {
        /// Search query (e.g., "where do we handle authentication?")
        query: String,

        /// Maximum total results to return
        #[arg(short = 'm', long, default_value = "25")]
        max_results: usize,

        /// Maximum matches to show per file
        #[arg(long, default_value = "1")]
        per_file: usize,

        /// Show full chunk content instead of snippets
        #[arg(short, long)]
        content: bool,

        /// Show relevance scores
        #[arg(long)]
        scores: bool,

        /// Show file paths only (like grep -l)
        #[arg(long)]
        compact: bool,

        /// Force re-index changed files before searching
        #[arg(short, long)]
        sync: bool,

        /// Output JSON for agents
        #[arg(long)]
        json: bool,

        /// Path to search in (defaults to current directory)
        #[arg(long)]
        path: Option<PathBuf>,

        /// Use vector-only search (disable hybrid FTS)
        #[arg(long)]
        vector_only: bool,

        /// RRF k parameter for score fusion (default 60, research-recommended)
        #[arg(long, default_value = "60")]
        rrf_k: f32,

        /// Enable neural reranking for better accuracy (uses Jina Reranker)
        #[arg(long)]
        rerank: bool,

        /// Number of top results to rerank (default 50)
        #[arg(long, default_value = "50")]
        rerank_top: usize,

        /// Filter results to files under this path (e.g., "src/")
        #[arg(long)]
        filter_path: Option<String>,

        /// Execution provider for embeddings (cpu, auto, cuda, tensorrt, coreml, directml)
        /// If not specified, uses config file or defaults to cpu
        #[arg(long)]
        provider: Option<String>,

        /// GPU device ID to use (for CUDA/TensorRT)
        #[arg(long)]
        device_id: Option<i32>,
    },

    /// Index the repository
    Index {
        /// Path to index (defaults to current directory)
        path: Option<PathBuf>,

        /// Show what would be indexed without actually indexing
        #[arg(long)]
        dry_run: bool,

        /// Force full re-index (delete existing database first)
        #[arg(short, long)]
        force: bool,

        /// Index to global database in home directory instead of local .demongrep.db
        #[arg(short = 'g', long)]
        global: bool,

        /// Execution provider for embeddings (cpu, auto, cuda, tensorrt, coreml, directml)
        /// If not specified, uses config file or defaults to cpu
        #[arg(long)]
        provider: Option<String>,

        /// GPU device ID to use (for CUDA/TensorRT)
        #[arg(long)]
        device_id: Option<i32>,
    },

    /// Run a background server with live file watching
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value = "4444")]
        port: u16,

        /// Path to serve (defaults to current directory)
        path: Option<PathBuf>,

        /// Execution provider for embeddings (cpu, auto, cuda, tensorrt, coreml, directml)
        /// If not specified, uses config file or defaults to cpu
        #[arg(long)]
        provider: Option<String>,

        /// GPU device ID to use (for CUDA/TensorRT)
        #[arg(long)]
        device_id: Option<i32>,
    },

    /// List all indexed repositories
    List,

    /// Show statistics about the vector database
    Stats {
        /// Path to show stats for (defaults to current directory)
        path: Option<PathBuf>,
    },

    /// Clear the vector database
    Clear {
        /// Path to clear (defaults to current directory)
        path: Option<PathBuf>,

        /// Skip confirmation prompt
        #[arg(short = 'y', long)]
        yes: bool,

        /// Project name or path to clear (looks up in global projects.json)
        #[arg(short = 'p', long)]
        project: Option<String>,
    },

    /// Check installation health
    Doctor,

    /// Download embedding models
    Setup {
        /// Model to download (defaults to mxbai-embed-xsmall-v1)
        #[arg(long)]
        model: Option<String>,
    },

    /// Start MCP server for Claude Code integration
    Mcp {
        /// Path to project (defaults to current directory)
        path: Option<PathBuf>,

        /// Execution provider for embeddings (cpu, auto, cuda, tensorrt, coreml, directml)
        /// If not specified, uses config file or defaults to cpu
        #[arg(long)]
        provider: Option<String>,

        /// GPU device ID to use (for CUDA/TensorRT)
        #[arg(long)]
        device_id: Option<i32>,
    },
}

pub async fn run() -> Result<()> {
    let cli = Cli::parse();

    // Load config file (uses defaults if file doesn't exist)
    let config = Config::load().unwrap_or_else(|e| {
        eprintln!("Warning: Failed to load config file: {}", e);
        Config::default()
    });

    // Merge model: CLI > config > default
    let model_str = cli.model.as_ref()
        .or(config.embedding.model.as_ref());
    let model_type = model_str.and_then(|m| ModelType::from_str(m));
    if model_str.is_some() && model_type.is_none() {
        eprintln!("Unknown model: '{}'. Available models:", model_str.unwrap());
        eprintln!("  minilm-l6, minilm-l6-q, minilm-l12, minilm-l12-q, paraphrase-minilm");
        eprintln!("  bge-small, bge-small-q, bge-base, nomic-v1, nomic-v1.5, nomic-v1.5-q");
        eprintln!("  jina-code, e5-multilingual, mxbai-large, modernbert-large");
        std::process::exit(1);
    }

    // Merge batch_size: CLI > config
    let batch_size = cli.batch_size.or(config.embedding.batch_size);

    // Merge backend: CLI > config > auto-detect
    let backend = if let Some(ref backend_str) = cli.backend {
        // Explicit CLI flag - use it
        EmbeddingBackend::from_str(backend_str).unwrap_or_else(|e| {
            eprintln!("Error: {}", e);
            eprintln!("Available backends: fastembed, ollama");
            std::process::exit(1);
        })
    } else if config.embedding.backend != "fastembed" {
        // Config file specifies a backend - use it
        EmbeddingBackend::from_str(&config.embedding.backend).unwrap_or_else(|e| {
            eprintln!("Error: {}", e);
            eprintln!("Available backends: fastembed, ollama");
            std::process::exit(1);
        })
    } else {
        // No explicit backend - auto-detect Ollama if available
        #[cfg(feature = "ollama")]
        {
            if is_ollama_available(&config.embedding.ollama.url) {
                eprintln!("📡 Auto-detected Ollama at {} - using Ollama backend", config.embedding.ollama.url);
                eprintln!("   (Use --backend fastembed to override)");
                EmbeddingBackend::Ollama
            } else {
                EmbeddingBackend::FastEmbed
            }
        }
        #[cfg(not(feature = "ollama"))]
        {
            EmbeddingBackend::FastEmbed
        }
    };

    // Create a mutable config clone for potential ollama model override
    let mut config = config;

    // Override ollama model if specified via CLI
    if let Some(ollama_model) = &cli.ollama_model {
        config.embedding.ollama.model = ollama_model.clone();
    }

    // Helper to get provider string from command or config
    let get_provider_str = |cli_provider: &Option<String>| -> String {
        cli_provider.clone().unwrap_or_else(|| config.embedding.provider.clone())
    };

    // Helper to get device_id from command or config
    let get_device_id = |cli_device: Option<i32>| -> i32 {
        cli_device.unwrap_or(config.embedding.device_id)
    };

    // Parse provider from CLI or config
    let (provider_type, device_id) = match &cli.command {
        Commands::Search { provider, device_id, .. } => {
            let provider_str = get_provider_str(provider);
            let dev_id = get_device_id(*device_id);
            let ptype = provider_str.parse::<ExecutionProviderType>().unwrap_or_else(|_| {
                eprintln!("Unknown execution provider: '{}'. Available providers:", provider_str);
                eprintln!("  cpu, auto, cuda, tensorrt, coreml, directml");
                std::process::exit(1);
            });
            (ptype, dev_id)
        }
        Commands::Index { provider, device_id, .. } => {
            let provider_str = get_provider_str(provider);
            let dev_id = get_device_id(*device_id);
            let ptype = provider_str.parse::<ExecutionProviderType>().unwrap_or_else(|_| {
                eprintln!("Unknown execution provider: '{}'. Available providers:", provider_str);
                eprintln!("  cpu, auto, cuda, tensorrt, coreml, directml");
                std::process::exit(1);
            });
            (ptype, dev_id)
        }
        Commands::Serve { provider, device_id, .. } => {
            let provider_str = get_provider_str(provider);
            let dev_id = get_device_id(*device_id);
            let ptype = provider_str.parse::<ExecutionProviderType>().unwrap_or_else(|_| {
                eprintln!("Unknown execution provider: '{}'. Available providers:", provider_str);
                eprintln!("  cpu, auto, cuda, tensorrt, coreml, directml");
                std::process::exit(1);
            });
            (ptype, dev_id)
        }
        Commands::Mcp { provider, device_id, .. } => {
            let provider_str = get_provider_str(provider);
            let dev_id = get_device_id(*device_id);
            let ptype = provider_str.parse::<ExecutionProviderType>().unwrap_or_else(|_| {
                eprintln!("Unknown execution provider: '{}'. Available providers:", provider_str);
                eprintln!("  cpu, auto, cuda, tensorrt, coreml, directml");
                std::process::exit(1);
            });
            (ptype, dev_id)
        }
        _ => {
            // For commands that don't use embedding (Doctor, Setup, etc.)
            let provider_str = config.embedding.provider.clone();
            let ptype = provider_str.parse::<ExecutionProviderType>().unwrap_or(ExecutionProviderType::Cpu);
            (ptype, config.embedding.device_id)
        }
    };

    // Set quiet mode if requested
    if cli.quiet {
        crate::output::set_quiet(true);
    }

    match cli.command {
        Commands::Search {
            query,
            max_results,
            per_file,
            content,
            scores,
            compact,
            sync,
            json,
            path,
            vector_only,
            rrf_k,
            rerank,
            rerank_top,
            filter_path,
            provider: _,
            device_id: _,
        } => {
            // Auto-enable quiet mode for JSON output
            if json {
                crate::output::set_quiet(true);
            }
            crate::search::search(
                &query,
                max_results,
                per_file,
                content,
                scores,
                compact,
                sync,
                json,
                path,
                filter_path,
                model_type,
                vector_only,
                rrf_k,
                rerank,
                rerank_top,
                provider_type,
                Some(device_id),
                batch_size,
                backend,
                &config,
            )
            .await
        }
        Commands::Index {
            path,
            dry_run,
            force,
            global,
            provider: _,
            device_id: _,
        } => crate::index::index(path, dry_run, force, global, model_type, provider_type, Some(device_id), batch_size, backend, &config).await,
        Commands::Serve { port, path, provider: _, device_id: _ } => {
            crate::server::serve(port, path, model_type, provider_type, Some(device_id), batch_size, backend, &config).await
        }
        Commands::List => crate::index::list().await,
        Commands::Stats { path } => crate::index::stats(path).await,
        Commands::Clear { path, yes, project } => crate::index::clear(path, yes, project).await,
        Commands::Doctor => crate::cli::doctor::run().await,
        Commands::Setup { model } => crate::cli::setup::run(model).await,
        Commands::Mcp { path, provider: _, device_id: _ } => {
            crate::mcp::run_mcp_server(path, provider_type, Some(device_id), batch_size, backend, &config).await
        }
    }
}

mod doctor;
mod setup;
