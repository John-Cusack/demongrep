# demongrep

**Fast, local semantic code search powered by Rust.**

Search your codebase using natural language queries like *"where do we handle authentication?"* — all running locally with no API calls.

## Features

- **Semantic Search** — Natural language queries that understand code meaning
- **Hybrid Search** — Combines vector similarity + BM25 full-text search with RRF fusion
- **Neural Reranking** — Optional second-pass reranking with Jina Reranker for higher accuracy
- **Smart Chunking** — Tree-sitter AST-aware chunking that preserves functions, classes, methods
- **Context Windows** — Shows surrounding code (3 lines before/after) for better understanding
- **Local & Private** — All processing happens locally using ONNX models, no data leaves your machine
- **Fast** — Sub-second search after initial model load, incremental indexing
- **GPU Acceleration** — Optional CUDA, TensorRT, DirectML, and **Ollama** (recommended) support for faster indexing
- **Multiple Interfaces** — CLI, HTTP server, and MCP server for Claude Code integration

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command Reference](#command-reference)
  - [search](#search)
  - [index](#index)
  - [serve](#serve)
  - [mcp](#mcp)
  - [stats](#stats)
  - [clear](#clear)
  - [list](#list)
  - [doctor](#doctor)
  - [setup](#setup)
- [Global Options](#global-options)
- [Search Modes](#search-modes)
- [MCP Server (Claude Code & OpenCode)](#mcp-server-claude-code-integration)
- [HTTP Server API](#http-server-api)
- [Database Management](#database-management)
- [Supported Languages](#supported-languages)
- [Embedding Models](#embedding-models)
- [GPU Acceleration](#gpu-acceleration)
- [Ollama Backend](#ollama-backend-recommended-for-gpu)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y build-essential protobuf-compiler libssl-dev pkg-config
```

#### Linux (Fedora/RHEL)
```bash
sudo dnf install -y gcc protobuf-compiler openssl-devel pkg-config
```

#### macOS
```bash
brew install protobuf openssl pkg-config
```

#### Windows
```powershell
# Using winget
winget install -e --id Google.Protobuf

# Or using chocolatey
choco install protoc
```

### Building from Source

```bash
# Clone the repository
git clone https://github.com/yxanul/demongrep.git
cd demongrep

# Build release binary
cargo build --release

# The binary is at target/release/demongrep
# Optionally, copy to your PATH:
sudo cp target/release/demongrep /usr/local/bin/
```

### Verify Installation

```bash
demongrep --version
demongrep doctor  # Check system health
```

---

## Quick Start

```bash
# 1. Navigate to your project
cd /path/to/your/project

# 2. Index the codebase (first time only, ~30-60s for medium projects)
demongrep index

# 3. Search with natural language
demongrep search "where do we handle authentication?"

# 4. Search with better accuracy (slower)
demongrep search "error handling" --rerank
```

---

## Command Reference

### search

Search the codebase using natural language queries.

```bash
demongrep search <QUERY> [OPTIONS]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `<QUERY>` | Natural language search query (e.g., "where do we handle authentication?") |

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--max-results` | `-m` | 25 | Maximum total results to return |
| `--per-file` | | 1 | Maximum matches to show per file |
| `--content` | `-c` | | Show full chunk content instead of snippets |
| `--scores` | | | Show relevance scores and timing information |
| `--compact` | | | Show file paths only (like `grep -l`) |
| `--sync` | `-s` | | Re-index changed files before searching |
| `--json` | | | Output results as JSON (for scripting/agents) |
| `--path` | | `.` | Path to search in |
| `--filter-path` | | | Only show results from files under this path (e.g., `src/`) |
| `--vector-only` | | | Disable hybrid search, use vector similarity only |
| `--rerank` | | | Enable neural reranking for better accuracy (~1.7s extra) |
| `--rerank-top` | | 50 | Number of candidates to rerank |
| `--rrf-k` | | 20 | RRF fusion parameter (higher = more weight to rank position) |

#### Examples

```bash
# Basic search
demongrep search "database connection pooling"

# Show full code content with context
demongrep search "error handling" --content

# Get JSON output for scripting
demongrep search "authentication" --json -m 10

# Search only in src/api directory
demongrep search "validation" --filter-path src/api

# High-accuracy search with reranking
demongrep search "complex algorithm" --rerank

# Quick search with scores
demongrep search "config loading" --scores

# Re-index changed files, then search
demongrep search "new feature" --sync

# File paths only
demongrep search "tests" --compact
```

---

### index

Index a codebase for semantic search.

```bash
demongrep index [PATH] [OPTIONS]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `[PATH]` | Path to index (defaults to current directory) |

#### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--dry-run` | | Preview what would be indexed without indexing |
| `--global` | `-g` | Index to global database in home directory |
| `--provider` | | Execution provider: `cpu`, `auto`, `cuda`, `tensorrt`, `coreml`, `directml` |
| `--device-id` | | GPU device ID for CUDA/TensorRT (default: 0) |
| `--batch-size` | | Override batch size for embedding |

#### Examples

```bash
# Index current directory
demongrep index

# Index a specific project
demongrep index /path/to/project

# Preview files to be indexed
demongrep index --dry-run

# Force complete re-index (delete and rebuild)
demongrep index --force

# Index with a specific model
demongrep index --model jina-code

# Index using GPU acceleration (CUDA)
demongrep index --provider cuda

# Index using specific GPU device
demongrep index --provider cuda --device-id 1
```

#### What Gets Indexed

- All text files respecting `.gitignore`
- Custom ignore patterns from `.demongrepignore` or `.osgrepignore`
- Skips binary files, `node_modules/`, `.git/`, etc.

#### Index Location

The index is stored in `.demongrep.db/` directory inside your project root.

---

### serve

Run an HTTP server with live file watching for continuous indexing.

```bash
demongrep serve [PATH] [OPTIONS]
```

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--port` | `-p` | 4444 | Port to listen on |

#### Examples

```bash
# Start server on default port (4444)
demongrep serve

# Start server on custom port
demongrep serve --port 3333

# Serve a specific project
demongrep serve /path/to/project --port 8080
```

The server automatically re-indexes files when they change (with 300ms debouncing).

---

### mcp

Start an MCP (Model Context Protocol) server for Claude Code integration.

```bash
demongrep mcp [PATH]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `[PATH]` | Path to project (defaults to current directory) |

See [MCP Server section](#mcp-server-claude-code-integration) for detailed setup.

---

### stats

Show statistics about the indexed database.

```bash
demongrep stats [PATH]
```

#### Output

```
📊 Database Statistics
============================================================
💾 Database: /path/to/project/.demongrep.db

Vector Store:
   Total chunks: 731
   Total files: 45
   Indexed: ✅ Yes
   Dimensions: 384

Storage:
   Database size: 12.34 MB
   Avg per chunk: 17.28 KB
```

---

### clear

Delete the index database.

```bash
demongrep clear [PATH] [OPTIONS]
```

#### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--yes` | `-y` | Skip confirmation prompt |

#### Examples

```bash
# Clear with confirmation
demongrep clear

# Clear without confirmation (for scripts)
demongrep clear -y

# Clear a specific project's index
demongrep clear /path/to/project -y
```

---

### list

List all indexed repositories (searches for `.demongrep.db` directories).

```bash
demongrep list
```

---

### doctor

Check installation health and system requirements.

```bash
demongrep doctor
```

---

### setup

Pre-download embedding models.

```bash
demongrep setup [OPTIONS]
```

#### Options

| Option | Description |
|--------|-------------|
| `--model` | Specific model to download (defaults to default model) |

---

## Global Options

These options work with all commands:

| Option | Short | Description |
|--------|-------|-------------|
| `--verbose` | `-v` | Enable verbose/debug output |
| `--quiet` | `-q` | Suppress informational output (only results/errors) |
| `--model` | | Override embedding model |
| `--store` | | Override store name |
| `--help` | `-h` | Show help |
| `--version` | `-V` | Show version |

---

## Search Modes

demongrep supports three search modes with different accuracy/speed tradeoffs:

### 1. Hybrid Search (Default)

Combines vector similarity with BM25 full-text search using Reciprocal Rank Fusion (RRF).

```bash
demongrep search "query"
```

- **Speed**: ~75ms
- **Best for**: Most queries, balances semantic understanding with keyword matching

### 2. Vector-Only Search

Pure semantic similarity search using embeddings.

```bash
demongrep search "query" --vector-only
```

- **Speed**: ~72ms
- **Best for**: Conceptual queries where exact keywords don't matter

### 3. Hybrid + Neural Reranking

Two-stage search: hybrid retrieval followed by cross-encoder reranking.

```bash
demongrep search "query" --rerank
```

- **Speed**: ~1.8s (adds ~1.7s for reranking)
- **Best for**: When accuracy matters more than speed

---

## MCP Server (Claude Code & OpenCode Integration)

demongrep can act as an MCP server, allowing AI coding assistants to search your codebase semantically.

### Claude Code Integration

#### Setup

1. **Build demongrep** and note the binary path:
   ```bash
   cargo build --release
   # Binary at: /path/to/demongrep/target/release/demongrep
   ```

2. **Index your project**:
   ```bash
   cd /path/to/your/project
   demongrep index
   ```

3. **Configure Claude Code**

   Edit `~/.config/claude-code/config.json` (Linux/Mac) or the appropriate config location:

   ```json
   {
     "mcpServers": {
       "demongrep": {
         "command": "/absolute/path/to/demongrep",
         "args": ["mcp", "/absolute/path/to/your/project"]
       }
     }
   }
   ```

4. **Restart Claude Code**

#### Available MCP Tools

| Tool | Parameters | Description |
|------|------------|-------------|
| `semantic_search` | `query`, `limit` | Search code semantically |
| `get_file_chunks` | `path` | Get all indexed chunks from a file |
| `index_status` | | Check if index exists and get stats |

#### Example Usage

Once configured, Claude Code can use commands like:
- *"Search for authentication handling"*
- *"Find all chunks in src/auth.rs"*
- *"Check if the index is ready"*

### OpenCode Integration

demongrep also works with [OpenCode](https://opencode.ai) as an MCP server.

1. **Index your project**:
   ```bash
   cd /path/to/your/project
   demongrep index
   ```

2. **Configure OpenCode**

   Edit `~/.config/opencode/opencode.json`:

   ```json
   {
     "$schema": "https://opencode.ai/config.json",
     "mcp": {
       "demongrep": {
         "type": "local",
         "command": ["/absolute/path/to/demongrep", "mcp"],
         "enabled": true,
         "timeout": 30000
       }
     }
   }
   ```

   Without a path argument, demongrep uses the current working directory, which is typically what you want for per-project search.

3. **Start OpenCode** in an indexed project directory

The following tools will be available:
- `demongrep/semantic_search` - Natural language code search
- `demongrep/get_file_chunks` - Get indexed chunks from a file
- `demongrep/index_status` - Check index statistics

#### Per-Project Configuration

For project-specific config, create `opencode.json` in your project root:

```json
{
  "mcp": {
    "demongrep": {
      "type": "local",
      "command": ["/absolute/path/to/demongrep", "mcp", "."],
      "enabled": true
    }
  }
}
```

---

## HTTP Server API

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check (returns `{"status": "ok"}`) |
| GET | `/status` | Index statistics |
| POST | `/search` | Search the codebase |

### Search API

**Request:**
```bash
curl -X POST http://localhost:4444/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "authentication",
    "limit": 10
  }'
```

**Response:**
```json
{
  "results": [
    {
      "path": "src/auth/handler.rs",
      "start_line": 45,
      "end_line": 67,
      "kind": "Function",
      "content": "pub fn authenticate(...) { ... }",
      "score": 0.89,
      "signature": "fn authenticate(credentials: &Credentials) -> Result<User>"
    }
  ],
  "query": "authentication",
  "total_results": 1
}
```

---

## Database Management

### Index Location

Each project has its own index at `<project_root>/.demongrep.db/`

### Re-indexing

```bash
# Incremental update (only changed files)
demongrep search "query" --sync

# Or explicitly re-index
demongrep index

# Full rebuild (delete and recreate)
demongrep index --force
```

### Delete Index

```bash
# Interactive
demongrep clear

# Non-interactive
demongrep clear -y

# Or manually
rm -rf .demongrep.db/
```

### Check Index Status

```bash
demongrep stats
```

---

## Supported Languages

### Full Semantic Chunking (Tree-sitter AST)

These languages have full AST-aware chunking that extracts functions, classes, methods, etc.:

| Language | Extensions |
|----------|------------|
| Rust | `.rs` |
| Python | `.py`, `.pyw`, `.pyi` |
| JavaScript | `.js`, `.mjs`, `.cjs` |
| TypeScript | `.ts`, `.mts`, `.cts`, `.tsx`, `.jsx` |

### Indexed (Line-based Chunking)

These languages are indexed with fallback line-based chunking:

| Language | Extensions |
|----------|------------|
| Go | `.go` |
| Java | `.java` |
| C | `.c`, `.h` |
| C++ | `.cpp`, `.cc`, `.cxx`, `.hpp` |
| C# | `.cs` |
| Ruby | `.rb`, `.rake` |
| PHP | `.php` |
| Swift | `.swift` |
| Kotlin | `.kt`, `.kts` |
| Shell | `.sh`, `.bash`, `.zsh` |
| Markdown | `.md`, `.markdown`, `.txt` |
| JSON | `.json` |
| YAML | `.yaml`, `.yml` |
| TOML | `.toml` |
| SQL | `.sql` |
| HTML | `.html`, `.htm` |
| CSS | `.css`, `.scss`, `.sass`, `.less` |

---

## Embedding Models

### Available Models

Models marked with ✅ in both columns can be used interchangeably between FastEmbed and Ollama backends.

| Name | FastEmbed ID | Ollama ID | Dims | Notes |
|------|--------------|-----------|------|-------|
| **All-MiniLM-L6** | `minilm-l6-q` | `all-minilm` | 384 | **Default** - Fast, works on both backends |
| MiniLM-L6 | `minilm-l6` | `all-minilm` | 384 | Non-quantized version |
| MiniLM-L12 | `minilm-l12-q` | — | 384 | Higher quality, FastEmbed only |
| BGE Small | `bge-small-q` | — | 384 | FastEmbed only |
| **Nomic Embed** | `nomic-v1.5` | `nomic-embed-text` | 768 | Works on both, good quality |
| BGE Base | `bge-base` | — | 768 | FastEmbed only |
| Jina Code | `jina-code` | — | 768 | Code-optimized, FastEmbed only |
| **MxBai Large** | `mxbai-large` | `mxbai-embed-large` | 1024 | Works on both, highest quality |
| **BGE Large** | `bge-large` | `bge-large` | 1024 | Works on both |
| BGE-M3 | — | `bge-m3` | 1024 | Ollama only, multilingual |
| E5 Multilingual | `e5-multilingual` | — | 384 | FastEmbed only, multilingual |

**Recommended models (work on both backends):**
- `all-minilm` / `minilm-l6-q` (384d) — **Default**, fastest, good for most codebases
- `nomic-embed-text` / `nomic-v1.5` (768d) — Better quality, long context
- `mxbai-embed-large` / `mxbai-large` (1024d) — Highest quality, slower

### Changing Models

```bash
# FastEmbed backend
demongrep index --model minilm-l6-q

# Ollama backend
demongrep index --backend ollama --ollama-model all-minilm

# Search must use same model/backend as index
demongrep search "query"
```

**Note:** The model used for indexing is saved in metadata. If you search with a different model, you may get poor results. Use `--force` to re-index with a new model.

---

## GPU Acceleration

demongrep supports GPU acceleration for faster embedding generation during indexing.

### Supported Providers

| Provider | Platform | Flag | Build Feature | Recommended |
|----------|----------|------|---------------|-------------|
| **Ollama** | All (with Ollama installed) | `--backend ollama` | `--features ollama` | ✅ **Best option** |
| CPU | All (Linux, macOS, Windows*) | `--provider cpu` | (default) | ✅ Good fallback |
| CUDA | Linux/Windows* + NVIDIA GPU | `--provider cuda` | `--features gpu-nvidia` | ⚠️ Modest gains |
| TensorRT | Linux/Windows* + NVIDIA GPU | `--provider tensorrt` | `--features tensorrt` | ⚠️ Complex setup |
| CoreML | macOS (Intel & Apple Silicon) | `--provider coreml` | `--features gpu-apple` | ❌ **Not recommended** |
| DirectML | Windows* (DirectX 12 GPU) | `--provider directml` | `--features gpu-windows` | ❓ Untested |

*Windows support has not been tested.

**⚠️ CoreML Warning:** Apple's CoreML is optimized for CNNs (image models), not transformer-based embedding models. Many transformer operations fall back to CPU, making CoreML often **slower than pure CPU**. Use Ollama instead on macOS - it uses Metal directly with optimized shaders and is ~10x faster.

### Building with GPU Support

#### Quick Reference

| Platform | Build Command | Notes |
|----------|---------------|-------|
| **All platforms (recommended)** | `cargo build --release --features ollama` | Use Ollama for GPU |
| macOS | `cargo build --release --features gpu-apple` | Includes Ollama (CoreML not recommended) |
| Linux (NVIDIA GPU) | `cargo build --release --features "gpu-nvidia,ollama"` | Ollama + CUDA fallback |
| CPU only (all platforms) | `cargo build --release` | Basic, no GPU |
| Windows (DirectX 12)* | `cargo build --release --features gpu-windows` | Untested |

*Windows support has not been tested.

**Recommendation:** Build with `--features ollama` and install [Ollama](https://ollama.ai) for the best GPU acceleration on all platforms.

#### Detailed Build Instructions

**Basic CPU build (all platforms):**
```bash
cargo build --release
```
This works on Linux, macOS (Intel & Apple Silicon), and Windows. No GPU features needed.

**macOS (Intel & Apple Silicon):**
```bash
# Recommended: Ollama support (uses Metal GPU via Ollama)
cargo build --release --features ollama

# Alternative: includes Ollama + CoreML (CoreML not recommended)
cargo build --release --features gpu-apple
```
**Note:** On macOS, always use Ollama for GPU acceleration. CoreML is not optimized for transformers and is often slower than CPU. The `gpu-apple` feature includes CoreML as a fallback, but Ollama with Metal is ~10x faster.

**Linux/Windows with NVIDIA GPU:**
```bash
# CUDA support (requires NVIDIA drivers)
cargo build --release --features gpu-nvidia

# TensorRT support (requires TensorRT SDK installed)
cargo build --release --features tensorrt
```

**Windows with DirectX 12 GPU (untested):**
```bash
cargo build --release --features gpu-windows
```

### Usage

```bash
# Auto-detect best available provider
demongrep index --provider auto

# Use CUDA explicitly
demongrep index --provider cuda

# Use specific GPU device
demongrep index --provider cuda --device-id 1

# Check available providers
demongrep doctor
```

### Adaptive Batching

demongrep uses intelligent adaptive batching to maximize throughput:

- **Chunks sorted by length** — Similar-sized chunks are batched together to minimize padding waste
- **Dynamic batch sizes** — Short chunks get large batches, long chunks get small batches
- **Token budget** — Batches are sized to stay within optimal memory limits (10K tokens default)

This results in ~30% faster indexing compared to fixed batch sizes.

### Performance Tips

1. **All platforms**: Install [Ollama](https://ollama.ai) and build with `--features ollama` — **~10x faster** than other options
2. **macOS**: **Use Ollama, NOT CoreML** — CoreML is optimized for CNNs, not transformers, and is often slower than CPU
3. **NVIDIA GPUs**: Ollama is still faster than CUDA/TensorRT for embeddings; use `--provider cuda` only as fallback
4. **Check GPU memory**: Use `nvidia-smi` (NVIDIA) or Activity Monitor (macOS) — model uses ~500MB
5. **Tune token budget**: Set `DEMONGREP_TOKEN_BUDGET` environment variable if needed

---

## Ollama Backend (Recommended for GPU)

demongrep supports [Ollama](https://ollama.ai) as an alternative embedding backend. **This is the recommended approach for GPU acceleration** as it provides significantly better performance than ONNX Runtime.

### Why Ollama?

| Backend | nomic-embed-text (768d) | Notes |
|---------|-------------------------|-------|
| FastEmbed CPU | 5.6 chunks/sec | Slow for large models |
| FastEmbed CUDA | 6.8 chunks/sec | Minimal GPU benefit |
| Ollama CPU | 2.4 chunks/sec | Model stays warm |
| **Ollama GPU** | **52.2 chunks/sec** | **~10x faster than FastEmbed** |

With the same model, Ollama with GPU is **~10x faster for indexing** and **~3x faster for search**.

**Why not CoreML?** Apple's CoreML is optimized for CNNs (image models), not transformer-based embedding models. Many transformer operations fall back to CPU, and data transfer overhead can make it slower than pure CPU. Ollama uses Metal directly with hand-optimized shaders for transformers, which is why it's much faster on Mac.

### Setup

#### 1. Install Ollama

**macOS/Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Or with Docker (recommended for servers):**
```bash
# With GPU support (NVIDIA)
docker run -d --gpus all --name ollama -p 11434:11434 ollama/ollama

# CPU only
docker run -d --name ollama -p 11434:11434 ollama/ollama
```

#### 2. Pull an embedding model

```bash
ollama pull all-minilm          # 384 dims, default, fastest
# or
ollama pull nomic-embed-text    # 768 dims, better quality
# or
ollama pull mxbai-embed-large   # 1024 dims, highest quality
```

#### 3. Build demongrep with Ollama support

```bash
# macOS (gpu-apple already includes ollama)
cargo build --release --features gpu-apple

# Linux/Windows
cargo build --release --features ollama
```

#### 4. Index and search

**Ollama is auto-detected!** When built with `--features ollama` and Ollama is running, demongrep automatically uses it:

```bash
# Auto-detects Ollama if running
demongrep index

# Output:
# 📡 Auto-detected Ollama at http://localhost:11434 - using Ollama backend
#    (Use --backend fastembed to override)
```

You can also be explicit:

```bash
# Explicitly use Ollama backend
demongrep index --backend ollama

# Force FastEmbed even if Ollama is running
demongrep index --backend fastembed

# Use a different Ollama model
demongrep index --ollama-model mxbai-embed-large
```

### Available Ollama Embedding Models

**Models are downloaded automatically!** When you specify a model, demongrep will pull it from Ollama if not already installed.

| Model | Dimensions | FastEmbed Equivalent | Notes |
|-------|------------|---------------------|-------|
| `all-minilm` | 384 | `minilm-l6-q` | **Default** - fast, cross-compatible |
| `nomic-embed-text` | 768 | `nomic-v1.5` | Better quality, cross-compatible |
| `mxbai-embed-large` | 1024 | `mxbai-large` | Highest quality, cross-compatible |
| `bge-large` | 1024 | `bge-large` | High quality, cross-compatible |
| `bge-m3` | 1024 | — | Ollama only, multilingual |
| `snowflake-arctic-embed` | 1024 | — | Ollama only |

**Recommended:** Use `all-minilm` (default) for the best balance of speed and cross-backend compatibility.

**Using a model:**

```bash
# Just specify it - demongrep will auto-pull if needed
demongrep index --ollama-model all-minilm

# Or manually pull first
ollama pull all-minilm

# Set default in config (~/.demongrep/config.toml)
[embedding.ollama]
model = "all-minilm"
```

**Check available models:**

```bash
# List models installed in Ollama
ollama list

# Check demongrep configuration
demongrep doctor
```

### Configuration

Add to `~/.demongrep/config.toml`:

```toml
[embedding]
backend = "ollama"  # Use Ollama by default

[embedding.ollama]
url = "http://localhost:11434"  # Ollama server URL
model = "all-minilm"             # Embedding model (384d, cross-compatible)
timeout = 30                     # Request timeout (seconds)
parallelism = 8                  # Parallel HTTP requests
```

### Performance Comparison

Benchmarked on RTX A2000 8GB with 1394 code chunks:

| Metric | FastEmbed (CPU) | Ollama (GPU) | Speedup |
|--------|-----------------|--------------|---------|
| Indexing | 5.6 c/s | 52.2 c/s | **9.3x** |
| Search | 1.27s | 0.43s | **3.0x** |
| Index 1394 chunks | 264s | 27s | **9.8x** |

### Troubleshooting

**"Cannot connect to Ollama"**
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve
# or
docker start ollama
```

**"Model not found"**
```bash
# Pull the model first
ollama pull nomic-embed-text
```

**Dimension mismatch error**
```bash
# Re-index with --force when switching backends/models
demongrep index --backend ollama --force
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEMONGREP_TOKEN_BUDGET` | Token budget for adaptive batching | 10000 |
| `DEMONGREP_BATCH_SIZE` | Override batch size (legacy) | Auto |
| `RUST_LOG` | Logging level | `demongrep=info` |

### Ignore Files

Create `.demongrepignore` in your project root:

```gitignore
# Ignore test fixtures
**/fixtures/**
**/testdata/**

# Ignore generated code
**/generated/**
*.generated.ts

# Ignore large files
*.min.js
*.bundle.js
```

demongrep also respects `.gitignore` and `.osgrepignore` files.

---

## How It Works

### 1. File Discovery
- Walks directory respecting `.gitignore` and custom ignore files
- Detects language from file extensions
- Skips binary files automatically

### 2. Semantic Chunking
- Parses code with tree-sitter (native Rust implementation)
- Extracts semantic units: functions, classes, methods, structs, traits, impls
- Preserves metadata: signatures, docstrings, context breadcrumbs
- Falls back to line-based chunking for unsupported languages

### 3. Embedding Generation
- Uses fastembed with ONNX Runtime
- GPU acceleration via CUDA, TensorRT, CoreML, or DirectML
- Adaptive token-budget batching: sorts chunks by length, dynamically sizes batches
- SHA-256 content hashing for incremental change detection

### 4. Vector Storage
- arroy for approximate nearest neighbor search
- LMDB for ACID transactions and persistence
- Single `.demongrep.db/` directory per project

### 5. Search
- Query embedding → Vector search → BM25 search → RRF fusion → (Optional) Reranking

---

## Troubleshooting

### "No database found"

```bash
# Index the project first
demongrep index
```

### Search returns poor results

1. **Check if index is stale:**
   ```bash
   demongrep search "query" --sync
   ```

2. **Try different search mode:**
   ```bash
   demongrep search "query" --rerank
   ```

3. **Rebuild index:**
   ```bash
   demongrep index --force
   ```

### Model mismatch warning

If you indexed with one model and search with another:
```bash
# Re-index with the model you want to use
demongrep index --force --model minilm-l6-q
```

### Out of memory during indexing

```bash
# Reduce token budget
DEMONGREP_TOKEN_BUDGET=5000 demongrep index

# Or use legacy batch size override
DEMONGREP_BATCH_SIZE=32 demongrep index
```

### GPU not detected

```bash
# Check GPU status
demongrep doctor

# Verify CUDA is available
nvidia-smi

# Fall back to CPU
demongrep index --provider cpu
```

### Server won't start (port in use)

```bash
# Use a different port
demongrep serve --port 5555
```

---

## Development

```bash
# Debug build
cargo build

# Release build
cargo build --release

# Run tests
cargo test

# Format code
cargo fmt

# Lint
cargo clippy
```

### Debug Logging

```bash
RUST_LOG=demongrep=debug demongrep search "query"
RUST_LOG=demongrep::embed=trace demongrep index
```

---

## License

[Apache-2.0](LICENSE) - See [NOTICE](NOTICE) for attribution.

---

## Contributing

Contributions welcome! See [TODO.md](TODO.md) for planned features.
