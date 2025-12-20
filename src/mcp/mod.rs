//! MCP (Model Context Protocol) server for Claude Code integration
//!
//! Exposes demongrep's semantic search capabilities via the MCP protocol,
//! allowing AI assistants like Claude to search codebases during conversations.
//!
//! **Now supports dual-database search**: Searches both local and global databases automatically.

use anyhow::Result;
use rmcp::{
    handler::server::router::tool::ToolRouter,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content, ServerCapabilities, ServerInfo},
    schemars::JsonSchema,
    tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Mutex;

use crate::embed::{EmbeddingService, ModelType};
use crate::index::get_search_db_paths;
use crate::vectordb::VectorStore;

/// Demongrep MCP service with dual-database support
pub struct DemongrepService {
    tool_router: ToolRouter<DemongrepService>,
    db_paths: Vec<PathBuf>,  // Changed: now supports multiple databases
    model_type: ModelType,
    dimensions: usize,
    // Lazily initialized on first search
    embedding_service: Mutex<Option<EmbeddingService>>,
}

impl std::fmt::Debug for DemongrepService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DemongrepService")
            .field("db_paths", &self.db_paths)
            .field("model_type", &self.model_type)
            .field("dimensions", &self.dimensions)
            .finish()
    }
}

// === Tool Request/Response Types ===

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SemanticSearchRequest {
    /// The search query (natural language or code snippet)
    pub query: String,

    /// Maximum number of results to return (default: 10)
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetFileChunksRequest {
    /// Path to the file (relative to project root)
    pub path: String,
}

#[derive(Debug, Serialize)]
pub struct SearchResultItem {
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub kind: String,
    pub content: String,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_prev: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_next: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub database: Option<String>,  // NEW: which database the result came from
}

#[derive(Debug, Serialize)]
pub struct IndexStatusResponse {
    pub indexed: bool,
    pub total_chunks: usize,
    pub total_files: usize,
    pub local_chunks: usize,  // NEW
    pub local_files: usize,   // NEW
    pub global_chunks: usize, // NEW
    pub global_files: usize,  // NEW
    pub model: String,
    pub dimensions: usize,
    pub databases: Vec<String>,  // NEW: list of database paths
    pub databases_available: usize,  // NEW
}

// === Tool Router Implementation ===

#[tool_router]
impl DemongrepService {
    /// Create a new DemongrepService with dual-database support
    pub fn new(db_paths: Vec<PathBuf>) -> Result<Self> {
        if db_paths.is_empty() {
            return Err(anyhow::anyhow!("No databases available"));
        }

        // Read model metadata from first available database
        let (model_type, dimensions) = Self::read_metadata(&db_paths[0])
            .unwrap_or_else(|| (ModelType::default(), 384));

        Ok(Self {
            tool_router: Self::tool_router(),
            db_paths,
            model_type,
            dimensions,
            embedding_service: Mutex::new(None),
        })
    }

    /// Read metadata from a database
    fn read_metadata(db_path: &PathBuf) -> Option<(ModelType, usize)> {
        let metadata_path = db_path.join("metadata.json");
        if metadata_path.exists() {
            let content = std::fs::read_to_string(&metadata_path).ok()?;
            let json: serde_json::Value = serde_json::from_str(&content).ok()?;
            let model_name = json
                .get("model_short_name")
                .and_then(|v| v.as_str())
                .unwrap_or("minilm-l6");
            let dims = json
                .get("dimensions")
                .and_then(|v| v.as_u64())
                .unwrap_or(384) as usize;
            let mt = ModelType::from_str(model_name).unwrap_or_default();
            Some((mt, dims))
        } else {
            None
        }
    }

    /// Get or initialize the embedding service
    fn get_embedding_service(&self) -> Result<std::sync::MutexGuard<Option<EmbeddingService>>> {
        let mut guard = self.embedding_service.lock().unwrap();
        if guard.is_none() {
            *guard = Some(EmbeddingService::with_model(self.model_type)?);
        }
        Ok(guard)
    }

    /// Search across all available databases
    fn search_all_databases(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<crate::vectordb::SearchResult>> {
        let mut all_results = Vec::new();

        for db_path in &self.db_paths {
            if !db_path.exists() {
                continue;
            }

            match VectorStore::new(db_path, self.dimensions) {
                Ok(store) => {
                    if let Ok(mut results) = store.search(query_embedding, limit) {
                        all_results.append(&mut results);
                    }
                }
                Err(e) => {
                    eprintln!("Warning: Failed to open database {}: {}", db_path.display(), e);
                }
            }
        }

        // Sort by score and limit
        all_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(limit);

        Ok(all_results)
    }

    #[tool(description = "Search the codebase using semantic similarity. Searches both local and global databases. Returns code chunks that are semantically similar to the query.")]
    async fn semantic_search(
        &self,
        Parameters(request): Parameters<SemanticSearchRequest>,
    ) -> Result<CallToolResult, McpError> {
        let limit = request.limit.unwrap_or(10);

        // Check if any database exists
        let available_dbs: Vec<&PathBuf> = self.db_paths.iter().filter(|p| p.exists()).collect();
        
        if available_dbs.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "Error: No index found. Run 'demongrep index' or 'demongrep index --global' first to index the codebase.",
            )]));
        }

        // Get embedding service and embed query
        let mut service_guard = match self.get_embedding_service() {
            Ok(g) => g,
            Err(e) => {
                return Ok(CallToolResult::success(vec![Content::text(format!(
                    "Error initializing embedding service: {}",
                    e
                ))]));
            }
        };

        let service = service_guard.as_mut().unwrap();
        let query_embedding = match service.embed_query(&request.query) {
            Ok(e) => e,
            Err(e) => {
                return Ok(CallToolResult::success(vec![Content::text(format!(
                    "Error embedding query: {}",
                    e
                ))]));
            }
        };

        // Search across all databases
        let results = match self.search_all_databases(&query_embedding, limit) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::success(vec![Content::text(format!(
                    "Error searching: {}",
                    e
                ))]));
            }
        };

        if results.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No results found for the query.",
            )]));
        }

        // Convert to response format
        let items: Vec<SearchResultItem> = results
            .into_iter()
            .map(|r| {
                // Determine which database this came from
                let database = if r.path.contains(".demongrep.db") {
                    Some("local".to_string())
                } else {
                    Some("global".to_string())
                };

                SearchResultItem {
                    path: r.path,
                    start_line: r.start_line,
                    end_line: r.end_line,
                    kind: r.kind,
                    content: r.content,
                    score: r.score,
                    signature: r.signature,
                    context_prev: r.context_prev,
                    context_next: r.context_next,
                    database,
                }
            })
            .collect();

        let json = serde_json::to_string_pretty(&items).unwrap_or_else(|_| "[]".to_string());
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(description = "Get all indexed chunks from a specific file. Searches across all databases. Useful for understanding the structure of a file.")]
    async fn get_file_chunks(
        &self,
        Parameters(request): Parameters<GetFileChunksRequest>,
    ) -> Result<CallToolResult, McpError> {
        let mut all_file_chunks: Vec<SearchResultItem> = Vec::new();

        // Search across all databases
        for db_path in &self.db_paths {
            if !db_path.exists() {
                continue;
            }

            let store = match VectorStore::new(db_path, self.dimensions) {
                Ok(s) => s,
                Err(_) => continue,
            };

            let stats = match store.stats() {
                Ok(s) => s,
                Err(_) => continue,
            };

            // Collect chunks for the requested file
            for id in 0..stats.total_chunks as u32 {
                if let Ok(Some(chunk)) = store.get_chunk(id) {
                    // Normalize paths for comparison
                    let chunk_path = chunk.path.trim_start_matches("./");
                    let req_path = request.path.trim_start_matches("./");

                    if chunk_path == req_path || chunk.path == request.path {
                        let database = if db_path.ends_with(".demongrep.db") {
                            Some("local".to_string())
                        } else {
                            Some("global".to_string())
                        };

                        all_file_chunks.push(SearchResultItem {
                            path: chunk.path,
                            start_line: chunk.start_line,
                            end_line: chunk.end_line,
                            kind: chunk.kind,
                            content: chunk.content,
                            score: 1.0,
                            signature: chunk.signature,
                            context_prev: chunk.context_prev,
                            context_next: chunk.context_next,
                            database,
                        });
                    }
                }
            }
        }

        // Sort by start line
        all_file_chunks.sort_by_key(|c| c.start_line);

        if all_file_chunks.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(format!(
                "No chunks found for file: {}",
                request.path
            ))]));
        }

        let json =
            serde_json::to_string_pretty(&all_file_chunks).unwrap_or_else(|_| "[]".to_string());
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(description = "Get the status of the semantic search index including model info and statistics from all databases.")]
    async fn index_status(&self) -> Result<CallToolResult, McpError> {
        let mut total_chunks = 0;
        let mut total_files = 0;
        let mut local_chunks = 0;
        let mut local_files = 0;
        let mut global_chunks = 0;
        let mut global_files = 0;
        let mut indexed = false;
        let mut db_paths_str = Vec::new();

        // Collect stats from all databases
        for db_path in &self.db_paths {
            if !db_path.exists() {
                continue;
            }

            db_paths_str.push(db_path.display().to_string());

            if let Ok(store) = VectorStore::new(db_path, self.dimensions) {
                if let Ok(stats) = store.stats() {
                    total_chunks += stats.total_chunks;
                    total_files += stats.total_files;
                    indexed = indexed || stats.indexed;

                    // Categorize by database type
                    if db_path.ends_with(".demongrep.db") {
                        local_chunks += stats.total_chunks;
                        local_files += stats.total_files;
                    } else {
                        global_chunks += stats.total_chunks;
                        global_files += stats.total_files;
                    }
                }
            }
        }

        let response = IndexStatusResponse {
            indexed,
            total_chunks,
            total_files,
            local_chunks,
            local_files,
            global_chunks,
            global_files,
            model: self.model_type.short_name().to_string(),
            dimensions: self.dimensions,
            databases: db_paths_str.clone(),
            databases_available: db_paths_str.len(),
        };

        let json = serde_json::to_string_pretty(&response).unwrap_or_else(|_| "{}".to_string());
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }
}

// === Server Handler Implementation ===

#[tool_handler]
impl ServerHandler for DemongrepService {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: rmcp::model::Implementation {
                name: "demongrep".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                title: None,
                icons: None,
                website_url: None,
            },
            instructions: Some(
                "Demongrep is a semantic code search tool with dual-database support. \
                 Use semantic_search to find code by meaning (searches both local and global databases), \
                 get_file_chunks to see all chunks in a file, and index_status \
                 to check if the index is ready and see stats from all databases."
                    .to_string(),
            ),
            ..Default::default()
        }
    }
}

/// Run the MCP server using stdio transport with dual-database support
pub async fn run_mcp_server(path: Option<PathBuf>) -> Result<()> {
    use rmcp::{transport::stdio, ServiceExt};

    // Use get_search_db_paths to find all available databases
    let db_paths = get_search_db_paths(path)?;

    if db_paths.is_empty() {
        eprintln!("Error: No databases found!");
        eprintln!("Run 'demongrep index' or 'demongrep index --global' first.");
        return Err(anyhow::anyhow!("No databases found"));
    }

    eprintln!("Starting demongrep MCP server...");
    eprintln!("Databases found:");
    for db_path in &db_paths {
        let db_type = if db_path.ends_with(".demongrep.db") {
            "Local"
        } else {
            "Global"
        };
        eprintln!("  {} {}", db_type, db_path.display());
    }

    let service = DemongrepService::new(db_paths)?;

    // Serve using stdio transport
    let server = service.serve(stdio()).await?;

    eprintln!("MCP server ready. Waiting for requests...");

    // Wait for shutdown
    server.waiting().await?;

    Ok(())
}
