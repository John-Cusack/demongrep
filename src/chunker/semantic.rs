use super::{Chunk, ChunkKind, Chunker, DEFAULT_CONTEXT_LINES};
use crate::chunker::extractor::{get_extractor, LanguageExtractor};
use crate::chunker::parser::CodeParser;
use crate::file::Language;
use anyhow::Result;
use std::path::Path;
use tree_sitter::Node;

/// Smart semantic chunker using tree-sitter and language-specific extractors
pub struct SemanticChunker {
    parser: CodeParser,
    max_chunk_lines: usize,
    max_chunk_chars: usize,
    overlap_lines: usize,
    context_lines: usize,
}

impl SemanticChunker {
    pub fn new(max_chunk_lines: usize, max_chunk_chars: usize, overlap_lines: usize) -> Self {
        Self {
            parser: CodeParser::new(),
            max_chunk_lines,
            max_chunk_chars,
            overlap_lines,
            context_lines: DEFAULT_CONTEXT_LINES,
        }
    }

    /// Set the number of context lines to extract before/after each chunk
    pub fn with_context_lines(mut self, lines: usize) -> Self {
        self.context_lines = lines;
        self
    }

    /// Chunk a file using semantic analysis
    pub fn chunk_semantic(
        &mut self,
        language: Language,
        path: &Path,
        content: &str,
    ) -> Result<Vec<Chunk>> {
        // 1. Detect if SQL file is actually dbt (contains Jinja templating)
        let effective_language = if language == Language::Sql && Language::detect_dbt_from_content(content) {
            Language::Dbt
        } else {
            language
        };

        // 2. Check if we have an extractor for this language
        let extractor = match get_extractor(effective_language) {
            Some(ext) => ext,
            None => {
                // Fall back to simple chunking for unsupported languages
                return Ok(self.fallback_chunk(path, content));
            }
        };

        // 3. Parse the code (use effective_language for proper grammar)
        let parsed = self.parser.parse(effective_language, content)?;

        // 4. Visit AST and extract chunks
        let mut definition_chunks = Vec::new();
        let mut gap_tracker = GapTracker::new(content);

        let file_context = format!("File: {}", path.display());
        self.visit_node(
            parsed.root_node(),
            parsed.source().as_bytes(),
            &*extractor,
            &[file_context],
            &mut definition_chunks,
            &mut gap_tracker,
        );

        // 5. Extract gap chunks (code between definitions)
        let gap_chunks = gap_tracker.extract_gaps(path);

        // 6. Combine and sort all chunks by position
        let mut all_chunks = definition_chunks;
        all_chunks.extend(gap_chunks);
        all_chunks.sort_by_key(|c| c.start_line);

        // 7. Populate context windows (lines before/after each chunk)
        let source_lines: Vec<&str> = content.lines().collect();
        self.populate_context_windows(&mut all_chunks, &source_lines);

        // 8. Split oversized chunks
        let final_chunks = all_chunks
            .into_iter()
            .flat_map(|c| self.split_if_needed(c))
            .collect();

        Ok(final_chunks)
    }

    /// Populate context_prev and context_next for each chunk
    fn populate_context_windows(&self, chunks: &mut [Chunk], source_lines: &[&str]) {
        let total_lines = source_lines.len();

        for chunk in chunks.iter_mut() {
            // Extract context_prev (N lines before start_line)
            if chunk.start_line > 0 && self.context_lines > 0 {
                let prev_start = chunk.start_line.saturating_sub(self.context_lines);
                let prev_end = chunk.start_line;
                if prev_start < prev_end && prev_end <= total_lines {
                    let prev_lines = &source_lines[prev_start..prev_end];
                    let prev_content = prev_lines.join("\n");
                    if !prev_content.trim().is_empty() {
                        chunk.context_prev = Some(prev_content);
                    }
                }
            }

            // Extract context_next (N lines after end_line)
            if chunk.end_line < total_lines && self.context_lines > 0 {
                let next_start = chunk.end_line;
                let next_end = (chunk.end_line + self.context_lines).min(total_lines);
                if next_start < next_end {
                    let next_lines = &source_lines[next_start..next_end];
                    let next_content = next_lines.join("\n");
                    if !next_content.trim().is_empty() {
                        chunk.context_next = Some(next_content);
                    }
                }
            }
        }
    }

    /// Recursively visit AST nodes and extract chunks
    fn visit_node(
        &self,
        node: Node,
        source: &[u8],
        extractor: &dyn LanguageExtractor,
        context_stack: &[String],
        chunks: &mut Vec<Chunk>,
        gap_tracker: &mut GapTracker,
    ) {
        // Check if this node is a definition
        let is_definition = extractor.definition_types().contains(&node.kind());

        if is_definition {
            // Mark this range as covered (not a gap)
            gap_tracker.mark_covered(
                node.start_position().row,
                node.end_position().row,
            );

            // Extract metadata using the language extractor
            let kind = extractor.classify(node);
            let name = extractor.extract_name(node, source);
            let signature = extractor.extract_signature(node, source);
            let docstring = extractor.extract_docstring(node, source);

            // Build label for context breadcrumb
            let label = extractor.build_label(node, source)
                .or_else(|| name.as_ref().map(|n| format!("{:?}: {}", kind, n)))
                .unwrap_or_else(|| format!("{:?}", kind));

            // Build new context stack
            let mut new_context = context_stack.to_vec();
            new_context.push(label);

            // Extract content (without docstring if we have it separate)
            let content = match node.utf8_text(source) {
                Ok(text) => text.to_string(),
                Err(_) => return, // Skip if we can't extract text
            };

            // Create chunk
            let path_str = context_stack.first()
                .map(|s| s.strip_prefix("File: ").unwrap_or(s))
                .unwrap_or("")
                .to_string();

            let mut chunk = Chunk::new(
                content.clone(),
                node.start_position().row,
                node.end_position().row + 1, // tree-sitter uses 0-based, we use line count
                kind,
                path_str,
            );
            chunk.context = new_context.clone();
            chunk.signature = signature;
            chunk.docstring = docstring;
            chunk.string_literals = Chunk::extract_string_literals(&content);

            chunks.push(chunk);

            // Visit children with updated context
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                self.visit_node(child, source, extractor, &new_context, chunks, gap_tracker);
            }
        } else {
            // Not a definition, just visit children with same context
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                self.visit_node(child, source, extractor, context_stack, chunks, gap_tracker);
            }
        }
    }

    /// Fallback chunking for unsupported languages
    fn fallback_chunk(&self, path: &Path, content: &str) -> Vec<Chunk> {
        let lines: Vec<&str> = content.lines().collect();
        let mut chunks = Vec::new();
        let stride = (self.max_chunk_lines - self.overlap_lines).max(1);

        let path_str = path.to_string_lossy().to_string();
        let context = vec![format!("File: {}", path_str)];

        let mut i = 0;
        while i < lines.len() {
            let end = (i + self.max_chunk_lines).min(lines.len());
            let chunk_lines = &lines[i..end];

            if !chunk_lines.is_empty() {
                let content = chunk_lines.join("\n");
                let mut chunk = Chunk::new(content.clone(), i, end, ChunkKind::Block, path_str.clone());
                chunk.context = context.clone();
                chunk.string_literals = Chunk::extract_string_literals(&content);
                chunks.push(chunk);
            }

            i += stride;
        }

        chunks
    }

    /// Split a chunk if it exceeds size limits
    fn split_if_needed(&self, chunk: Chunk) -> Vec<Chunk> {
        let line_count = chunk.line_count();
        let char_count = chunk.size_bytes();

        // Check if splitting is needed
        if line_count <= self.max_chunk_lines && char_count <= self.max_chunk_chars {
            return vec![chunk];
        }

        // Need to split
        let lines: Vec<&str> = chunk.content.lines().collect();
        let mut split_chunks = Vec::new();
        let stride = (self.max_chunk_lines - self.overlap_lines).max(1);

        let mut i = 0;
        let mut split_index = 0;

        while i < lines.len() {
            let end = (i + self.max_chunk_lines).min(lines.len());
            let chunk_lines = &lines[i..end];

            if !chunk_lines.is_empty() {
                let content = chunk_lines.join("\n");
                let mut split_chunk = Chunk::new(
                    content,
                    chunk.start_line + i,
                    chunk.start_line + end,
                    chunk.kind,
                    chunk.path.clone(),
                );

                // Preserve metadata
                split_chunk.context = chunk.context.clone();
                split_chunk.signature = chunk.signature.clone();
                split_chunk.docstring = if split_index == 0 {
                    chunk.docstring.clone() // Only first chunk gets docstring
                } else {
                    None
                };
                split_chunk.is_complete = false;
                split_chunk.split_index = Some(split_index);

                split_chunks.push(split_chunk);
                split_index += 1;
            }

            i += stride;
        }

        // Add header to split chunks to indicate they're partial
        let total_parts = split_chunks.len();
        for chunk in &mut split_chunks {
            if let Some(idx) = chunk.split_index {
                let header = format!(
                    "// [Part {}/{}] {}\n",
                    idx + 1,
                    total_parts,
                    chunk.signature.as_ref().unwrap_or(&"(continued)".to_string())
                );
                chunk.content = header + &chunk.content;
            }
        }

        split_chunks
    }
}

impl Chunker for SemanticChunker {
    fn chunk_file(&self, path: &Path, content: &str) -> Result<Vec<Chunk>> {
        // Detect language from path
        let language = Language::from_path(path);

        // Can't use &mut self in trait method, so we need a workaround
        // Create a temporary parser for this call
        let mut temp_chunker = SemanticChunker::new(
            self.max_chunk_lines,
            self.max_chunk_chars,
            self.overlap_lines,
        );

        temp_chunker.chunk_semantic(language, path, content)
    }
}

/// Helper to track gaps (code between definitions)
struct GapTracker<'a> {
    content: &'a str,
    lines: Vec<&'a str>,
    covered: Vec<bool>, // covered[i] = true if line i is part of a definition
}

impl<'a> GapTracker<'a> {
    fn new(content: &'a str) -> Self {
        let lines: Vec<&str> = content.lines().collect();
        let covered = vec![false; lines.len()];

        Self {
            content,
            lines,
            covered,
        }
    }

    /// Mark a range of lines as covered by a definition
    fn mark_covered(&mut self, start_line: usize, end_line: usize) {
        for i in start_line..=end_line.min(self.covered.len().saturating_sub(1)) {
            if i < self.covered.len() {
                self.covered[i] = true;
            }
        }
    }

    /// Extract gap chunks (uncovered regions)
    fn extract_gaps(&self, path: &Path) -> Vec<Chunk> {
        let mut gaps = Vec::new();
        let path_str = path.to_string_lossy().to_string();
        let context = vec![format!("File: {}", path_str)];

        let mut gap_start: Option<usize> = None;

        for (i, &is_covered) in self.covered.iter().enumerate() {
            if !is_covered {
                // Start or continue a gap
                if gap_start.is_none() {
                    gap_start = Some(i);
                }
            } else {
                // End of gap
                if let Some(start) = gap_start {
                    // Extract gap content
                    let gap_lines = &self.lines[start..i];
                    let gap_content = gap_lines.join("\n");

                    // Only create chunk if gap is not empty/whitespace
                    if !gap_content.trim().is_empty() {
                        let kind = Self::classify_gap(&gap_content);
                        let mut chunk = Chunk::new(
                            gap_content.clone(),
                            start,
                            i,
                            kind,
                            path_str.clone(),
                        );
                        chunk.context = context.clone();
                        chunk.string_literals = Chunk::extract_string_literals(&gap_content);
                        gaps.push(chunk);
                    }

                    gap_start = None;
                }
            }
        }

        // Handle final gap (if file ends with gap)
        if let Some(start) = gap_start {
            let gap_lines = &self.lines[start..];
            let gap_content = gap_lines.join("\n");

            if !gap_content.trim().is_empty() {
                let kind = Self::classify_gap(&gap_content);
                let mut chunk = Chunk::new(
                    gap_content.clone(),
                    start,
                    self.lines.len(),
                    kind,
                    path_str.clone(),
                );
                chunk.context = context.clone();
                chunk.string_literals = Chunk::extract_string_literals(&gap_content);
                gaps.push(chunk);
            }
        }

        gaps
    }

    /// Classify what kind of gap this is
    fn classify_gap(content: &str) -> ChunkKind {
        let trimmed = content.trim();

        // Check if it's mostly imports
        let import_count = trimmed.lines()
            .filter(|line| {
                let line = line.trim();
                line.starts_with("import ") ||
                line.starts_with("from ") ||
                line.starts_with("use ") ||
                line.starts_with("#include")
            })
            .count();

        if import_count > trimmed.lines().count() / 2 {
            return ChunkKind::Block; // Could add ChunkKind::Imports later
        }

        // Check if it's module-level docs
        if trimmed.starts_with("//!") || trimmed.starts_with("/*!") {
            return ChunkKind::Block; // Could add ChunkKind::ModuleDocs later
        }

        ChunkKind::Block
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_chunker_creation() {
        let chunker = SemanticChunker::new(100, 2000, 10);
        assert_eq!(chunker.max_chunk_lines, 100);
        assert_eq!(chunker.max_chunk_chars, 2000);
        assert_eq!(chunker.overlap_lines, 10);
    }

    #[test]
    fn test_chunk_rust_code() {
        let mut chunker = SemanticChunker::new(100, 2000, 10);

        let rust_code = r#"
/// This is a doc comment
fn hello_world() {
    println!("Hello, world!");
}

fn add(a: i32, b: i32) -> i32 {
    a + b
}

struct Point {
    x: f64,
    y: f64,
}
"#;

        let path = Path::new("test.rs");
        let chunks = chunker.chunk_semantic(Language::Rust, path, rust_code).unwrap();

        // Should have at least 3 definition chunks (2 functions + 1 struct)
        assert!(chunks.len() >= 3, "Expected at least 3 chunks, got {}", chunks.len());

        // Check that we have function chunks
        let function_chunks: Vec<_> = chunks.iter()
            .filter(|c| c.kind == ChunkKind::Function)
            .collect();
        assert!(function_chunks.len() >= 2, "Expected at least 2 function chunks");

        // Check that first function has signature
        let hello_chunk = function_chunks.iter()
            .find(|c| c.content.contains("hello_world"));
        assert!(hello_chunk.is_some(), "Should find hello_world function");

        if let Some(chunk) = hello_chunk {
            assert!(chunk.signature.is_some(), "Should have signature");
            assert!(chunk.signature.as_ref().unwrap().contains("fn hello_world"));
        }
    }

    #[test]
    fn test_chunk_python_code() {
        let mut chunker = SemanticChunker::new(100, 2000, 10);

        let python_code = r#"
def hello():
    """Say hello"""
    print("Hello!")

class Calculator:
    """A simple calculator"""

    def add(self, a, b):
        """Add two numbers"""
        return a + b
"#;

        let path = Path::new("test.py");
        let chunks = chunker.chunk_semantic(Language::Python, path, python_code).unwrap();

        // Should have at least 2 chunks (function + class)
        assert!(chunks.len() >= 2, "Expected at least 2 chunks");

        // Check for docstrings
        let chunks_with_docs: Vec<_> = chunks.iter()
            .filter(|c| c.docstring.is_some())
            .collect();
        assert!(!chunks_with_docs.is_empty(), "Should have chunks with docstrings");
    }

    #[test]
    fn test_chunk_unsupported_language() {
        let mut chunker = SemanticChunker::new(100, 2000, 10);

        let content = "Some random text file\nWith multiple lines\nThat should be chunked\nAs fallback";
        let path = Path::new("test.txt");

        let chunks = chunker.chunk_semantic(Language::Unknown, path, content).unwrap();

        // Should use fallback chunking
        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|c| c.kind == ChunkKind::Block));
    }

    #[test]
    fn test_gap_tracking() {
        let content = "line 0\nline 1\nline 2\nline 3\nline 4";
        let mut tracker = GapTracker::new(content);

        // Mark lines 1-2 as covered
        tracker.mark_covered(1, 2);

        // Should have gaps: [0], [3-4]
        let path = Path::new("test.txt");
        let gaps = tracker.extract_gaps(path);

        assert_eq!(gaps.len(), 2, "Should have 2 gaps");
        assert_eq!(gaps[0].start_line, 0);
        assert_eq!(gaps[0].end_line, 1);
        assert_eq!(gaps[1].start_line, 3);
        assert_eq!(gaps[1].end_line, 5);
    }

    #[test]
    fn test_chunk_splitting() {
        let chunker = SemanticChunker::new(5, 100, 1); // Very small limit

        let large_content = (0..20).map(|i| format!("line {}", i)).collect::<Vec<_>>().join("\n");
        let chunk = Chunk::new(
            large_content,
            0,
            20,
            ChunkKind::Function,
            "test.rs".to_string(),
        );

        let splits = chunker.split_if_needed(chunk);

        // Should be split into multiple chunks
        assert!(splits.len() > 1, "Should split large chunk");

        // All splits should be marked as incomplete
        for split in &splits {
            assert!(!split.is_complete, "Split chunks should be marked incomplete");
            assert!(split.split_index.is_some(), "Split chunks should have index");
        }
    }

    #[test]
    fn test_context_breadcrumbs() {
        let mut chunker = SemanticChunker::new(100, 2000, 10);

        let rust_code = r#"
impl MyStruct {
    fn method(&self) {
        println!("method");
    }
}
"#;

        let path = Path::new("test.rs");
        let chunks = chunker.chunk_semantic(Language::Rust, path, rust_code).unwrap();

        // Find method chunk
        let method_chunk = chunks.iter()
            .find(|c| c.kind == ChunkKind::Method);

        if let Some(chunk) = method_chunk {
            // Should have context: File > Impl > Method
            assert!(chunk.context.len() >= 2, "Should have nested context");
            assert!(chunk.context[0].contains("File:"));
        }
    }
}

#[test]
fn test_chunk_dbt_model() {
    let dbt_content = r#"{{ config(
    materialized='incremental',
    unique_key='id',
    schema='analytics'
) }}

-- This model aggregates customer orders
WITH customers AS (
    SELECT *
    FROM {{ ref('stg_customers') }}
    WHERE is_active = true
),

orders AS (
    SELECT *
    FROM {{ ref('stg_orders') }}
    WHERE order_date >= '2024-01-01'
),

payments AS (
    SELECT *
    FROM {{ source('stripe', 'payments') }}
)

SELECT
    c.customer_id,
    c.customer_name,
    COUNT(DISTINCT o.order_id) as total_orders,
    SUM(p.amount) as total_spent
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
LEFT JOIN payments p ON o.order_id = p.order_id
GROUP BY 1, 2
"#;

    // Verify it's detected as dbt
    assert!(Language::detect_dbt_from_content(dbt_content), "Should be detected as dbt");
    
    let mut chunker = SemanticChunker::new(100, 2000, 10);
    let path = std::path::Path::new("models/customer_orders.sql");
    
    let chunks = chunker.chunk_semantic(Language::Sql, path, dbt_content).unwrap();
    
    println!("\n=== DBT Chunking Test Results ===");
    println!("Number of chunks: {}", chunks.len());
    
    for (i, chunk) in chunks.iter().enumerate() {
        println!("\n--- Chunk {} ---", i + 1);
        println!("Kind: {:?}", chunk.kind);
        println!("Lines: {}-{}", chunk.start_line, chunk.end_line);
        if let Some(sig) = &chunk.signature {
            println!("Signature: {}", sig);
        }
        if let Some(doc) = &chunk.docstring {
            println!("Docstring: {}", doc);
        }
        println!("Content (first 200 chars):\n{}", 
            chunk.content.chars().take(200).collect::<String>());
    }
    
    assert!(!chunks.is_empty(), "Should produce at least one chunk");
}

#[test]
fn test_chunk_dbt_macro() {
    let dbt_macro = r#"{% macro generate_schema_name(custom_schema_name, node) %}
    {%- set default_schema = target.schema -%}
    {%- if custom_schema_name is none -%}
        {{ default_schema }}
    {%- else -%}
        {{ default_schema }}_{{ custom_schema_name | trim }}
    {%- endif -%}
{% endmacro %}

{% macro cents_to_dollars(column_name) %}
    ({{ column_name }} / 100)::numeric(16,2)
{% endmacro %}

{% macro get_payment_methods() %}
    {{ return(['credit_card', 'bank_transfer', 'gift_card']) }}
{% endmacro %}
"#;

    assert!(Language::detect_dbt_from_content(dbt_macro), "Should be detected as dbt");
    
    let mut chunker = SemanticChunker::new(100, 2000, 10);
    let path = std::path::Path::new("macros/utils.sql");
    
    let chunks = chunker.chunk_semantic(Language::Sql, path, dbt_macro).unwrap();
    
    println!("\n=== DBT Macro Chunking Test Results ===");
    println!("Number of chunks: {}", chunks.len());
    
    for (i, chunk) in chunks.iter().enumerate() {
        println!("\n--- Chunk {} ---", i + 1);
        println!("Kind: {:?}", chunk.kind);
        println!("Lines: {}-{}", chunk.start_line, chunk.end_line);
        if let Some(sig) = &chunk.signature {
            println!("Signature: {}", sig);
        }
        println!("Content:\n{}", chunk.content);
    }
    
    assert!(!chunks.is_empty(), "Should produce at least one chunk");
}

#[test]
fn test_chunk_plain_sql() {
    let sql_content = r#"-- Create customers table
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create orders table  
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customers(id),
    total DECIMAL(10,2),
    status VARCHAR(50),
    order_date DATE
);

-- View for active customers with orders
CREATE VIEW active_customers AS
SELECT 
    c.id,
    c.name,
    COUNT(o.id) as order_count,
    SUM(o.total) as total_spent
FROM customers c
JOIN orders o ON c.id = o.customer_id
WHERE o.status = 'completed'
GROUP BY c.id, c.name;
"#;

    // Verify it's NOT detected as dbt
    assert!(!Language::detect_dbt_from_content(sql_content), "Should NOT be detected as dbt");
    
    let mut chunker = SemanticChunker::new(100, 2000, 10);
    let path = std::path::Path::new("schema.sql");
    
    let chunks = chunker.chunk_semantic(Language::Sql, path, sql_content).unwrap();
    
    println!("\n=== Plain SQL Chunking Test Results ===");
    println!("Number of chunks: {}", chunks.len());
    
    for (i, chunk) in chunks.iter().enumerate() {
        println!("\n--- Chunk {} ---", i + 1);
        println!("Kind: {:?}", chunk.kind);
        println!("Lines: {}-{}", chunk.start_line, chunk.end_line);
        if let Some(sig) = &chunk.signature {
            println!("Signature: {}", sig);
        }
        println!("Content:\n{}", chunk.content);
    }
    
    assert!(!chunks.is_empty(), "Should produce at least one chunk");
}

#[test]
fn test_chunk_sql_with_functions() {
    let sql_content = r#"-- Helper function to calculate tax
CREATE FUNCTION calculate_tax(amount DECIMAL(10,2), rate DECIMAL(5,2))
RETURNS DECIMAL(10,2)
AS $$
BEGIN
    RETURN amount * rate;
END;
$$ LANGUAGE plpgsql;

-- Get customer total with tax
CREATE FUNCTION get_customer_total(customer_id INT)
RETURNS TABLE(subtotal DECIMAL, tax DECIMAL, total DECIMAL)
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        SUM(amount) as subtotal,
        calculate_tax(SUM(amount), 0.08) as tax,
        SUM(amount) + calculate_tax(SUM(amount), 0.08) as total
    FROM orders
    WHERE customer_id = customer_id;
END;
$$ LANGUAGE plpgsql;
"#;

    assert!(!Language::detect_dbt_from_content(sql_content), "Should NOT be dbt");
    
    let mut chunker = SemanticChunker::new(100, 2000, 10);
    let path = std::path::Path::new("functions.sql");
    
    let chunks = chunker.chunk_semantic(Language::Sql, path, sql_content).unwrap();
    
    println!("\n=== SQL Functions Test ===");
    for (i, chunk) in chunks.iter().enumerate() {
        println!("Chunk {}: {:?} - {:?}", i + 1, chunk.kind, chunk.signature);
    }
    
    // Should have function chunks
    let function_chunks: Vec<_> = chunks.iter()
        .filter(|c| c.kind == ChunkKind::Function)
        .collect();
    
    assert!(function_chunks.len() >= 1, "Should have at least one function chunk");
}

#[test]
fn test_chunk_sql_with_ctes() {
    let sql_content = r#"WITH 
monthly_sales AS (
    SELECT 
        DATE_TRUNC('month', order_date) as month,
        SUM(amount) as total_sales
    FROM orders
    GROUP BY 1
),
quarterly_sales AS (
    SELECT 
        DATE_TRUNC('quarter', month) as quarter,
        SUM(total_sales) as total_sales
    FROM monthly_sales
    GROUP BY 1
)
SELECT * FROM quarterly_sales;
"#;

    let mut chunker = SemanticChunker::new(100, 2000, 10);
    let path = std::path::Path::new("report.sql");
    
    let chunks = chunker.chunk_semantic(Language::Sql, path, sql_content).unwrap();
    
    println!("\n=== SQL CTEs Test ===");
    for (i, chunk) in chunks.iter().enumerate() {
        println!("Chunk {}: {:?} lines {}-{}", i + 1, chunk.kind, chunk.start_line, chunk.end_line);
    }
    
    assert!(!chunks.is_empty(), "Should produce chunks");
}

#[test]
fn test_chunk_dbt_snapshot() {
    let dbt_snapshot = r#"{% snapshot orders_snapshot %}

{{ config(
    target_schema='snapshots',
    unique_key='order_id',
    strategy='timestamp',
    updated_at='updated_at',
) }}

SELECT 
    order_id,
    customer_id,
    status,
    amount,
    updated_at
FROM {{ source('raw', 'orders') }}

{% endsnapshot %}
"#;

    assert!(Language::detect_dbt_from_content(dbt_snapshot), "Should be detected as dbt");
    
    let mut chunker = SemanticChunker::new(100, 2000, 10);
    let path = std::path::Path::new("snapshots/orders_snapshot.sql");
    
    let chunks = chunker.chunk_semantic(Language::Sql, path, dbt_snapshot).unwrap();
    
    println!("\n=== dbt Snapshot Test ===");
    for (i, chunk) in chunks.iter().enumerate() {
        println!("Chunk {}: {:?}", i + 1, chunk.kind);
        if let Some(sig) = &chunk.signature {
            println!("  Signature: {}", sig);
        }
    }
    
    assert!(!chunks.is_empty(), "Should produce chunks");
}

#[test]
fn test_chunk_dbt_incremental_model() {
    let dbt_incremental = r#"{{ config(
    materialized='incremental',
    unique_key='event_id',
    incremental_strategy='merge',
    on_schema_change='sync_all_columns'
) }}

WITH source_events AS (
    SELECT *
    FROM {{ source('events', 'raw_events') }}
    {% if is_incremental() %}
    WHERE event_timestamp > (SELECT MAX(event_timestamp) FROM {{ this }})
    {% endif %}
),

enriched AS (
    SELECT 
        e.*,
        u.user_name,
        u.user_segment
    FROM source_events e
    LEFT JOIN {{ ref('dim_users') }} u ON e.user_id = u.user_id
)

SELECT * FROM enriched
"#;

    assert!(Language::detect_dbt_from_content(dbt_incremental), "Should be detected as dbt");
    
    let mut chunker = SemanticChunker::new(100, 2000, 10);
    let path = std::path::Path::new("models/events/fct_events.sql");
    
    let chunks = chunker.chunk_semantic(Language::Sql, path, dbt_incremental).unwrap();
    
    println!("\n=== dbt Incremental Model Test ===");
    for (i, chunk) in chunks.iter().enumerate() {
        println!("Chunk {}: {:?} lines {}-{}", i + 1, chunk.kind, chunk.start_line, chunk.end_line);
        if let Some(sig) = &chunk.signature {
            println!("  Signature: {}", sig);
        }
    }
    
    // Check that signature contains expected dbt metadata
    let has_config = chunks.iter().any(|c| {
        c.signature.as_ref().map_or(false, |s| s.contains("materialized='incremental'"))
    });
    
    let has_refs = chunks.iter().any(|c| {
        c.signature.as_ref().map_or(false, |s| s.contains("dim_users"))
    });
    
    let has_sources = chunks.iter().any(|c| {
        c.signature.as_ref().map_or(false, |s| s.contains("events.raw_events"))
    });
    
    assert!(has_config || has_refs || has_sources, 
        "Should extract dbt metadata (config, refs, or sources)");
}

#[test]
fn test_chunk_dbt_test_file() {
    let dbt_test = r#"{% test not_null_where(model, column_name, where_clause) %}

SELECT *
FROM {{ model }}
WHERE {{ column_name }} IS NULL
AND {{ where_clause }}

{% endtest %}

{% test accepted_range(model, column_name, min_value, max_value) %}

SELECT *
FROM {{ model }}
WHERE {{ column_name }} < {{ min_value }}
   OR {{ column_name }} > {{ max_value }}

{% endtest %}
"#;

    assert!(Language::detect_dbt_from_content(dbt_test), "Should be detected as dbt");
    
    let mut chunker = SemanticChunker::new(100, 2000, 10);
    let path = std::path::Path::new("tests/generic/custom_tests.sql");
    
    let chunks = chunker.chunk_semantic(Language::Sql, path, dbt_test).unwrap();
    
    println!("\n=== dbt Test File Test ===");
    for (i, chunk) in chunks.iter().enumerate() {
        println!("Chunk {}: {:?}", i + 1, chunk.kind);
    }
    
    assert!(!chunks.is_empty(), "Should produce chunks");
}

#[test]
fn test_plain_sql_vs_dbt_detection() {
    // Plain SQL - should NOT be detected as dbt
    let plain_sql_examples = vec![
        "SELECT * FROM users WHERE status = 'active'",
        "CREATE TABLE orders (id INT, amount DECIMAL)",
        "INSERT INTO logs VALUES (1, 'test', NOW())",
        "UPDATE users SET status = 'inactive' WHERE last_login < '2024-01-01'",
        "-- Comment with {{ curly braces }} but no dbt pattern",
    ];
    
    for sql in plain_sql_examples {
        assert!(!Language::detect_dbt_from_content(sql), 
            "Should NOT detect as dbt: {}", sql.chars().take(50).collect::<String>());
    }
    
    // dbt - SHOULD be detected
    let dbt_examples = vec![
        "SELECT * FROM {{ ref('users') }}",
        "{{ config(materialized='table') }} SELECT 1",
        "SELECT * FROM {{ source('raw', 'data') }}",
        "{% macro test() %} SELECT 1 {% endmacro %}",
        "{% if is_incremental() %} WHERE id > 0 {% endif %}",
    ];
    
    for dbt in dbt_examples {
        assert!(Language::detect_dbt_from_content(dbt), 
            "SHOULD detect as dbt: {}", dbt.chars().take(50).collect::<String>());
    }
}
