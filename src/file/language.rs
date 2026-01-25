use std::path::Path;

/// Supported programming languages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Go,
    Java,
    C,
    Cpp,
    CSharp,
    Ruby,
    Php,
    Swift,
    Kotlin,
    Shell,
    Markdown,
    Json,
    Yaml,
    Toml,
    Sql,
    Dbt,
    Html,
    Css,
    Unknown,
}

impl Language {
    /// Detect language from file extension
    pub fn from_path(path: &Path) -> Self {
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        Self::from_extension(extension)
    }

    /// Detect language from extension string
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "rs" => Self::Rust,
            "py" | "pyw" | "pyi" => Self::Python,
            "js" | "mjs" | "cjs" => Self::JavaScript,
            "ts" | "mts" | "cts" => Self::TypeScript,
            "tsx" | "jsx" => Self::TypeScript, // Treat JSX/TSX as TypeScript
            "go" => Self::Go,
            "java" => Self::Java,
            "c" | "h" => Self::C,
            "cpp" | "cc" | "cxx" | "hpp" | "hxx" => Self::Cpp,
            "cs" => Self::CSharp,
            "rb" | "rake" => Self::Ruby,
            "php" => Self::Php,
            "swift" => Self::Swift,
            "kt" | "kts" => Self::Kotlin,
            "sh" | "bash" | "zsh" => Self::Shell,
            "md" | "markdown" | "txt" => Self::Markdown, // Treat txt as markdown-like
            "json" => Self::Json,
            "yaml" | "yml" => Self::Yaml,
            "toml" => Self::Toml,
            "sql" => Self::Sql,
            "html" | "htm" => Self::Html,
            "css" | "scss" | "sass" | "less" => Self::Css,
            _ => Self::Unknown,
        }
    }

    /// Check if this language is supported for semantic chunking
    pub fn supports_tree_sitter(&self) -> bool {
        matches!(
            self,
            Self::Rust
                | Self::Python
                | Self::JavaScript
                | Self::TypeScript
                | Self::CSharp
                | Self::Go
                | Self::Java
                | Self::C
                | Self::Cpp
                | Self::Ruby
                | Self::Php
                | Self::Shell
                | Self::Sql
                | Self::Dbt
        )
    }

    /// Detect if SQL content is actually a dbt model (contains Jinja templating)
    ///
    /// dbt files use Jinja2 templating with patterns like:
    /// - `{{ ref('model') }}` - model references
    /// - `{{ source('schema', 'table') }}` - source references
    /// - `{{ config(...) }}` - configuration blocks
    /// - `{% macro name(...) %}` - macro definitions
    pub fn detect_dbt_from_content(content: &str) -> bool {
        // Check for common dbt/Jinja patterns
        let has_jinja_expr = content.contains("{{") && content.contains("}}");
        let has_jinja_stmt = content.contains("{%") && content.contains("%}");
        let has_dbt_ref = content.contains("ref(") || content.contains("ref (");
        let has_dbt_source = content.contains("source(") || content.contains("source (");
        let has_dbt_config = content.contains("config(") || content.contains("config (");

        // Must have Jinja syntax AND at least one dbt-specific pattern
        (has_jinja_expr || has_jinja_stmt) && (has_dbt_ref || has_dbt_source || has_dbt_config || has_jinja_stmt)
    }

    /// Check if this is a text-based language (should be indexed)
    pub fn is_indexable(&self) -> bool {
        !matches!(self, Self::Unknown)
    }

    /// Get the language name as a string
    pub fn name(&self) -> &'static str {
        match self {
            Self::Rust => "Rust",
            Self::Python => "Python",
            Self::JavaScript => "JavaScript",
            Self::TypeScript => "TypeScript",
            Self::Go => "Go",
            Self::Java => "Java",
            Self::C => "C",
            Self::Cpp => "C++",
            Self::CSharp => "C#",
            Self::Ruby => "Ruby",
            Self::Php => "PHP",
            Self::Swift => "Swift",
            Self::Kotlin => "Kotlin",
            Self::Shell => "Shell",
            Self::Markdown => "Markdown",
            Self::Json => "JSON",
            Self::Yaml => "YAML",
            Self::Toml => "TOML",
            Self::Sql => "SQL",
            Self::Dbt => "dbt",
            Self::Html => "HTML",
            Self::Css => "CSS",
            Self::Unknown => "Unknown",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_rust_detection() {
        assert_eq!(Language::from_extension("rs"), Language::Rust);
        assert_eq!(
            Language::from_path(&PathBuf::from("main.rs")),
            Language::Rust
        );
    }

    #[test]
    fn test_python_detection() {
        assert_eq!(Language::from_extension("py"), Language::Python);
        assert_eq!(Language::from_extension("pyi"), Language::Python);
    }

    #[test]
    fn test_typescript_detection() {
        assert_eq!(Language::from_extension("ts"), Language::TypeScript);
        assert_eq!(Language::from_extension("tsx"), Language::TypeScript);
        assert_eq!(Language::from_extension("jsx"), Language::TypeScript);
    }

    #[test]
    fn test_tree_sitter_support() {
        assert!(Language::Rust.supports_tree_sitter());
        assert!(Language::Python.supports_tree_sitter());
        assert!(Language::TypeScript.supports_tree_sitter());
        assert!(!Language::Markdown.supports_tree_sitter());
        assert!(!Language::Json.supports_tree_sitter());
    }

    #[test]
    fn test_indexable() {
        assert!(Language::Rust.is_indexable());
        assert!(Language::Markdown.is_indexable());
        assert!(!Language::Unknown.is_indexable());
    }

    #[test]
    fn test_sql_tree_sitter_support() {
        assert!(Language::Sql.supports_tree_sitter());
        assert!(Language::Dbt.supports_tree_sitter());
    }

    #[test]
    fn test_dbt_detection_with_ref() {
        let dbt_content = r#"
            {{ config(materialized='table') }}

            SELECT *
            FROM {{ ref('upstream_model') }}
            WHERE status = 'active'
        "#;
        assert!(Language::detect_dbt_from_content(dbt_content));
    }

    #[test]
    fn test_dbt_detection_with_source() {
        let dbt_content = r#"
            SELECT *
            FROM {{ source('raw', 'customers') }}
        "#;
        assert!(Language::detect_dbt_from_content(dbt_content));
    }

    #[test]
    fn test_dbt_detection_with_macro() {
        let dbt_content = r#"
            {% macro my_macro(arg1) %}
                SELECT {{ arg1 }}
            {% endmacro %}
        "#;
        assert!(Language::detect_dbt_from_content(dbt_content));
    }

    #[test]
    fn test_plain_sql_not_detected_as_dbt() {
        let sql_content = r#"
            CREATE TABLE customers (
                id INT PRIMARY KEY,
                name VARCHAR(255)
            );

            SELECT * FROM customers WHERE status = 'active';
        "#;
        assert!(!Language::detect_dbt_from_content(sql_content));
    }

    #[test]
    fn test_sql_with_curly_braces_not_dbt() {
        // SQL with JSON/JSONB that has curly braces but no dbt patterns
        let sql_content = r#"
            SELECT data->>'name' as name
            FROM events
            WHERE data @> '{"type": "click"}'
        "#;
        assert!(!Language::detect_dbt_from_content(sql_content));
    }
}
