//! Reranking and result fusion strategies
//!
//! Provides RRF (Reciprocal Rank Fusion) for combining vector and FTS results,
//! and neural reranking using cross-encoder models for improved accuracy.

mod neural;

use std::collections::HashMap;

use crate::fts::FtsResult;
use crate::vectordb::SearchResult;

pub use neural::NeuralReranker;

/// Default RRF k parameter (research recommends k=60 for better recall)
pub const DEFAULT_RRF_K: f32 = 60.0;

/// Fused search result combining vector and FTS scores
#[derive(Debug, Clone)]
pub struct FusedResult {
    /// Chunk ID
    pub chunk_id: u32,
    /// Combined RRF score
    pub rrf_score: f32,
    /// Original vector similarity score (if present)
    pub vector_score: Option<f32>,
    /// Original FTS/BM25 score (if present)
    pub fts_score: Option<f32>,
    /// Vector rank (1-indexed, None if not in vector results)
    pub vector_rank: Option<usize>,
    /// FTS rank (1-indexed, None if not in FTS results)
    pub fts_rank: Option<usize>,
}

/// Reciprocal Rank Fusion (RRF) for combining search results
///
/// RRF formula: score = sum(1 / (k + rank)) for each ranking list
/// where k is a constant (default 20) and rank is 1-indexed position.
///
/// This is a proven technique for combining multiple ranking signals
/// without needing to normalize scores across different systems.
pub fn rrf_fusion(
    vector_results: &[SearchResult],
    fts_results: &[FtsResult],
    k: f32,
) -> Vec<FusedResult> {
    // Maps chunk_id -> (rrf_score, vector_score, fts_score, vector_rank, fts_rank)
    let mut scores: HashMap<u32, (f32, Option<f32>, Option<f32>, Option<usize>, Option<usize>)> =
        HashMap::new();

    // Process vector results
    for (rank, result) in vector_results.iter().enumerate() {
        let chunk_id = result.id;
        let rrf_score = 1.0 / (k + rank as f32 + 1.0);

        let entry = scores.entry(chunk_id).or_insert((0.0, None, None, None, None));
        entry.0 += rrf_score;
        entry.1 = Some(result.score);
        entry.3 = Some(rank + 1);
    }

    // Process FTS results
    for (rank, result) in fts_results.iter().enumerate() {
        let chunk_id = result.chunk_id;
        let rrf_score = 1.0 / (k + rank as f32 + 1.0);

        let entry = scores.entry(chunk_id).or_insert((0.0, None, None, None, None));
        entry.0 += rrf_score;
        entry.2 = Some(result.score);
        entry.4 = Some(rank + 1);
    }

    // Convert to FusedResult and sort by RRF score
    let mut results: Vec<FusedResult> = scores
        .into_iter()
        .map(|(chunk_id, (rrf_score, vector_score, fts_score, vector_rank, fts_rank))| {
            FusedResult {
                chunk_id,
                rrf_score,
                vector_score,
                fts_score,
                vector_rank,
                fts_rank,
            }
        })
        .collect();

    // Sort by RRF score descending
    results.sort_by(|a, b| b.rrf_score.partial_cmp(&a.rrf_score).unwrap_or(std::cmp::Ordering::Equal));

    results
}

/// Query types for adaptive weighting
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QueryType {
    /// Single identifier or code-like token (e.g., "HashMap", "process_payment")
    Keyword,
    /// Natural language question (e.g., "how do I authenticate users")
    Semantic,
    /// Mixed query with both code and natural language
    Hybrid,
}

/// Classify query type based on heuristics
pub fn classify_query(query: &str) -> QueryType {
    let words: Vec<&str> = query.split_whitespace().collect();

    // Single word or very short = likely keyword search
    if words.len() == 1 {
        return QueryType::Keyword;
    }

    // Contains question words = semantic
    let semantic_markers = ["how", "what", "where", "why", "when", "which", "can", "does", "is", "are", "should", "could", "would"];
    if words.iter().any(|w| semantic_markers.contains(&w.to_lowercase().as_str())) {
        return QueryType::Semantic;
    }

    // Contains code-like tokens = keyword-leaning
    // These patterns strongly indicate code searches
    if query.contains("::") || query.contains("()") || query.contains("->") || query.contains("=>") {
        return QueryType::Keyword;
    }

    // Generic type patterns like Result<T, E> or Vec<String>
    if query.contains('<') && query.contains('>') {
        return QueryType::Keyword;
    }

    // CamelCase or snake_case patterns suggest keyword
    let has_camel_case = query.chars().any(|c| c.is_uppercase()) &&
                         query.chars().any(|c| c.is_lowercase()) &&
                         !query.contains(' ');
    let has_snake_case = query.contains('_') && !query.contains(' ');

    if has_camel_case || has_snake_case {
        return QueryType::Keyword;
    }

    // Short phrases (2-3 words) without semantic markers lean toward hybrid
    if words.len() <= 3 {
        return QueryType::Hybrid;
    }

    // Longer natural language phrases = semantic
    if words.len() >= 5 {
        return QueryType::Semantic;
    }

    QueryType::Hybrid
}

/// Get alpha weight for vector results based on query type
/// Alpha = weight for vector search (1 - alpha = weight for FTS)
pub fn query_alpha(query_type: QueryType) -> f32 {
    match query_type {
        QueryType::Keyword => 0.3,   // 30% vector, 70% FTS (favor exact matching)
        QueryType::Semantic => 0.8,  // 80% vector, 20% FTS (favor semantic similarity)
        QueryType::Hybrid => 0.5,    // 50% each (balanced)
    }
}

/// Reciprocal Rank Fusion with query-dependent weighting
///
/// This version applies alpha weighting to balance vector and FTS contributions
/// based on query type classification.
pub fn weighted_rrf_fusion(
    vector_results: &[SearchResult],
    fts_results: &[FtsResult],
    k: f32,
    alpha: f32,
) -> Vec<FusedResult> {
    // Maps chunk_id -> (rrf_score, vector_score, fts_score, vector_rank, fts_rank)
    let mut scores: HashMap<u32, (f32, Option<f32>, Option<f32>, Option<usize>, Option<usize>)> =
        HashMap::new();

    // Process vector results with alpha weight
    for (rank, result) in vector_results.iter().enumerate() {
        let chunk_id = result.id;
        let rrf_score = alpha * (1.0 / (k + rank as f32 + 1.0));

        let entry = scores.entry(chunk_id).or_insert((0.0, None, None, None, None));
        entry.0 += rrf_score;
        entry.1 = Some(result.score);
        entry.3 = Some(rank + 1);
    }

    // Process FTS results with (1 - alpha) weight
    for (rank, result) in fts_results.iter().enumerate() {
        let chunk_id = result.chunk_id;
        let rrf_score = (1.0 - alpha) * (1.0 / (k + rank as f32 + 1.0));

        let entry = scores.entry(chunk_id).or_insert((0.0, None, None, None, None));
        entry.0 += rrf_score;
        entry.2 = Some(result.score);
        entry.4 = Some(rank + 1);
    }

    // Convert to FusedResult and sort by RRF score
    let mut results: Vec<FusedResult> = scores
        .into_iter()
        .map(|(chunk_id, (rrf_score, vector_score, fts_score, vector_rank, fts_rank))| {
            FusedResult {
                chunk_id,
                rrf_score,
                vector_score,
                fts_score,
                vector_rank,
                fts_rank,
            }
        })
        .collect();

    // Sort by RRF score descending
    results.sort_by(|a, b| b.rrf_score.partial_cmp(&a.rrf_score).unwrap_or(std::cmp::Ordering::Equal));

    results
}

/// Simple vector-only pass-through (no fusion)
pub fn vector_only(vector_results: &[SearchResult]) -> Vec<FusedResult> {
    vector_results
        .iter()
        .enumerate()
        .map(|(rank, result)| FusedResult {
            chunk_id: result.id,
            rrf_score: result.score,
            vector_score: Some(result.score),
            fts_score: None,
            vector_rank: Some(rank + 1),
            fts_rank: None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vector_result(id: u32, score: f32) -> SearchResult {
        SearchResult {
            id,
            content: format!("content {}", id),
            path: format!("file_{}.rs", id),
            start_line: 1,
            end_line: 10,
            kind: "function".to_string(),
            signature: None,
            docstring: None,
            context: None,
            hash: format!("hash_{}", id),
            distance: 1.0 - score,
            score,
            context_prev: None,
            context_next: None,
        }
    }

    fn make_fts_result(id: u32, score: f32) -> FtsResult {
        FtsResult {
            chunk_id: id,
            score,
        }
    }

    #[test]
    fn test_rrf_fusion_basic() {
        let vector_results = vec![
            make_vector_result(1, 0.9),
            make_vector_result(2, 0.8),
            make_vector_result(3, 0.7),
        ];

        let fts_results = vec![
            make_fts_result(2, 10.0), // ID 2 is top in FTS
            make_fts_result(1, 8.0),
            make_fts_result(4, 6.0),  // ID 4 only in FTS
        ];

        let fused = rrf_fusion(&vector_results, &fts_results, 20.0);

        // ID 2 should be top (rank 1 in FTS, rank 2 in vector)
        // ID 1 should be second (rank 1 in vector, rank 2 in FTS)
        assert!(!fused.is_empty());

        // Find IDs 1 and 2
        let id1 = fused.iter().find(|r| r.chunk_id == 1).unwrap();
        let id2 = fused.iter().find(|r| r.chunk_id == 2).unwrap();

        // Both should have contributions from both sources
        assert!(id1.vector_rank.is_some());
        assert!(id1.fts_rank.is_some());
        assert!(id2.vector_rank.is_some());
        assert!(id2.fts_rank.is_some());

        // ID 4 should only be in FTS
        let id4 = fused.iter().find(|r| r.chunk_id == 4).unwrap();
        assert!(id4.vector_rank.is_none());
        assert!(id4.fts_rank.is_some());
    }

    #[test]
    fn test_rrf_score_calculation() {
        // With k=20:
        // Rank 1: 1/(20+1) = 0.0476
        // Rank 2: 1/(20+2) = 0.0454
        let vector_results = vec![make_vector_result(1, 0.9)];
        let fts_results = vec![make_fts_result(1, 10.0)];

        let fused = rrf_fusion(&vector_results, &fts_results, 20.0);

        assert_eq!(fused.len(), 1);
        let result = &fused[0];

        // Should be sum of both contributions
        let expected = 1.0 / 21.0 + 1.0 / 21.0;
        assert!((result.rrf_score - expected).abs() < 0.0001);
    }

    #[test]
    fn test_vector_only() {
        let vector_results = vec![
            make_vector_result(1, 0.9),
            make_vector_result(2, 0.8),
        ];

        let results = vector_only(&vector_results);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk_id, 1);
        assert_eq!(results[0].rrf_score, 0.9);
        assert!(results[0].fts_score.is_none());
    }

    #[test]
    fn test_query_classification_keyword() {
        // Single tokens
        assert_eq!(classify_query("HashMap"), QueryType::Keyword);
        assert_eq!(classify_query("processPayment"), QueryType::Keyword);

        // Code-like patterns
        assert_eq!(classify_query("std::collections::HashMap"), QueryType::Keyword);
        assert_eq!(classify_query("process_payment"), QueryType::Keyword);
        assert_eq!(classify_query("foo()"), QueryType::Keyword);
        assert_eq!(classify_query("Result<T, E>"), QueryType::Keyword);
    }

    #[test]
    fn test_query_classification_semantic() {
        // Question patterns
        assert_eq!(classify_query("how do I authenticate users"), QueryType::Semantic);
        assert_eq!(classify_query("what is the best way to handle errors"), QueryType::Semantic);
        assert_eq!(classify_query("where are database connections managed"), QueryType::Semantic);
        assert_eq!(classify_query("why does this function return None"), QueryType::Semantic);

        // Long natural language
        assert_eq!(classify_query("find the code that processes payment transactions"), QueryType::Semantic);
    }

    #[test]
    fn test_query_classification_hybrid() {
        // Mixed short phrases
        assert_eq!(classify_query("sort function"), QueryType::Hybrid);
        assert_eq!(classify_query("authentication handler"), QueryType::Hybrid);
        assert_eq!(classify_query("database connection pool"), QueryType::Hybrid);
    }

    #[test]
    fn test_query_alpha_values() {
        assert!((query_alpha(QueryType::Keyword) - 0.3).abs() < 0.001);
        assert!((query_alpha(QueryType::Semantic) - 0.8).abs() < 0.001);
        assert!((query_alpha(QueryType::Hybrid) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_weighted_rrf_keyword_query() {
        // For keyword queries (alpha=0.3), FTS should dominate
        let vector_results = vec![
            make_vector_result(1, 0.9),  // Top in vector
            make_vector_result(2, 0.8),
        ];

        let fts_results = vec![
            make_fts_result(2, 10.0),  // Top in FTS
            make_fts_result(1, 8.0),
        ];

        let alpha = 0.3;  // Keyword query
        let fused = weighted_rrf_fusion(&vector_results, &fts_results, 60.0, alpha);

        // With alpha=0.3, FTS has 70% weight, so ID 2 (top in FTS) should rank higher
        assert_eq!(fused[0].chunk_id, 2, "FTS top result should win with low alpha");
    }

    #[test]
    fn test_weighted_rrf_semantic_query() {
        // For semantic queries (alpha=0.8), vector should dominate
        let vector_results = vec![
            make_vector_result(1, 0.9),  // Top in vector
            make_vector_result(2, 0.8),
        ];

        let fts_results = vec![
            make_fts_result(2, 10.0),  // Top in FTS
            make_fts_result(1, 8.0),
        ];

        let alpha = 0.8;  // Semantic query
        let fused = weighted_rrf_fusion(&vector_results, &fts_results, 60.0, alpha);

        // With alpha=0.8, vector has 80% weight, so ID 1 (top in vector) should rank higher
        assert_eq!(fused[0].chunk_id, 1, "Vector top result should win with high alpha");
    }

    #[test]
    fn test_weighted_rrf_balanced() {
        // For balanced queries (alpha=0.5), both contribute equally
        let vector_results = vec![make_vector_result(1, 0.9)];
        let fts_results = vec![make_fts_result(1, 10.0)];

        let alpha = 0.5;
        let fused = weighted_rrf_fusion(&vector_results, &fts_results, 60.0, alpha);

        // With k=60, rank 1 contribution = 1/61
        // Expected: 0.5 * (1/61) + 0.5 * (1/61) = 1/61
        let expected = 1.0 / 61.0;
        assert!((fused[0].rrf_score - expected).abs() < 0.0001);
    }

    #[test]
    fn test_default_rrf_k_is_60() {
        assert_eq!(DEFAULT_RRF_K, 60.0);
    }
}
