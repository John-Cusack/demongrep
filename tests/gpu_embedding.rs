//! Integration tests for GPU embedding functionality
//!
//! These tests verify that execution providers work correctly and
//! that batch sizes are optimized per provider.

use demongrep::embed::{detect_best_provider, ExecutionProviderType, FastEmbedder, ModelType, EmbeddingService};

// ============================================================================
// Default Behavior Tests
// ============================================================================

#[test]
fn test_cpu_is_default_provider() {
    // Ensure CPU is the default provider for backward compatibility
    assert_eq!(ExecutionProviderType::default(), ExecutionProviderType::Cpu);
}

#[test]
fn test_default_provider_is_not_auto() {
    // Auto should NOT be the default - users must opt-in to GPU
    assert_ne!(ExecutionProviderType::default(), ExecutionProviderType::Auto);
}

// ============================================================================
// ExecutionProviderType Tests
// ============================================================================

#[test]
fn test_execution_provider_from_str() {
    // Test valid providers
    assert_eq!(
        "cpu".parse::<ExecutionProviderType>().unwrap(),
        ExecutionProviderType::Cpu
    );
    assert_eq!(
        "auto".parse::<ExecutionProviderType>().unwrap(),
        ExecutionProviderType::Auto
    );

    // Test case insensitive
    assert_eq!(
        "CPU".parse::<ExecutionProviderType>().unwrap(),
        ExecutionProviderType::Cpu
    );
    assert_eq!(
        "Auto".parse::<ExecutionProviderType>().unwrap(),
        ExecutionProviderType::Auto
    );

    // Test invalid provider
    assert!("invalid".parse::<ExecutionProviderType>().is_err());
    assert!("gpu".parse::<ExecutionProviderType>().is_err());
    assert!("".parse::<ExecutionProviderType>().is_err());
}

#[test]
fn test_execution_provider_from_str_with_device() {
    assert_eq!(
        ExecutionProviderType::from_str_with_device("cpu", 0).unwrap(),
        ExecutionProviderType::Cpu
    );
    assert_eq!(
        ExecutionProviderType::from_str_with_device("auto", 1).unwrap(),
        ExecutionProviderType::Auto
    );
}

// ============================================================================
// Optimal Batch Size Tests
// ============================================================================

#[test]
fn test_optimal_batch_size_cpu() {
    // All providers now return 32 (empirically optimal based on benchmarks)
    assert_eq!(ExecutionProviderType::Cpu.optimal_batch_size(384), 32);
    assert_eq!(ExecutionProviderType::Cpu.optimal_batch_size(768), 32);
    assert_eq!(ExecutionProviderType::Cpu.optimal_batch_size(1024), 32);
}

#[test]
fn test_optimal_batch_size_auto() {
    // All providers now return 32 (empirically optimal based on benchmarks)
    assert_eq!(ExecutionProviderType::Auto.optimal_batch_size(384), 32);
    assert_eq!(ExecutionProviderType::Auto.optimal_batch_size(768), 32);
    assert_eq!(ExecutionProviderType::Auto.optimal_batch_size(1024), 32);
}

#[cfg(feature = "cuda")]
#[test]
fn test_optimal_batch_size_cuda() {
    let cuda = ExecutionProviderType::Cuda { device_id: 0 };
    // All providers now return 32 (empirically optimal based on benchmarks)
    assert_eq!(cuda.optimal_batch_size(384), 32);
    assert_eq!(cuda.optimal_batch_size(768), 32);
    assert_eq!(cuda.optimal_batch_size(1024), 32);
}

#[cfg(feature = "tensorrt")]
#[test]
fn test_optimal_batch_size_tensorrt() {
    let trt = ExecutionProviderType::TensorRt { device_id: 0 };
    // All providers now return 32 (empirically optimal based on benchmarks)
    assert_eq!(trt.optimal_batch_size(384), 32);
    assert_eq!(trt.optimal_batch_size(768), 32);
    assert_eq!(trt.optimal_batch_size(1024), 32);
}

#[cfg(feature = "coreml")]
#[test]
fn test_optimal_batch_size_coreml() {
    // All providers now return 32 (empirically optimal based on benchmarks)
    assert_eq!(ExecutionProviderType::CoreMl.optimal_batch_size(384), 32);
    assert_eq!(ExecutionProviderType::CoreMl.optimal_batch_size(768), 32);
    assert_eq!(ExecutionProviderType::CoreMl.optimal_batch_size(1024), 32);
}

#[cfg(feature = "directml")]
#[test]
fn test_optimal_batch_size_directml() {
    let dml = ExecutionProviderType::DirectMl { device_id: 0 };
    // All providers now return 32 (empirically optimal based on benchmarks)
    assert_eq!(dml.optimal_batch_size(384), 32);
    assert_eq!(dml.optimal_batch_size(768), 32);
    assert_eq!(dml.optimal_batch_size(1024), 32);
}

#[test]
fn test_optimal_batch_size_scales_with_dimensions() {
    // All providers should use smaller batches for larger models
    let providers = vec![
        ExecutionProviderType::Cpu,
        ExecutionProviderType::Auto,
    ];

    for provider in providers {
        let small = provider.optimal_batch_size(384);
        let medium = provider.optimal_batch_size(768);
        let large = provider.optimal_batch_size(1024);

        assert!(small >= medium, "Small model batch should be >= medium for {:?}", provider);
        assert!(medium >= large, "Medium model batch should be >= large for {:?}", provider);
    }
}

// ============================================================================
// Provider Detection Tests
// ============================================================================

#[test]
fn test_detect_best_provider_returns_valid() {
    let provider = detect_best_provider();
    // Should never return Auto
    assert_ne!(provider, ExecutionProviderType::Auto);
}

#[test]
fn test_detect_best_provider_cpu_fallback() {
    // Without GPU features enabled, should fall back to CPU
    #[cfg(not(any(feature = "cuda", feature = "tensorrt", feature = "coreml", feature = "directml")))]
    {
        let provider = detect_best_provider();
        assert_eq!(provider, ExecutionProviderType::Cpu);
    }
}

#[test]
fn test_provider_name() {
    assert_eq!(ExecutionProviderType::Cpu.name(), "CPU");
    assert_eq!(ExecutionProviderType::Auto.name(), "Auto");

    #[cfg(feature = "cuda")]
    assert_eq!(ExecutionProviderType::Cuda { device_id: 0 }.name(), "CUDA");

    #[cfg(feature = "tensorrt")]
    assert_eq!(ExecutionProviderType::TensorRt { device_id: 0 }.name(), "TensorRT");

    #[cfg(feature = "coreml")]
    assert_eq!(ExecutionProviderType::CoreMl.name(), "CoreML");

    #[cfg(feature = "directml")]
    assert_eq!(ExecutionProviderType::DirectMl { device_id: 0 }.name(), "DirectML");
}

// ============================================================================
// Embedder Integration Tests (require model download)
// ============================================================================

#[test]
#[ignore] // Requires model download
fn test_embedder_stores_provider() {
    let embedder = FastEmbedder::with_model_and_provider(
        ModelType::AllMiniLML6V2Q,
        ExecutionProviderType::Cpu,
        None,
    )
    .unwrap();
    assert_eq!(embedder.provider(), ExecutionProviderType::Cpu);
}

#[test]
#[ignore] // Requires model download
fn test_embedder_resolves_auto_provider() {
    let embedder = FastEmbedder::with_model_and_provider(
        ModelType::AllMiniLML6V2Q,
        ExecutionProviderType::Auto,
        None,
    )
    .unwrap();

    // Auto should resolve to something concrete (not Auto)
    let provider = embedder.provider();
    assert_ne!(provider, ExecutionProviderType::Auto);
}

#[test]
#[ignore] // Requires model download
fn test_auto_fallback_to_cpu() {
    // Test that auto mode gracefully falls back to CPU
    let model_type = ModelType::default();
    let embedder =
        FastEmbedder::with_model_and_provider(model_type, ExecutionProviderType::Auto, None);

    // Should succeed even if no GPU is available
    assert!(embedder.is_ok(), "Auto provider should fall back to CPU");
}

#[test]
#[ignore] // Requires model download
fn test_embedding_service_with_batch_size() {
    let service = EmbeddingService::with_model_and_provider(
        ModelType::AllMiniLML6V2Q,
        ExecutionProviderType::Cpu,
        None,
        Some(128), // explicit batch size
    );
    assert!(service.is_ok());
}

#[test]
#[ignore] // Requires model download
fn test_embedding_service_uses_optimal_batch_size() {
    // Create service without explicit batch size
    let service = EmbeddingService::with_model_and_provider(
        ModelType::AllMiniLML6V2Q,
        ExecutionProviderType::Cpu,
        None,
        None, // should use optimal for CPU
    );
    assert!(service.is_ok());
    assert_eq!(service.unwrap().dimensions(), 384);
}

// ============================================================================
// CUDA-specific Tests (require CUDA hardware)
// ============================================================================

#[cfg(feature = "cuda")]
mod cuda_tests {
    use super::*;
    use demongrep::embed::is_cuda_available;

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_cuda_embedding() {
        if !is_cuda_available() {
            eprintln!("CUDA not available, skipping test");
            return;
        }

        let model_type = ModelType::default();
        let embedder = FastEmbedder::with_model_and_provider(
            model_type,
            ExecutionProviderType::Cuda { device_id: 0 },
            Some(0),
        );

        assert!(
            embedder.is_ok(),
            "CUDA provider should work when CUDA is available"
        );

        let mut embedder = embedder.unwrap();
        assert_eq!(embedder.provider(), ExecutionProviderType::Cuda { device_id: 0 });

        let embedding = embedder.embed_one("Test text for CUDA embedding").unwrap();
        assert!(!embedding.is_empty(), "Embedding should not be empty");
        assert_eq!(embedding.len(), 384);
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_cuda_batch_embedding() {
        if !is_cuda_available() {
            eprintln!("CUDA not available, skipping test");
            return;
        }

        let mut embedder = FastEmbedder::with_model_and_provider(
            ModelType::AllMiniLML6V2Q,
            ExecutionProviderType::Cuda { device_id: 0 },
            Some(0),
        )
        .unwrap();

        // Create a batch of texts
        let texts: Vec<String> = (0..100)
            .map(|i| format!("Test text number {} for batch embedding", i))
            .collect();

        let embeddings = embedder.embed_batch(texts).unwrap();
        assert_eq!(embeddings.len(), 100);
        for emb in &embeddings {
            assert_eq!(emb.len(), 384);
        }
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_cuda_uses_optimal_batch_size() {
        if !is_cuda_available() {
            eprintln!("CUDA not available, skipping test");
            return;
        }

        let embedder = FastEmbedder::with_model_and_provider(
            ModelType::AllMiniLML6V2Q,
            ExecutionProviderType::Cuda { device_id: 0 },
            Some(0),
        )
        .unwrap();

        // CUDA should use larger batch sizes than CPU
        let optimal = embedder.provider().optimal_batch_size(384);
        assert_eq!(optimal, 1024, "CUDA should use batch size 1024 for 384-dim models");
    }
}

// ============================================================================
// Environment Variable Override Tests
// ============================================================================

#[test]
#[ignore] // Requires model download
fn test_env_var_batch_size_override() {
    // Set environment variable
    std::env::set_var("DEMONGREP_BATCH_SIZE", "512");

    let mut embedder = FastEmbedder::with_model_and_provider(
        ModelType::AllMiniLML6V2Q,
        ExecutionProviderType::Cpu,
        None,
    )
    .unwrap();

    // Generate some embeddings - env var should be used internally
    let texts = vec!["test".to_string(); 10];
    let result = embedder.embed_batch(texts);
    assert!(result.is_ok());

    // Clean up
    std::env::remove_var("DEMONGREP_BATCH_SIZE");
}

#[test]
#[ignore] // Requires model download
fn test_env_var_invalid_batch_size() {
    // Set invalid environment variable
    std::env::set_var("DEMONGREP_BATCH_SIZE", "not_a_number");

    let mut embedder = FastEmbedder::with_model_and_provider(
        ModelType::AllMiniLML6V2Q,
        ExecutionProviderType::Cpu,
        None,
    )
    .unwrap();

    // Should fall back to default (256) when env var is invalid
    let texts = vec!["test".to_string(); 10];
    let result = embedder.embed_batch(texts);
    assert!(result.is_ok());

    // Clean up
    std::env::remove_var("DEMONGREP_BATCH_SIZE");
}
