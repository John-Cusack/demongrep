use demongrep::embed::{detect_best_provider, ExecutionProviderType, FastEmbedder, ModelType};

#[test]
fn test_execution_provider_from_str() {
    // Test valid providers
    assert_eq!(
        ExecutionProviderType::from_str("cpu", 0).unwrap(),
        ExecutionProviderType::Cpu
    );
    assert_eq!(
        ExecutionProviderType::from_str("auto", 0).unwrap(),
        ExecutionProviderType::Auto
    );

    // Test case insensitive
    assert_eq!(
        ExecutionProviderType::from_str("CPU", 0).unwrap(),
        ExecutionProviderType::Cpu
    );
    assert_eq!(
        ExecutionProviderType::from_str("Auto", 0).unwrap(),
        ExecutionProviderType::Auto
    );

    // Test invalid provider
    assert!(ExecutionProviderType::from_str("invalid", 0).is_err());
}

#[test]
fn test_optimal_batch_size_by_provider() {
    // Test CPU batch sizes
    assert_eq!(ExecutionProviderType::Cpu.optimal_batch_size(384), 256);
    assert_eq!(ExecutionProviderType::Cpu.optimal_batch_size(768), 128);
    assert_eq!(ExecutionProviderType::Cpu.optimal_batch_size(1024), 64);

    // Test CUDA batch sizes (when feature is enabled)
    #[cfg(feature = "cuda")]
    {
        assert_eq!(
            ExecutionProviderType::Cuda { device_id: 0 }.optimal_batch_size(384),
            1024
        );
        assert_eq!(
            ExecutionProviderType::Cuda { device_id: 0 }.optimal_batch_size(768),
            512
        );
        assert_eq!(
            ExecutionProviderType::Cuda { device_id: 0 }.optimal_batch_size(1024),
            256
        );
    }

    // Test Auto batch sizes
    assert_eq!(ExecutionProviderType::Auto.optimal_batch_size(384), 1024);
    assert_eq!(ExecutionProviderType::Auto.optimal_batch_size(768), 512);
    assert_eq!(ExecutionProviderType::Auto.optimal_batch_size(1024), 256);
}

#[test]
fn test_detect_best_provider_cpu_only() {
    // This test assumes no GPU is available
    // The detect_best_provider function should return CPU
    let provider = detect_best_provider();

    // On systems without GPU features, it should be CPU
    match provider {
        ExecutionProviderType::Cpu => {} // Expected
        _ => panic!("Expected CPU provider, got {:?}", provider),
    }
}

#[test]
fn test_auto_fallback_to_cpu() {
    // Test that auto mode gracefully falls back to CPU
    // This is the default behavior
    let model_type = ModelType::default();
    let embedder =
        FastEmbedder::with_model_and_provider(model_type, ExecutionProviderType::Auto, None);

    // Should succeed even if no GPU is available
    assert!(embedder.is_ok(), "Auto provider should fall back to CPU");
}

// This test is ignored by default and should only run on systems with CUDA
#[test]
#[ignore]
#[cfg(feature = "cuda")]
fn test_cuda_embedding() {
    // Skip test if CUDA is not available
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
    let embedding = embedder.embed_one("Test text for CUDA embedding").unwrap();
    assert!(!embedding.is_empty(), "Embedding should not be empty");
}
