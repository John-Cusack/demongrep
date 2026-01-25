//! Batch size configuration for embedding
//!
//! Based on benchmarks, small batch sizes (32) perform best for both CPU and GPU.
//! Users can override via config file if they want to experiment.

/// Default batch size - empirically determined to be optimal across providers
///
/// Benchmarks showed that batch_size=32 consistently outperforms larger batches:
/// - GPU (CUDA): 161.7 chunks/sec @ batch=32 vs 75.8 chunks/sec @ batch=1024
/// - CPU: 124.6 chunks/sec @ batch=32 vs 67.4 chunks/sec @ batch=256
///
/// This is counterintuitive but likely due to:
/// - Better cache utilization with smaller batches
/// - Less padding waste (shorter max sequence per batch)
/// - More efficient memory access patterns
pub const DEFAULT_BATCH_SIZE: usize = 32;

/// Get the default batch size for embedding
///
/// Returns 32, which benchmarks showed to be optimal for both CPU and GPU.
/// Users can override this via the config file's `embedding.batch_size` setting.
pub fn default_batch_size() -> usize {
    DEFAULT_BATCH_SIZE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_batch_size() {
        assert_eq!(default_batch_size(), 32);
    }
}
