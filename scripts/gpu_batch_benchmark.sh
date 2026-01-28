#!/bin/bash
# Quick GPU batch size benchmark
# Tests different batch sizes on CUDA to find optimal

set -e

echo "========================================"
echo "GPU Batch Size Benchmark"
echo "========================================"
echo ""

# Check VRAM
echo "GPU Memory:"
nvidia-smi --query-gpu=memory.free,memory.total,name --format=csv -i 0
echo ""

DEMONGREP="./target/release/demongrep"

# Create test corpus if needed
TEST_DIR="./test_corpus_gpu"
if [ ! -d "$TEST_DIR" ]; then
    echo "Creating test corpus (200 files)..."
    mkdir -p "$TEST_DIR"
    for i in $(seq 1 200); do
        cat > "$TEST_DIR/test_$i.rs" << EOF
//! Module $i - handles processing for component $i
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for processor $i
#[derive(Debug, Clone)]
pub struct Config$i {
    pub max_retries: usize,
    pub timeout_ms: u64,
    pub buffer_size: usize,
}

impl Default for Config$i {
    fn default() -> Self {
        Self {
            max_retries: 3,
            timeout_ms: 5000,
            buffer_size: 1024,
        }
    }
}

/// Process data with index $i
pub async fn process_$i(data: &[u8], config: &Config$i) -> Result<Vec<u8>, Error> {
    let mut output = Vec::with_capacity(data.len());
    for (idx, byte) in data.iter().enumerate() {
        if idx % config.buffer_size == 0 {
            tokio::task::yield_now().await;
        }
        output.push(byte.wrapping_add($i as u8));
    }
    Ok(output)
}

/// Helper struct for module $i
pub struct Helper$i {
    pub value: i32,
    pub name: String,
    cache: HashMap<String, Vec<u8>>,
}

impl Helper$i {
    pub fn new(value: i32) -> Self {
        Self {
            value,
            name: format!("helper_{}", $i),
            cache: HashMap::new(),
        }
    }

    pub fn process(&mut self, input: &str) -> String {
        let key = input.to_string();
        if !self.cache.contains_key(&key) {
            self.cache.insert(key.clone(), input.as_bytes().to_vec());
        }
        format!("{}: {} (cached: {})", self.name, input, self.cache.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_process_$i() {
        let config = Config$i::default();
        let data = vec![1, 2, 3, 4, 5];
        let result = process_$i(&data, &config).await.unwrap();
        assert_eq!(result.len(), 5);
    }
}
EOF
    done
    echo "Created 200 test files"
fi

echo ""

# Function to run benchmark
run_benchmark() {
    local batch_size=$1
    local provider=$2

    rm -rf "$TEST_DIR/.demongrep.db" 2>/dev/null || true

    # Time the indexing
    local start=$(date +%s.%N)
    $DEMONGREP index "$TEST_DIR" --provider "$provider" --batch-size "$batch_size" 2>&1 | grep -E "(chunks|Embedding|batch)" || true
    local end=$(date +%s.%N)
    local duration=$(echo "$end - $start" | bc)

    echo "  Time: ${duration}s"
    echo ""
}

echo "========================================"
echo "CUDA Benchmarks (384-dim model)"
echo "========================================"

for batch in 32 64 128 256 512 1024 2048; do
    echo ""
    echo "--- Batch size: $batch ---"
    run_benchmark $batch cuda
done

echo ""
echo "========================================"
echo "CPU Baseline (for comparison)"
echo "========================================"

for batch in 32 64 256; do
    echo ""
    echo "--- CPU Batch size: $batch ---"
    run_benchmark $batch cpu
done

echo ""
echo "========================================"
echo "Benchmark Complete"
echo "========================================"

# Cleanup
rm -rf "$TEST_DIR"
