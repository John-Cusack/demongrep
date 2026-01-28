#!/bin/bash
# Test whether sorting chunks by length improves embedding performance

set -e

echo "========================================"
echo "Sorted vs Unsorted Batching Test"
echo "========================================"
echo ""

DEMONGREP="./target/release/demongrep"

# Create test corpus with varying chunk sizes
TEST_DIR="./test_corpus_sort"
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

echo "Creating test corpus with varying file sizes..."

# Create SHORT files (small chunks)
for i in $(seq 1 50); do
    cat > "$TEST_DIR/short_$i.rs" << EOF
fn short_$i() -> i32 { $i }
EOF
done

# Create MEDIUM files
for i in $(seq 1 50); do
    cat > "$TEST_DIR/medium_$i.rs" << EOF
/// Medium function $i with some documentation
pub fn medium_process_$i(input: &str) -> Result<String, Error> {
    let validated = validate_input(input)?;
    let transformed = transform_data(validated)?;
    Ok(transformed.to_string())
}
EOF
done

# Create LONG files
for i in $(seq 1 50); do
    cat > "$TEST_DIR/long_$i.rs" << EOF
/// Long module $i with extensive implementation
/// This module handles complex processing logic
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct LongProcessor$i {
    pub config: ProcessorConfig,
    pub cache: HashMap<String, Vec<u8>>,
    pub metrics: ProcessorMetrics,
}

impl LongProcessor$i {
    pub fn new(config: ProcessorConfig) -> Self {
        Self {
            config,
            cache: HashMap::new(),
            metrics: ProcessorMetrics::default(),
        }
    }

    pub async fn process(&mut self, input: &[u8]) -> Result<Vec<u8>, ProcessError> {
        self.metrics.increment_calls();
        let key = self.compute_cache_key(input);

        if let Some(cached) = self.cache.get(&key) {
            self.metrics.increment_cache_hits();
            return Ok(cached.clone());
        }

        let result = self.do_expensive_computation(input).await?;
        self.cache.insert(key, result.clone());
        Ok(result)
    }

    fn compute_cache_key(&self, input: &[u8]) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(input);
        format!("{:x}", hasher.finalize())
    }

    async fn do_expensive_computation(&self, input: &[u8]) -> Result<Vec<u8>, ProcessError> {
        let mut output = Vec::with_capacity(input.len() * 2);
        for (idx, byte) in input.iter().enumerate() {
            if idx % self.config.checkpoint_interval == 0 {
                tokio::task::yield_now().await;
            }
            output.push(byte.wrapping_add(1));
            output.push(byte.wrapping_sub(1));
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_long_processor_$i() {
        let config = ProcessorConfig::default();
        let mut processor = LongProcessor$i::new(config);
        let input = vec![1, 2, 3, 4, 5];
        let result = processor.process(&input).await.unwrap();
        assert_eq!(result.len(), 10);
    }
}
EOF
done

echo "Created 150 files (50 short, 50 medium, 50 long)"
echo ""

# Function to run test
run_test() {
    local batch_size=$1
    local label=$2

    rm -rf "$TEST_DIR/.demongrep.db" 2>/dev/null || true

    local start=$(date +%s.%N)
    $DEMONGREP index "$TEST_DIR" --provider cuda --batch-size "$batch_size" 2>&1 | grep -E "(chunks|Embedded)" || true
    local end=$(date +%s.%N)
    local duration=$(echo "$end - $start" | bc)

    echo "  [$label] batch=$batch_size: ${duration}s"
}

echo "========================================"
echo "Test 1: Files in creation order (mixed sizes)"
echo "========================================"
# Files are named short_*, medium_*, long_* so ls will group them
# but within each group they're ordered 1-50

run_test 32 "mixed"
run_test 64 "mixed"
run_test 128 "mixed"
run_test 256 "mixed"

echo ""
echo "========================================"
echo "Test 2: Rename files to force sorted order (short->long)"
echo "========================================"

# Rename to force alphabetical = size order
cd "$TEST_DIR"
for i in $(seq 1 50); do
    mv "short_$i.rs" "a_short_$(printf '%03d' $i).rs" 2>/dev/null || true
    mv "medium_$i.rs" "b_medium_$(printf '%03d' $i).rs" 2>/dev/null || true
    mv "long_$i.rs" "c_long_$(printf '%03d' $i).rs" 2>/dev/null || true
done
cd - > /dev/null

run_test 32 "sorted"
run_test 64 "sorted"
run_test 128 "sorted"
run_test 256 "sorted"

echo ""
echo "========================================"
echo "Test 3: Reverse order (long->short)"
echo "========================================"

cd "$TEST_DIR"
for f in a_short_*.rs; do
    [ -f "$f" ] && mv "$f" "z_${f#a_}" 2>/dev/null || true
done
for f in b_medium_*.rs; do
    [ -f "$f" ] && mv "$f" "y_${f#b_}" 2>/dev/null || true
done
for f in c_long_*.rs; do
    [ -f "$f" ] && mv "$f" "x_${f#c_}" 2>/dev/null || true
done
cd - > /dev/null

run_test 32 "reverse"
run_test 64 "reverse"
run_test 128 "reverse"
run_test 256 "reverse"

echo ""
echo "========================================"
echo "Test Complete"
echo "========================================"

# Cleanup
rm -rf "$TEST_DIR"
