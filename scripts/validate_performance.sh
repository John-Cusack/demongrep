#!/bin/bash
# Performance validation script for GPU embedding changes
#
# Usage: ./scripts/validate_performance.sh [test_directory]
#
# This script validates that the GPU embedding batch size optimization
# provides measurable performance improvements.

set -e

echo "========================================"
echo "GPU Embedding Performance Validation"
echo "========================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we have a test corpus
TEST_DIR="${1:-./test_corpus}"
if [ ! -d "$TEST_DIR" ]; then
    echo -e "${YELLOW}Creating test corpus...${NC}"
    mkdir -p "$TEST_DIR"

    # Generate some test files
    for i in $(seq 1 100); do
        cat > "$TEST_DIR/test_$i.rs" << EOF
//! Test module $i
use std::collections::HashMap;

/// Process data with index $i
pub fn process_$i(data: &[u8]) -> Vec<u8> {
    data.iter().map(|b| b.wrapping_add($i as u8)).collect()
}

/// Helper struct for module $i
pub struct Helper$i {
    pub value: i32,
    pub name: String,
}

impl Helper$i {
    pub fn new(value: i32) -> Self {
        Self { value, name: format!("helper_{}", $i) }
    }

    pub fn process(&self, input: &str) -> String {
        format!("{}: {}", self.name, input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_$i() {
        let data = vec![1, 2, 3, 4, 5];
        let result = process_$i(&data);
        assert_eq!(result.len(), 5);
    }
}
EOF
    done
    echo -e "${GREEN}Created 100 test files${NC}"
fi

echo ""
echo "Building demongrep in release mode with CUDA support..."
cargo build --features cuda --release 2>/dev/null

DEMONGREP="./target/release/demongrep"

# Clean up any existing index
rm -rf "$TEST_DIR/.demongrep.db" 2>/dev/null || true

echo ""
echo "========================================"
echo "Test 1: CPU with default batch size"
echo "========================================"

# Index with CPU provider
START=$(date +%s.%N)
$DEMONGREP index "$TEST_DIR" --provider cpu 2>&1 | grep -E "(chunks|seconds|batch|Embedding)" || true
END=$(date +%s.%N)
CPU_DEFAULT_TIME=$(echo "$END - $START" | bc)
echo -e "${GREEN}CPU default indexing time: ${CPU_DEFAULT_TIME}s${NC}"

# Clean up
rm -rf "$TEST_DIR/.demongrep.db"

echo ""
echo "========================================"
echo "Test 2: CPU with small batch size (64)"
echo "========================================"

START=$(date +%s.%N)
$DEMONGREP index "$TEST_DIR" --provider cpu --batch-size 64 2>&1 | grep -E "(chunks|seconds|batch|Embedding)" || true
END=$(date +%s.%N)
CPU_64_TIME=$(echo "$END - $START" | bc)
echo -e "${GREEN}CPU (batch=64) indexing time: ${CPU_64_TIME}s${NC}"

rm -rf "$TEST_DIR/.demongrep.db"

echo ""
echo "========================================"
echo "Test 3: CPU with optimal batch size (256)"
echo "========================================"

START=$(date +%s.%N)
$DEMONGREP index "$TEST_DIR" --provider cpu --batch-size 256 2>&1 | grep -E "(chunks|seconds|batch|Embedding)" || true
END=$(date +%s.%N)
CPU_256_TIME=$(echo "$END - $START" | bc)
echo -e "${GREEN}CPU (batch=256) indexing time: ${CPU_256_TIME}s${NC}"

rm -rf "$TEST_DIR/.demongrep.db"

# Check for CUDA
HAS_CUDA=false
if $DEMONGREP doctor 2>&1 | grep -qi "CUDA.*Available"; then
    HAS_CUDA=true
fi

if [ "$HAS_CUDA" = true ]; then
    echo ""
    echo "========================================"
    echo "Test 4: CUDA with optimal batch size"
    echo "========================================"

    START=$(date +%s.%N)
    $DEMONGREP index "$TEST_DIR" --provider cuda 2>&1 | grep -E "(chunks|seconds|batch|Embedding)" || true
    END=$(date +%s.%N)
    CUDA_TIME=$(echo "$END - $START" | bc)
    echo -e "${GREEN}CUDA optimal indexing time: ${CUDA_TIME}s${NC}"

    rm -rf "$TEST_DIR/.demongrep.db"

    echo ""
    echo "========================================"
    echo "Test 5: CUDA with suboptimal batch size (256)"
    echo "========================================"

    START=$(date +%s.%N)
    $DEMONGREP index "$TEST_DIR" --provider cuda --batch-size 256 2>&1 | grep -E "(chunks|seconds|batch|Embedding)" || true
    END=$(date +%s.%N)
    CUDA_256_TIME=$(echo "$END - $START" | bc)
    echo -e "${GREEN}CUDA (batch=256) indexing time: ${CUDA_256_TIME}s${NC}"

    rm -rf "$TEST_DIR/.demongrep.db"

    echo ""
    echo "========================================"
    echo "Performance Summary"
    echo "========================================"
    echo ""
    echo "CPU Batch Size Comparison:"
    echo "  batch=64:  ${CPU_64_TIME}s"
    echo "  batch=256: ${CPU_256_TIME}s"
    if command -v bc &> /dev/null && [ -n "$CPU_64_TIME" ] && [ -n "$CPU_256_TIME" ]; then
        SPEEDUP=$(echo "scale=2; $CPU_64_TIME / $CPU_256_TIME" | bc 2>/dev/null || echo "N/A")
        echo -e "  ${GREEN}Speedup: ${SPEEDUP}x${NC}"
    fi
    echo ""
    echo "CPU vs CUDA Comparison:"
    echo "  CPU (optimal):  ${CPU_256_TIME}s"
    echo "  CUDA (optimal): ${CUDA_TIME}s"
    if command -v bc &> /dev/null && [ -n "$CPU_256_TIME" ] && [ -n "$CUDA_TIME" ]; then
        GPU_SPEEDUP=$(echo "scale=2; $CPU_256_TIME / $CUDA_TIME" | bc 2>/dev/null || echo "N/A")
        echo -e "  ${GREEN}GPU Speedup: ${GPU_SPEEDUP}x${NC}"
    fi
    echo ""
    echo "CUDA Batch Size Comparison:"
    echo "  batch=256 (suboptimal): ${CUDA_256_TIME}s"
    echo "  batch=1024 (optimal):   ${CUDA_TIME}s"
    if command -v bc &> /dev/null && [ -n "$CUDA_256_TIME" ] && [ -n "$CUDA_TIME" ]; then
        CUDA_SPEEDUP=$(echo "scale=2; $CUDA_256_TIME / $CUDA_TIME" | bc 2>/dev/null || echo "N/A")
        echo -e "  ${GREEN}Optimal vs Suboptimal: ${CUDA_SPEEDUP}x${NC}"
    fi
else
    echo ""
    echo -e "${YELLOW}CUDA not available, skipping GPU tests${NC}"
    echo ""
    echo "========================================"
    echo "Performance Summary (CPU only)"
    echo "========================================"
    echo ""
    echo "CPU Batch Size Comparison:"
    echo "  batch=64:  ${CPU_64_TIME}s"
    echo "  batch=256: ${CPU_256_TIME}s"
    if command -v bc &> /dev/null && [ -n "$CPU_64_TIME" ] && [ -n "$CPU_256_TIME" ]; then
        SPEEDUP=$(echo "scale=2; $CPU_64_TIME / $CPU_256_TIME" | bc 2>/dev/null || echo "N/A")
        echo -e "  ${GREEN}Speedup: ${SPEEDUP}x${NC}"
    fi
fi

echo ""
echo -e "${GREEN}Validation complete!${NC}"

# Clean up test corpus if we created it
if [ "$1" = "" ]; then
    echo ""
    echo "Cleaning up test corpus..."
    rm -rf "$TEST_DIR"
fi
