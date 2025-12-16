#!/bin/bash
# Thinking Model Quality Rubric Test Script
# Runs all T1/T2/T3 questions and captures results with timing
# Supports multiple configurations: baseline, MoE reduction
set -euo pipefail

# Configuration
MODEL="${1:-}"
MODEL_NAME="${2:-unknown}"
MODEL_ARCH="${3:-dense}"  # dense, qwen3moe
OUTPUT_DIR="/tmp/claude/thinking_rubric_results"
LLAMA_COMPLETION="/mnt/raid0/llm/llama.cpp/build/bin/llama-completion"
TIMEOUT=120

# Quirk: Don't pipe output directly - causes "error: invalid argument:"
# Quirk: Model uses <think>...</think> blocks for reasoning
# Quirk: Auto-enables conversation mode (has chat template)

if [[ -z "$MODEL" ]]; then
    echo "Usage: $0 <model.gguf> <model_name> [arch]"
    echo ""
    echo "Architecture types:"
    echo "  dense       - Standard dense model (no MoE optimization)"
    echo "  qwen3moe    - Qwen3-MoE model (tests baseline + MoE reduction)"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/Qwen3-4B-Thinking.gguf Qwen3-4B-Thinking dense"
    echo "  $0 /path/to/Qwen3-30B-A3B-Thinking.gguf Qwen3-30B-Thinking qwen3moe"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Determine configurations to test based on architecture
declare -a CONFIGS
case "$MODEL_ARCH" in
    dense)
        CONFIGS=("baseline")
        ;;
    qwen3moe)
        CONFIGS=("baseline" "moe4")
        ;;
    *)
        echo "Unknown architecture: $MODEL_ARCH, using baseline only"
        CONFIGS=("baseline")
        ;;
esac

echo "=============================================="
echo "Thinking Model Quality Rubric"
echo "Model: $MODEL_NAME"
echo "Model file: $MODEL"
echo "Architecture: $MODEL_ARCH"
echo "Configurations to test: ${CONFIGS[*]}"
echo "Date: $(date)"
echo "=============================================="

# Function to get MoE override based on config and arch
get_moe_override() {
    local config="$1"
    local arch="$2"

    if [[ "$config" == "baseline" ]]; then
        echo ""
        return
    fi

    local expert_count="${config#moe}"  # Extract number from moe4, moe6, etc.

    case "$arch" in
        qwen3moe)
            echo "--override-kv qwen3moe.expert_used_count=int:$expert_count"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Function to run a test and extract timing
run_test() {
    local test_name="$1"
    local prompt="$2"
    local config="$3"
    local moe_override="$4"
    local output_file="$OUTPUT_DIR/${MODEL_NAME}_${config}_${test_name}.txt"

    echo ""
    echo "--- Running $test_name ($config) ---"

    # Write prompt to temp file (avoids shell escaping issues)
    echo "$prompt" > "/tmp/claude/rubric_prompt.txt"

    # Run model and capture output
    timeout "$TIMEOUT" "$LLAMA_COMPLETION" \
        -m "$MODEL" \
        -t 96 -n 512 --temp 0.6 \
        $moe_override \
        -f "/tmp/claude/rubric_prompt.txt" \
        > "$output_file" 2>&1 || true

    # Extract timing
    local speed=$(grep "eval time" "$output_file" | grep -oP '\d+\.\d+ tokens per second' | tail -1 || echo "N/A")

    echo "Speed: $speed"
    echo "Output saved to: $output_file"

    # Show the answer (after </think> if present)
    echo "--- Answer ---"
    if grep -q "</think>" "$output_file"; then
        sed -n '/<\/think>/,/EOF by user/p' "$output_file" | head -20
    else
        tail -30 "$output_file" | head -20
    fi
}

# Run tests for each configuration
for CONFIG in "${CONFIGS[@]}"; do
    MOE_OVERRIDE=$(get_moe_override "$CONFIG" "$MODEL_ARCH")

    echo ""
    echo "##############################################"
    echo "# Configuration: $CONFIG"
    [[ -n "$MOE_OVERRIDE" ]] && echo "# Override: $MOE_OVERRIDE"
    echo "##############################################"

    # T1: Baseline questions
    echo ""
    echo "========== TIER 1 (Baseline) =========="

    run_test "t1_q1_algorithm" "Sort 10 mostly-sorted items. Quicksort or insertion sort? One sentence." "$CONFIG" "$MOE_OVERRIDE"

    run_test "t1_q2_threadsafe" "Is self.count += 1 thread-safe in Python? One sentence." "$CONFIG" "$MOE_OVERRIDE"

    # T2: Medium-Hard questions
    echo ""
    echo "========== TIER 2 (Medium-Hard) =========="

    run_test "t2_q1_dict_reuse" "Python function called 1000x/sec creates a 1KB dict each call.
Better to: A) Pre-allocate global dict and clear() each time, or B) Create new dict each time?
Explain memory and performance implications in 2-3 sentences." "$CONFIG" "$MOE_OVERRIDE"

    run_test "t2_q2_cache_bug" "Find the bug in this cache:

cache = {}
lock = threading.Lock()

def get_cached(key, compute_fn):
    if key in cache:
        return cache[key]
    with lock:
        result = compute_fn()
        cache[key] = result
        return result

One paragraph explanation." "$CONFIG" "$MOE_OVERRIDE"

    run_test "t2_q3_api_design" "For a library function that might fail, which return style is best?
A) return (success: bool, data, error_msg)
B) return {\"success\": bool, \"data\": ..., \"error\": ...}
C) raise Exception on error, return data on success
Brief reasoning for a Python library consumed by external users." "$CONFIG" "$MOE_OVERRIDE"

    # T3: Very Hard questions
    echo ""
    echo "========== TIER 3 (Very Hard) =========="

    run_test "t3_q1_dependency" "Service startup constraints:
- A must start before B
- B must start before C or D (either one)
- C and D cannot start simultaneously (resource conflict)
- E requires both C AND D to be running

What is the minimum number of sequential startup phases?
List the phases." "$CONFIG" "$MOE_OVERRIDE"

    run_test "t3_q2_vector_clock" "Vector clock merge:
P1 has clock [2,0,1], P2 has clock [1,2,0], P3 has clock [0,0,2].
1) P1 sends message to P2. What is P2's clock after receiving?
2) P2 then sends to P3. What is P3's clock after receiving?
Show work." "$CONFIG" "$MOE_OVERRIDE"

    run_test "t3_q3_type_system" "Consider this function:

from typing import Sequence, TypeVar
T = TypeVar('T')

def first_or_default(items: Sequence[T], default: T) -> T:
    return items[0] if items else default

What happens with: first_or_default([], None)?
Is the type signature correct? What is the subtle issue?" "$CONFIG" "$MOE_OVERRIDE"

    run_test "t3_q4_probability" "Load balancer randomly routes to 3 servers with equal probability.
Server latencies: S1=10ms, S2=50ms, S3=100ms.
What is the expected MEDIAN latency over many requests?" "$CONFIG" "$MOE_OVERRIDE"

    # Configuration Summary
    echo ""
    echo "=============================================="
    echo "CONFIG $CONFIG COMPLETE"
    echo "=============================================="
    echo "Speed summary for $MODEL_NAME ($CONFIG):"
    for f in "$OUTPUT_DIR"/${MODEL_NAME}_${CONFIG}_*.txt; do
        if [[ -f "$f" ]]; then
            test_name=$(basename "$f" .txt | sed "s/${MODEL_NAME}_${CONFIG}_//")
            speed=$(grep "eval time" "$f" 2>/dev/null | grep -oP '\d+\.\d+ tokens per second' | tail -1 || echo "N/A")
            echo "  $test_name: $speed"
        fi
    done
done

# Final Summary - Compare configurations
echo ""
echo "=============================================="
echo "THINKING RUBRIC TEST COMPLETE - ALL CONFIGURATIONS"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""

# Create comparison table if multiple configs
if [[ ${#CONFIGS[@]} -gt 1 ]]; then
    echo "Speed Comparison (tokens/second):"
    echo "-------------------------------------------"
    printf "%-20s" "Test"
    for cfg in "${CONFIGS[@]}"; do
        printf "%-15s" "$cfg"
    done
    echo ""
    echo "-------------------------------------------"

    for test in t1_q1_algorithm t1_q2_threadsafe t2_q1_dict_reuse t2_q2_cache_bug t2_q3_api_design t3_q1_dependency t3_q2_vector_clock t3_q3_type_system t3_q4_probability; do
        printf "%-20s" "$test"
        for cfg in "${CONFIGS[@]}"; do
            f="$OUTPUT_DIR/${MODEL_NAME}_${cfg}_${test}.txt"
            if [[ -f "$f" ]]; then
                speed=$(grep "eval time" "$f" 2>/dev/null | grep -oP '\d+\.\d+' | tail -1 || echo "N/A")
                printf "%-15s" "$speed"
            else
                printf "%-15s" "N/A"
            fi
        done
        echo ""
    done
fi
