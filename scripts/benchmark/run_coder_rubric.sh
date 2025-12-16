#!/bin/bash
# Coder Model Quality Rubric Test Script
# Runs all T1/T2/T3 coding questions and captures results with timing
# Supports multiple configurations: baseline, MoE reduction
set -euo pipefail

# Configuration
MODEL="${1:-}"
MODEL_NAME="${2:-unknown}"
MODEL_ARCH="${3:-dense}"  # dense, qwen3moe, qwen3vlmoe, etc.
OUTPUT_DIR="/tmp/claude/coder_rubric_results"
LLAMA_COMPLETION="/mnt/raid0/llm/llama.cpp/build/bin/llama-completion"
TIMEOUT=180

if [[ -z "$MODEL" ]]; then
    echo "Usage: $0 <model.gguf> <model_name> [arch]"
    echo ""
    echo "Architecture types:"
    echo "  dense       - Standard dense model (no MoE optimization)"
    echo "  qwen3moe    - Qwen3-MoE model (tests baseline + MoE reduction)"
    echo "  qwen3vlmoe  - Qwen3-VL-MoE model (tests baseline + MoE reduction)"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/Qwen2.5-Coder-32B-Q4.gguf Qwen2.5-Coder-32B dense"
    echo "  $0 /path/to/Qwen3-Coder-30B-A3B-Q4.gguf Qwen3-Coder-30B qwen3moe"
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
    qwen3vlmoe)
        CONFIGS=("baseline" "moe4")
        ;;
    *)
        echo "Unknown architecture: $MODEL_ARCH, using baseline only"
        CONFIGS=("baseline")
        ;;
esac

echo "=============================================="
echo "Coder Model Quality Rubric"
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
        qwen3vlmoe)
            echo "--override-kv qwen3vlmoe.expert_used_count=int:$expert_count"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Function to run a coding test and extract timing
run_coder_test() {
    local test_name="$1"
    local prompt="$2"
    local config="$3"
    local moe_override="$4"
    local output_file="$OUTPUT_DIR/${MODEL_NAME}_${config}_${test_name}.txt"

    echo ""
    echo "--- Running $test_name ($config) ---"

    # Write prompt to temp file
    echo "$prompt" > "/tmp/claude/coder_prompt.txt"

    # Run model and capture output
    timeout "$TIMEOUT" "$LLAMA_COMPLETION" \
        -m "$MODEL" \
        -t 96 -n 1024 --temp 0.3 \
        $moe_override \
        -f "/tmp/claude/coder_prompt.txt" \
        > "$output_file" 2>&1 || true

    # Extract timing
    local speed=$(grep "eval time" "$output_file" | grep -oP '\d+\.\d+ tokens per second' | tail -1 || echo "N/A")

    echo "Speed: $speed"
    echo "Output saved to: $output_file"

    # Show the code output (filter out llama.cpp noise)
    echo "--- Code Output ---"
    grep -v "^llama\|^load\|^print_info\|^common\|^sampler\|^generate\|^system_info\|^main:\|^==\|^>" "$output_file" | tail -40 | head -30
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

    run_coder_test "t1_q1_factorial" \
"Write a Python function that returns the factorial of a number n. Include type hints.
Just the function, no explanation needed." "$CONFIG" "$MOE_OVERRIDE"

    run_coder_test "t1_q2_reverse_words" \
"Write a function that reverses each word in a string but keeps word order.
Example: 'hello world' -> 'olleh dlrow'
Just the function, no explanation needed." "$CONFIG" "$MOE_OVERRIDE"

    run_coder_test "t1_q3_stack" \
"Write a Python class for a Stack with push, pop, peek, and is_empty methods.
Just the class, no explanation needed." "$CONFIG" "$MOE_OVERRIDE"

    # T2: Medium-Hard questions
    echo ""
    echo "========== TIER 2 (Medium-Hard) =========="

    run_coder_test "t2_q1_palindrome" \
"Write a function to find the longest palindromic substring in a string.
Return the first one if there are multiple of the same length.
Handle edge cases. Just the function." "$CONFIG" "$MOE_OVERRIDE"

    run_coder_test "t2_q2_async_fetch" \
"Write a Python async function that fetches multiple URLs concurrently.
Return a dict mapping URL to status code. Handle 5s timeout per request.
Use aiohttp. Just the function." "$CONFIG" "$MOE_OVERRIDE"

    run_coder_test "t2_q3_lru_cache" \
"Implement an LRU cache class with get(key) and put(key, value) methods.
Both should be O(1). Capacity is passed to __init__.
Just the class." "$CONFIG" "$MOE_OVERRIDE"

    # T3: Hard questions
    echo ""
    echo "========== TIER 3 (Hard) =========="

    run_coder_test "t3_q1_ip_addresses" \
"Write a function that finds all valid IP addresses from a string of digits.
Example: '25525511135' -> ['255.255.11.135', '255.255.111.35']
Just the function." "$CONFIG" "$MOE_OVERRIDE"

    run_coder_test "t3_q2_rate_limiter" \
"Write a thread-safe rate limiter class using the token bucket algorithm.
Allow N requests per minute. Include allow_request() -> bool and reset().
Just the class." "$CONFIG" "$MOE_OVERRIDE"

    run_coder_test "t3_q3_bug_fix" \
"Fix the bug in this merge intervals code:
\`\`\`python
def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for i in range(1, len(intervals)):
        if intervals[i][0] <= merged[-1][1]:
            merged[-1][1] = intervals[i][1]
        else:
            merged.append(intervals[i])
    return merged
\`\`\`
Explain the bug and provide the fixed code." "$CONFIG" "$MOE_OVERRIDE"

    run_coder_test "t3_q4_retry_decorator" \
"Write a Python decorator @retry(max_attempts=3, delay=1.0, backoff=2.0)
that retries a function on exception with exponential backoff.
Should work with both sync and async functions.
Just the decorator code." "$CONFIG" "$MOE_OVERRIDE"

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
echo "CODER RUBRIC TEST COMPLETE - ALL CONFIGURATIONS"
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

    for test in t1_q1_factorial t1_q2_reverse_words t1_q3_stack t2_q1_palindrome t2_q2_async_fetch t2_q3_lru_cache t3_q1_ip_addresses t3_q2_rate_limiter t3_q3_bug_fix t3_q4_retry_decorator; do
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
