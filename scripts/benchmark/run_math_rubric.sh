#!/bin/bash
# Math Model Quality Rubric Test Script
# Runs all T1/T2/T3 mathematical reasoning questions
# Supports multiple configurations: baseline, MoE reduction
set -euo pipefail

# Configuration
MODEL="${1:-}"
MODEL_NAME="${2:-unknown}"
MODEL_ARCH="${3:-dense}"  # dense, qwen3moe
OUTPUT_DIR="/tmp/claude/math_rubric_results"
LLAMA_COMPLETION="/mnt/raid0/llm/llama.cpp/build/bin/llama-completion"
TIMEOUT=180  # Math problems may need more time for reasoning

if [[ -z "$MODEL" ]]; then
    echo "Usage: $0 <model.gguf> <model_name> [arch]"
    echo ""
    echo "Architecture types:"
    echo "  dense       - Standard dense model (no MoE optimization)"
    echo "  qwen3moe    - Qwen3-MoE model (tests baseline + MoE reduction)"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/Qwen2.5-Math-7B-Instruct.gguf Qwen2.5-Math-7B dense"
    echo "  $0 /path/to/Qwen3-30B-A3B-Instruct.gguf Qwen3-30B qwen3moe"
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
echo "Math Model Quality Rubric"
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

    local expert_count="${config#moe}"

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

    # Write prompt to temp file
    echo "$prompt" > "/tmp/claude/math_prompt.txt"

    # Run model and capture output (longer timeout for math)
    timeout "$TIMEOUT" "$LLAMA_COMPLETION" \
        -m "$MODEL" \
        -t 96 -n 1024 --temp 0.1 \
        $moe_override \
        -f "/tmp/claude/math_prompt.txt" \
        > "$output_file" 2>&1 || true

    # Extract timing
    local speed=$(grep "eval time" "$output_file" | grep -oP '\d+\.\d+ tokens per second' | tail -1 || echo "N/A")

    echo "Speed: $speed"
    echo "Output saved to: $output_file"

    # Show the answer
    echo "--- Answer ---"
    grep -v "^llama\|^load\|^print_info\|^common\|^sampler\|^generate\|^system_info\|^main:" "$output_file" | tail -30 | head -25
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

    run_test "t1_q1_arithmetic" \
"Calculate: 847 × 23 + 156 ÷ 4 - 89

Show your work step by step, then give the final answer." "$CONFIG" "$MOE_OVERRIDE"

    run_test "t1_q2_algebra" \
"Solve for x: 3x + 7 = 22

Show each step." "$CONFIG" "$MOE_OVERRIDE"

    run_test "t1_q3_conversion" \
"Convert 2.5 kilometers to:
1. meters
2. centimeters
3. miles (use 1 mile = 1.609 km)

Show the conversion calculation for each." "$CONFIG" "$MOE_OVERRIDE"

    # T2: Medium-Hard questions
    echo ""
    echo "========== TIER 2 (Medium-Hard) =========="

    run_test "t2_q1_word_problem" \
"A store offers a 20% discount on all items. After the discount, a 8% sales tax is applied. If an item originally costs \$150:

1. What is the price after discount?
2. What is the final price after tax?
3. What percentage of the original price is the final price?

Show your work for each part." "$CONFIG" "$MOE_OVERRIDE"

    run_test "t2_q2_system_equations" \
"Solve the system of equations:
2x + 3y = 13
4x - y = 5

Find the values of x and y. Show all steps." "$CONFIG" "$MOE_OVERRIDE"

    run_test "t2_q3_probability" \
"A bag contains 5 red balls, 3 blue balls, and 2 green balls.
If you draw 2 balls without replacement:

1. What is the probability both are red?
2. What is the probability of getting one red and one blue (in any order)?

Show your probability calculations." "$CONFIG" "$MOE_OVERRIDE"

    # T3: Hard questions
    echo ""
    echo "========== TIER 3 (Hard) =========="

    run_test "t3_q1_optimization" \
"A farmer has 200 meters of fencing to enclose a rectangular field that borders a river (no fencing needed on the river side).

What dimensions maximize the enclosed area? What is the maximum area?

Set up the optimization problem and solve it." "$CONFIG" "$MOE_OVERRIDE"

    run_test "t3_q2_proof" \
"Prove that the sum of the first n positive integers is n(n+1)/2.

Use mathematical induction. Show:
1. Base case
2. Inductive hypothesis
3. Inductive step" "$CONFIG" "$MOE_OVERRIDE"

    run_test "t3_q3_calculus" \
"Evaluate the definite integral:
∫₀² (3x² + 2x - 1) dx

Show your work:
1. Find the antiderivative
2. Evaluate at the bounds
3. Calculate the final answer" "$CONFIG" "$MOE_OVERRIDE"

    run_test "t3_q4_statistics" \
"Given the dataset: 12, 15, 18, 22, 25, 28, 30, 35

Calculate:
1. Mean
2. Median
3. Standard deviation (population)
4. Is there any outlier using the 1.5×IQR rule?

Show all calculations." "$CONFIG" "$MOE_OVERRIDE"

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

# Final Summary
echo ""
echo "=============================================="
echo "MATH RUBRIC TEST COMPLETE - ALL CONFIGURATIONS"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"
