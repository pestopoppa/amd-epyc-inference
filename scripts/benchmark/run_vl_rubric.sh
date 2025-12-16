#!/bin/bash
# VL Model Quality Rubric Test Script
# Runs all T1/T2/T3 visual questions and captures results with timing
# Supports multiple configurations: baseline, MoE reduction
set -euo pipefail

# Configuration
MODEL="${1:-}"
MMPROJ="${2:-}"
MODEL_NAME="${3:-unknown}"
MODEL_ARCH="${4:-dense}"  # dense, qwen3vlmoe
OUTPUT_DIR="/tmp/claude/vl_rubric_results"
IMAGE_DIR="/mnt/raid0/llm/claude/test_images/vl_rubric"
LLAMA_MTMD="/mnt/raid0/llm/llama.cpp/build/bin/llama-mtmd-cli"
TIMEOUT=120

if [[ -z "$MODEL" ]] || [[ -z "$MMPROJ" ]]; then
    echo "Usage: $0 <model.gguf> <mmproj.gguf> <model_name> [arch]"
    echo ""
    echo "Architecture types:"
    echo "  dense       - Standard dense model (no MoE optimization)"
    echo "  qwen3vlmoe  - Qwen3-VL-MoE model (tests baseline + MoE reduction)"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/Qwen2.5-VL-7B.gguf /path/to/mmproj.gguf Qwen2.5-VL-7B dense"
    echo "  $0 /path/to/Qwen3-VL-30B-A3B.gguf /path/to/mmproj.gguf Qwen3-VL-30B qwen3vlmoe"
    echo ""
    echo "Available VL models:"
    find /mnt/raid0/llm -name "*VL*.gguf" -o -name "*vl*.gguf" 2>/dev/null | grep -v mmproj | head -20
    echo ""
    echo "Available mmproj files:"
    find /mnt/raid0/llm -name "*mmproj*.gguf" 2>/dev/null | head -20
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Determine configurations to test based on architecture
declare -a CONFIGS
case "$MODEL_ARCH" in
    dense)
        CONFIGS=("baseline")
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
echo "VL Model Quality Rubric"
echo "Model: $MODEL_NAME"
echo "Model file: $MODEL"
echo "MMProj: $MMPROJ"
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
        qwen3vlmoe)
            echo "--override-kv qwen3vlmoe.expert_used_count=int:$expert_count"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Function to run a VL test and extract timing
run_vl_test() {
    local test_name="$1"
    local image="$2"
    local prompt="$3"
    local config="$4"
    local moe_override="$5"
    local output_file="$OUTPUT_DIR/${MODEL_NAME}_${config}_${test_name}.txt"

    echo ""
    echo "--- Running $test_name ($config) ---"
    echo "Image: $image"
    echo "Prompt: $prompt"

    # Run model and capture output
    timeout "$TIMEOUT" "$LLAMA_MTMD" \
        -m "$MODEL" \
        --mmproj "$MMPROJ" \
        --image "$image" \
        -t 96 -n 256 --temp 0.3 \
        $moe_override \
        -p "$prompt" \
        > "$output_file" 2>&1 || true

    # Extract timing
    local speed=$(grep "eval time" "$output_file" | grep -oP '\d+\.\d+ tokens per second' | tail -1 || echo "N/A")

    echo "Speed: $speed"
    echo "Output saved to: $output_file"

    # Show the answer (last 20 lines before timing info)
    echo "--- Answer ---"
    grep -v "^llama\|^load\|^print_info\|^common\|^sampler\|^generate\|^system_info\|^main:\|^==" "$output_file" | tail -20 | head -15
}

# Verify images exist
if [[ ! -d "$IMAGE_DIR" ]]; then
    echo "ERROR: Test images not found at $IMAGE_DIR"
    echo "Run: python scripts/benchmark/generate_vl_test_images.py"
    exit 1
fi

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

    run_vl_test "t1_q1_ocr" \
        "$IMAGE_DIR/text_simple.png" \
        "What text is shown in this image? Just give the exact text." \
        "$CONFIG" "$MOE_OVERRIDE"

    run_vl_test "t1_q2_shapes" \
        "$IMAGE_DIR/shapes_basic.png" \
        "How many shapes are in this image? List them with colors." \
        "$CONFIG" "$MOE_OVERRIDE"

    run_vl_test "t1_q3_icon" \
        "$IMAGE_DIR/icon_folder.png" \
        "What does this icon represent?" \
        "$CONFIG" "$MOE_OVERRIDE"

    # T2: Medium-Hard questions
    echo ""
    echo "========== TIER 2 (Medium-Hard) =========="

    run_vl_test "t2_q1_chart" \
        "$IMAGE_DIR/chart_bar.png" \
        "What is the value of bar B? Which bar has the highest value?" \
        "$CONFIG" "$MOE_OVERRIDE"

    run_vl_test "t2_q2_invoice" \
        "$IMAGE_DIR/doc_invoice.png" \
        "Extract the total amount from this invoice." \
        "$CONFIG" "$MOE_OVERRIDE"

    run_vl_test "t2_q3_code" \
        "$IMAGE_DIR/code_python.png" \
        "What does this code do? Is there a bug?" \
        "$CONFIG" "$MOE_OVERRIDE"

    # T3: Hard questions
    echo ""
    echo "========== TIER 3 (Hard) =========="

    run_vl_test "t3_q1_math" \
        "$IMAGE_DIR/math_equation.png" \
        "Solve the equation shown in the image. Show your work." \
        "$CONFIG" "$MOE_OVERRIDE"

    run_vl_test "t3_q2_flowchart" \
        "$IMAGE_DIR/diagram_flowchart.png" \
        "Trace the path if input > 10 and flag = true. What path do you take?" \
        "$CONFIG" "$MOE_OVERRIDE"

    run_vl_test "t3_q3_diff" \
        "$IMAGE_DIR/diff_images.png" \
        "Find all differences between Image A and Image B." \
        "$CONFIG" "$MOE_OVERRIDE"

    run_vl_test "t3_q4_puzzle" \
        "$IMAGE_DIR/puzzle_grid.png" \
        "What shape should go in the empty cell marked with '?'? Explain the pattern." \
        "$CONFIG" "$MOE_OVERRIDE"

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
echo "VL RUBRIC TEST COMPLETE - ALL CONFIGURATIONS"
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

    for test in t1_q1_ocr t1_q2_shapes t1_q3_icon t2_q1_chart t2_q2_invoice t2_q3_code t3_q1_math t3_q2_flowchart t3_q3_diff t3_q4_puzzle; do
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
