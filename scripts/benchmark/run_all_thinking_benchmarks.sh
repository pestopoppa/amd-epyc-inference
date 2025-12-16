#!/bin/bash
# Run Thinking model quality+speed benchmarks on all available thinking models
# Tests each model with baseline + optimization configurations where applicable
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
THINKING_RUBRIC="$SCRIPT_DIR/run_thinking_rubric.sh"

echo "=============================================="
echo "Thinking Model Benchmark Suite"
echo "Date: $(date)"
echo "=============================================="

# Define all thinking models with their architecture
# Format: model_path|architecture
declare -A THINKING_MODELS=(
    # Dense models (baseline only)
    ["Qwen3-4B-Thinking-Q8"]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-4B-Thinking-2507-GGUF/Qwen3-4B-Thinking-2507-Q8_0.gguf|dense"

    # MoE models (baseline + moe4)
    ["Qwen3-30B-A3B-Thinking-Q4"]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Thinking-2507-GGUF/Qwen3-30B-A3B-Thinking-2507-Q4_K_S.gguf|qwen3moe"
    ["Qwen3-30B-A3B-Thinking-Q8"]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Thinking-2507-GGUF/Qwen3-30B-A3B-Thinking-2507-Q8_0.gguf|qwen3moe"

    # DeepSeek R1 Distill (dense)
    ["DeepSeek-R1-Distill-7B"]="/mnt/raid0/llm/models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf|dense"
    ["DeepSeek-R1-Distill-14B"]="/mnt/raid0/llm/models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf|dense"
    ["DeepSeek-R1-Distill-32B"]="/mnt/raid0/llm/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf|dense"

    # QwQ (dense)
    ["QwQ-32B-Q4"]="/mnt/raid0/llm/models/QwQ-32B-Preview-Q4_K_M.gguf|dense"
)

# Order of testing (smallest to largest)
MODEL_ORDER=(
    "Qwen3-4B-Thinking-Q8"
    "DeepSeek-R1-Distill-7B"
    "DeepSeek-R1-Distill-14B"
    "Qwen3-30B-A3B-Thinking-Q4"
    "Qwen3-30B-A3B-Thinking-Q8"
    "DeepSeek-R1-Distill-32B"
    "QwQ-32B-Q4"
)

# Track which models were tested
declare -a TESTED_MODELS=()

# Run tests
for model_name in "${MODEL_ORDER[@]}"; do
    if [[ -v THINKING_MODELS[$model_name] ]]; then
        IFS='|' read -r model_path arch <<< "${THINKING_MODELS[$model_name]}"

        echo ""
        echo "######################################################"
        echo "# Testing: $model_name"
        echo "# Architecture: $arch"
        echo "######################################################"

        if [[ -f "$model_path" ]]; then
            "$THINKING_RUBRIC" "$model_path" "$model_name" "$arch"
            TESTED_MODELS+=("$model_name")
        else
            echo "SKIPPED: Model not found"
            echo "  Path: $model_path"
        fi
    fi
done

echo ""
echo "=============================================="
echo "ALL THINKING BENCHMARKS COMPLETE"
echo "=============================================="
echo ""
echo "Results in: /tmp/claude/thinking_rubric_results/"
echo ""
echo "Models tested: ${TESTED_MODELS[*]}"
echo ""
echo "Summary of all models (all configurations):"

for model_name in "${TESTED_MODELS[@]}"; do
    if [[ -v THINKING_MODELS[$model_name] ]]; then
        IFS='|' read -r _ arch <<< "${THINKING_MODELS[$model_name]}"

        # Determine configs tested
        if [[ "$arch" == "qwen3moe" ]]; then
            configs=("baseline" "moe4")
        else
            configs=("baseline")
        fi

        for cfg in "${configs[@]}"; do
            echo ""
            echo "=== $model_name ($cfg) ==="
            for f in /tmp/claude/thinking_rubric_results/${model_name}_${cfg}_*.txt 2>/dev/null; do
                if [[ -f "$f" ]]; then
                    test_name=$(basename "$f" .txt | sed "s/${model_name}_${cfg}_//")
                    speed=$(grep "eval time" "$f" 2>/dev/null | grep -oP '\d+\.\d+ tokens per second' | tail -1 || echo "N/A")
                    echo "  $test_name: $speed"
                fi
            done
        done
    fi
done

# Generate comparison table for MoE models
echo ""
echo "=============================================="
echo "MoE OPTIMIZATION COMPARISON"
echo "=============================================="
for model_name in "${TESTED_MODELS[@]}"; do
    if [[ -v THINKING_MODELS[$model_name] ]]; then
        IFS='|' read -r _ arch <<< "${THINKING_MODELS[$model_name]}"

        if [[ "$arch" == "qwen3moe" ]]; then
            echo ""
            echo "=== $model_name: baseline vs moe4 ==="
            echo "-------------------------------------------"
            printf "%-20s %-15s %-15s %-10s\n" "Test" "baseline" "moe4" "Speedup"
            echo "-------------------------------------------"

            for test in t1_q1_algorithm t1_q2_threadsafe t2_q1_dict_reuse t2_q2_cache_bug t2_q3_api_design t3_q1_dependency t3_q2_vector_clock t3_q3_type_system t3_q4_probability; do
                base_file="/tmp/claude/thinking_rubric_results/${model_name}_baseline_${test}.txt"
                moe_file="/tmp/claude/thinking_rubric_results/${model_name}_moe4_${test}.txt"

                base_speed="N/A"
                moe_speed="N/A"
                speedup="N/A"

                if [[ -f "$base_file" ]]; then
                    base_speed=$(grep "eval time" "$base_file" 2>/dev/null | grep -oP '\d+\.\d+' | tail -1 || echo "N/A")
                fi
                if [[ -f "$moe_file" ]]; then
                    moe_speed=$(grep "eval time" "$moe_file" 2>/dev/null | grep -oP '\d+\.\d+' | tail -1 || echo "N/A")
                fi

                # Calculate speedup if both values are numeric
                if [[ "$base_speed" != "N/A" ]] && [[ "$moe_speed" != "N/A" ]]; then
                    speedup=$(echo "scale=2; $moe_speed / $base_speed" | bc)
                    speedup="${speedup}x"
                fi

                printf "%-20s %-15s %-15s %-10s\n" "$test" "$base_speed" "$moe_speed" "$speedup"
            done
        fi
    fi
done
