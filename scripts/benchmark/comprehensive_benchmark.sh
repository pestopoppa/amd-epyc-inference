#!/bin/bash
set -euo pipefail

# Comprehensive Benchmark Script for AMD EPYC 9655
# Tests: Baseline, Lookup, External Draft, Hard Mask (MoE)
# Skips already-completed tests

LLAMA_DIR="/mnt/raid0/llm/llama.cpp/build/bin"
LOG_DIR="/mnt/raid0/llm/LOGS/benchmarks"
RESULTS_CSV="$LOG_DIR/comprehensive_results_$(date +%Y%m%d_%H%M%S).csv"
EXISTING_CSV="$LOG_DIR/optimization_results_20251215_045816.csv"

mkdir -p "$LOG_DIR"

# Initialize CSV
echo "timestamp,model,method,draft,prompt_type,speed_tps,accept_pct,n_accept,notes" > "$RESULTS_CSV"

# Model paths
declare -A MODELS=(
    # Dense 32B
    ["Qwen2.5-Coder-32B"]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-32B-GGUF/Qwen2.5-Coder-32B-Q4_K_M.gguf"
    ["DeepSeek-R1-32B"]="/mnt/raid0/llm/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf"
    ["Qwen3-32B"]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-32B-GGUF/Qwen3-32B-Q4_K_M.gguf"
    # Dense 70B+
    ["Meta-Llama-3.1-70B-Instruct"]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf"
    ["Meta-Llama-3-70B-Instruct"]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF/Meta-Llama-3-70B-Instruct-Q4_K_M.gguf"
    ["Hermes-4-70B"]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Hermes-4-70B-GGUF/Hermes-4-70B-Q4_K_M.gguf"
    ["DeepSeek-R1-Llama-70B"]="/mnt/raid0/llm/lmstudio/models/unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF/DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf"
    ["Qwen2.5-72B"]="/mnt/raid0/llm/lmstudio/models/mradermacher/Qwen2.5-72B-GGUF/Qwen2.5-72B.Q4_K_M.gguf"
    ["Qwen2.5-72B-Instruct"]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-72B-Instruct-GGUF/Qwen2.5-72B-Instruct-Q4_K_M.gguf"
    ["Qwen2.5-Math-72B"]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Math-72B-Instruct-GGUF/Qwen2.5-Math-72B-Instruct-Q4_K_M.gguf"
    ["Gemma-3-27B"]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/gemma-3-27B-it-qat-GGUF/gemma-3-27B-it-QAT-Q4_0.gguf"
    # MoE 30B
    ["Qwen3-VL-30B-A3B"]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf"
    ["Qwen3-Coder-30B-A3B"]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
    # MoE 80B+
    ["Qwen3-Next-80B-A3B"]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf"
    ["Qwen3-235B-A22B"]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-Q4_K_M-00001-of-00004.gguf"
    # Large MoE (multi-file)
    ["GLM-4.6-355B"]="/mnt/raid0/llm/lmstudio/models/unsloth/GLM-4.6-GGUF/GLM-4.6-Q4_K_S-00001-of-00005.gguf"
    ["Qwen3-Coder-480B-A35B"]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00008.gguf"
    ["Qwen3-VL-235B-A22B-Thinking"]="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3-VL-235B-A22B-Thinking-GGUF/Qwen3-VL-235B-A22B-Thinking-Q4_K_S-00001-of-00003.gguf"
)

# Draft models
DRAFT_QWEN_05B="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q8_0.gguf"
DRAFT_QWEN3_06B="/mnt/raid0/llm/models/Qwen_Qwen3-0.6B-Q8_0.gguf"
DRAFT_QWEN3_VL_2B="/mnt/raid0/llm/lmstudio/models/mradermacher/Qwen3_VL_2B-GGUF/Qwen3_VL_2B.Q4_K_M.gguf"

# MoE models (for hard mask)
MOE_MODELS=("Qwen3-VL-30B-A3B" "Qwen3-Coder-30B-A3B" "Qwen3-Next-80B-A3B" "Qwen3-235B-A22B" "GLM-4.6-355B" "Qwen3-Coder-480B-A35B" "Qwen3-VL-235B-A22B-Thinking")

# MoE architecture prefixes for --override-kv
declare -A MOE_ARCH_PREFIX=(
    ["Qwen3-VL-30B-A3B"]="qwen3vlmoe"
    ["Qwen3-Coder-30B-A3B"]="qwen3moe"
    ["Qwen3-Next-80B-A3B"]="qwen3next"
    ["Qwen3-235B-A22B"]="qwen3moe"
    ["GLM-4.6-355B"]="glm4"
    ["Qwen3-Coder-480B-A35B"]="qwen3moe"
    ["Qwen3-VL-235B-A22B-Thinking"]="qwen3vlmoe"
)

# Test prompts
PROMPT_SUMMARIZE="Summarize the following text in 3 bullet points:\n\nThe AMD EPYC 9655 Turin processor represents a significant advancement in server CPU technology. Built on the Zen 5 architecture, it features 96 cores and 192 threads, offering unprecedented parallel processing capabilities. The processor supports DDR5-5600 memory across 12 channels, providing approximately 460 GB/s of memory bandwidth. One of the key improvements in Zen 5 is the true 512-bit AVX-512 implementation, which is not double-pumped like previous generations. This makes it particularly effective for AI inference workloads that can leverage wide vector operations. The processor is manufactured on TSMC's 4nm process node, offering improved power efficiency compared to previous generations."

PROMPT_CODE="Write a Python function that implements binary search on a sorted array. Include docstring and type hints."

PROMPT_EDIT="Edit this code to add error handling:\ndef divide(a, b):\n    return a / b\n\nAdd try-except block, handle ZeroDivisionError, and return None on error."

# Check if test already completed
check_existing() {
    local model="$1"
    local method="$2"
    local prompt_type="$3"

    if [[ -f "$EXISTING_CSV" ]]; then
        # Check if we have a non-zero, non-skipped result
        if grep -q ",$model,$method,.*,$prompt_type,success," "$EXISTING_CSV" 2>/dev/null; then
            local speed=$(grep ",$model,$method,.*,$prompt_type,success," "$EXISTING_CSV" | tail -1 | cut -d',' -f7)
            if [[ "$speed" != "0" ]] && [[ -n "$speed" ]]; then
                echo "SKIP: $model/$method/$prompt_type already done (${speed} t/s)"
                return 0
            fi
        fi
    fi
    return 1
}

# Run baseline benchmark
run_baseline() {
    local name="$1"
    local model_path="$2"

    if ! [[ -f "$model_path" ]]; then
        echo "SKIP: $name - model file not found"
        return
    fi

    echo "=== Baseline: $name ==="

    local output
    output=$(OMP_NUM_THREADS=1 numactl --interleave=all \
        "$LLAMA_DIR/llama-bench" \
        -m "$model_path" \
        -t 96 -p 512 -n 128 -r 3 2>&1) || true

    # Parse tg128 speed from markdown table output
    local tg_speed=$(echo "$output" | grep "tg128" | sed 's/.*tg128[^|]*|[[:space:]]*//' | awk '{print $1}')

    if [[ -n "$tg_speed" ]] && [[ "$tg_speed" != "0" ]]; then
        echo "$(date +%Y-%m-%d\ %H:%M:%S),$name,baseline,none,tg128,$tg_speed,0,0,ok" >> "$RESULTS_CSV"
        echo "  -> $tg_speed t/s"
    else
        echo "$(date +%Y-%m-%d\ %H:%M:%S),$name,baseline,none,tg128,0,0,0,parse_error" >> "$RESULTS_CSV"
        echo "  -> PARSE ERROR"
        echo "$output" | tail -10
    fi
}

# Run lookup benchmark
run_lookup() {
    local name="$1"
    local model_path="$2"
    local prompt_type="$3"
    local prompt="$4"

    if ! [[ -f "$model_path" ]]; then
        return
    fi

    if check_existing "$name" "lookup" "$prompt_type"; then
        return
    fi

    echo "=== Lookup: $name ($prompt_type) ==="

    local tmpfile=$(mktemp /mnt/raid0/llm/tmp/prompt_XXXXXX.txt)
    echo -e "$prompt" > "$tmpfile"

    local output
    output=$(timeout 180 OMP_NUM_THREADS=1 numactl --interleave=all \
        "$LLAMA_DIR/llama-lookup" \
        -m "$model_path" \
        -f "$tmpfile" \
        --draft-max 16 \
        -t 96 -n 200 --temp 0 2>&1) || true

    rm -f "$tmpfile"

    # Parse output
    local speed=$(echo "$output" | grep -oP 'speed:\s*[\d.]+\s*t/s' | grep -oP '[\d.]+' | tail -1)
    local accept=$(echo "$output" | grep -oP 'accept:\s*[\d.]+%' | grep -oP '[\d.]+' | tail -1)
    local n_accept=$(echo "$output" | grep -oP 'n_accept\s*=\s*\d+' | grep -oP '\d+' | tail -1)

    if [[ -n "$speed" ]]; then
        echo "$(date +%Y-%m-%d\ %H:%M:%S),$name,lookup,none,$prompt_type,$speed,${accept:-0},${n_accept:-0},ok" >> "$RESULTS_CSV"
        echo "  -> $speed t/s (accept: ${accept:-0}%)"
    else
        # Fallback: try to extract from generation stats
        speed=$(echo "$output" | grep -oP 'generation:\s*[\d.]+\s*tokens/s' | grep -oP '[\d.]+' | head -1)
        if [[ -n "$speed" ]]; then
            echo "$(date +%Y-%m-%d\ %H:%M:%S),$name,lookup,none,$prompt_type,$speed,0,0,fallback" >> "$RESULTS_CSV"
            echo "  -> $speed t/s (fallback parse)"
        else
            echo "$(date +%Y-%m-%d\ %H:%M:%S),$name,lookup,none,$prompt_type,0,0,0,error" >> "$RESULTS_CSV"
            echo "  -> ERROR"
        fi
    fi
}

# Run external draft benchmark
run_external_draft() {
    local name="$1"
    local model_path="$2"
    local draft_path="$3"
    local draft_name="$4"
    local prompt_type="$5"
    local prompt="$6"

    if ! [[ -f "$model_path" ]] || ! [[ -f "$draft_path" ]]; then
        return
    fi

    if check_existing "$name" "external_draft" "$prompt_type"; then
        return
    fi

    echo "=== External Draft: $name + $draft_name ($prompt_type) ==="

    local output
    output=$(timeout 300 OMP_NUM_THREADS=1 numactl --interleave=all \
        "$LLAMA_DIR/llama-speculative" \
        -m "$model_path" \
        -md "$draft_path" \
        --draft-max 16 \
        -t 96 -n 200 --temp 0 \
        -p "$prompt" 2>&1) || true

    # Parse output
    local speed=$(echo "$output" | grep -oP 'speed:\s*[\d.]+\s*t/s' | grep -oP '[\d.]+' | tail -1)
    local accept=$(echo "$output" | grep -oP 'accept:\s*[\d.]+%' | grep -oP '[\d.]+' | tail -1)
    local n_accept=$(echo "$output" | grep -oP 'n_accept\s*=\s*\d+' | grep -oP '\d+' | tail -1)

    if [[ -n "$speed" ]]; then
        echo "$(date +%Y-%m-%d\ %H:%M:%S),$name,external_draft,$draft_name,$prompt_type,$speed,${accept:-0},${n_accept:-0},ok" >> "$RESULTS_CSV"
        echo "  -> $speed t/s (accept: ${accept:-0}%)"
    else
        echo "$(date +%Y-%m-%d\ %H:%M:%S),$name,external_draft,$draft_name,$prompt_type,0,0,0,error" >> "$RESULTS_CSV"
        echo "  -> ERROR or vocab mismatch"
    fi
}

# Run hard mask benchmark (MoE only)
run_hard_mask() {
    local name="$1"
    local model_path="$2"
    local n_expert="$3"

    if ! [[ -f "$model_path" ]]; then
        return
    fi

    if check_existing "$name" "hard_mask_${n_expert}" "tg128"; then
        return
    fi

    echo "=== Hard Mask: $name (n_expert=$n_expert) ==="

    local tmpfile=$(mktemp /mnt/raid0/llm/tmp/prompt_XXXXXX.txt)
    echo -e "$PROMPT_SUMMARIZE" > "$tmpfile"

    local output
    output=$(timeout 300 OMP_NUM_THREADS=1 numactl --interleave=all \
        "$LLAMA_DIR/llama-cli" \
        -m "$model_path" \
        --moe-n-expert "$n_expert" \
        -f "$tmpfile" \
        -t 96 -n 128 --temp 0 \
        --no-display-prompt --simple-io --no-warmup 2>&1) || true

    rm -f "$tmpfile"

    # Parse generation speed
    local speed=$(echo "$output" | grep -oP 'generation:\s*[\d.]+\s*tokens/s' | grep -oP '[\d.]+' | head -1)

    if [[ -n "$speed" ]]; then
        echo "$(date +%Y-%m-%d\ %H:%M:%S),$name,hard_mask_${n_expert},none,summarize,$speed,0,0,ok" >> "$RESULTS_CSV"
        echo "  -> $speed t/s"
    else
        echo "$(date +%Y-%m-%d\ %H:%M:%S),$name,hard_mask_${n_expert},none,summarize,0,0,0,error" >> "$RESULTS_CSV"
        echo "  -> ERROR"
    fi
}

# Run layer skip benchmark
run_layer_skip() {
    local name="$1"
    local model_path="$2"
    local n_layers="$3"

    if ! [[ -f "$model_path" ]]; then
        return
    fi

    if check_existing "$name" "layer_skip_${n_layers}" "tg128"; then
        return
    fi

    echo "=== Layer Skip: $name (n_layer_exit=$n_layers) ==="

    local tmpfile=$(mktemp /mnt/raid0/llm/tmp/prompt_XXXXXX.txt)
    echo -e "$PROMPT_SUMMARIZE" > "$tmpfile"

    local output
    output=$(timeout 300 OMP_NUM_THREADS=1 numactl --interleave=all \
        "$LLAMA_DIR/llama-cli" \
        -m "$model_path" \
        --n-layer-exit "$n_layers" \
        -f "$tmpfile" \
        -t 96 -n 128 --temp 0 \
        --no-display-prompt --simple-io --no-warmup 2>&1) || true

    rm -f "$tmpfile"

    # Parse generation speed
    local speed=$(echo "$output" | grep -oP 'generation:\s*[\d.]+\s*tokens/s' | grep -oP '[\d.]+' | head -1)

    if [[ -n "$speed" ]]; then
        echo "$(date +%Y-%m-%d\ %H:%M:%S),$name,layer_skip_${n_layers},none,summarize,$speed,0,0,ok" >> "$RESULTS_CSV"
        echo "  -> $speed t/s"
    else
        echo "$(date +%Y-%m-%d\ %H:%M:%S),$name,layer_skip_${n_layers},none,summarize,0,0,0,error" >> "$RESULTS_CSV"
        echo "  -> ERROR"
    fi
}

# Main execution
echo "Starting comprehensive benchmark at $(date)"
echo "Results will be saved to: $RESULTS_CSV"
echo ""

# Phase 1: Baselines (all models)
echo "=========================================="
echo "PHASE 1: Baseline Benchmarks"
echo "=========================================="

for name in "${!MODELS[@]}"; do
    if check_existing "$name" "baseline" "tg128"; then
        continue
    fi
    run_baseline "$name" "${MODELS[$name]}"
done

# Phase 2: Lookup tests (all models, all prompt types)
echo ""
echo "=========================================="
echo "PHASE 2: Lookup Benchmarks"
echo "=========================================="

for name in "${!MODELS[@]}"; do
    run_lookup "$name" "${MODELS[$name]}" "summarize" "$PROMPT_SUMMARIZE"
    run_lookup "$name" "${MODELS[$name]}" "code" "$PROMPT_CODE"
    run_lookup "$name" "${MODELS[$name]}" "edit" "$PROMPT_EDIT"
done

# Phase 3: External draft (Qwen family only)
echo ""
echo "=========================================="
echo "PHASE 3: External Draft Benchmarks"
echo "=========================================="

# Qwen2.5 family with 0.5B draft
for name in "Qwen2.5-Coder-32B" "Qwen2.5-72B" "Qwen2.5-72B-Instruct" "Qwen2.5-Math-72B"; do
    if [[ -n "${MODELS[$name]:-}" ]]; then
        run_external_draft "$name" "${MODELS[$name]}" "$DRAFT_QWEN_05B" "Qwen2.5-0.5B" "summarize" "$PROMPT_SUMMARIZE"
        run_external_draft "$name" "${MODELS[$name]}" "$DRAFT_QWEN_05B" "Qwen2.5-0.5B" "code" "$PROMPT_CODE"
        run_external_draft "$name" "${MODELS[$name]}" "$DRAFT_QWEN_05B" "Qwen2.5-0.5B" "edit" "$PROMPT_EDIT"
    fi
done

# Qwen3 family with 0.6B draft
for name in "Qwen3-32B"; do
    if [[ -n "${MODELS[$name]:-}" ]]; then
        run_external_draft "$name" "${MODELS[$name]}" "$DRAFT_QWEN3_06B" "Qwen3-0.6B" "summarize" "$PROMPT_SUMMARIZE"
        run_external_draft "$name" "${MODELS[$name]}" "$DRAFT_QWEN3_06B" "Qwen3-0.6B" "code" "$PROMPT_CODE"
    fi
done

# Phase 4: Hard mask (MoE models only)
echo ""
echo "=========================================="
echo "PHASE 4: Hard Mask Benchmarks (MoE)"
echo "=========================================="

for name in "${MOE_MODELS[@]}"; do
    if [[ -n "${MODELS[$name]:-}" ]]; then
        run_hard_mask "$name" "${MODELS[$name]}" 4
    fi
done

# Phase 5: Layer skip (all large models)
echo ""
echo "=========================================="
echo "PHASE 5: Layer Skip Benchmarks"
echo "=========================================="

# Layer skip configurations: model name -> (total_layers, test_layers)
# 70B models have ~80 layers, test 50% (40 layers)
# 32B models have ~64 layers, test 50% (32 layers)
# MoE models have ~28-94 layers, test 50%

declare -A MODEL_LAYERS=(
    ["Meta-Llama-3.1-70B-Instruct"]=80
    ["Meta-Llama-3-70B-Instruct"]=80
    ["Hermes-4-70B"]=80
    ["DeepSeek-R1-Llama-70B"]=80
    ["Qwen2.5-72B"]=80
    ["Qwen2.5-72B-Instruct"]=80
    ["Qwen2.5-Math-72B"]=80
    ["Qwen2.5-Coder-32B"]=64
    ["DeepSeek-R1-32B"]=64
    ["Qwen3-32B"]=64
    ["Gemma-3-27B"]=46
    ["Qwen3-VL-30B-A3B"]=28
    ["Qwen3-Coder-30B-A3B"]=28
    ["Qwen3-Next-80B-A3B"]=28
    ["Qwen3-235B-A22B"]=94
    ["GLM-4.6-355B"]=80
    ["Qwen3-Coder-480B-A35B"]=94
    ["Qwen3-VL-235B-A22B-Thinking"]=94
)

for name in "${!MODEL_LAYERS[@]}"; do
    if [[ -n "${MODELS[$name]:-}" ]]; then
        total_layers="${MODEL_LAYERS[$name]}"
        half_layers=$((total_layers / 2))
        run_layer_skip "$name" "${MODELS[$name]}" "$half_layers"
    fi
done

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo "Results saved to: $RESULTS_CSV"
echo ""
cat "$RESULTS_CSV" | column -t -s','
