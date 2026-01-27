#!/bin/bash
set -x

# Quality evaluation for optimization techniques
# Compares baseline vs optimized outputs for quality degradation

LLAMA_CPP="/mnt/raid0/llm/llama.cpp/build/bin"
LOG_DIR="/mnt/raid0/llm/LOGS/quality"
mkdir -p "$LOG_DIR"

PROMPT_FACTUAL="What is the capital of France? Answer in one word."
PROMPT_REASONING="If a train travels 60 miles in 1 hour, how far will it travel in 2.5 hours? Show your calculation."
PROMPT_CODE="Write a Python function to calculate factorial. Include docstring."
PROMPT_SUMMARIZE="Summarize in one sentence: The AMD EPYC 9655 Turin processor is built on Zen 5 architecture with 96 cores."

# Function to get output text
get_output() {
  local model_path="$1"
  local prompt="$2"
  local extra_args="$3"

  local tmpfile

  tmpfile=$(mktemp /mnt/raid0/llm/tmp/prompt_XXXXXX.txt)
  echo -e "$prompt" >"$tmpfile"

  local raw_output
  raw_output=$(OMP_NUM_THREADS=1 numactl --interleave=all \
    "$LLAMA_CPP/llama-completion" \
    -m "$model_path" \
    -f "$tmpfile" \
    -t 96 \
    -n 100 \
    --temp 0 \
    $extra_args 2>/dev/null)

  rm -f "$tmpfile"

  # Extract just the assistant response (after "assistant" line, before "> EOF")
  echo "$raw_output" | sed -n '/^assistant$/,/^> EOF/p' | grep -v "^assistant$" | grep -v "^> EOF"
}

# Function to compare outputs
compare_quality() {
  local name="$1"
  local model_path="$2"
  local opt_args="$3"
  local opt_name="$4"

  echo "=========================================="
  echo "Quality Check: $name ($opt_name)"
  echo "=========================================="

  for prompt_name in "factual" "reasoning" "code" "summarize"; do
    case $prompt_name in
      factual) prompt="$PROMPT_FACTUAL" ;;
      reasoning) prompt="$PROMPT_REASONING" ;;
      code) prompt="$PROMPT_CODE" ;;
      summarize) prompt="$PROMPT_SUMMARIZE" ;;
    esac

    echo ""
    echo "--- $prompt_name ---"
    echo "Baseline:"
    baseline_output=$(get_output "$model_path" "$prompt" "")
    echo "$baseline_output"

    echo ""
    echo "Optimized ($opt_name):"
    opt_output=$(get_output "$model_path" "$prompt" "$opt_args")
    echo "$opt_output"

    # Check if outputs match
    if [[ "$baseline_output" == "$opt_output" ]]; then
      echo "✅ MATCH: Outputs identical"
    else
      echo "⚠️ DIFFERENT: Outputs differ"
      # Save for manual review
      echo "Baseline: $baseline_output" >>"$LOG_DIR/${name}_${opt_name}_${prompt_name}.diff"
      echo "Optimized: $opt_output" >>"$LOG_DIR/${name}_${opt_name}_${prompt_name}.diff"
    fi
  done
}

echo "=========================================="
echo "Running Quality Evaluation"
echo "=========================================="

# ============================================================
# TEST 1: Hard Mask Quality (MoE - may degrade)
# ============================================================
echo ""
echo "=== HARD MASK QUALITY (MoE) ==="

# Qwen3-Coder-30B-A3B with 4 experts (baseline uses 8)
compare_quality "Qwen3-Coder-30B-A3B" \
  "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf" \
  "--moe-n-expert 4" \
  "hard_mask_4"

# ============================================================
# TEST 2: Layer Skip Quality (Dense - may degrade)
# ============================================================
echo ""
echo "=== LAYER SKIP QUALITY (Dense) ==="

# Qwen2.5-Coder-32B with 32 layers (baseline uses 64)
compare_quality "Qwen2.5-Coder-32B" \
  "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-32B-GGUF/Qwen2.5-Coder-32B-Q4_K_M.gguf" \
  "--n-layer-exit 32" \
  "layer_skip_32"

# ============================================================
# TEST 3: Combined Optimizations Quality
# ============================================================
echo ""
echo "=== COMBINED QUALITY (Hard Mask + Layer Skip on Different Models) ==="

# Note: Most combinations should preserve quality since speculative decoding
# uses verification. But hard_mask and layer_skip modify the model itself.

echo ""
echo "=========================================="
echo "Quality checks complete!"
echo "Review differences in: $LOG_DIR/"
echo "=========================================="
