#!/bin/bash
set -x

# Run remaining optimization benchmarks
# This script tests lookup, external draft, hard mask, and layer skip combinations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment library for path variables
# shellcheck source=../lib/env.sh
source "${SCRIPT_DIR}/../lib/env.sh"

LLAMA_CPP="${LLAMA_CPP_BIN}"
BENCH_LOG_DIR="${LOG_DIR}/benchmarks"
RESULTS_CSV="$BENCH_LOG_DIR/optimization_results_20251215_045816.csv"

# Ensure log directory exists
mkdir -p "$BENCH_LOG_DIR"

# Function to check if test already done
check_existing() {
  local model="$1"
  local method="$2"
  local prompt_type="$3"
  if [[ -f "$RESULTS_CSV" ]]; then
    if grep -q ",${model},${method},.*,${prompt_type},success," "$RESULTS_CSV" 2>/dev/null; then
      local speed
      speed=$(grep ",${model},${method},.*,${prompt_type},success," "$RESULTS_CSV" | tail -1 | cut -d',' -f7)
      if [[ "$speed" != "0" ]] && [[ -n "$speed" ]]; then
        echo "SKIP: $model/$method/$prompt_type already done ($speed t/s)"
        return 0
      fi
    fi
  fi
  return 1
}

# Function to run lookup benchmark
run_lookup() {
  local name="$1"
  local model_path="$2"
  local prompt_type="$3"
  local prompt="$4"

  [[ -f "$model_path" ]] || {
    echo "Model not found: $model_path"
    return 1
  }
  check_existing "$name" "lookup" "$prompt_type" && return

  echo "=== Lookup: $name ($prompt_type) ==="
  local tmpfile
  tmpfile=$(mktemp /mnt/raid0/llm/tmp/prompt_XXXXXX.txt)
  echo -e "$prompt" >"$tmpfile"

  local output
  output=$(OMP_NUM_THREADS=1 numactl --interleave=all \
    "$LLAMA_CPP/llama-lookup" \
    -m "$model_path" \
    -f "$tmpfile" \
    --draft-max 16 \
    -t 96 \
    -n 200 \
    --temp 0 2>&1) || true

  rm -f "$tmpfile"

  local speed

  speed=$(echo "$output" | grep -oP 'speed:\s*[\d.]+\s*t/s' | grep -oP '[\d.]+' | tail -1)
  local accept
  accept=$(echo "$output" | grep -oP 'accept:\s*[\d.]+%' | grep -oP '[\d.]+' | tail -1)
  local n_accept
  n_accept=$(echo "$output" | grep -oP 'n_accept\s*=\s*\d+' | grep -oP '\d+' | tail -1)

  if [[ -n "$speed" ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$name,lookup,none,$prompt_type,success,$speed,${accept:-0},${n_accept:-0}" >>"$RESULTS_CSV"
    echo "  -> $speed t/s (accept: ${accept:-0}%)"
  else
    # Try alternate parsing for generation speed
    speed=$(echo "$output" | grep -oP 'generation:\s*[\d.]+\s*tokens/s' | grep -oP '[\d.]+' | head -1)
    if [[ -n "$speed" ]]; then
      echo "$(date '+%Y-%m-%d %H:%M:%S'),$name,lookup,none,$prompt_type,success,$speed,${accept:-0},${n_accept:-0}" >>"$RESULTS_CSV"
      echo "  -> $speed t/s"
    else
      echo "$(date '+%Y-%m-%d %H:%M:%S'),$name,lookup,none,$prompt_type,failed,0,0,0" >>"$RESULTS_CSV"
      echo "  -> FAILED"
    fi
  fi
}

# Function to run external draft benchmark
run_external_draft() {
  local name="$1"
  local model_path="$2"
  local draft_name="$3"
  local draft_path="$4"
  local prompt_type="$5"
  local prompt="$6"

  [[ -f "$model_path" ]] || {
    echo "Model not found: $model_path"
    return 1
  }
  [[ -f "$draft_path" ]] || {
    echo "Draft not found: $draft_path"
    return 1
  }
  check_existing "$name" "external_draft" "$prompt_type" && return

  echo "=== External Draft: $name + $draft_name ($prompt_type) ==="
  local tmpfile
  tmpfile=$(mktemp /mnt/raid0/llm/tmp/prompt_XXXXXX.txt)
  echo -e "$prompt" >"$tmpfile"

  local output
  output=$(OMP_NUM_THREADS=1 numactl --interleave=all \
    "$LLAMA_CPP/llama-speculative" \
    -m "$model_path" \
    -md "$draft_path" \
    -f "$tmpfile" \
    --draft-max 16 \
    -t 96 \
    -n 200 \
    --temp 0 2>&1) || true

  rm -f "$tmpfile"

  local speed

  speed=$(echo "$output" | grep -oP 'speed:\s*[\d.]+\s*t/s' | grep -oP '[\d.]+' | tail -1)
  local accept
  accept=$(echo "$output" | grep -oP 'accept:\s*[\d.]+%' | grep -oP '[\d.]+' | tail -1)

  if [[ -z "$speed" ]]; then
    speed=$(echo "$output" | grep -oP 'generation:\s*[\d.]+\s*tokens/s' | grep -oP '[\d.]+' | head -1)
  fi

  if [[ -n "$speed" ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$name,external_draft,$draft_name,$prompt_type,success,$speed,${accept:-0},0" >>"$RESULTS_CSV"
    echo "  -> $speed t/s (accept: ${accept:-0}%)"
  else
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$name,external_draft,$draft_name,$prompt_type,failed,0,0,0" >>"$RESULTS_CSV"
    echo "  -> FAILED"
  fi
}

# Function to run hard mask benchmark (MoE models)
run_hard_mask() {
  local name="$1"
  local model_path="$2"
  local n_expert="$3"
  local prompt_type="$4"
  local prompt="$5"

  [[ -f "$model_path" ]] || {
    echo "Model not found: $model_path"
    return 1
  }
  check_existing "$name" "hard_mask_${n_expert}" "$prompt_type" && return

  echo "=== Hard Mask ($n_expert experts): $name ($prompt_type) ==="
  local tmpfile
  tmpfile=$(mktemp /mnt/raid0/llm/tmp/prompt_XXXXXX.txt)
  echo -e "$prompt" >"$tmpfile"

  # NOTE: Use llama-completion instead of llama-cli - llama-cli gets stuck with --moe-n-expert
  local output
  output=$(OMP_NUM_THREADS=1 numactl --interleave=all \
    "$LLAMA_CPP/llama-completion" \
    -m "$model_path" \
    -f "$tmpfile" \
    --moe-n-expert "$n_expert" \
    -t 96 \
    -n 128 \
    --temp 0 2>&1) || true

  rm -f "$tmpfile"

  # Parse speed from llama-completion output (format: "X.XX tokens per second")
  local speed
  speed=$(echo "$output" | grep -oP '[\d.]+\s*tokens per second' | grep -oP '[\d.]+' | tail -1)
  if [[ -z "$speed" ]]; then
    speed=$(echo "$output" | grep -oP 'eval time.*=.*\(\s*[\d.]+' | grep -oP '[\d.]+' | tail -1)
  fi

  if [[ -n "$speed" ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$name,hard_mask_${n_expert},none,$prompt_type,success,$speed,0,0" >>"$RESULTS_CSV"
    echo "  -> $speed t/s"
  else
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$name,hard_mask_${n_expert},none,$prompt_type,failed,0,0,0" >>"$RESULTS_CSV"
    echo "  -> FAILED"
  fi
}

# Function to run layer skip benchmark
run_layer_skip() {
  local name="$1"
  local model_path="$2"
  local n_layers="$3"
  local prompt_type="$4"
  local prompt="$5"

  [[ -f "$model_path" ]] || {
    echo "Model not found: $model_path"
    return 1
  }
  check_existing "$name" "layer_skip_${n_layers}" "$prompt_type" && return

  echo "=== Layer Skip ($n_layers layers): $name ($prompt_type) ==="
  local tmpfile
  tmpfile=$(mktemp /mnt/raid0/llm/tmp/prompt_XXXXXX.txt)
  echo -e "$prompt" >"$tmpfile"

  local output
  output=$(OMP_NUM_THREADS=1 numactl --interleave=all \
    "$LLAMA_CPP/llama-cli" \
    -m "$model_path" \
    -f "$tmpfile" \
    --n-layer-exit "$n_layers" \
    -t 96 \
    -n 128 \
    --temp 0 \
    --no-display-prompt \
    --simple-io \
    --no-warmup 2>&1) || true

  rm -f "$tmpfile"

  local speed

  speed=$(echo "$output" | grep -oP 'speed:\s*[\d.]+\s*t/s' | grep -oP '[\d.]+' | tail -1)
  if [[ -z "$speed" ]]; then
    speed=$(echo "$output" | grep -oP 'generation:\s*[\d.]+\s*tokens/s' | grep -oP '[\d.]+' | head -1)
  fi
  if [[ -z "$speed" ]]; then
    speed=$(echo "$output" | grep -oP 'eval time.*=.*\(\s*[\d.]+' | grep -oP '[\d.]+' | tail -1)
  fi

  if [[ -n "$speed" ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$name,layer_skip_${n_layers},none,$prompt_type,success,$speed,0,0" >>"$RESULTS_CSV"
    echo "  -> $speed t/s"
  else
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$name,layer_skip_${n_layers},none,$prompt_type,failed,0,0,0" >>"$RESULTS_CSV"
    echo "  -> FAILED"
  fi
}

# Test prompts
PROMPT_SUMMARIZE="Summarize the following text in 3 bullet points:\n\nThe AMD EPYC 9655 Turin processor represents a significant advancement in server CPU technology. Built on the Zen 5 architecture, it features 96 cores and 192 threads, offering unprecedented parallel processing capabilities. The processor supports DDR5-5600 memory across 12 channels, providing approximately 460 GB/s of memory bandwidth. One of the key improvements in Zen 5 is the true 512-bit AVX-512 implementation, which is not double-pumped like previous generations. This makes it particularly effective for AI inference workloads that can leverage wide vector operations. The processor is manufactured on TSMC's 4nm process node, offering improved power efficiency compared to previous generations."

PROMPT_CODE="Write a Python function that implements binary search on a sorted array. Include docstring and type hints."

PROMPT_EDIT="Edit this code to add error handling:\ndef divide(a, b):\n    return a / b\n\nAdd try-except block, handle ZeroDivisionError, and return None on error."

echo "=========================================="
echo "Running remaining optimization benchmarks"
echo "=========================================="

# ============================================================
# PHASE 2: LOOKUP TESTS (models missing lookup results)
# ============================================================
echo ""
echo "=== PHASE 2: LOOKUP TESTS ==="

# 70B Dense models - need lookup tests
for prompt_type in summarize code edit; do
  case $prompt_type in
    summarize) prompt="$PROMPT_SUMMARIZE" ;;
    code) prompt="$PROMPT_CODE" ;;
    edit) prompt="$PROMPT_EDIT" ;;
  esac

  run_lookup "Hermes-4-70B" "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Hermes-4-70B-GGUF/Hermes-4-70B-Q4_K_M.gguf" "$prompt_type" "$prompt"
  run_lookup "DeepSeek-R1-Llama-70B" "/mnt/raid0/llm/lmstudio/models/unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF/DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf" "$prompt_type" "$prompt"
  run_lookup "Meta-Llama-3-70B-Instruct" "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF/Meta-Llama-3-70B-Instruct-Q4_K_M.gguf" "$prompt_type" "$prompt"
  run_lookup "Meta-Llama-3.1-70B-Instruct" "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf" "$prompt_type" "$prompt"
  run_lookup "Qwen2.5-72B" "/mnt/raid0/llm/lmstudio/models/mradermacher/Qwen2.5-72B-GGUF/Qwen2.5-72B.Q4_K_M.gguf" "$prompt_type" "$prompt"
  run_lookup "Qwen2.5-72B-Instruct" "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-72B-Instruct-GGUF/Qwen2.5-72B-Instruct-Q4_K_M.gguf" "$prompt_type" "$prompt"
  run_lookup "Gemma-3-27B" "/mnt/raid0/llm/lmstudio/models/lmstudio-community/gemma-3-27B-it-qat-GGUF/gemma-3-27B-it-QAT-Q4_0.gguf" "$prompt_type" "$prompt"
  run_lookup "DeepSeek-R1-32B" "/mnt/raid0/llm/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf" "$prompt_type" "$prompt"
  run_lookup "GLM-4.6-355B" "/mnt/raid0/llm/lmstudio/models/unsloth/GLM-4.6-GGUF/GLM-4.6-Q4_K_S-00001-of-00005.gguf" "$prompt_type" "$prompt"
done

# ============================================================
# PHASE 3: EXTERNAL DRAFT TESTS
# ============================================================
echo ""
echo "=== PHASE 3: EXTERNAL DRAFT TESTS ==="

# NOTE: Qwen2.5-72B spec decode tests DISABLED
# All 72B models show ~2% acceptance with ALL tested draft models
# See model_registry.yaml runtime_quirks for full details
# Tested drafts: Qwen2.5-Coder-0.5B, Qwen2.5-0.5B, Qwen2.5-Math-1.5B - all ~2%
# Use baseline or prompt lookup instead for 72B models

# Qwen3-32B + Qwen3-0.6B
DRAFT_QWEN3_06B="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"
for prompt_type in summarize code edit; do
  case $prompt_type in
    summarize) prompt="$PROMPT_SUMMARIZE" ;;
    code) prompt="$PROMPT_CODE" ;;
    edit) prompt="$PROMPT_EDIT" ;;
  esac

  run_external_draft "Qwen3-32B" "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-32B-GGUF/Qwen3-32B-Q4_K_M.gguf" "Qwen3-0.6B" "$DRAFT_QWEN3_06B" "$prompt_type" "$prompt"
done

# ============================================================
# PHASE 4: HARD MASK TESTS (MoE models)
# ============================================================
echo ""
echo "=== PHASE 4: HARD MASK TESTS (MoE) ==="

# Test with 4 experts (50% reduction for 8-expert models)
for prompt_type in summarize code; do
  case $prompt_type in
    summarize) prompt="$PROMPT_SUMMARIZE" ;;
    code) prompt="$PROMPT_CODE" ;;
  esac

  run_hard_mask "Qwen3-VL-30B-A3B" "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf" 4 "$prompt_type" "$prompt"
  run_hard_mask "Qwen3-Coder-30B-A3B" "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf" 4 "$prompt_type" "$prompt"
  run_hard_mask "Qwen3-Next-80B-A3B" "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf" 4 "$prompt_type" "$prompt"
  run_hard_mask "Qwen3-235B-A22B" "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-Q4_K_M-00001-of-00004.gguf" 4 "$prompt_type" "$prompt"
done

# ============================================================
# PHASE 5: LAYER SKIP TESTS - DISABLED (produces garbage output)
# ============================================================
# IMPORTANT: Standalone layer skip (--n-layer-exit) destroys output quality.
# Testing showed that even 97% of layers produces garbage (e.g., "巴黎巴黎巴黎...").
#
# Layer skip is designed for CAS-Spec/CLaSp speculative decoding where:
# 1. Draft phase uses early exit (fast, low quality)
# 2. Verify phase uses full layers (ensures quality)
#
# Current llama.cpp --n-layer-exit applies globally, not per-phase.
# Until proper CAS-Spec support is added, layer skip tests are disabled.
#
# See Quality Evaluation Results in research_report.md for details.
echo ""
echo "=== PHASE 5: LAYER SKIP TESTS (DISABLED - see comments) ==="
echo "Layer skip standalone produces garbage output. Skipping."
echo "Use layer skip only within CAS-Spec/CLaSp speculative framework."

echo ""
echo "=========================================="
echo "Benchmark complete!"
echo "Results: $RESULTS_CSV"
echo "=========================================="
