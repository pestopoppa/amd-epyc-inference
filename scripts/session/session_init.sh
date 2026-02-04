#!/bin/bash
# session_init.sh — Initialize agent session with full context
# Run at the start of every Claude Code session
#
# This script:
# 1. Sets critical environment variables (storage constraint)
# 2. Starts logging session
# 3. Discovers all available models
# 4. Identifies untested models
# 5. Loads research report context

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment library first (sets all path variables)
source "$SCRIPT_DIR/../lib/env.sh"

source "$SCRIPT_DIR/agent_log.sh"

LOGS_DIR="${LLM_ROOT}/LOGS"
INVENTORY_FILE="$LOGS_DIR/model_inventory.json"
TESTED_FILE="$LOGS_DIR/tested_models.json"
RESEARCH_REPORT="$LOGS_DIR/research_report.md"
UNTESTED_FILE="$LOGS_DIR/untested_models.txt"

echo "=============================================="
echo "AGENT SESSION INITIALIZATION"
echo "Timestamp: $(date)"
echo "=============================================="

# ============================================
# 0. CRITICAL: VERIFY STORAGE CONSTRAINTS
# ============================================
# Note: env.sh already exports all storage paths (HF_HOME, TMPDIR, etc.)
echo ""
echo "--- Storage Constraints (from env.sh) ---"
echo "All caches/temp files MUST use ${LLM_ROOT}/"

# npm cache not in env.sh, add it
export npm_config_cache="${CACHE_DIR}/npm"

# Create directories if they don't exist
ensure_dir "${HF_HOME}"
ensure_dir "${HF_DATASETS_CACHE}"
ensure_dir "${PIP_CACHE_DIR}"
ensure_dir "${npm_config_cache}"
ensure_dir "${TMPDIR}"
ensure_dir "$LOGS_DIR"

echo "  HF_HOME=$HF_HOME"
echo "  TMPDIR=$TMPDIR"
echo "  PIP_CACHE_DIR=$PIP_CACHE_DIR"
echo ""
echo "⛔ NEVER write to /home/, /tmp/, /var/, or ~/.cache/"
echo "✅ ALL files must be under ${LLM_ROOT}/"

agent_session_start "New optimization session"
agent_observe "storage_constraint" "All writes to /mnt/raid0/ only"

# ============================================
# 0.5 VERIFY LLAMA.CPP BRANCH (CRITICAL)
# ============================================
echo ""
echo "--- Verifying llama.cpp branch ---"
VERIFY_SCRIPT="$SCRIPT_DIR/verify_llama_cpp.sh"
if [[ -x "$VERIFY_SCRIPT" ]]; then
  if ! "$VERIFY_SCRIPT"; then
    echo ""
    echo "⛔ WARNING: llama.cpp branch verification failed!"
    echo "   Production inference may use wrong binary."
    echo "   Fix before running benchmarks or live inference."
    echo ""
    agent_observe "llama_cpp_branch" "VERIFICATION FAILED - wrong branch or missing binary"
  else
    agent_observe "llama_cpp_branch" "production-consolidated verified"
  fi
else
  echo "  Verification script not found (skip in dev environments)"
fi

# ============================================
# 0.6 VERIFY DEPENDENCIES (UV)
# ============================================
echo ""
echo "--- Checking Python dependencies ---"
if command -v uv &>/dev/null; then
  if [[ -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
    cd "${PROJECT_ROOT}"
    if uv sync --dry-run 2>&1 | grep -q "Would install"; then
      echo "⚠ Dependencies out of sync. Running: uv sync"
      uv sync 2>&1 | tail -5
    else
      echo "✓ Dependencies up to date"
    fi
  fi
else
  echo "  uv not installed (using pip)"
fi

# ============================================
# 1. MODEL DISCOVERY
# ============================================
agent_task_start "Discover models" "Scanning all model directories"

echo ""
echo "--- Scanning for models ---"

# Initialize inventory file
cat >"$INVENTORY_FILE" <<'HEADER'
{
  "timestamp": "TIMESTAMP_PLACEHOLDER",
  "models": {
HEADER
sed -i "s/TIMESTAMP_PLACEHOLDER/$(date -Iseconds)/" "$INVENTORY_FILE"

first_model=true

# Scan HuggingFace models
echo "Scanning ${LLM_ROOT}/hf/..."
if [ -d "${LLM_ROOT}/hf" ]; then
  for dir in "${LLM_ROOT}"/hf/*/; do
    if [ -d "$dir" ]; then
      model_name=$(basename "$dir")
      model_path="${dir%/}"

      if [ "$first_model" = false ]; then
        echo "," >>"$INVENTORY_FILE"
      fi
      first_model=false

      cat >>"$INVENTORY_FILE" <<EOF
    "$model_path": {
      "name": "$model_name",
      "format": "huggingface",
      "type": "source"
    }
EOF
      echo "  [HF] $model_name"
      agent_observe "model_hf" "$model_path"
    fi
  done
fi

# Scan GGUF models in ${MODELS_DIR}
echo "Scanning ${MODELS_DIR}/..."
if [ -d "${MODELS_DIR}" ]; then
  while IFS= read -r -d '' gguf; do
    model_name=$(basename "$gguf")
    model_path="$gguf"
    size_bytes=$(stat -c%s "$gguf" 2>/dev/null || echo "0")

    if [ "$first_model" = false ]; then
      echo "," >>"$INVENTORY_FILE"
    fi
    first_model=false

    cat >>"$INVENTORY_FILE" <<EOF
    "$model_path": {
      "name": "$model_name",
      "format": "gguf",
      "type": "converted",
      "size_bytes": $size_bytes
    }
EOF
    echo "  [GGUF] $model_name"
    agent_observe "model_gguf" "$model_path"
  done < <(find "${MODELS_DIR}" -name "*.gguf" -type f -print0 2>/dev/null)
fi

# Scan LM Studio models
echo "Scanning ${MODEL_BASE}/..."
if [ -d "${MODEL_BASE}" ]; then
  while IFS= read -r -d '' gguf; do
    model_name=$(basename "$gguf")
    model_path="$gguf"
    size_bytes=$(stat -c%s "$gguf" 2>/dev/null || echo "0")

    if [ "$first_model" = false ]; then
      echo "," >>"$INVENTORY_FILE"
    fi
    first_model=false

    cat >>"$INVENTORY_FILE" <<EOF
    "$model_path": {
      "name": "$model_name",
      "format": "gguf",
      "type": "lmstudio",
      "size_bytes": $size_bytes
    }
EOF
    echo "  [LMStudio] $model_name"
    agent_observe "model_lmstudio" "$model_path"
  done < <(find "${MODEL_BASE}" -name "*.gguf" -type f -print0 2>/dev/null)
fi

# Close JSON
cat >>"$INVENTORY_FILE" <<'FOOTER'
  }
}
FOOTER

agent_task_end "Discover models" "success"

# ============================================
# 2. IDENTIFY UNTESTED MODELS
# ============================================
agent_task_start "Identify untested models" "Cross-referencing with test history"

# Initialize tested models file if it doesn't exist
if [ ! -f "$TESTED_FILE" ]; then
  echo '{"tested": []}' >"$TESTED_FILE"
fi

# Extract all model paths from inventory
all_models=$(grep -oP '^\s*"/mnt/raid0/llm[^"]+' "$INVENTORY_FILE" | tr -d ' "' | sort -u)

# Extract tested model paths
tested_models=$(jq -r '.tested[]?.path // empty' "$TESTED_FILE" 2>/dev/null | sort -u)

# Find untested models
echo "$all_models" | while read -r model; do
  if [ -n "$model" ] && ! echo "$tested_models" | grep -qF "$model"; then
    echo "$model"
  fi
done >"$UNTESTED_FILE"

untested_count=$(wc -l <"$UNTESTED_FILE" | tr -d ' ')
echo ""
echo "--- Untested Models: $untested_count ---"
if [ "$untested_count" -gt 0 ]; then
  cat "$UNTESTED_FILE"
  agent_observe "untested_models_count" "$untested_count"
  agent_observe "untested_models_file" "$UNTESTED_FILE"
fi

agent_task_end "Identify untested models" "success"

# ============================================
# 3. LOAD RESEARCH REPORT SUMMARY
# ============================================
echo ""
echo "--- Research Report Status ---"

if [ -f "$RESEARCH_REPORT" ]; then
  echo "Research report exists: $RESEARCH_REPORT"
  echo "Last modified: $(stat -c%y "$RESEARCH_REPORT" | cut -d'.' -f1)"

  # Show executive summary if present
  if grep -q "## Executive Summary" "$RESEARCH_REPORT"; then
    echo ""
    echo "Executive Summary:"
    sed -n '/## Executive Summary/,/^## /p' "$RESEARCH_REPORT" | head -20
  fi

  agent_observe "research_report" "exists"
else
  echo "No research report found. Will create after first test."
  agent_observe "research_report" "missing"
fi

# ============================================
# 4. SESSION SUMMARY
# ============================================
echo ""
echo "=============================================="
echo "SESSION READY"
echo "=============================================="
echo "Models discovered: $(grep -c '"format":' "$INVENTORY_FILE" 2>/dev/null || echo 0)"
echo "Untested models: $untested_count"
echo "Logs directory: $LOGS_DIR"
echo ""
echo "PRIORITY ACTIONS:"
if [ "$untested_count" -gt 0 ]; then
  echo "  1. Test untested models with adaptive speculative decoding"
  echo "     Models: $(head -3 "$UNTESTED_FILE" | tr '\n' ' ')"
fi
echo "  2. Continue current research track"
echo "  3. Update research report after each test"
echo ""
echo "Key files:"
echo "  - Model inventory: $INVENTORY_FILE"
echo "  - Untested models: $UNTESTED_FILE"
echo "  - Research report: $RESEARCH_REPORT"
echo "  - Audit log: $LOGS_DIR/agent_audit.log"
echo "=============================================="
