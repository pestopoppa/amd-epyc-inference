#!/bin/bash
# inference_guard.sh — Pre-run check for nightshift
#
# Checks if heavy inference is running (llama-server processes using significant RAM).
# When inference is detected, limits nightshift to analysis-only tasks that won't
# attempt test execution (which would fail the 100GB-free-RAM conftest guard anyway).
#
# Usage:
#   source scripts/nightshift/inference_guard.sh
#   # Sets NIGHTSHIFT_INFERENCE_ACTIVE=1 and NIGHTSHIFT_TASK_FILTER if inference detected
#
# Called by: scripts/nightshift/run_wrapper.sh

set -euo pipefail

# Threshold: if llama-server processes use more than this (in GB), consider inference active
INFERENCE_RAM_THRESHOLD_GB="${NIGHTSHIFT_INFERENCE_THRESHOLD_GB:-200}"

# Analysis-only tasks (safe to run during inference — no test execution)
ANALYSIS_ONLY_TASKS="dead-code,test-gap,security-footgun,perf-regression,doc-drift,docs-backfill,skill-groom"

get_llama_server_rss_gb() {
  local total_kb=0
  while IFS= read -r rss; do
    total_kb=$((total_kb + rss))
  done < <(pgrep -f 'llama-server|llama.cpp' 2>/dev/null | xargs -I{} awk '/^VmRSS:/{print $2}' /proc/{}/status 2>/dev/null || true)
  echo $((total_kb / 1024 / 1024))
}

check_inference_load() {
  local rss_gb
  rss_gb=$(get_llama_server_rss_gb)

  if [[ "$rss_gb" -ge "$INFERENCE_RAM_THRESHOLD_GB" ]]; then
    export NIGHTSHIFT_INFERENCE_ACTIVE=1
    export NIGHTSHIFT_INFERENCE_RSS_GB="$rss_gb"
    export NIGHTSHIFT_TASK_FILTER="$ANALYSIS_ONLY_TASKS"
    echo "[inference_guard] Inference active: ${rss_gb}GB RSS from llama-server processes"
    echo "[inference_guard] Restricting to analysis-only tasks: $ANALYSIS_ONLY_TASKS"
    return 0
  else
    export NIGHTSHIFT_INFERENCE_ACTIVE=0
    export NIGHTSHIFT_INFERENCE_RSS_GB="$rss_gb"
    unset NIGHTSHIFT_TASK_FILTER 2>/dev/null || true
    echo "[inference_guard] No heavy inference detected (${rss_gb}GB < ${INFERENCE_RAM_THRESHOLD_GB}GB threshold)"
    echo "[inference_guard] All tasks eligible"
    return 0
  fi
}

# Run if sourced or executed directly
check_inference_load
