#!/bin/bash
# run_wrapper.sh — Nightshift runner with inference guard
#
# This script is called by the systemd timer/service instead of
# `nightshift run` directly. It checks inference load and adjusts
# which tasks nightshift can run.
#
# Usage:
#   scripts/nightshift/run_wrapper.sh [extra nightshift args...]
#
# Environment:
#   NIGHTSHIFT_MAX_PROJECTS  — max projects per run (default: 3)
#   NIGHTSHIFT_MAX_TASKS     — max tasks per project (default: 2)
#   NIGHTSHIFT_INFERENCE_THRESHOLD_GB — RAM threshold for inference detection (default: 200)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKTREE="/mnt/raid0/llm/claude-nightshift"
LOG_DIR="$PROJECT_ROOT/logs/nightshift"

mkdir -p "$LOG_DIR"

LOGFILE="$LOG_DIR/$(date +%Y-%m-%d_%H%M%S).log"

{
  echo "=== Nightshift Run: $(date -Iseconds) ==="
  echo "Project root: $PROJECT_ROOT"
  echo "Worktree: $WORKTREE"

  # 0. Sync worktree to latest main (so nightshift sees current code)
  if [[ -d "$WORKTREE" ]]; then
    echo "[wrapper] Syncing worktree to latest main..."
    git -C "$WORKTREE" fetch origin main 2>/dev/null || true
    git -C "$WORKTREE" checkout --detach origin/main 2>/dev/null ||
      git -C "$WORKTREE" checkout --detach main 2>/dev/null || true
    echo "[wrapper] Worktree at: $(git -C "$WORKTREE" rev-parse --short HEAD)"
  else
    echo "[wrapper] ERROR: worktree not found at $WORKTREE"
    echo "[wrapper] Create with: git -C $PROJECT_ROOT worktree add --detach $WORKTREE HEAD"
    exit 1
  fi

  # 1. Check inference load
  source "$SCRIPT_DIR/inference_guard.sh"

  # 2. Build nightshift command
  MAX_PROJECTS="${NIGHTSHIFT_MAX_PROJECTS:-3}"
  MAX_TASKS="${NIGHTSHIFT_MAX_TASKS:-2}"

  NIGHTSHIFT_CMD=(
    nightshift run
    --yes
    --max-projects "$MAX_PROJECTS"
    --max-tasks "$MAX_TASKS"
  )

  # If inference is active, run only on this project with analysis-only tasks
  if [[ "${NIGHTSHIFT_INFERENCE_ACTIVE:-0}" == "1" ]]; then
    echo "[wrapper] Inference mode: limiting to analysis-only tasks on this project"
    IFS=',' read -ra TASKS <<<"$NIGHTSHIFT_TASK_FILTER"
    for task in "${TASKS[@]}"; do
      echo "[wrapper] Running analysis task: $task"
      nightshift run --yes --project "$WORKTREE" --task "$task" "$@" 2>&1 || {
        echo "[wrapper] Task $task failed or skipped (budget/cooldown), continuing..."
      }
    done
  else
    echo "[wrapper] Full mode: running all eligible tasks"
    "${NIGHTSHIFT_CMD[@]}" "$@" 2>&1
  fi

  echo "=== Nightshift Run Complete: $(date -Iseconds) ==="
} 2>&1 | tee "$LOGFILE"

# Prune logs older than 30 days
find "$LOG_DIR" -name "*.log" -mtime +30 -delete 2>/dev/null || true
