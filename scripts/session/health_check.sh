#!/bin/bash
# health_check.sh - Pre-session system health check
# Usage: bash scripts/session/health_check.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/env.sh"

echo "=============================================="
echo "Pre-Session Health Check"
echo "=============================================="
echo ""

PASS=0
WARN=0
FAIL=0

check() {
  local test_name="$1"
  local condition="$2"
  local fail_msg="${3:-}"

  if eval "$condition"; then
    echo "âœ… PASS: $test_name"
    ((PASS++))
    return 0
  else
    if [[ -n "$fail_msg" ]]; then
      echo "❌ FAIL: $test_name - $fail_msg"
      ((FAIL++))
    else
      echo "⚠️  WARN: $test_name"
      ((WARN++))
    fi
    return 1
  fi
}

# ============================================
# 1. FILESYSTEM CHECKS
# ============================================

echo "--- Filesystem Health ---"

ROOT_USAGE=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
check "Root FS usage <70%" "[ $ROOT_USAGE -lt 70 ]" "Currently at ${ROOT_USAGE}%"

RAID_AVAIL=$(df /mnt/raid0 | awk 'NR==2 {print $4}')
check "RAID0 available >100GB" "[ $RAID_AVAIL -gt 102400 ]"

check "/mnt/raid0/llm exists" "[ -d /mnt/raid0/llm ]" "Create with: mkdir -p /mnt/raid0/llm"

check "/tmp/claude bind-mounted" "mountpoint -q /tmp/claude 2>/dev/null" "Not mounted - use claude_safe_start.sh"

echo ""

# ============================================
# 2. ENVIRONMENT VARIABLES
# ============================================

echo "--- Environment Variables ---"

check "TMPDIR set" "[ -n \"\${TMPDIR:-}\" ]" "Set via env.sh or export TMPDIR=${TMP_DIR}"

if [ -n "${TMPDIR:-}" ]; then
  check "TMPDIR under LLM_ROOT" "[[ \"$TMPDIR\" == ${LLM_ROOT}/* ]]" "Currently: $TMPDIR"
fi

check "HF_HOME set" "[ -n \"\${HF_HOME:-}\" ]" "Set via env.sh or export HF_HOME=${HF_HOME}"

if [ -n "${HF_HOME:-}" ]; then
  check "HF_HOME under LLM_ROOT" "[[ \"$HF_HOME\" == ${LLM_ROOT}/* ]]" "Currently: $HF_HOME"
fi

echo ""

# ============================================
# 3. REQUIRED DIRECTORIES
# ============================================

echo "--- Required Directories ---"

check "${TMP_DIR} exists" "[ -d ${TMP_DIR} ]"
check "${CACHE_DIR} exists" "[ -d ${CACHE_DIR} ]"
check "${LLM_ROOT}/LOGS exists" "[ -d ${LLM_ROOT}/LOGS ]"
check "${MODELS_DIR} exists" "[ -d ${MODELS_DIR} ]"

echo ""

# ============================================
# 4. PROCESS CHECKS
# ============================================

echo "--- Process Status ---"

if pgrep -f "claude" >/dev/null; then
  echo "⚠️  WARN: Claude process already running"
  ps aux | grep -i claude | grep -v grep
  ((WARN++))
else
  echo "âœ… PASS: No Claude processes running"
  ((PASS++))
fi

if pgrep -f "monitor_storage" >/dev/null; then
  echo "âœ… PASS: Storage monitor is running"
  ((PASS++))
else
  echo "⚠️  WARN: Storage monitor not running - consider starting it"
  ((WARN++))
fi

echo ""

# ============================================
# 5. SYSTEM RESOURCES
# ============================================

echo "--- System Resources ---"

MEM_AVAIL=$(free -g | awk 'NR==2 {print $7}')
check "Available RAM >100GB" "[ $MEM_AVAIL -gt 100 ]"

CPU_GOVERNOR=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
check "CPU governor is 'performance'" "[ \"$CPU_GOVERNOR\" == \"performance\" ]" "Currently: $CPU_GOVERNOR"

echo ""

# ============================================
# 6. SUMMARY
# ============================================

echo "=============================================="
echo "Health Check Summary"
echo "=============================================="
echo "  Passed:   $PASS âœ…"
echo "  Warnings: $WARN ⚠️"
echo "  Failed:   $FAIL ❌"
echo ""

if [ $FAIL -gt 0 ]; then
  echo "🚨 CRITICAL ISSUES DETECTED"
  echo ""
  echo "Recommended actions:"
  if [ $ROOT_USAGE -ge 70 ]; then
    echo "  1. Run emergency_cleanup.sh to free root FS"
  fi
  if ! mountpoint -q /tmp/claude 2>/dev/null; then
    echo "  2. Start Claude via: bash ${PROJECT_ROOT}/scripts/session/claude_safe_start.sh"
  fi
  if [ "$CPU_GOVERNOR" != "performance" ]; then
    echo "  3. Set CPU governor: echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
  fi
  echo ""
  exit 1
elif [ $WARN -gt 0 ]; then
  echo "⚠️  WARNINGS PRESENT - Review above"
  echo ""
  echo "Recommended actions:"
  if ! pgrep -f "monitor_storage" >/dev/null; then
    echo "  • Start monitor: bash ${PROJECT_ROOT}/scripts/session/monitor_storage.sh &"
  fi
  echo ""
  exit 0
else
  echo "âœ… ALL CHECKS PASSED - System ready for Claude session"
  echo ""
  echo "Start Claude Code:"
  echo "  bash ${PROJECT_ROOT}/scripts/session/claude_safe_start.sh"
  echo ""
  exit 0
fi
