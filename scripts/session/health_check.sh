#!/bin/bash
# health_check.sh - Pre-session system health check
# Usage: bash /mnt/raid0/llm/UTILS/health_check.sh

set -euo pipefail

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

check "TMPDIR set" "[ -n \"\${TMPDIR:-}\" ]" "Set via wrapper or export TMPDIR=/mnt/raid0/llm/tmp"

if [ -n "${TMPDIR:-}" ]; then
    check "TMPDIR on RAID0" "[[ \"$TMPDIR\" == /mnt/raid0/* ]]" "Currently: $TMPDIR"
fi

check "HF_HOME set" "[ -n \"\${HF_HOME:-}\" ]" "Set via wrapper or export HF_HOME=/mnt/raid0/llm/cache/huggingface"

if [ -n "${HF_HOME:-}" ]; then
    check "HF_HOME on RAID0" "[[ \"$HF_HOME\" == /mnt/raid0/* ]]" "Currently: $HF_HOME"
fi

echo ""

# ============================================
# 3. REQUIRED DIRECTORIES
# ============================================

echo "--- Required Directories ---"

check "/mnt/raid0/llm/tmp exists" "[ -d /mnt/raid0/llm/tmp ]"
check "/mnt/raid0/llm/cache exists" "[ -d /mnt/raid0/llm/cache ]"
check "/mnt/raid0/llm/LOGS exists" "[ -d /mnt/raid0/llm/LOGS ]"
check "/mnt/raid0/llm/models exists" "[ -d /mnt/raid0/llm/models ]"

echo ""

# ============================================
# 4. PROCESS CHECKS
# ============================================

echo "--- Process Status ---"

if pgrep -f "claude" > /dev/null; then
    echo "⚠️  WARN: Claude process already running"
    ps aux | grep -i claude | grep -v grep
    ((WARN++))
else
    echo "âœ… PASS: No Claude processes running"
    ((PASS++))
fi

if pgrep -f "monitor_storage" > /dev/null; then
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
        echo "  2. Start Claude via: bash /mnt/raid0/llm/UTILS/claude_safe_start.sh"
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
    if ! pgrep -f "monitor_storage" > /dev/null; then
        echo "  • Start monitor: bash /mnt/raid0/llm/UTILS/monitor_storage.sh &"
    fi
    echo ""
    exit 0
else
    echo "âœ… ALL CHECKS PASSED - System ready for Claude session"
    echo ""
    echo "Start Claude Code:"
    echo "  bash /mnt/raid0/llm/UTILS/claude_safe_start.sh"
    echo ""
    exit 0
fi
