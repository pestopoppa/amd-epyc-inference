#!/bin/bash
# emergency_cleanup.sh - Clean up /tmp/claude and free root FS space
# Usage: sudo bash scripts/session/emergency_cleanup.sh

set -euo pipefail

echo "=============================================="
echo "Emergency Root FS Cleanup"
echo "=============================================="
echo ""

# Check current usage
ROOT_BEFORE=$(df / | awk 'NR==2 {print $5}')
echo "Root FS usage before cleanup: $ROOT_BEFORE"
echo ""

# ============================================
# 1. STOP ANY RUNNING CLAUDE PROCESSES
# ============================================

echo "Checking for running Claude processes..."
if pgrep -f claude >/dev/null; then
  echo "⚠️  Found running Claude processes. Kill them? (y/n)"
  read -r response
  if [[ "$response" == "y" ]]; then
    sudo pkill -f claude || true
    sleep 2
    echo "✓ Processes stopped"
  fi
fi

# ============================================
# 2. UNMOUNT /tmp/claude IF MOUNTED
# ============================================

if mountpoint -q /tmp/claude 2>/dev/null; then
  echo "Unmounting /tmp/claude bind mount..."
  sudo umount /tmp/claude
  echo "✓ Unmounted"
fi

# ============================================
# 3. REMOVE /tmp/claude ENTIRELY
# ============================================

if [ -d /tmp/claude ]; then
  echo ""
  echo "Analyzing /tmp/claude contents..."
  du -sh /tmp/claude 2>/dev/null || echo "  (cannot access)"
  du -sh /tmp/claude/* 2>/dev/null | head -10 || true
  echo ""

  echo "⚠️  This will DELETE /tmp/claude and ALL contents."
  echo "Continue? (y/n)"
  read -r response
  if [[ "$response" == "y" ]]; then
    echo "Removing /tmp/claude..."
    sudo rm -rf /tmp/claude
    echo "✓ Removed"
  else
    echo "Aborted. /tmp/claude preserved."
    exit 0
  fi
fi

# ============================================
# 4. CLEAN OTHER /tmp CRUFT
# ============================================

echo ""
echo "Cleaning other temporary files..."

# Remove old Python bytecode
sudo find /tmp -name "*.pyc" -type f -delete 2>/dev/null || true
sudo find /tmp -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Remove old pip builds
sudo rm -rf /tmp/pip-* 2>/dev/null || true

# Remove npm cache if present
sudo rm -rf /tmp/npm-* 2>/dev/null || true

echo "✓ Cleaned temporary files"

# ============================================
# 5. VERIFY RESULTS
# ============================================

echo ""
ROOT_AFTER=$(df / | awk 'NR==2 {print $5}')
echo "Root FS usage after cleanup: $ROOT_AFTER (was $ROOT_BEFORE)"
echo ""

# Verify /tmp/claude is gone
if [ -d /tmp/claude ]; then
  echo "⚠️  WARNING: /tmp/claude still exists!"
else
  echo "✓ /tmp/claude successfully removed"
fi

echo ""
echo "=============================================="
echo "Cleanup complete."
echo "Next steps:"
echo "  1. Use claude_safe_start.sh to start Claude Code"
echo "  2. Monitor root FS: watch -n 5 'df -h /'"
echo "=============================================="
