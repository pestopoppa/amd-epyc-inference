#!/bin/bash
# claude_safe_start.sh - Safe startup wrapper for Claude Code
# Enforces storage constraints BEFORE Claude Code initializes
# Usage: ./scripts/session/claude_safe_start.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment library for path variables (this sets all the env vars we need)
# shellcheck source=../lib/env.sh
source "${SCRIPT_DIR}/../lib/env.sh"

echo "=============================================="
echo "Claude Code Safe Startup Wrapper"
echo "=============================================="
echo ""

# ============================================
# 1. FORCE ALL WRITES TO /mnt/raid0/ (from env.sh)
# ============================================

# Additional vars not in env.sh
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"

# Package managers
export TORCH_HOME="${LLM_ROOT}/cache/torch"
export npm_config_cache="${LLM_ROOT}/cache/npm"
export CARGO_HOME="${LLM_ROOT}/cache/cargo"

# XDG
export XDG_CACHE_HOME=/mnt/raid0/llm/cache
export XDG_DATA_HOME=/mnt/raid0/llm/data

# Claude Code specific (attempt to override default)
export CLAUDE_WORKSPACE=/mnt/raid0/llm/claude_workspace
export CLAUDE_TEMP=/mnt/raid0/llm/tmp

echo "✓ Environment variables set to redirect ALL writes to /mnt/raid0/"
echo ""

# ============================================
# 2. CREATE REQUIRED DIRECTORIES
# ============================================

echo "Creating required directories..."
mkdir -p /mnt/raid0/llm/tmp
mkdir -p /mnt/raid0/llm/cache/{huggingface,pip,npm,cargo,torch}
mkdir -p /mnt/raid0/llm/data
mkdir -p /mnt/raid0/llm/claude_workspace
mkdir -p /mnt/raid0/llm/LOGS

echo "✓ Directories created"
echo ""

# ============================================
# 3. VERIFY NO SPACE ON ROOT (DEFENSIVE)
# ============================================

ROOT_USAGE=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$ROOT_USAGE" -gt 80 ]; then
  echo "⚠️  WARNING: Root filesystem is ${ROOT_USAGE}% full"
  echo "   Continuing, but monitoring required."
fi

echo "Root filesystem usage: ${ROOT_USAGE}%"
echo "RAID0 available: $(df -h /mnt/raid0 | awk 'NR==2 {print $4}')"
echo ""

# ============================================
# 4. BIND MOUNT /tmp/claude (NUCLEAR OPTION)
# ============================================

# This forces ANY writes to /tmp/claude to actually go to /mnt/raid0/
if [ ! -d /tmp/claude ]; then
  sudo mkdir -p /tmp/claude
fi

# Check if already mounted
if ! mountpoint -q /tmp/claude 2>/dev/null; then
  echo "Creating bind mount: /tmp/claude -> /mnt/raid0/llm/tmp/claude"
  sudo mkdir -p /mnt/raid0/llm/tmp/claude
  sudo mount --bind /mnt/raid0/llm/tmp/claude /tmp/claude
  echo "✓ Bind mount active"
else
  echo "✓ /tmp/claude already bind-mounted"
fi

echo ""

# ============================================
# 5. START CLAUDE CODE
# ============================================

echo "=============================================="
echo "Starting Claude Code..."
echo "=============================================="
echo ""
echo "Storage constraints active:"
echo "  • TMPDIR=/mnt/raid0/llm/tmp"
echo "  • HF_HOME=/mnt/raid0/llm/cache/huggingface"
echo "  • PIP_CACHE_DIR=/mnt/raid0/llm/cache/pip"
echo "  • /tmp/claude bind-mounted to /mnt/raid0/"
echo ""
echo "ALL writes will go to /mnt/raid0/ - root FS is protected."
echo ""

# Source logging library
source "${SCRIPT_DIR}/../utils/agent_log.sh"
agent_session_start "Claude Code session with enforced storage constraints"
agent_observe "tmpdir" "$TMPDIR"
agent_observe "hf_home" "$HF_HOME"
agent_observe "bind_mount" "/tmp/claude -> ${LLM_ROOT}/tmp/claude"

# Start Claude Code in the project directory
cd "${LLM_ROOT}"
exec claude "$@"
