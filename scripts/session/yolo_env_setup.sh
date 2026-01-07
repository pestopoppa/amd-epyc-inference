#!/bin/bash
# YOLO Agent Environment Setup Script
# Run this at the start of any YOLO agent session
# Usage: source scripts/session/yolo_env_setup.sh

set -euo pipefail

echo "=== YOLO Agent Environment Setup ==="
echo ""

# 1. Set environment variables (from CLAUDE.md)
export HF_HOME=/mnt/raid0/llm/cache/huggingface
export TRANSFORMERS_CACHE=/mnt/raid0/llm/cache/huggingface
export HF_DATASETS_CACHE=/mnt/raid0/llm/cache/huggingface/datasets
export PIP_CACHE_DIR=/mnt/raid0/llm/cache/pip
export TMPDIR=/mnt/raid0/llm/tmp
export XDG_CACHE_HOME=/mnt/raid0/llm/claude/cache
export XDG_DATA_HOME=/mnt/raid0/llm/claude/share
export XDG_STATE_HOME=/mnt/raid0/llm/claude/state

# Add uv to PATH
export PATH="/mnt/raid0/llm/tools:$PATH"

echo "✓ Environment variables set"

# 2. Change to project directory
cd /mnt/raid0/llm/claude
echo "✓ Working directory: $(pwd)"

# 3. Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "Creating virtual environment with uv..."
    /mnt/raid0/llm/tools/uv venv .venv
    source .venv/bin/activate
    echo "Installing dependencies..."
    /mnt/raid0/llm/tools/uv pip install -e .
    echo "✓ Virtual environment created and dependencies installed"
fi

# 4. Verify critical imports
python3 -c "
import sys
try:
    import optuna
    import sklearn
    import httpx
    import fastapi
    import numpy
    import yaml
    print('✓ All critical imports OK')
except ImportError as e:
    print(f'✗ Missing import: {e}')
    sys.exit(1)
"

# 5. Verify model registry is readable
if [ -f "orchestration/model_registry.yaml" ]; then
    python3 -c "import yaml; yaml.safe_load(open('orchestration/model_registry.yaml'))" && \
        echo "✓ Model registry valid" || \
        echo "✗ Model registry invalid"
else
    echo "✗ Model registry not found"
fi

# 6. Verify dev models exist
MODEL_BASE="/mnt/raid0/llm/lmstudio/models"
DEV_MODEL="lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q8_0.gguf"

if [ -f "${MODEL_BASE}/${DEV_MODEL}" ]; then
    echo "✓ Dev model exists: ${DEV_MODEL}"
else
    echo "✗ Dev model NOT found: ${MODEL_BASE}/${DEV_MODEL}"
fi

# 7. Check llama.cpp binaries
LLAMA_BIN="/mnt/raid0/llm/llama.cpp/build/bin"
for binary in llama-server llama-cli llama-speculative; do
    if [ -x "${LLAMA_BIN}/${binary}" ]; then
        echo "✓ Binary OK: ${binary}"
    else
        echo "✗ Binary missing: ${binary}"
    fi
done

# 8. Check ports
echo ""
echo "Port status:"
for port in 8080 8082 8000; do
    if lsof -i :$port >/dev/null 2>&1; then
        echo "  ⚠ Port $port IN USE"
    else
        echo "  ✓ Port $port available"
    fi
done

# 9. Memory check
free_gb=$(free -g | awk '/^Mem:/{print $7}')
echo ""
echo "Memory: ${free_gb}GB available"

# 10. Summary
echo ""
echo "=== Environment Ready ==="
echo "Python: $(python3 --version)"
echo "uv: $(/mnt/raid0/llm/tools/uv --version)"
echo ""
echo "Next steps:"
echo "  1. Read the plan: cat /home/daniele/.claude/plans/twinkly-sniffing-crescent.md"
echo "  2. Start Phase 0 preflight checks"
echo ""
