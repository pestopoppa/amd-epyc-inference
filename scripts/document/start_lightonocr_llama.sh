#!/bin/bash
# Start LightOnOCR-2 GGUF server with parallel worker pool
#
# Usage: ./start_lightonocr_llama.sh [--workers N] [--threads N] [--port N]
#
# Default: 4 workers with 24 threads each = 96 total threads

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../.."

# Default configuration
WORKERS="${LIGHTONOCR_WORKERS:-4}"
THREADS="${LIGHTONOCR_THREADS:-24}"
PORT="${LIGHTONOCR_PORT:-9001}"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--workers N] [--threads N] [--port N]"
      exit 1
      ;;
  esac
done

# Set environment
export HF_HOME=/mnt/raid0/llm/cache/huggingface
export LIGHTONOCR_WORKERS="$WORKERS"
export LIGHTONOCR_THREADS="$THREADS"

echo "=== LightOnOCR-2 GGUF Server ==="
echo "Workers: $WORKERS"
echo "Threads per worker: $THREADS"
echo "Total threads: $((WORKERS * THREADS))"
echo "Port: $PORT"
echo ""

# Activate venv if available
if [[ -f /mnt/raid0/llm/venv/bin/activate ]]; then
  source /mnt/raid0/llm/venv/bin/activate
fi

# Start server
exec python3 "${PROJECT_ROOT}/src/services/lightonocr_llama_server.py" \
  --port "$PORT" \
  --workers "$WORKERS" \
  --threads "$THREADS"
