#!/bin/bash
# Start LightOnOCR document processing server
set -euo pipefail

PORT="${LIGHTONOCR_PORT:-9001}"
THREADS="${LIGHTONOCR_THREADS:-4}"

# Set thread counts
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Check port
if lsof -i ":$PORT" >/dev/null 2>&1; then
  echo "ERROR: Port $PORT is already in use"
  exit 1
fi

# Activate venv
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
fi

echo "Starting LightOnOCR server on port $PORT with $THREADS threads..."

# Run server
exec python "$PROJECT_ROOT/src/services/lightonocr_server.py" --port "$PORT"
