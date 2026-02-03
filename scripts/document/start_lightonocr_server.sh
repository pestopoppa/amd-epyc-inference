#!/bin/bash
# Start LightOnOCR document processing server
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment library for path variables
# shellcheck source=../lib/env.sh
source "${SCRIPT_DIR}/../lib/env.sh"

PORT="${LIGHTONOCR_PORT:-9001}"
THREADS="${LIGHTONOCR_THREADS:-4}"

# Set thread counts
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

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
