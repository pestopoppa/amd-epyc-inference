#!/bin/bash
set -euo pipefail

# Reset episodic memory to empty state WITHOUT breaking the running API.
#
# What it does:
#   1. Truncates the SQLite memories table (preserves schema)
#   2. Resets FAISS index to empty (rewrites embeddings.faiss + id_map.npy)
#   3. Archives checkpoint JSONL files (which contain seen question IDs)
#   4. Clears seen_questions.jsonl so seeding can re-sample all questions
#   5. Sends SIGHUP to the API process so it reinitializes its in-memory state
#
# What it does NOT do:
#   - Delete database files (which breaks the running API)
#   - Touch sessions.db (session history is independent of episodic memory)
#   - Require the API to be stopped
#
# Usage:
#   ./scripts/session/reset_episodic_memory.sh          # Reset everything
#   ./scripts/session/reset_episodic_memory.sh --keep-seen  # Keep seen_questions

MEMORY_DIR="/mnt/raid0/llm/claude/orchestration/repl_memory/sessions"
EVAL_DIR="/mnt/raid0/llm/claude/benchmarks/results/eval"
DB_PATH="$MEMORY_DIR/episodic.db"
FAISS_PATH="$MEMORY_DIR/embeddings.faiss"
IDMAP_PATH="$MEMORY_DIR/id_map.npy"
SEEN_PATH="$EVAL_DIR/seen_questions.jsonl"

# Match orchestrator API environment for consistency with seeding infra
export HF_HOME="/mnt/raid0/llm/cache/huggingface"
export TMPDIR="/mnt/raid0/llm/tmp"
export ORCHESTRATOR_CACHING="1"
export ORCHESTRATOR_STREAMING="1"
export ORCHESTRATOR_MOCK_MODE="0"
export ORCHESTRATOR_REAL_MODE="1"
export ORCHESTRATOR_SCRIPTS="1"
export ORCHESTRATOR_REACT_MODE="1"
export ORCHESTRATOR_MEMRL="1"
export ORCHESTRATOR_TOOLS="1"
export ORCHESTRATOR_GENERATION_MONITOR="1"
export ORCHESTRATOR_UVICORN_WORKERS="1"

KEEP_SEEN=false
if [[ "${1:-}" == "--keep-seen" ]]; then
    KEEP_SEEN=true
fi

echo "=== Episodic Memory Reset ==="

get_api_pid() {
    local pid=""
    if command -v lsof >/dev/null 2>&1; then
        pid=$(lsof -ti :8000 2>/dev/null || true)
    fi
    if [[ -z "$pid" ]] && command -v fuser >/dev/null 2>&1; then
        pid=$(fuser -n tcp 8000 2>/dev/null | awk '{print $1}' || true)
    fi
    echo "$pid"
}

restart_api() {
    local log_file="/mnt/raid0/llm/claude/logs/orchestrator_autolaunch.log"
    cd /mnt/raid0/llm/claude
    python3 -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --log-level warning \
        >> "$log_file" 2>&1 &
    sleep 3
    if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "  API: restarted OK"
    else
        echo "  API: WARNING — failed to restart, check logs"
    fi
}

# 0. Stop API before touching on-disk state (avoid concurrent reads)
API_PID=$(get_api_pid)
if [[ -n "$API_PID" ]]; then
    echo "  API (PID $API_PID): stopping before reset..."
    kill "$API_PID" 2>/dev/null || true
    sleep 2
fi

# 1. Clear SQLite memories table
python3 -c "
import sqlite3
from pathlib import Path

db_path = Path('$DB_PATH')
if db_path.exists():
    conn = sqlite3.connect(db_path)
    count = conn.execute('SELECT COUNT(*) FROM memories;').fetchone()[0]
    conn.execute('DELETE FROM memories;')
    conn.commit()
    conn.close()
    print(f'  episodic.db: cleared {count} memories (schema preserved)')
else:
    print('  episodic.db: not found (will be created on API start)')
"

# 2. Reset FAISS index to empty (embedding dim derived from config)
python3 -c "
import sys
sys.path.insert(0, '/mnt/raid0/llm/claude')
from pathlib import Path
import numpy as np
try:
    from orchestration.repl_memory.embedder import EmbeddingConfig
    dim = EmbeddingConfig().embedding_dim
    import faiss
    index = faiss.IndexFlatIP(dim)
    faiss.write_index(index, '$FAISS_PATH')
    np.save('$IDMAP_PATH', np.array([], dtype=object))
    print(f'  FAISS index: reset to empty ({dim}-dim)')
except ImportError:
    print('  FAISS: not installed, skipping index reset')
    print('  (index will be recreated on next API start)')
"

# 3. Archive checkpoint JSONL files (contain seen question IDs)
if [[ "$KEEP_SEEN" == "false" ]]; then
    checkpoint_count=$(find "$EVAL_DIR" -maxdepth 1 -name "*.jsonl" -type f 2>/dev/null | wc -l)
    if [[ "$checkpoint_count" -gt 0 ]]; then
        archive_dir="$EVAL_DIR/archive_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$archive_dir"
        mv "$EVAL_DIR"/*.jsonl "$archive_dir/" 2>/dev/null || true
        echo "  checkpoints: archived $checkpoint_count files to $(basename "$archive_dir")"
    else
        echo "  checkpoints: none found"
    fi

    # Recreate empty seen_questions.jsonl
    touch "$SEEN_PATH"
    echo "  seen_questions.jsonl: reset"
else
    echo "  checkpoints: kept (--keep-seen)"
    echo "  seen_questions.jsonl: kept (--keep-seen)"
fi

# 4. Restart API to pick up empty state
echo "  API: restarting to pick up empty state..."
restart_api

echo "=== Done. Ready for seeding. ==="
