#!/bin/bash
set -euo pipefail

# Reset episodic memory to empty state WITHOUT breaking the running API.
#
# What it does:
#   1. Truncates the SQLite memories table (preserves schema)
#   2. Resets FAISS index to empty (rewrites embeddings.faiss + id_map.npy)
#   3. Clears seen_questions.jsonl so seeding can re-sample all questions
#   4. Sends SIGHUP to the API process so it reinitializes its in-memory state
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

KEEP_SEEN=false
if [[ "${1:-}" == "--keep-seen" ]]; then
    KEEP_SEEN=true
fi

echo "=== Episodic Memory Reset ==="

# 1. Clear SQLite memories table
if [[ -f "$DB_PATH" ]]; then
    count=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM memories;" 2>/dev/null || echo "0")
    sqlite3 "$DB_PATH" "DELETE FROM memories;"
    echo "  episodic.db: cleared $count memories (schema preserved)"
else
    echo "  episodic.db: not found (will be created on API start)"
fi

# 2. Reset FAISS index to empty
python3 -c "
import sys
sys.path.insert(0, '/mnt/raid0/llm/claude')
from pathlib import Path
import numpy as np
try:
    import faiss
    index = faiss.IndexFlatIP(896)  # Qwen2.5-0.5B hidden dim
    faiss.write_index(index, '$FAISS_PATH')
    np.save('$IDMAP_PATH', np.array([], dtype=object))
    print('  FAISS index: reset to empty (896-dim)')
except ImportError:
    print('  FAISS: not installed, skipping index reset')
    print('  (index will be recreated on next API start)')
"

# 3. Clear seen questions
if [[ "$KEEP_SEEN" == "false" ]]; then
    if [[ -f "$SEEN_PATH" ]]; then
        count=$(wc -l < "$SEEN_PATH" 2>/dev/null || echo "0")
        truncate -s 0 "$SEEN_PATH"
        echo "  seen_questions.jsonl: cleared $count entries"
    else
        echo "  seen_questions.jsonl: not found"
    fi
else
    echo "  seen_questions.jsonl: kept (--keep-seen)"
fi

# 4. Signal API to reinitialize (if running)
API_PID=$(lsof -ti :8000 2>/dev/null || true)
if [[ -n "$API_PID" ]]; then
    echo "  API (PID $API_PID): restarting to pick up empty state..."
    kill "$API_PID" 2>/dev/null || true
    sleep 2
    # Relaunch API
    cd /mnt/raid0/llm/claude
    python3 -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --log-level warning \
        >> /mnt/raid0/llm/claude/logs/orchestrator_autolaunch.log 2>&1 &
    sleep 3
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "  API: restarted OK"
    else
        echo "  API: WARNING — failed to restart, check logs"
    fi
else
    echo "  API: not running (no restart needed)"
fi

echo "=== Done. Ready for seeding. ==="
