#!/bin/bash
set -euo pipefail

# Reset ALL episodic memory stores to empty state.
#
# What it does:
#   1. Stops the API (avoids concurrent reads on on-disk state)
#   2. Truncates the SQLite memories table (preserves schema)
#   3. Resets FAISS index to empty (rewrites embeddings.faiss + id_map.npy)
#   4. Deletes Kuzu FailureGraph database (dangling MemoryLink refs)
#   5. Deletes Kuzu HypothesisGraph database (dangling evidence refs)
#   6. Clears SkillBank (skills.db + skill FAISS — provenance orphaned)
#   7. Removes legacy session_embeddings.npy
#   8. Clears replay/meta-agent archive.db (deterministic candidate history reset)
#   9. Archives checkpoint JSONL files (which contain seen question IDs)
#  10. Clears seen_questions.jsonl so seeding can re-sample all questions
#  11. Restarts the API to pick up empty state
#
# What it does NOT do:
#   - Delete episodic.db itself (preserves schema, only truncates rows)
#   - Touch sessions.db (session history is independent of episodic memory)
#   - Touch model servers (only the orchestrator API)
#
# Usage:
#   ./scripts/session/reset_episodic_memory.sh                      # Reset everything
#   ./scripts/session/reset_episodic_memory.sh --keep-seen          # Keep seen_questions
#   ./scripts/session/reset_episodic_memory.sh --keep-skills        # Keep SkillBank
#   ./scripts/session/reset_episodic_memory.sh --keep-archive       # Keep replay archive
#   ./scripts/session/reset_episodic_memory.sh --keep-seen --keep-skills --keep-archive

MEMORY_DIR="/mnt/raid0/llm/claude/orchestration/repl_memory/sessions"
KUZU_DIR="/mnt/raid0/llm/claude/orchestration/repl_memory/kuzu_db"
META_ARCHIVE_DIR="/mnt/raid0/llm/claude/orchestration/repl_memory/meta_archive"
EVAL_DIR="/mnt/raid0/llm/claude/benchmarks/results/eval"
DB_PATH="$MEMORY_DIR/episodic.db"
FAISS_PATH="$MEMORY_DIR/embeddings.faiss"
IDMAP_PATH="$MEMORY_DIR/id_map.npy"
SKILLS_DB_PATH="$MEMORY_DIR/skills.db"
SKILL_FAISS_PATH="$MEMORY_DIR/skill_embeddings.faiss"
SKILL_IDMAP_PATH="$MEMORY_DIR/skill_id_map.npy"
SESSION_EMBEDDINGS_PATH="$MEMORY_DIR/session_embeddings.npy"
ARCHIVE_DB_PATH="$META_ARCHIVE_DIR/archive.db"
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
KEEP_SKILLS=false
KEEP_ARCHIVE=false
for arg in "$@"; do
  case "$arg" in
    --keep-seen) KEEP_SEEN=true ;;
    --keep-skills) KEEP_SKILLS=true ;;
    --keep-archive) KEEP_ARCHIVE=true ;;
    *)
      echo "Unknown flag: $arg"
      exit 1
      ;;
  esac
done

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
    >>"$log_file" 2>&1 &
  sleep 3
  if curl -s http://127.0.0.1:8000/health >/dev/null 2>&1; then
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

# 3. Delete Kuzu FailureGraph (contains MemoryLink refs → now dangling)
if [[ -e "$KUZU_DIR/failure_graph" ]]; then
  size=$(du -sh "$KUZU_DIR/failure_graph" 2>/dev/null | cut -f1)
  rm -rf "$KUZU_DIR/failure_graph"
  echo "  FailureGraph: deleted ($size, will be recreated on API start)"
else
  echo "  FailureGraph: not found (clean)"
fi

# 4. Delete Kuzu HypothesisGraph (contains evidence refs → now dangling)
if [[ -e "$KUZU_DIR/hypothesis_graph" ]]; then
  size=$(du -sh "$KUZU_DIR/hypothesis_graph" 2>/dev/null | cut -f1)
  rm -rf "$KUZU_DIR/hypothesis_graph"
  echo "  HypothesisGraph: deleted ($size, will be recreated on API start)"
else
  echo "  HypothesisGraph: not found (clean)"
fi

# 5. Clear SkillBank (skills.db + skill FAISS index)
if [[ "$KEEP_SKILLS" == "false" ]]; then
  python3 -c "
import sqlite3
from pathlib import Path

db_path = Path('$SKILLS_DB_PATH')
if db_path.exists():
    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute('SELECT COUNT(*) FROM skills;').fetchone()[0]
        conn.execute('DELETE FROM skills;')
        conn.commit()
        print(f'  skills.db: cleared {count} skills (schema preserved)')
    except Exception as e:
        print(f'  skills.db: no skills table ({e})')
    conn.close()
else:
    print('  skills.db: not found (clean)')
"
  # Reset skill FAISS if present
  if [[ -f "$SKILL_FAISS_PATH" ]]; then
    python3 -c "
import sys, numpy as np
sys.path.insert(0, '/mnt/raid0/llm/claude')
try:
    import faiss
    from orchestration.repl_memory.embedder import EmbeddingConfig
    dim = EmbeddingConfig().embedding_dim
    index = faiss.IndexFlatIP(dim)
    faiss.write_index(index, '$SKILL_FAISS_PATH')
    np.save('$SKILL_IDMAP_PATH', np.array([], dtype=object))
    print(f'  skill FAISS: reset to empty ({dim}-dim)')
except ImportError:
    print('  skill FAISS: skipped (faiss not installed)')
"
  else
    echo "  skill FAISS: not found (clean)"
  fi
else
  echo "  skills.db: kept (--keep-skills)"
  echo "  skill FAISS: kept (--keep-skills)"
fi

# 6. Remove legacy session_embeddings.npy
if [[ -f "$SESSION_EMBEDDINGS_PATH" ]]; then
  size=$(du -sh "$SESSION_EMBEDDINGS_PATH" 2>/dev/null | cut -f1)
  rm -f "$SESSION_EMBEDDINGS_PATH"
  echo "  session_embeddings.npy: removed ($size)"
else
  echo "  session_embeddings.npy: not found (clean)"
fi

# 7. Clear replay/meta-agent archive (candidate history)
if [[ "$KEEP_ARCHIVE" == "false" ]]; then
  if [[ -f "$ARCHIVE_DB_PATH" ]]; then
    size=$(du -sh "$ARCHIVE_DB_PATH" 2>/dev/null | cut -f1)
    rm -f "$ARCHIVE_DB_PATH"
    echo "  replay archive: deleted archive.db ($size)"
  else
    echo "  replay archive: archive.db not found (clean)"
  fi
else
  echo "  replay archive: kept (--keep-archive)"
fi

# 8. Archive checkpoint JSONL files (contain seen question IDs)
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

# 9. Restart API to pick up empty state
echo "  API: restarting to pick up empty state..."
restart_api

echo "=== Done. Ready for seeding. ==="
