"""
EpisodicStore: SQLite-backed episodic memory with numpy embeddings.

Stores memories as (embedding, action, outcome, q_value) tuples with efficient
retrieval by embedding similarity and Q-value ranking.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Default paths (on RAID array per CLAUDE.md requirements)
DEFAULT_DB_PATH = Path("/mnt/raid0/llm/claude/orchestration/repl_memory/episodic.db")
DEFAULT_EMBEDDINGS_PATH = Path("/mnt/raid0/llm/claude/orchestration/repl_memory/embeddings.npy")


@dataclass
class MemoryEntry:
    """A single episodic memory entry."""

    id: str
    embedding: np.ndarray  # Task embedding vector
    action: str  # Routing decision, escalation action, or exploration code
    action_type: str  # "routing", "escalation", or "exploration"
    context: Dict[str, Any]  # Original task context (task_type, objective, etc.)
    outcome: Optional[str] = None  # "success", "failure", or None if pending
    q_value: float = 0.5  # Initial Q-value (neutral)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    update_count: int = 0  # Number of Q-value updates

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage (embedding stored separately)."""
        return {
            "id": self.id,
            "action": self.action,
            "action_type": self.action_type,
            "context": self.context,
            "outcome": self.outcome,
            "q_value": self.q_value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "update_count": self.update_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], embedding: np.ndarray) -> MemoryEntry:
        """Deserialize from storage."""
        return cls(
            id=data["id"],
            embedding=embedding,
            action=data["action"],
            action_type=data["action_type"],
            context=data["context"],
            outcome=data.get("outcome"),
            q_value=data["q_value"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            update_count=data.get("update_count", 0),
        )


class EpisodicStore:
    """
    SQLite-backed episodic memory store with numpy embeddings.

    Memory layout:
    - SQLite: Metadata (action, context, q_value, timestamps)
    - Numpy: Embeddings (memory-mapped for efficient similarity search)

    Supports:
    - Store new memories
    - Retrieve by embedding similarity
    - Update Q-values
    - Query by action type
    """

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        embeddings_path: Path = DEFAULT_EMBEDDINGS_PATH,
        embedding_dim: int = 896,  # Qwen2.5-0.5B hidden dim
    ):
        self.db_path = db_path
        self.embeddings_path = embeddings_path
        self.embedding_dim = embedding_dim

        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

        # Load or create embeddings array
        self._load_embeddings()

    def _init_db(self) -> None:
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    embedding_idx INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    context TEXT NOT NULL,
                    outcome TEXT,
                    q_value REAL DEFAULT 0.5,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    update_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_action_type ON memories(action_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_q_value ON memories(q_value DESC)
            """)
            conn.commit()

    def _load_embeddings(self) -> None:
        """Load or create embeddings array (memory-mapped)."""
        if self.embeddings_path.exists():
            # Load existing embeddings
            self._embeddings = np.load(self.embeddings_path, mmap_mode="r+")
            self._next_idx = len(self._embeddings)
        else:
            # Create new embeddings array (start with 1000 slots)
            initial_size = 1000
            self._embeddings = np.zeros((initial_size, self.embedding_dim), dtype=np.float32)
            np.save(self.embeddings_path, self._embeddings)
            self._embeddings = np.load(self.embeddings_path, mmap_mode="r+")
            self._next_idx = 0

    def _grow_embeddings(self) -> None:
        """Double the embeddings array size when full."""
        current_size = len(self._embeddings)
        new_size = current_size * 2

        # Create new array with doubled size
        new_embeddings = np.zeros((new_size, self.embedding_dim), dtype=np.float32)
        new_embeddings[:current_size] = self._embeddings[:]

        # Save and reload
        np.save(self.embeddings_path, new_embeddings)
        self._embeddings = np.load(self.embeddings_path, mmap_mode="r+")

    def store(
        self,
        embedding: np.ndarray,
        action: str,
        action_type: str,
        context: Dict[str, Any],
        outcome: Optional[str] = None,
        initial_q: float = 0.5,
    ) -> str:
        """
        Store a new memory entry.

        Args:
            embedding: Task embedding vector
            action: The action taken (routing, escalation decision, or exploration code)
            action_type: "routing", "escalation", or "exploration"
            context: Original task context
            outcome: Optional immediate outcome
            initial_q: Initial Q-value

        Returns:
            Memory ID
        """
        memory_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        # Grow embeddings array if needed
        if self._next_idx >= len(self._embeddings):
            self._grow_embeddings()

        # Store embedding
        embedding_idx = self._next_idx
        self._embeddings[embedding_idx] = embedding.astype(np.float32)
        self._next_idx += 1

        # Flush to disk
        self._embeddings.flush()

        # Store metadata
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO memories
                (id, embedding_idx, action, action_type, context, outcome, q_value, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    memory_id,
                    embedding_idx,
                    action,
                    action_type,
                    json.dumps(context),
                    outcome,
                    initial_q,
                    now,
                    now,
                ),
            )
            conn.commit()

        return memory_id

    def retrieve_by_similarity(
        self,
        query_embedding: np.ndarray,
        k: int = 20,
        action_type: Optional[str] = None,
        min_q_value: float = 0.0,
    ) -> List[MemoryEntry]:
        """
        Retrieve memories by embedding similarity (Phase 1 of two-phase retrieval).

        Args:
            query_embedding: Query embedding vector
            k: Number of candidates to retrieve
            action_type: Optional filter by action type
            min_q_value: Minimum Q-value threshold

        Returns:
            List of MemoryEntry sorted by similarity (descending)
        """
        # Get candidate indices from database
        with sqlite3.connect(self.db_path) as conn:
            if action_type:
                rows = conn.execute(
                    """
                    SELECT id, embedding_idx, action, action_type, context, outcome, q_value,
                           created_at, updated_at, update_count
                    FROM memories
                    WHERE action_type = ? AND q_value >= ?
                """,
                    (action_type, min_q_value),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, embedding_idx, action, action_type, context, outcome, q_value,
                           created_at, updated_at, update_count
                    FROM memories
                    WHERE q_value >= ?
                """,
                    (min_q_value,),
                ).fetchall()

        if not rows:
            return []

        # Compute similarities
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        similarities = []

        for row in rows:
            embedding_idx = row[1]
            if embedding_idx < len(self._embeddings):
                mem_embedding = self._embeddings[embedding_idx]
                mem_norm = mem_embedding / (np.linalg.norm(mem_embedding) + 1e-8)
                similarity = float(np.dot(query_norm, mem_norm))
                similarities.append((similarity, row))

        # Sort by similarity and take top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_k = similarities[:k]

        # Convert to MemoryEntry objects
        results = []
        for sim, row in top_k:
            entry = MemoryEntry(
                id=row[0],
                embedding=self._embeddings[row[1]],
                action=row[2],
                action_type=row[3],
                context=json.loads(row[4]),
                outcome=row[5],
                q_value=row[6],
                created_at=datetime.fromisoformat(row[7]),
                updated_at=datetime.fromisoformat(row[8]),
                update_count=row[9],
            )
            results.append(entry)

        return results

    def update_q_value(
        self,
        memory_id: str,
        reward: float,
        learning_rate: float = 0.1,
    ) -> float:
        """
        Update Q-value for a memory using TD-learning style update.

        Q(m) ← Q(m) + α(r - Q(m))

        Args:
            memory_id: Memory ID to update
            reward: Observed reward (0-1 scale)
            learning_rate: Learning rate α

        Returns:
            New Q-value
        """
        now = datetime.utcnow().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Get current Q-value
            row = conn.execute(
                "SELECT q_value, update_count FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()

            if not row:
                raise ValueError(f"Memory {memory_id} not found")

            old_q, update_count = row

            # TD-style update
            new_q = old_q + learning_rate * (reward - old_q)

            # Clamp to [0, 1]
            new_q = max(0.0, min(1.0, new_q))

            # Update database
            conn.execute(
                """
                UPDATE memories
                SET q_value = ?, updated_at = ?, update_count = ?
                WHERE id = ?
            """,
                (new_q, now, update_count + 1, memory_id),
            )
            conn.commit()

        return new_q

    def get_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory by ID."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, embedding_idx, action, action_type, context, outcome, q_value,
                       created_at, updated_at, update_count
                FROM memories
                WHERE id = ?
            """,
                (memory_id,),
            ).fetchone()

        if not row:
            return None

        return MemoryEntry(
            id=row[0],
            embedding=self._embeddings[row[1]],
            action=row[2],
            action_type=row[3],
            context=json.loads(row[4]),
            outcome=row[5],
            q_value=row[6],
            created_at=datetime.fromisoformat(row[7]),
            updated_at=datetime.fromisoformat(row[8]),
            update_count=row[9],
        )

    def count(self, action_type: Optional[str] = None) -> int:
        """Count memories, optionally filtered by action type."""
        with sqlite3.connect(self.db_path) as conn:
            if action_type:
                row = conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE action_type = ?",
                    (action_type,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0] if row else 0

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            by_type = conn.execute(
                """
                SELECT action_type, COUNT(*), AVG(q_value)
                FROM memories
                GROUP BY action_type
            """
            ).fetchall()
            avg_q = conn.execute("SELECT AVG(q_value) FROM memories").fetchone()[0]

        return {
            "total_memories": total,
            "by_action_type": {row[0]: {"count": row[1], "avg_q": row[2]} for row in by_type},
            "overall_avg_q": avg_q or 0.0,
            "embeddings_capacity": len(self._embeddings),
            "embeddings_used": self._next_idx,
        }
