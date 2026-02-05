"""
EpisodicStore: SQLite-backed episodic memory with FAISS or numpy embeddings.

Stores memories as (embedding, action, outcome, q_value) tuples with efficient
retrieval by embedding similarity and Q-value ranking.

Supports two embedding backends:
- FAISS (default): O(log n) search, ~70x faster at scale
- NumPy (legacy): O(n) search, for migration/fallback

Enhanced with optional graph integration:
- FailureGraph: Tracks failure patterns and mitigations
- HypothesisGraph: Tracks action-task confidence
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from .faiss_store import FAISSEmbeddingStore, NumpyEmbeddingStore
    from .failure_graph import FailureGraph
    from .hypothesis_graph import HypothesisGraph

logger = logging.getLogger(__name__)

# Default paths (on RAID array per CLAUDE.md requirements)
DEFAULT_DB_PATH = Path("/mnt/raid0/llm/claude/orchestration/repl_memory/sessions")
DEFAULT_EMBEDDINGS_PATH = Path("/mnt/raid0/llm/claude/orchestration/repl_memory/embeddings.npy")


@dataclass
class MemoryEntry:
    """A single episodic memory entry."""

    id: str
    embedding: Optional[np.ndarray]  # Task embedding vector (may be None if not loaded)
    action: str  # Routing decision, escalation action, or exploration code
    action_type: str  # "routing", "escalation", or "exploration"
    context: Dict[str, Any]  # Original task context (task_type, objective, etc.)
    outcome: Optional[str] = None  # "success", "failure", or None if pending
    q_value: float = 0.5  # Initial Q-value (neutral)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    update_count: int = 0  # Number of Q-value updates
    similarity_score: float = 0.0  # Similarity score from retrieval (set by retrieve methods)

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
    SQLite-backed episodic memory store with FAISS or numpy embeddings.

    Memory layout:
    - SQLite: Metadata (action, context, q_value, timestamps)
    - FAISS/Numpy: Embeddings (FAISS default for O(log n) search)

    Supports:
    - Store new memories
    - Retrieve by embedding similarity
    - Update Q-values
    - Query by action type

    Backend selection:
    - use_faiss=True (default): FAISS IndexFlatIP with L2 normalization
    - use_faiss=False: Legacy NumPy mmap (O(n) search, for migration)
    """

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        embeddings_path: Optional[Path] = None,
        embedding_dim: int = 1024,  # BGE-large-en-v1.5 embedding dim
        use_faiss: bool = True,
        flush_interval: float = 10.0,  # Write-behind flush interval (seconds)
    ):
        self.db_path = Path(db_path)
        self.embedding_dim = embedding_dim
        self.use_faiss = use_faiss

        # Write-behind state
        self._dirty = False
        self._flush_interval = flush_interval
        self._flush_task: Optional[asyncio.Task] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # For FAISS, db_path is a directory containing both SQLite and FAISS files
        # For NumPy, we keep backward compatibility with separate paths
        if use_faiss:
            self.storage_dir = self.db_path
            self.sqlite_path = self.storage_dir / "episodic.db"
        else:
            self.storage_dir = self.db_path.parent if self.db_path.suffix == ".db" else self.db_path
            self.sqlite_path = self.db_path if self.db_path.suffix == ".db" else self.db_path / "episodic.db"
            self.embeddings_path = embeddings_path or DEFAULT_EMBEDDINGS_PATH

        # Ensure directories exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

        # Initialize embedding store (FAISS or NumPy)
        self._init_embedding_store()

    def _init_db(self) -> None:
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.sqlite_path) as conn:
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
            # Additional indexes for performance optimization
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_outcome ON memories(outcome)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_action_outcome ON memories(action_type, outcome)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_type_q ON memories(action_type, q_value DESC)
            """)
            conn.commit()

    def _init_embedding_store(self) -> None:
        """Initialize FAISS or NumPy embedding store."""
        if self.use_faiss:
            from .faiss_store import FAISSEmbeddingStore
            self._embedding_store: Union[FAISSEmbeddingStore, NumpyEmbeddingStore] = (
                FAISSEmbeddingStore(path=self.storage_dir, dim=self.embedding_dim)
            )
            logger.info("Initialized FAISS embedding store at %s", self.storage_dir)
        else:
            from .faiss_store import NumpyEmbeddingStore
            self._embedding_store = NumpyEmbeddingStore(
                path=self.storage_dir, dim=self.embedding_dim
            )
            logger.info("Initialized NumPy embedding store at %s", self.storage_dir)

    # Legacy compatibility properties for NumPy backend
    @property
    def _embeddings(self) -> np.ndarray:
        """Legacy access to embeddings array (NumPy backend only)."""
        if hasattr(self._embedding_store, "_embeddings"):
            return self._embedding_store._embeddings
        raise AttributeError("FAISS backend does not expose raw embeddings array")

    @property
    def _next_idx(self) -> int:
        """Legacy access to next index (NumPy backend only)."""
        return self._embedding_store.count

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

        # Store embedding in FAISS/NumPy
        embedding_idx = self._embedding_store.add(memory_id, embedding)

        # Store metadata in SQLite
        with sqlite3.connect(self.sqlite_path) as conn:
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

        # Mark dirty for write-behind (10s flush interval)
        self._dirty = True
        self._schedule_flush()

        return memory_id

    def store_immediate(
        self,
        embedding: np.ndarray,
        action: str,
        action_type: str,
        context: Dict[str, Any],
        outcome: Optional[str] = None,
        initial_q: float = 0.5,
    ) -> str:
        """
        Store a new memory with immediate FAISS persistence (ACID-critical).

        Use this for critical memories that must survive a crash.
        Normal store() uses write-behind for better throughput.

        Args:
            embedding: Task embedding vector
            action: The action taken
            action_type: "routing", "escalation", or "exploration"
            context: Original task context
            outcome: Optional immediate outcome
            initial_q: Initial Q-value

        Returns:
            Memory ID
        """
        memory_id = self.store(embedding, action, action_type, context, outcome, initial_q)
        self.flush()  # Synchronous flush
        return memory_id

    def _schedule_flush(self) -> None:
        """Schedule a write-behind flush if not already scheduled."""
        try:
            loop = asyncio.get_running_loop()
            if self._flush_task is None or self._flush_task.done():
                self._flush_task = loop.create_task(self._flush_loop())
        except RuntimeError:
            # No running event loop - we're in sync context
            # Flush will happen on close() or store_immediate()
            pass

    async def _flush_loop(self) -> None:
        """Background flush loop (10s interval)."""
        while True:
            await asyncio.sleep(self._flush_interval)
            if self._dirty:
                await asyncio.to_thread(self._embedding_store.save)
                self._dirty = False
                logger.debug("Write-behind flush completed")

    def flush(self) -> None:
        """Synchronous flush of embedding store to disk."""
        if self._dirty:
            self._embedding_store.save()
            self._dirty = False
            logger.debug("Synchronous flush completed")

    def close(self) -> None:
        """Close the store, flushing any pending writes."""
        # Cancel flush task if running
        if self._flush_task is not None and not self._flush_task.done():
            self._flush_task.cancel()
            self._flush_task = None

        # Synchronous flush on shutdown
        self.flush()

    def retrieve_by_similarity(
        self,
        query_embedding: np.ndarray,
        k: int = 20,
        action_type: Optional[str] = None,
        min_q_value: float = 0.0,
    ) -> List[MemoryEntry]:
        """
        Retrieve memories by embedding similarity (two-phase retrieval).

        Phase 1: FAISS/NumPy search for top candidates by embedding similarity
        Phase 2: Filter and enrich with SQLite metadata

        Args:
            query_embedding: Query embedding vector
            k: Number of candidates to retrieve
            action_type: Optional filter by action type
            min_q_value: Minimum Q-value threshold

        Returns:
            List of MemoryEntry sorted by similarity (descending)
        """
        # Phase 1: Embedding search (FAISS or NumPy)
        # Over-fetch to account for filtering
        candidates = self._embedding_store.search(query_embedding, k=k * 2)

        if not candidates:
            return []

        # Build score lookup
        score_map = {memory_id: score for memory_id, score in candidates}
        memory_ids = list(score_map.keys())

        # Phase 2: Fetch and filter metadata from SQLite
        placeholders = ",".join("?" * len(memory_ids))
        query = f"""
            SELECT id, embedding_idx, action, action_type, context, outcome, q_value,
                   created_at, updated_at, update_count
            FROM memories
            WHERE id IN ({placeholders})
        """
        params: list = list(memory_ids)

        # Add filters
        filters = []
        if action_type:
            filters.append("action_type = ?")
            params.append(action_type)
        if min_q_value > 0:
            filters.append("q_value >= ?")
            params.append(min_q_value)

        if filters:
            query += " AND " + " AND ".join(filters)

        with sqlite3.connect(self.sqlite_path) as conn:
            rows = conn.execute(query, params).fetchall()

        if not rows:
            return []

        # Combine with similarity scores and build MemoryEntry objects
        results = []
        for row in rows:
            memory_id = row[0]
            embedding_idx = row[1]

            # Get embedding if available (None for FAISS to save memory)
            embedding = None
            if hasattr(self._embedding_store, "get_embedding"):
                embedding = self._embedding_store.get_embedding(embedding_idx)

            entry = MemoryEntry(
                id=memory_id,
                embedding=embedding,
                action=row[2],
                action_type=row[3],
                context=json.loads(row[4]),
                outcome=row[5],
                q_value=row[6],
                created_at=datetime.fromisoformat(row[7]),
                updated_at=datetime.fromisoformat(row[8]),
                update_count=row[9],
                similarity_score=score_map.get(memory_id, 0.0),
            )
            results.append(entry)

        # Sort by similarity score and return top k
        results.sort(key=lambda m: m.similarity_score, reverse=True)
        return results[:k]

    def update_q_value(
        self,
        memory_id: str,
        reward: float,
        learning_rate: float = 0.1,
        temporal_decay_rate: float | None = None,
    ) -> float:
        """
        Update Q-value for a memory using TD-learning style update.

        If temporal_decay_rate is provided, the stored Q-value is first decayed
        toward neutral (0.5) based on elapsed days since last update:

            Q_decayed = 0.5 + (Q_old - 0.5) * decay_rate ^ days_elapsed

        Then the standard TD update is applied:

            Q(m) ← Q_decayed + α(r - Q_decayed)

        Args:
            memory_id: Memory ID to update
            reward: Observed reward (0-1 scale)
            learning_rate: Learning rate α
            temporal_decay_rate: Optional decay rate per day (e.g. 0.99).
                None disables decay. Applied before TD update.

        Returns:
            New Q-value
        """
        now = datetime.utcnow()
        now_iso = now.isoformat()

        with sqlite3.connect(self.sqlite_path) as conn:
            # Get current Q-value and updated_at for decay calculation
            row = conn.execute(
                "SELECT q_value, update_count, updated_at FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()

            if not row:
                raise ValueError(f"Memory {memory_id} not found")

            old_q, update_count, updated_at_str = row

            # Apply temporal decay toward neutral (0.5) if configured
            if temporal_decay_rate is not None and updated_at_str:
                try:
                    updated_at = datetime.fromisoformat(updated_at_str)
                    days_elapsed = (now - updated_at).total_seconds() / 86400.0
                    if days_elapsed > 0:
                        decay_factor = temporal_decay_rate ** days_elapsed
                        old_q = 0.5 + (old_q - 0.5) * decay_factor
                except (ValueError, TypeError):
                    pass  # Skip decay on unparseable timestamps

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
                (new_q, now_iso, update_count + 1, memory_id),
            )
            conn.commit()

        return new_q

    def get_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory by ID."""
        with sqlite3.connect(self.sqlite_path) as conn:
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

        # Get embedding if available
        embedding = None
        embedding_idx = row[1]
        if hasattr(self._embedding_store, "get_embedding"):
            embedding = self._embedding_store.get_embedding(embedding_idx)

        return MemoryEntry(
            id=row[0],
            embedding=embedding,
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
        with sqlite3.connect(self.sqlite_path) as conn:
            if action_type:
                row = conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE action_type = ?",
                    (action_type,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0] if row else 0

    def count_by_combo(
        self, action: str, task_type: Optional[str] = None,
    ) -> int:
        """Count memories matching action and optional task_type in context.

        Uses SQLite json_extract() to filter by task_type stored in the
        JSON context column. Requires SQLite 3.38+ (2022).

        Args:
            action: Action string to match (e.g. "frontdoor:direct").
            task_type: Optional task_type value in context JSON.

        Returns:
            Count of matching memories.
        """
        with sqlite3.connect(self.sqlite_path) as conn:
            if task_type:
                row = conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE action = ? "
                    "AND json_extract(context, '$.task_type') = ?",
                    (action, task_type),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE action = ?",
                    (action,),
                ).fetchone()
        return row[0] if row else 0

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with sqlite3.connect(self.sqlite_path) as conn:
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
            "embeddings_count": self._embedding_store.count,
            "backend": "faiss" if self.use_faiss else "numpy",
        }

    def get_q_outliers(
        self,
        low_threshold: float = 0.3,
        high_threshold: float = 0.8,
        limit: int = 1000,
    ) -> List[MemoryEntry]:
        """
        Get memories with extreme Q-values (outliers) for graph linkage.

        Returns memories where Q < low_threshold (failures) or Q > high_threshold (successes).
        These are the most informative memories for graph seeding.

        Args:
            low_threshold: Q-values below this are considered failures (default 0.3)
            high_threshold: Q-values above this are considered successes (default 0.8)
            limit: Maximum number of memories to return

        Returns:
            List of MemoryEntry sorted by Q-value (lowest first, then highest)
        """
        with sqlite3.connect(self.sqlite_path) as conn:
            rows = conn.execute(
                """
                SELECT id, embedding_idx, action, action_type, context, outcome, q_value,
                       created_at, updated_at, update_count
                FROM memories
                WHERE q_value < ? OR q_value > ?
                ORDER BY
                    CASE WHEN q_value < ? THEN 0 ELSE 1 END,  -- Failures first
                    ABS(q_value - 0.5) DESC  -- Most extreme first
                LIMIT ?
                """,
                (low_threshold, high_threshold, low_threshold, limit),
            ).fetchall()

        results = []
        for row in rows:
            embedding = None
            embedding_idx = row[1]
            if hasattr(self._embedding_store, "get_embedding"):
                embedding = self._embedding_store.get_embedding(embedding_idx)

            results.append(
                MemoryEntry(
                    id=row[0],
                    embedding=embedding,
                    action=row[2],
                    action_type=row[3],
                    context=json.loads(row[4]),
                    outcome=row[5],
                    q_value=row[6],
                    created_at=datetime.fromisoformat(row[7]),
                    updated_at=datetime.fromisoformat(row[8]),
                    update_count=row[9],
                )
            )

        return results

    def get_action_q_summary(self) -> Dict[str, tuple]:
        """Get Q-value statistics grouped by action.

        Returns:
            Dict mapping action -> (count, mean_q, std_q).
            Useful for understanding learned routing preferences.
        """
        with sqlite3.connect(self.sqlite_path) as conn:
            rows = conn.execute(
                """
                SELECT action,
                       COUNT(*) as n,
                       AVG(q_value) as mean_q,
                       -- SQLite doesn't have STDDEV, compute manually
                       CASE
                           WHEN COUNT(*) <= 1 THEN 0.0
                           ELSE SQRT(
                               SUM((q_value - sub.avg) * (q_value - sub.avg)) / (COUNT(*) - 1)
                           )
                       END as std_q
                FROM memories
                JOIN (SELECT action as a, AVG(q_value) as avg FROM memories GROUP BY action) sub
                  ON memories.action = sub.a
                GROUP BY action
                ORDER BY mean_q DESC
                """,
            ).fetchall()

        return {
            row[0]: (row[1], row[2], row[3])
            for row in rows
        }


# Symptom patterns for failure detection
SYMPTOM_PATTERNS: Dict[str, str] = {
    "timeout": r"timeout|timed out|deadline exceeded",
    "0% acceptance": r"0%.*accept|acceptance.*0|acceptance rate.*0",
    "SIGSEGV": r"sigsegv|segmentation fault|signal 11",
    "OOM": r"out of memory|oom|memory exhausted|cannot allocate",
    "BOS mismatch": r"bos.*mismatch|token.*mismatch|bos_token",
    "SWA incompatibility": r"swa|sliding window|kv_cache.*slot",
    "vocab mismatch": r"vocab.*mismatch|vocabulary.*size",
    "empty output": r"empty output|no output|output.*empty",
    "JSON parse error": r"json.*parse|invalid json|json.*error",
    "connection refused": r"connection refused|cannot connect|econnrefused",
}


def extract_symptoms(context: Dict[str, Any], outcome: str) -> List[str]:
    """
    Extract failure symptoms from context and outcome.

    Uses regex patterns to identify known failure modes.

    Args:
        context: Task context dictionary
        outcome: Outcome string (may contain error details)

    Returns:
        List of detected symptom names
    """
    symptoms = []
    error_text = str(context.get("error", "")) + str(outcome)
    error_text = error_text.lower()

    for symptom, pattern in SYMPTOM_PATTERNS.items():
        if re.search(pattern, error_text, re.IGNORECASE):
            symptoms.append(symptom)

    return symptoms or ["unknown"]


class GraphEnhancedStore:
    """
    Wrapper around EpisodicStore with graph integration.

    Provides async graph updates after storing memories:
    - FailureGraph: Records failures with symptoms
    - HypothesisGraph: Updates action-task confidence

    Graph updates happen asynchronously after the response is returned
    to avoid impacting latency.
    """

    def __init__(
        self,
        episodic_store: EpisodicStore,
        failure_graph: Optional["FailureGraph"] = None,
        hypothesis_graph: Optional["HypothesisGraph"] = None,
    ):
        self.store = episodic_store
        self.failure_graph = failure_graph
        self.hypothesis_graph = hypothesis_graph
        self._pending_updates: List[asyncio.Task] = []

    def store_with_graphs(
        self,
        embedding: np.ndarray,
        action: str,
        action_type: str,
        context: Dict[str, Any],
        outcome: Optional[str] = None,
        initial_q: float = 0.5,
        task_type: Optional[str] = None,
    ) -> str:
        """
        Store a memory and schedule async graph updates.

        Args:
            embedding: Task embedding vector
            action: The action taken
            action_type: "routing", "escalation", or "exploration"
            context: Original task context
            outcome: Optional outcome ("success", "failure", or error details)
            initial_q: Initial Q-value
            task_type: Task type for hypothesis tracking

        Returns:
            Memory ID
        """
        # Store in episodic memory (synchronous)
        memory_id = self.store.store(
            embedding=embedding,
            action=action,
            action_type=action_type,
            context=context,
            outcome=outcome,
            initial_q=initial_q,
        )

        # Schedule async graph updates
        if outcome is not None:
            try:
                # Try to get or create an event loop
                try:
                    loop = asyncio.get_running_loop()
                    task = loop.create_task(
                        self._update_graphs(memory_id, action, action_type, context, outcome, task_type)
                    )
                    self._pending_updates.append(task)
                except RuntimeError:
                    # No running loop - run synchronously in a new loop
                    asyncio.run(
                        self._update_graphs(memory_id, action, action_type, context, outcome, task_type)
                    )
            except Exception as e:
                logger.warning("Failed to schedule graph updates: %s", e)

        return memory_id

    async def _update_graphs(
        self,
        memory_id: str,
        action: str,
        action_type: str,
        context: Dict[str, Any],
        outcome: str,
        task_type: Optional[str],
    ) -> None:
        """
        Update failure and hypothesis graphs asynchronously.

        Args:
            memory_id: Memory ID to link
            action: The action taken
            action_type: Action type
            context: Task context
            outcome: Outcome string
            task_type: Task type for hypothesis
        """
        is_failure = outcome.lower() in ("failure", "error") or "error" in outcome.lower()

        # Update failure graph
        if self.failure_graph is not None and is_failure:
            try:
                symptoms = extract_symptoms(context, outcome)
                self.failure_graph.record_failure(
                    memory_id=memory_id,
                    symptoms=symptoms,
                    description=f"{action} failed with: {outcome[:100]}",
                    severity=3,  # Default medium severity
                )
                logger.debug("Recorded failure with symptoms: %s", symptoms)
            except Exception as e:
                logger.warning("Failed to update failure graph: %s", e)

        # Update hypothesis graph (success or failure)
        if self.hypothesis_graph is not None and task_type:
            try:
                # Get or create hypothesis for action-task combination
                hypothesis_id = self.hypothesis_graph.get_or_create_hypothesis(
                    action=action,
                    task_type=task_type,
                    memory_id=memory_id,
                )
                # Add evidence
                evidence_outcome = "success" if not is_failure else "failure"
                self.hypothesis_graph.add_evidence(
                    hypothesis_id=hypothesis_id,
                    outcome=evidence_outcome,
                    source=memory_id,
                )
                logger.debug("Updated hypothesis confidence for %s|%s", action, task_type)
            except Exception as e:
                logger.warning("Failed to update hypothesis graph: %s", e)

    def record_mitigation(
        self,
        failure_memory_id: str,
        action: str,
        worked: bool,
    ) -> Optional[str]:
        """
        Record a mitigation attempt for a prior failure.

        Args:
            failure_memory_id: ID of the failure memory
            action: The mitigation action taken
            worked: Whether the mitigation resolved the failure

        Returns:
            Mitigation ID or None if failure graph unavailable
        """
        if self.failure_graph is None:
            return None

        # Find the failure associated with this memory
        # (The failure graph links memories to failures)
        try:
            # Get symptoms from the failed memory to find the failure
            memory = self.store.get_by_id(failure_memory_id)
            if memory and memory.outcome:
                symptoms = extract_symptoms(memory.context, memory.outcome)
                failures = self.failure_graph.find_matching_failures(symptoms)
                if failures:
                    return self.failure_graph.record_mitigation(
                        failure_id=failures[0].id,
                        action=action,
                        worked=worked,
                    )
        except Exception as e:
            logger.warning("Failed to record mitigation: %s", e)

        return None

    async def flush_pending_updates(self) -> int:
        """
        Wait for all pending graph updates to complete.

        Returns:
            Number of updates completed
        """
        if not self._pending_updates:
            return 0

        completed = 0
        for task in self._pending_updates:
            try:
                await task
                completed += 1
            except Exception as e:
                logger.warning("Graph update failed: %s", e)

        self._pending_updates.clear()
        return completed

    # Delegate core methods to underlying store
    def retrieve_by_similarity(self, *args, **kwargs):
        return self.store.retrieve_by_similarity(*args, **kwargs)

    def update_q_value(self, *args, **kwargs):
        return self.store.update_q_value(*args, **kwargs)

    def get_by_id(self, memory_id: str):
        return self.store.get_by_id(memory_id)

    def count(self, action_type: Optional[str] = None) -> int:
        return self.store.count(action_type)

    def count_by_combo(self, action: str, task_type: Optional[str] = None) -> int:
        return self.store.count_by_combo(action, task_type)

    def get_stats(self) -> Dict[str, Any]:
        stats = self.store.get_stats()
        # Add graph stats if available
        if self.failure_graph:
            stats["failure_graph"] = self.failure_graph.get_stats()
        if self.hypothesis_graph:
            stats["hypothesis_graph"] = self.hypothesis_graph.get_stats()
        return stats
