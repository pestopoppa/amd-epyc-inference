"""SQLite-backed SessionStore implementation.

Follows the EpisodicStore pattern:
- SQLite for metadata (WAL mode for crash safety)
- Numpy array for embeddings (memory-mapped for efficiency)

Storage location: /workspace/orchestration/repl_memory/sessions/
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.models.document import DocumentPreprocessResult
    from src.session.document_cache import DocumentCache

from src.session.models import (
    Checkpoint,
    Finding,
    FindingSource,
    Session,
    SessionDocument,
    SessionStatus,
)
from src.session.protocol import BaseSessionStore, WhereFilter

logger = logging.getLogger(__name__)

# Default paths — sourced from centralized config (on RAID array per CLAUDE.md)
from src.config import get_config as _get_config

DEFAULT_SESSIONS_DIR = _get_config().paths.sessions_dir
DEFAULT_DB_PATH = DEFAULT_SESSIONS_DIR / "sessions.db"
DEFAULT_EMBEDDINGS_PATH = DEFAULT_SESSIONS_DIR / "session_embeddings.npy"


class SQLiteSessionStore(BaseSessionStore):
    """SQLite-backed session store with numpy embeddings.

    Memory layout:
    - SQLite: Session metadata, documents, findings, checkpoints, tags
    - Numpy: Embeddings (memory-mapped for efficient similarity search)

    Features:
    - WAL mode for crash safety
    - ChromaDB-compatible where filters
    - Embedding storage for future semantic search
    """

    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB_PATH,
        embeddings_path: Path | str = DEFAULT_EMBEDDINGS_PATH,
        embedding_dim: int = 1024,  # BGE-large embedding dim (matches TaskEmbedder)
    ):
        self.db_path = Path(db_path)
        self.embeddings_path = Path(embeddings_path)
        self.embedding_dim = embedding_dim
        self._embeddings_lock = threading.Lock()

        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

        # Load or create embeddings array
        self._load_embeddings()

        logger.info(f"SQLiteSessionStore initialized at {self.db_path}")

    def _init_db(self) -> None:
        """Initialize SQLite database schema with WAL mode."""
        with sqlite3.connect(self.db_path) as conn:
            # Enable WAL mode for better concurrency and crash safety
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")

            # Sessions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    project TEXT,
                    status TEXT DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    last_active TEXT NOT NULL,
                    last_checkpoint_at TEXT,
                    message_count INTEGER DEFAULT 0,
                    working_directory TEXT,
                    task_id TEXT,
                    resume_count INTEGER DEFAULT 0,
                    lineage TEXT,
                    embedding_id INTEGER,
                    summary TEXT,
                    last_topic TEXT
                )
            """)

            # Documents table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_documents (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    processed_at TEXT NOT NULL,
                    total_pages INTEGER DEFAULT 0,
                    cache_path TEXT
                )
            """)

            # Findings table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS findings (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    confirmed INTEGER DEFAULT 0,
                    tags TEXT,
                    source_file TEXT,
                    source_section_id TEXT,
                    source_page INTEGER,
                    source_turn INTEGER
                )
            """)

            # Checkpoints table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    context_hash TEXT NOT NULL,
                    artifacts TEXT NOT NULL,
                    execution_count INTEGER DEFAULT 0,
                    exploration_calls INTEGER DEFAULT 0,
                    message_count INTEGER DEFAULT 0,
                    trigger TEXT NOT NULL,
                    user_globals TEXT DEFAULT '{}',
                    variable_lineage TEXT DEFAULT '{}',
                    skipped_user_globals TEXT DEFAULT '[]'
                )
            """)

            # Tags table (many-to-many)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_tags (
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    tag TEXT NOT NULL,
                    PRIMARY KEY (session_id, tag)
                )
            """)

            # Embeddings metadata table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            # Indexes
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status, last_active DESC)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_session ON session_documents(session_id)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_findings_session ON findings(session_id)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_checkpoints_session ON checkpoints(session_id, created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_embeddings_type ON embeddings(content_type)"
            )

            conn.commit()
            self._migrate_schema(conn)

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """Apply additive schema migrations for existing databases."""
        columns = {
            (row["name"] if isinstance(row, sqlite3.Row) else row[1])
            for row in conn.execute("PRAGMA table_info(checkpoints)").fetchall()
        }
        if "user_globals" not in columns:
            conn.execute("ALTER TABLE checkpoints ADD COLUMN user_globals TEXT DEFAULT '{}'")
        if "variable_lineage" not in columns:
            conn.execute("ALTER TABLE checkpoints ADD COLUMN variable_lineage TEXT DEFAULT '{}'")
        if "skipped_user_globals" not in columns:
            conn.execute("ALTER TABLE checkpoints ADD COLUMN skipped_user_globals TEXT DEFAULT '[]'")
        conn.commit()

    def _load_embeddings(self) -> None:
        """Load or create embeddings array (memory-mapped)."""
        if self.embeddings_path.exists() and self.embeddings_path.stat().st_size > 0:
            try:
                self._embeddings = np.load(self.embeddings_path, mmap_mode="r+")
                if self._embeddings.shape[1] != self.embedding_dim:
                    logger.warning(
                        "Session embeddings dim mismatch (%s != %s); recreating file.",
                        self._embeddings.shape[1],
                        self.embedding_dim,
                    )
                    self.embeddings_path.unlink()
                else:
                    self._next_embedding_idx = len(self._embeddings)
                    return
            except (EOFError, ValueError) as e:
                logger.warning(f"Corrupt embeddings file, recreating: {e}")
                self.embeddings_path.unlink()

        # Create new embeddings array (start with 1000 slots)
        initial_size = 1000
        self._embeddings = np.zeros((initial_size, self.embedding_dim), dtype=np.float32)
        np.save(self.embeddings_path, self._embeddings)
        self._embeddings = np.load(self.embeddings_path, mmap_mode="r+")
        self._next_embedding_idx = 0

    def _grow_embeddings(self) -> None:
        """Double the embeddings array size when full.

        Must be called under ``_embeddings_lock``.
        """
        current_size = len(self._embeddings)
        new_size = current_size * 2

        # Create new array
        new_embeddings = np.zeros((new_size, self.embedding_dim), dtype=np.float32)
        new_embeddings[:current_size] = self._embeddings[:]

        # Save and reload
        np.save(self.embeddings_path, new_embeddings)
        self._embeddings = np.load(self.embeddings_path, mmap_mode="r+")
        logger.info(f"Grew embeddings array from {current_size} to {new_size}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _apply_where_filter(
        self, query: str, where: WhereFilter | None, params: list[Any]
    ) -> tuple[str, list[Any]]:
        """Apply ChromaDB-style where filter to query.

        Supports:
        - {"field": "value"} -> field = value
        - {"field": {"$in": [...]}} -> field IN (...)
        - {"field": {"$gte": N}} -> field >= N
        - {"field": {"$lte": N}} -> field <= N
        - {"field": {"$ne": "value"}} -> field != value
        """
        if not where:
            return query, params

        conditions = []
        for field, value in where.items():
            if isinstance(value, dict):
                # Operator filter
                for op, operand in value.items():
                    if op == "$in":
                        placeholders = ",".join("?" * len(operand))
                        conditions.append(f"{field} IN ({placeholders})")
                        params.extend(operand)
                    elif op == "$gte":
                        conditions.append(f"{field} >= ?")
                        params.append(operand)
                    elif op == "$lte":
                        conditions.append(f"{field} <= ?")
                        params.append(operand)
                    elif op == "$ne":
                        conditions.append(f"{field} != ?")
                        params.append(operand)
                    elif op == "$gt":
                        conditions.append(f"{field} > ?")
                        params.append(operand)
                    elif op == "$lt":
                        conditions.append(f"{field} < ?")
                        params.append(operand)
            else:
                # Simple equality
                conditions.append(f"{field} = ?")
                params.append(value)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        return query, params

    # =========================================================================
    # Session CRUD
    # =========================================================================

    def create_session(self, session: Session) -> Session:
        """Create a new session."""
        with self._get_connection() as conn:
            # Check for duplicate
            existing = conn.execute(
                "SELECT id FROM sessions WHERE id = ?", (session.id,)
            ).fetchone()
            if existing:
                raise ValueError(f"Session {session.id} already exists")

            conn.execute(
                """
                INSERT INTO sessions (
                    id, name, project, status, created_at, last_active,
                    last_checkpoint_at, message_count, working_directory,
                    task_id, resume_count, lineage, embedding_id, summary, last_topic
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    session.name,
                    session.project,
                    session.status.value,
                    session.created_at.isoformat(),
                    session.last_active.isoformat(),
                    session.last_checkpoint_at.isoformat() if session.last_checkpoint_at else None,
                    session.message_count,
                    session.working_directory,
                    session.task_id,
                    session.resume_count,
                    json.dumps(session.lineage),
                    session.embedding_id,
                    session.summary,
                    session.last_topic,
                ),
            )

            # Insert tags
            for tag in session.tags:
                conn.execute(
                    "INSERT OR IGNORE INTO session_tags (session_id, tag) VALUES (?, ?)",
                    (session.id, tag),
                )

            conn.commit()

        logger.debug(f"Created session {session.id}")
        return session

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()

            if not row:
                return None

            # Get tags
            tags = [
                r["tag"]
                for r in conn.execute(
                    "SELECT tag FROM session_tags WHERE session_id = ?", (session_id,)
                )
            ]

            return self._row_to_session(row, tags)

    def _row_to_session(self, row: sqlite3.Row, tags: list[str]) -> Session:
        """Convert a database row to a Session object."""
        return Session(
            id=row["id"],
            name=row["name"],
            project=row["project"],
            tags=tags,
            status=SessionStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            last_active=datetime.fromisoformat(row["last_active"]),
            last_checkpoint_at=(
                datetime.fromisoformat(row["last_checkpoint_at"])
                if row["last_checkpoint_at"]
                else None
            ),
            message_count=row["message_count"],
            working_directory=row["working_directory"],
            task_id=row["task_id"],
            resume_count=row["resume_count"],
            lineage=json.loads(row["lineage"]) if row["lineage"] else [],
            embedding_id=row["embedding_id"],
            summary=row["summary"],
            last_topic=row["last_topic"],
        )

    def update_session(self, session: Session) -> Session:
        """Update an existing session."""
        with self._get_connection() as conn:
            existing = conn.execute(
                "SELECT id FROM sessions WHERE id = ?", (session.id,)
            ).fetchone()
            if not existing:
                raise ValueError(f"Session {session.id} does not exist")

            conn.execute(
                """
                UPDATE sessions SET
                    name = ?, project = ?, status = ?, last_active = ?,
                    last_checkpoint_at = ?, message_count = ?, working_directory = ?,
                    task_id = ?, resume_count = ?, lineage = ?, embedding_id = ?,
                    summary = ?, last_topic = ?
                WHERE id = ?
                """,
                (
                    session.name,
                    session.project,
                    session.status.value,
                    session.last_active.isoformat(),
                    session.last_checkpoint_at.isoformat() if session.last_checkpoint_at else None,
                    session.message_count,
                    session.working_directory,
                    session.task_id,
                    session.resume_count,
                    json.dumps(session.lineage),
                    session.embedding_id,
                    session.summary,
                    session.last_topic,
                    session.id,
                ),
            )

            # Update tags
            conn.execute("DELETE FROM session_tags WHERE session_id = ?", (session.id,))
            for tag in session.tags:
                conn.execute(
                    "INSERT INTO session_tags (session_id, tag) VALUES (?, ?)",
                    (session.id, tag),
                )

            conn.commit()

        return session

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all associated data."""
        with self._get_connection() as conn:
            existing = conn.execute(
                "SELECT id FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if not existing:
                return False

            # Cascade deletes handle related data
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()

        logger.info(f"Deleted session {session_id}")
        return True

    # Valid column names for ORDER BY (whitelist to prevent SQL injection)
    _VALID_ORDER_COLUMNS = frozenset(
        {
            "id",
            "name",
            "project",
            "status",
            "created_at",
            "last_active",
            "last_checkpoint_at",
            "message_count",
            "working_directory",
            "task_id",
            "resume_count",
            "last_topic",
        }
    )

    def list_sessions(
        self,
        where: WhereFilter | None = None,
        order_by: str = "last_active",
        descending: bool = True,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Session]:
        """List sessions with optional filtering."""
        if order_by not in self._VALID_ORDER_COLUMNS:
            raise ValueError(
                f"Invalid order_by column: {order_by!r}. "
                f"Must be one of: {sorted(self._VALID_ORDER_COLUMNS)}"
            )

        query = "SELECT * FROM sessions"
        params: list[Any] = []

        query, params = self._apply_where_filter(query, where, params)

        order_dir = "DESC" if descending else "ASC"
        query += f" ORDER BY {order_by} {order_dir}"

        if limit:
            query += f" LIMIT {limit} OFFSET {offset}"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

            if not rows:
                return []

            # Batch-fetch all tags in one query instead of N+1
            session_ids = [row["id"] for row in rows]
            placeholders = ",".join("?" * len(session_ids))
            tag_rows = conn.execute(
                f"SELECT session_id, tag FROM session_tags WHERE session_id IN ({placeholders})",
                session_ids,
            ).fetchall()

            tags_by_session: dict[str, list[str]] = {}
            for tr in tag_rows:
                tags_by_session.setdefault(tr["session_id"], []).append(tr["tag"])

        return [
            self._row_to_session(row, tags_by_session.get(row["id"], []))
            for row in rows
        ]

    # =========================================================================
    # Documents
    # =========================================================================

    def add_document(self, document: SessionDocument) -> SessionDocument:
        """Add a document to a session."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO session_documents (
                    id, session_id, file_path, file_hash, processed_at, total_pages, cache_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.id,
                    document.session_id,
                    document.file_path,
                    document.file_hash,
                    document.processed_at.isoformat(),
                    document.total_pages,
                    document.cache_path,
                ),
            )
            conn.commit()

        return document

    def get_documents(self, session_id: str) -> list[SessionDocument]:
        """Get all documents for a session."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM session_documents WHERE session_id = ?", (session_id,)
            ).fetchall()

        return [
            SessionDocument(
                id=row["id"],
                session_id=row["session_id"],
                file_path=row["file_path"],
                file_hash=row["file_hash"],
                processed_at=datetime.fromisoformat(row["processed_at"]),
                total_pages=row["total_pages"],
                cache_path=row["cache_path"],
            )
            for row in rows
        ]

    def update_document(self, document: SessionDocument) -> SessionDocument:
        """Update document metadata."""
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE session_documents SET
                    file_hash = ?, processed_at = ?, total_pages = ?, cache_path = ?
                WHERE id = ?
                """,
                (
                    document.file_hash,
                    document.processed_at.isoformat(),
                    document.total_pages,
                    document.cache_path,
                    document.id,
                ),
            )
            conn.commit()

        return document

    # =========================================================================
    # Findings
    # =========================================================================

    def add_finding(self, finding: Finding) -> Finding:
        """Add a key finding to a session."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO findings (
                    id, session_id, content, source, created_at, confidence, confirmed,
                    tags, source_file, source_section_id, source_page, source_turn
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    finding.id,
                    finding.session_id,
                    finding.content,
                    finding.source.value,
                    finding.created_at.isoformat(),
                    finding.confidence,
                    1 if finding.confirmed else 0,
                    json.dumps(finding.tags),
                    finding.source_file,
                    finding.source_section_id,
                    finding.source_page,
                    finding.source_turn,
                ),
            )
            conn.commit()

        return finding

    def get_findings(self, session_id: str) -> list[Finding]:
        """Get all findings for a session."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM findings WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            ).fetchall()

        return [
            Finding(
                id=row["id"],
                session_id=row["session_id"],
                content=row["content"],
                source=FindingSource(row["source"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                confidence=row["confidence"],
                confirmed=bool(row["confirmed"]),
                tags=json.loads(row["tags"]) if row["tags"] else [],
                source_file=row["source_file"],
                source_section_id=row["source_section_id"],
                source_page=row["source_page"],
                source_turn=row["source_turn"],
            )
            for row in rows
        ]

    def update_finding(self, finding: Finding) -> Finding:
        """Update a finding."""
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE findings SET
                    content = ?, confidence = ?, confirmed = ?, tags = ?
                WHERE id = ?
                """,
                (
                    finding.content,
                    finding.confidence,
                    1 if finding.confirmed else 0,
                    json.dumps(finding.tags),
                    finding.id,
                ),
            )
            conn.commit()

        return finding

    def delete_finding(self, finding_id: str) -> bool:
        """Delete a finding."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM findings WHERE id = ?", (finding_id,))
            conn.commit()
            return cursor.rowcount > 0

    # =========================================================================
    # Checkpoints
    # =========================================================================

    def save_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint:
        """Save a checkpoint for crash recovery."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO checkpoints (
                    id, session_id, created_at, context_hash, artifacts,
                    execution_count, exploration_calls, message_count, trigger,
                    user_globals, variable_lineage, skipped_user_globals
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint.id,
                    checkpoint.session_id,
                    checkpoint.created_at.isoformat(),
                    checkpoint.context_hash,
                    json.dumps(checkpoint.artifacts),
                    checkpoint.execution_count,
                    checkpoint.exploration_calls,
                    checkpoint.message_count,
                    checkpoint.trigger,
                    json.dumps(checkpoint.user_globals),
                    json.dumps(checkpoint.variable_lineage),
                    json.dumps(checkpoint.skipped_user_globals),
                ),
            )
            conn.commit()

        # Update session's last_checkpoint_at
        session = self.get_session(checkpoint.session_id)
        if session:
            session.last_checkpoint_at = checkpoint.created_at
            self.update_session(session)

        return checkpoint

    def get_latest_checkpoint(self, session_id: str) -> Checkpoint | None:
        """Get the most recent checkpoint for a session."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM checkpoints
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()

        if not row:
            return None

        return Checkpoint(
            id=row["id"],
            session_id=row["session_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            context_hash=row["context_hash"],
            artifacts=json.loads(row["artifacts"]),
            execution_count=row["execution_count"],
            exploration_calls=row["exploration_calls"],
            message_count=row["message_count"],
            trigger=row["trigger"],
            user_globals=json.loads(row["user_globals"]) if row["user_globals"] else {},
            variable_lineage=(
                json.loads(row["variable_lineage"]) if row["variable_lineage"] else {}
            ),
            skipped_user_globals=(
                json.loads(row["skipped_user_globals"]) if row["skipped_user_globals"] else []
            ),
        )

    def get_checkpoints(self, session_id: str, limit: int = 10) -> list[Checkpoint]:
        """Get recent checkpoints for a session."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM checkpoints
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        return [
            Checkpoint(
                id=row["id"],
                session_id=row["session_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                context_hash=row["context_hash"],
                artifacts=json.loads(row["artifacts"]),
                execution_count=row["execution_count"],
                exploration_calls=row["exploration_calls"],
                message_count=row["message_count"],
                trigger=row["trigger"],
                user_globals=json.loads(row["user_globals"]) if row["user_globals"] else {},
                variable_lineage=(
                    json.loads(row["variable_lineage"]) if row["variable_lineage"] else {}
                ),
                skipped_user_globals=(
                    json.loads(row["skipped_user_globals"]) if row["skipped_user_globals"] else []
                ),
            )
            for row in rows
        ]

    # =========================================================================
    # Search
    # =========================================================================

    def search_sessions(
        self,
        query: str,
        search_in: list[str] | None = None,
        limit: int = 10,
    ) -> list[Session]:
        """Search sessions by text query (simple LIKE search)."""
        if search_in is None:
            search_in = ["name", "summary", "last_topic"]

        conditions = []
        params: list[Any] = []
        for field in search_in:
            conditions.append(f"{field} LIKE ?")
            params.append(f"%{query}%")

        sql = f"SELECT * FROM sessions WHERE {' OR '.join(conditions)} LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()

            if not rows:
                return []

            # Batch-fetch all tags in one query instead of N+1
            session_ids = [row["id"] for row in rows]
            placeholders = ",".join("?" * len(session_ids))
            tag_rows = conn.execute(
                f"SELECT session_id, tag FROM session_tags WHERE session_id IN ({placeholders})",
                session_ids,
            ).fetchall()

            tags_by_session: dict[str, list[str]] = {}
            for tr in tag_rows:
                tags_by_session.setdefault(tr["session_id"], []).append(tr["tag"])

        return [
            self._row_to_session(row, tags_by_session.get(row["id"], []))
            for row in rows
        ]

    def search_findings(
        self,
        query: str,
        session_id: str | None = None,
        limit: int = 10,
    ) -> list[Finding]:
        """Search findings by text query."""
        sql = "SELECT * FROM findings WHERE content LIKE ?"
        params: list[Any] = [f"%{query}%"]

        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)

        sql += " LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [
            Finding(
                id=row["id"],
                session_id=row["session_id"],
                content=row["content"],
                source=FindingSource(row["source"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                confidence=row["confidence"],
                confirmed=bool(row["confirmed"]),
                tags=json.loads(row["tags"]) if row["tags"] else [],
                source_file=row["source_file"],
                source_section_id=row["source_section_id"],
                source_page=row["source_page"],
                source_turn=row["source_turn"],
            )
            for row in rows
        ]

    # =========================================================================
    # Embeddings
    # =========================================================================

    def store_embedding(
        self,
        session_id: str,
        embedding: np.ndarray,
        content_type: str = "session",
    ) -> int:
        """Store an embedding for semantic search."""
        with self._embeddings_lock:
            # Ensure we have space
            if self._next_embedding_idx >= len(self._embeddings):
                self._grow_embeddings()

            # Store embedding
            idx = self._next_embedding_idx
            self._embeddings[idx] = embedding
            self._next_embedding_idx += 1

        # Save metadata
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO embeddings (id, session_id, content_type, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (idx, session_id, content_type, datetime.utcnow().isoformat()),
            )
            conn.commit()

        return idx

    def search_by_embedding(
        self,
        embedding: np.ndarray,
        content_type: str = "session",
        limit: int = 10,
        min_similarity: float = 0.3,
    ) -> list[tuple[str, float]]:
        """Search by embedding similarity using cosine similarity."""
        # Get valid embedding indices for this content type
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT id, session_id FROM embeddings WHERE content_type = ?",
                (content_type,),
            ).fetchall()

        if not rows:
            return []

        # Compute similarities
        indices = [r["id"] for r in rows]
        session_ids = [r["session_id"] for r in rows]

        # Normalize query embedding
        query_norm = embedding / (np.linalg.norm(embedding) + 1e-8)

        results = []
        for idx, sid in zip(indices, session_ids):
            stored_emb = self._embeddings[idx]
            stored_norm = stored_emb / (np.linalg.norm(stored_emb) + 1e-8)
            similarity = float(np.dot(query_norm, stored_norm))

            if similarity >= min_similarity:
                results.append((sid, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    # =========================================================================
    # Tags
    # =========================================================================

    def add_tag(self, session_id: str, tag: str) -> bool:
        """Add a tag to a session."""
        with self._get_connection() as conn:
            try:
                conn.execute(
                    "INSERT INTO session_tags (session_id, tag) VALUES (?, ?)",
                    (session_id, tag),
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False  # Already exists

    def remove_tag(self, session_id: str, tag: str) -> bool:
        """Remove a tag from a session."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM session_tags WHERE session_id = ? AND tag = ?",
                (session_id, tag),
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_sessions_by_tag(self, tag: str) -> list[Session]:
        """Get all sessions with a specific tag."""
        with self._get_connection() as conn:
            # Single JOIN query instead of 2N+1 separate queries
            rows = conn.execute(
                """
                SELECT s.* FROM sessions s
                INNER JOIN session_tags st ON s.id = st.session_id
                WHERE st.tag = ?
                """,
                (tag,),
            ).fetchall()

            if not rows:
                return []

            # Batch-fetch all tags for matched sessions
            session_ids = [row["id"] for row in rows]
            placeholders = ",".join("?" * len(session_ids))
            tag_rows = conn.execute(
                f"SELECT session_id, tag FROM session_tags WHERE session_id IN ({placeholders})",
                session_ids,
            ).fetchall()

            tags_by_session: dict[str, list[str]] = {}
            for tr in tag_rows:
                tags_by_session.setdefault(tr["session_id"], []).append(tr["tag"])

        return [
            self._row_to_session(row, tags_by_session.get(row["id"], []))
            for row in rows
        ]

    # =========================================================================
    # Document Caching Integration
    # =========================================================================

    def get_document_cache(self, session_id: str) -> "DocumentCache":
        """Get or create a document cache for a session.

        Args:
            session_id: The session ID.

        Returns:
            DocumentCache instance configured for this session.
        """
        from src.session.document_cache import DocumentCache

        return DocumentCache(session_id, session_store=self)

    def get_cached_preprocess_result(
        self, session_id: str, file_path: str
    ) -> "DocumentPreprocessResult | None":
        """Get cached preprocessing result for a document.

        This is a convenience method that checks if the source file
        is unchanged and returns the cached result if available.

        Args:
            session_id: The session ID.
            file_path: Path to the source document.

        Returns:
            Cached result if valid, None if not cached or stale.
        """
        from src.session.document_cache import DocumentCache

        cache = DocumentCache(session_id, session_store=self)
        return cache.get_cached(file_path)

    def cache_preprocess_result(
        self,
        session_id: str,
        file_path: str,
        result: "DocumentPreprocessResult",
    ) -> str:
        """Cache a preprocessing result for a document.

        This is a convenience method that stores the result and
        tracks the document in the session.

        Args:
            session_id: The session ID.
            file_path: Path to the source document.
            result: The preprocessing result to cache.

        Returns:
            File hash used as cache key.
        """
        from src.session.document_cache import DocumentCache

        cache = DocumentCache(session_id, session_store=self)
        return cache.cache_result(file_path, result, track_in_session=True)

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def close(self) -> None:
        """Close the store and flush embeddings to disk."""
        # Flush embeddings
        if hasattr(self, "_embeddings"):
            self._embeddings.flush()
        logger.info("SQLiteSessionStore closed")
