"""Document cache manager for session persistence.

Handles OCR result caching with:
- Hash-based source file change detection (SHA-256)
- Per-session SQLite storage for cached results
- Lazy loading to minimize memory usage

Storage: /workspace/orchestration/repl_memory/sessions/state/{session_id}/ocr_cache.db
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.models.document import DocumentPreprocessResult
from src.session.models import SessionDocument

if TYPE_CHECKING:
    from src.session.sqlite_store import SQLiteSessionStore

logger = logging.getLogger(__name__)

# Base directory for session state
SESSION_STATE_DIR = Path("/workspace/orchestration/repl_memory/sessions/state")


class DocumentCache:
    """Manages per-session document OCR caching.

    Each session gets its own SQLite database for OCR results.
    This allows efficient storage and retrieval without loading
    all cached data into memory.
    """

    def __init__(self, session_id: str, session_store: SQLiteSessionStore | None = None):
        """Initialize document cache for a session.

        Args:
            session_id: Session UUID.
            session_store: Optional SessionStore for document tracking.
        """
        self.session_id = session_id
        self.session_store = session_store

        # Session-specific cache directory
        self.state_dir = SESSION_STATE_DIR / session_id
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # OCR cache database
        self.cache_db_path = self.state_dir / "ocr_cache.db"
        self._init_cache_db()

    def _init_cache_db(self) -> None:
        """Initialize the cache database schema."""
        with sqlite3.connect(self.cache_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_cache (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    cached_at TEXT NOT NULL,
                    total_pages INTEGER,
                    result_json TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_path
                ON document_cache(file_path)
            """)
            conn.commit()

    def get_cached(self, file_path: str | Path) -> DocumentPreprocessResult | None:
        """Get cached preprocessing result if source unchanged.

        Args:
            file_path: Path to source document.

        Returns:
            Cached result if valid, None if cache miss or stale.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.debug(f"Source file not found: {file_path}")
            return None

        # Compute current hash
        current_hash = SessionDocument.compute_file_hash(file_path)

        # Check cache
        with sqlite3.connect(self.cache_db_path) as conn:
            row = conn.execute(
                "SELECT result_json FROM document_cache WHERE file_hash = ?",
                (current_hash,)
            ).fetchone()

        if row is None:
            logger.debug(f"Cache miss for {file_path.name}")
            return None

        # Deserialize cached result
        try:
            data = json.loads(row[0])
            result = DocumentPreprocessResult.from_cache_dict(data)
            logger.info(f"Cache hit for {file_path.name} ({result.total_pages} pages)")
            return result
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Invalid cache entry for {file_path}: {e}")
            self._remove_entry(current_hash)
            return None

    def cache_result(
        self,
        file_path: str | Path,
        result: DocumentPreprocessResult,
        track_in_session: bool = True,
    ) -> str:
        """Cache a preprocessing result.

        Args:
            file_path: Path to source document.
            result: Preprocessing result to cache.
            track_in_session: Whether to add to session's document list.

        Returns:
            File hash used as cache key.
        """
        file_path = Path(file_path)

        # Compute hash
        file_hash = SessionDocument.compute_file_hash(file_path)
        cached_at = datetime.utcnow().isoformat()

        # Serialize result (excludes image_base64)
        result_json = json.dumps(result.to_cache_dict())

        # Store in cache
        with sqlite3.connect(self.cache_db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO document_cache
                (file_hash, file_path, cached_at, total_pages, result_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (file_hash, str(file_path), cached_at, result.total_pages, result_json)
            )
            conn.commit()

        logger.info(f"Cached {file_path.name} ({result.total_pages} pages, {len(result_json)} bytes)")

        # Track in session store if requested
        if track_in_session and self.session_store:
            doc = SessionDocument(
                id=str(uuid.uuid4()),
                session_id=self.session_id,
                file_path=str(file_path),
                file_hash=file_hash,
                processed_at=datetime.utcnow(),
                total_pages=result.total_pages,
                cache_path=str(self.cache_db_path.relative_to(SESSION_STATE_DIR.parent)),
            )
            self.session_store.add_document(doc)

        return file_hash

    def check_changes(self) -> list[dict[str, Any]]:
        """Check all cached documents for source changes.

        Returns:
            List of change info dicts with keys:
                - file_path: Source file path
                - cached_hash: Hash when cached
                - current_hash: Current hash (None if missing)
                - changed: True if file changed
                - exists: True if file still exists
        """
        changes = []

        with sqlite3.connect(self.cache_db_path) as conn:
            rows = conn.execute(
                "SELECT file_hash, file_path FROM document_cache"
            ).fetchall()

        for cached_hash, file_path in rows:
            path = Path(file_path)
            exists = path.exists()

            if exists:
                current_hash = SessionDocument.compute_file_hash(path)
                changed = current_hash != cached_hash
            else:
                current_hash = None
                changed = True

            changes.append({
                "file_path": file_path,
                "cached_hash": cached_hash,
                "current_hash": current_hash,
                "changed": changed,
                "exists": exists,
            })

        return changes

    def invalidate(self, file_path: str | Path) -> bool:
        """Invalidate cache for a specific file.

        Args:
            file_path: Path to source document.

        Returns:
            True if entry was removed.
        """
        file_path = str(file_path)

        with sqlite3.connect(self.cache_db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM document_cache WHERE file_path = ?",
                (file_path,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def invalidate_all(self) -> int:
        """Clear all cached results.

        Returns:
            Number of entries removed.
        """
        with sqlite3.connect(self.cache_db_path) as conn:
            cursor = conn.execute("DELETE FROM document_cache")
            conn.commit()
            return cursor.rowcount

    def _remove_entry(self, file_hash: str) -> None:
        """Remove a cache entry by hash."""
        with sqlite3.connect(self.cache_db_path) as conn:
            conn.execute(
                "DELETE FROM document_cache WHERE file_hash = ?",
                (file_hash,)
            )
            conn.commit()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with keys: total_files, total_pages, cache_size_bytes
        """
        with sqlite3.connect(self.cache_db_path) as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) as total_files,
                    COALESCE(SUM(total_pages), 0) as total_pages,
                    COALESCE(SUM(LENGTH(result_json)), 0) as cache_size
                FROM document_cache
                """
            ).fetchone()

        return {
            "total_files": row[0],
            "total_pages": row[1],
            "cache_size_bytes": row[2],
        }

    def list_cached_files(self) -> list[dict[str, Any]]:
        """List all cached files.

        Returns:
            List of dicts with keys: file_path, file_hash, cached_at, total_pages
        """
        with sqlite3.connect(self.cache_db_path) as conn:
            rows = conn.execute(
                """
                SELECT file_path, file_hash, cached_at, total_pages
                FROM document_cache
                ORDER BY cached_at DESC
                """
            ).fetchall()

        return [
            {
                "file_path": row[0],
                "file_hash": row[1],
                "cached_at": row[2],
                "total_pages": row[3],
            }
            for row in rows
        ]


def get_document_cache(session_id: str, session_store: SQLiteSessionStore | None = None) -> DocumentCache:
    """Get or create document cache for a session.

    This is the main entry point for document caching.

    Args:
        session_id: Session UUID.
        session_store: Optional SessionStore for document tracking.

    Returns:
        DocumentCache instance.
    """
    return DocumentCache(session_id, session_store)
