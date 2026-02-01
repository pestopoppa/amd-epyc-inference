"""Content-addressed cache for two-stage summarization drafts.

This module provides caching for Stage 1 (frontdoor) outputs in the two-stage
summarization pipeline. Caching avoids redundant processing when the same
document is summarized multiple times.

Key features:
- Content-addressed: SHA256 hash of document content as key
- TTL-based expiration: Default 24 hours
- Disk-backed: Persists across server restarts
- Thread-safe: Safe for concurrent access
"""

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class CachedDraft:
    """Cached Stage 1 output."""

    draft_summary: str
    grep_hits: list[dict[str, Any]]
    figures: list[dict[str, Any]]
    timestamp: float
    context_tokens: int
    processing_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CachedDraft":
        return cls(**data)


class DraftCache:
    """Content-addressed cache for Stage 1 summarization drafts.

    Usage:
        cache = DraftCache()

        # Check cache
        key = cache.make_key(document_content)
        if cached := cache.get(key):
            draft, grep_hits, figures = cached.draft_summary, cached.grep_hits, cached.figures
        else:
            # Run Stage 1
            draft, grep_hits, figures = run_stage1(...)
            cache.set(key, CachedDraft(draft, grep_hits, figures, ...))
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        ttl_hours: float | None = None,
    ):
        from src.config import get_config
        _svc = get_config().services
        self.cache_dir = cache_dir or _svc.draft_cache_dir
        if ttl_hours is None:
            ttl_hours = _svc.draft_cache_ttl_hours
        self.ttl_seconds = ttl_hours * 3600
        self._lock = threading.Lock()

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def make_key(self, content: str) -> str:
        """Generate content-addressed key from document content.

        Args:
            content: The full document content to hash

        Returns:
            SHA256 hex digest of the content
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        """Get the filesystem path for a cache key."""
        # Use first 2 chars as subdirectory to avoid too many files in one dir
        subdir = key[:2]
        return self.cache_dir / subdir / f"{key}.json"

    def get(self, key: str) -> CachedDraft | None:
        """Retrieve a cached draft if it exists and hasn't expired.

        Args:
            key: The content hash key

        Returns:
            CachedDraft if found and valid, None otherwise
        """
        cache_path = self._cache_path(key)

        if not cache_path.exists():
            return None

        try:
            with self._lock:
                data = json.loads(cache_path.read_text())

            cached = CachedDraft.from_dict(data)

            # Check TTL
            age = time.time() - cached.timestamp
            if age > self.ttl_seconds:
                # Expired - remove and return None
                self._remove(key)
                return None

            return cached

        except (json.JSONDecodeError, KeyError, TypeError):
            # Corrupted cache entry - remove it
            self._remove(key)
            return None

    def set(self, key: str, draft: CachedDraft) -> None:
        """Store a draft in the cache.

        Args:
            key: The content hash key
            draft: The CachedDraft to store
        """
        cache_path = self._cache_path(key)

        # Ensure subdirectory exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            cache_path.write_text(json.dumps(draft.to_dict(), indent=2))

    def _remove(self, key: str) -> None:
        """Remove a cache entry."""
        cache_path = self._cache_path(key)
        try:
            cache_path.unlink(missing_ok=True)
        except OSError:
            pass

    def cleanup_expired(self) -> int:
        """Remove all expired entries from the cache.

        Returns:
            Number of entries removed
        """
        removed = 0
        now = time.time()

        for subdir in self.cache_dir.iterdir():
            if not subdir.is_dir():
                continue

            for cache_file in subdir.glob("*.json"):
                try:
                    data = json.loads(cache_file.read_text())
                    age = now - data.get("timestamp", 0)
                    if age > self.ttl_seconds:
                        cache_file.unlink()
                        removed += 1
                except (json.JSONDecodeError, OSError):
                    # Corrupted or inaccessible - remove
                    try:
                        cache_file.unlink()
                        removed += 1
                    except OSError:
                        pass

        return removed

    def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_entries = 0
        total_size = 0
        expired_entries = 0
        now = time.time()

        for subdir in self.cache_dir.iterdir():
            if not subdir.is_dir():
                continue

            for cache_file in subdir.glob("*.json"):
                total_entries += 1
                total_size += cache_file.stat().st_size

                try:
                    data = json.loads(cache_file.read_text())
                    age = now - data.get("timestamp", 0)
                    if age > self.ttl_seconds:
                        expired_entries += 1
                except (json.JSONDecodeError, OSError):
                    expired_entries += 1

        return {
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "valid_entries": total_entries - expired_entries,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
            "ttl_hours": self.ttl_seconds / 3600,
        }


# Global cache instance
_draft_cache: DraftCache | None = None
_draft_cache_lock = threading.Lock()


def get_draft_cache() -> DraftCache:
    """Get the global draft cache instance (thread-safe)."""
    global _draft_cache
    if _draft_cache is None:
        with _draft_cache_lock:
            if _draft_cache is None:
                _draft_cache = DraftCache()
    return _draft_cache
