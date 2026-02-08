"""Content-addressable LLM response cache (Lobster pattern).

SHA-256 keyed cache for complete prompt→response pairs. Distinct from
prefix caching (KV cache reuse) — this saves the full response to disk.

Only active when features().content_cache is True.

Design decisions:
- JSON files on disk (like Lobster), not SQLite — simpler, no schema migration
- TTL: 168h (1 week) default, configurable
- Max entries: 10K default with LRU eviction
- Skip caching when temperature > 0 (non-deterministic)
- Include model_hash to invalidate on model swap

Highest value targets (slowest models):
- architect_general (6.75 t/s)
- ingest_long_context (6.3 t/s)
- architect_coding (10.3 t/s)

Usage:
    from src.llm_cache import ContentAddressableCache

    cache = ContentAddressableCache("/mnt/raid0/llm/claude/cache/llm_responses")
    key = cache.make_key(prompt, role, n_tokens)

    # Check cache
    cached = cache.get(key)
    if cached is not None:
        return cached

    # After inference
    cache.put(key, response, metadata={"role": role})
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 168 * 3600  # 1 week
DEFAULT_MAX_ENTRIES = 10_000


@dataclass
class CacheEntry:
    """A single cached response."""

    key: str
    response: str
    created_at: float
    ttl_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds


class ContentAddressableCache:
    """SHA-256 keyed response cache for LLM calls.

    Files are stored as JSON in a flat directory structure with the cache
    key as filename. LRU eviction runs lazily on put() when max entries
    is exceeded.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        model_hash: str = "",
    ) -> None:
        self._dir = Path(cache_dir)
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._model_hash = model_hash
        self._hits = 0
        self._misses = 0
        self._dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(
        prompt: str,
        role: str,
        n_tokens: int,
        model_hash: str = "",
    ) -> str:
        """Generate SHA-256 cache key from prompt parameters.

        Args:
            prompt: The full prompt text.
            role: The role name.
            n_tokens: Max tokens to generate.
            model_hash: Model identifier for invalidation on swap.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        payload = f"{prompt}|{role}|{n_tokens}|{model_hash}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> str | None:
        """Look up a cached response.

        Args:
            key: Cache key from make_key().

        Returns:
            Cached response string, or None on miss/expired.
        """
        path = self._dir / f"{key}.json"
        if not path.exists():
            self._misses += 1
            return None

        try:
            data = json.loads(path.read_text())
            entry = CacheEntry(**data)

            if entry.is_expired:
                path.unlink(missing_ok=True)
                self._misses += 1
                return None

            # Update access time for LRU
            os.utime(path, None)
            self._hits += 1
            return entry.response

        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.debug("Cache read error for %s: %s", key[:12], e)
            path.unlink(missing_ok=True)
            self._misses += 1
            return None

    def put(
        self,
        key: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a response in the cache.

        Runs LRU eviction if max entries exceeded.

        Args:
            key: Cache key from make_key().
            response: The LLM response to cache.
            metadata: Optional metadata (role, timing, etc.).
        """
        entry = CacheEntry(
            key=key,
            response=response,
            created_at=time.time(),
            ttl_seconds=self._ttl,
            metadata=metadata or {},
        )

        path = self._dir / f"{key}.json"
        try:
            path.write_text(json.dumps(entry.__dict__, ensure_ascii=False))
        except OSError as e:
            logger.warning("Cache write failed for %s: %s", key[:12], e)
            return

        # Lazy LRU eviction
        self._maybe_evict()

    def _maybe_evict(self) -> None:
        """Evict oldest entries if over max_entries."""
        try:
            entries = list(self._dir.glob("*.json"))
        except OSError:
            return

        if len(entries) <= self._max_entries:
            return

        # Sort by access time (oldest first)
        entries.sort(key=lambda p: p.stat().st_atime)

        # Remove oldest until within limit
        to_remove = len(entries) - self._max_entries
        for path in entries[:to_remove]:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

        logger.debug("Cache eviction: removed %d entries", to_remove)

    def clear(self) -> None:
        """Remove all cached entries."""
        for path in self._dir.glob("*.json"):
            path.unlink(missing_ok=True)
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with hits, misses, hit_rate, entries, and size_bytes.
        """
        try:
            entries = list(self._dir.glob("*.json"))
            size = sum(p.stat().st_size for p in entries)
        except OSError:
            entries = []
            size = 0

        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "entries": len(entries),
            "size_bytes": size,
            "max_entries": self._max_entries,
            "ttl_seconds": self._ttl,
        }
