"""Accurate token counting via llama-server /tokenize endpoint.

Provides LlamaTokenizer for pre-flight token counting with LRU cache
and graceful fallback to character-count heuristic on timeout/error.
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict

import httpx

log = logging.getLogger(__name__)

# LRU cache capacity for token counts
_DEFAULT_CACHE_SIZE = 1000


class LlamaTokenizer:
    """Accurate token counting via llama-server /tokenize.

    Uses POST /tokenize on a running llama-server to get exact token
    counts.  Falls back to ``len(text) // 4`` on timeout or error.

    An internal LRU cache avoids repeated HTTP calls for the same text.
    Cache keys use a hash of the first 200 chars + total length to keep
    hashing cheap for large prompts.

    Args:
        base_url: llama-server base URL (e.g. ``http://localhost:8080``).
        timeout: HTTP timeout in seconds (default 0.5s).
        cache_size: Max entries in the LRU cache.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 0.5,
        cache_size: int = _DEFAULT_CACHE_SIZE,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._cache_size = cache_size
        self._cache: OrderedDict[str, int] = OrderedDict()
        self._client = httpx.Client(timeout=timeout)
        # Stats
        self.total_calls = 0
        self.cache_hits = 0
        self.fallback_count = 0

    def _cache_key(self, text: str) -> str:
        """Compute a cheap cache key from text prefix + length."""
        prefix = text[:200]
        raw = f"{prefix}|{len(text)}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> int | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, key: str, value: int) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def count_tokens(self, text: str) -> int:
        """Count tokens accurately via /tokenize, with LRU cache.

        Falls back to ``count_tokens_approx`` on any error.

        Args:
            text: Input text to tokenize.

        Returns:
            Token count.
        """
        self.total_calls += 1

        key = self._cache_key(text)
        cached = self._cache_get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        try:
            resp = self._client.post(
                f"{self.base_url}/tokenize",
                json={"content": text},
            )
            resp.raise_for_status()
            tokens = resp.json().get("tokens", [])
            count = len(tokens)
            self._cache_put(key, count)
            return count
        except Exception as exc:
            self.fallback_count += 1
            log.debug("Tokenizer fallback (server error): %s", exc)
            count = self.count_tokens_approx(text)
            self._cache_put(key, count)
            return count

    def count_tokens_approx(self, text: str) -> int:
        """Approximate token count using character heuristic.

        Args:
            text: Input text.

        Returns:
            Estimated token count (``len(text) // 4``).
        """
        return len(text) // 4

    def get_stats(self) -> dict[str, int | float]:
        """Return tokenizer usage statistics."""
        return {
            "total_calls": self.total_calls,
            "cache_hits": self.cache_hits,
            "cache_size": len(self._cache),
            "fallback_count": self.fallback_count,
            "cache_hit_rate": (
                self.cache_hits / self.total_calls if self.total_calls > 0 else 0.0
            ),
        }

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()
