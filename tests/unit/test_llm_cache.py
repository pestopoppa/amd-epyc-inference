#!/usr/bin/env python3
"""Tests for the content-addressable LLM cache."""

import json
import time

import pytest

from src.llm_cache import CacheEntry, ContentAddressableCache


@pytest.fixture
def cache_dir(tmp_path):
    """Create a temporary cache directory."""
    return tmp_path / "llm_cache"


@pytest.fixture
def cache(cache_dir):
    """Create a cache instance with short TTL for testing."""
    return ContentAddressableCache(cache_dir, ttl_seconds=3600, max_entries=5)


class TestMakeKey:
    """Tests for cache key generation."""

    def test_deterministic(self):
        key1 = ContentAddressableCache.make_key("prompt", "coder", 512)
        key2 = ContentAddressableCache.make_key("prompt", "coder", 512)
        assert key1 == key2

    def test_different_prompts_different_keys(self):
        key1 = ContentAddressableCache.make_key("prompt1", "coder", 512)
        key2 = ContentAddressableCache.make_key("prompt2", "coder", 512)
        assert key1 != key2

    def test_different_roles_different_keys(self):
        key1 = ContentAddressableCache.make_key("prompt", "coder", 512)
        key2 = ContentAddressableCache.make_key("prompt", "architect", 512)
        assert key1 != key2

    def test_different_n_tokens_different_keys(self):
        key1 = ContentAddressableCache.make_key("prompt", "coder", 512)
        key2 = ContentAddressableCache.make_key("prompt", "coder", 1024)
        assert key1 != key2

    def test_model_hash_affects_key(self):
        key1 = ContentAddressableCache.make_key("prompt", "coder", 512, model_hash="v1")
        key2 = ContentAddressableCache.make_key("prompt", "coder", 512, model_hash="v2")
        assert key1 != key2

    def test_key_is_hex_sha256(self):
        key = ContentAddressableCache.make_key("prompt", "coder", 512)
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)


class TestCacheGetPut:
    """Tests for cache get/put operations."""

    def test_miss_returns_none(self, cache):
        assert cache.get("nonexistent_key") is None

    def test_put_then_get(self, cache):
        key = ContentAddressableCache.make_key("test prompt", "coder", 512)
        cache.put(key, "test response")
        assert cache.get(key) == "test response"

    def test_overwrite(self, cache):
        key = ContentAddressableCache.make_key("test prompt", "coder", 512)
        cache.put(key, "response1")
        cache.put(key, "response2")
        assert cache.get(key) == "response2"

    def test_metadata_stored(self, cache, cache_dir):
        key = ContentAddressableCache.make_key("test", "coder", 512)
        cache.put(key, "response", metadata={"role": "coder", "tps": 33.0})

        path = cache_dir / f"{key}.json"
        data = json.loads(path.read_text())
        assert data["metadata"]["role"] == "coder"
        assert data["metadata"]["tps"] == 33.0


class TestCacheExpiry:
    """Tests for TTL-based expiry."""

    def test_expired_entry_returns_none(self, cache_dir):
        cache = ContentAddressableCache(cache_dir, ttl_seconds=0.01)
        key = ContentAddressableCache.make_key("test", "coder", 512)
        cache.put(key, "response")
        time.sleep(0.02)
        assert cache.get(key) is None

    def test_non_expired_entry_returns_value(self, cache):
        key = ContentAddressableCache.make_key("test", "coder", 512)
        cache.put(key, "response")
        assert cache.get(key) == "response"


class TestCacheEviction:
    """Tests for LRU eviction."""

    def test_eviction_at_max_entries(self, cache_dir):
        cache = ContentAddressableCache(cache_dir, max_entries=3)

        for i in range(5):
            key = ContentAddressableCache.make_key(f"prompt_{i}", "coder", 512)
            cache.put(key, f"response_{i}")

        entries = list(cache_dir.glob("*.json"))
        assert len(entries) <= 3


class TestCacheStats:
    """Tests for cache statistics."""

    def test_initial_stats(self, cache):
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["entries"] == 0

    def test_hit_miss_tracking(self, cache):
        key = ContentAddressableCache.make_key("test", "coder", 512)
        cache.get("nonexistent")  # miss
        cache.put(key, "response")
        cache.get(key)  # hit
        cache.get("another_miss")  # miss

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["hit_rate"] == pytest.approx(1 / 3)

    def test_clear(self, cache):
        key = ContentAddressableCache.make_key("test", "coder", 512)
        cache.put(key, "response")
        cache.clear()

        stats = cache.stats()
        assert stats["entries"] == 0
        assert stats["hits"] == 0


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_not_expired(self):
        entry = CacheEntry(
            key="test", response="r", created_at=time.time(), ttl_seconds=3600
        )
        assert entry.is_expired is False

    def test_expired(self):
        entry = CacheEntry(
            key="test", response="r", created_at=time.time() - 7200, ttl_seconds=3600
        )
        assert entry.is_expired is True
