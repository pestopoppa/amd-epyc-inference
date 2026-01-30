#!/usr/bin/env python3
"""Unit tests for prefix caching infrastructure.

Tests cover:
- Prompt canonicalization (Phase C)
- PrefixRouter with LRU eviction (Phase B)
- CachingBackend integration
- RadixCache (Phase D)
"""

import pytest
import time
from unittest.mock import MagicMock, patch

from src.prefix_cache import (
    canonicalize_prompt,
    create_prefix_filter,
    PrefixRouter,
    SlotState,
    CachingBackend,
)
from src.radix_cache import RadixCache, RadixNode, CacheEntry, TokenizedRadixCache
from src.model_server import InferenceRequest


# =============================================================================
# Prompt Canonicalization Tests (Phase C)
# =============================================================================


class TestCanonicalizePrompt:
    """Tests for canonicalize_prompt function."""

    def test_strips_trailing_whitespace(self):
        """Should strip trailing whitespace."""
        assert canonicalize_prompt("hello   ") == "hello"
        assert canonicalize_prompt("hello\n\n") == "hello"
        assert canonicalize_prompt("hello\t  \n") == "hello"

    def test_normalizes_crlf(self):
        """Should normalize CRLF to LF."""
        assert canonicalize_prompt("line1\r\nline2") == "line1\nline2"
        assert canonicalize_prompt("line1\rline2") == "line1\nline2"

    def test_normalizes_iso_timestamps(self):
        """Should normalize ISO timestamps to [TIMESTAMP]."""
        assert canonicalize_prompt(
            "Time: 2024-01-15T10:30:00Z"
        ) == "Time: [TIMESTAMP]"
        assert canonicalize_prompt(
            "Time: 2024-01-15T10:30:00.123Z"
        ) == "Time: [TIMESTAMP]"
        assert canonicalize_prompt(
            "Time: 2024-01-15T10:30:00+05:30"
        ) == "Time: [TIMESTAMP]"

    def test_normalizes_dates(self):
        """Should normalize dates to [DATE]."""
        assert canonicalize_prompt("Date: 2024-01-15") == "Date: [DATE]"
        assert canonicalize_prompt(
            "From 2024-01-15 to 2024-02-20"
        ) == "From [DATE] to [DATE]"

    def test_normalizes_uuids(self):
        """Should normalize UUIDs to [UUID]."""
        assert canonicalize_prompt(
            "ID: 550e8400-e29b-41d4-a716-446655440000"
        ) == "ID: [UUID]"
        assert canonicalize_prompt(
            "ids: ABC12345-DEF6-7890-ABCD-EF1234567890"
        ) == "ids: [UUID]"

    def test_collapses_multiple_blank_lines(self):
        """Should collapse multiple blank lines to single."""
        assert canonicalize_prompt("a\n\n\n\nb") == "a\n\nb"
        assert canonicalize_prompt("a\n\n\n\n\n\nb") == "a\n\nb"

    def test_combined_normalization(self):
        """Should apply all normalizations together."""
        input_prompt = (
            "Request at 2024-01-15T10:30:00Z\r\n"
            "ID: 550e8400-e29b-41d4-a716-446655440000\r\n"
            "\r\n"
            "\r\n"
            "\r\n"
            "Done   "
        )
        expected = (
            "Request at [TIMESTAMP]\n"
            "ID: [UUID]\n"
            "\n"
            "Done"
        )
        assert canonicalize_prompt(input_prompt) == expected


class TestCreatePrefixFilter:
    """Tests for create_prefix_filter function."""

    def test_single_pattern(self):
        """Should filter single pattern."""
        filter_fn = create_prefix_filter([r"\d{4}-\d{4}-\d{4}"])
        assert filter_fn("Card: 1234-5678-9012") == "Card: [FILTERED]"

    def test_multiple_patterns(self):
        """Should filter multiple patterns."""
        filter_fn = create_prefix_filter([
            r"password=\S+",
            r"token=\S+",
        ])
        result = filter_fn("password=secret123 token=abc456")
        assert result == "[FILTERED] [FILTERED]"

    def test_no_match(self):
        """Should leave non-matching text unchanged."""
        filter_fn = create_prefix_filter([r"\d{16}"])
        assert filter_fn("no numbers here") == "no numbers here"


# =============================================================================
# PrefixRouter Tests (Phase B)
# =============================================================================


class TestSlotState:
    """Tests for SlotState dataclass."""

    def test_hit_rate_calculation(self):
        """Should calculate hit rate correctly."""
        slot = SlotState(slot_id=0, hit_count=3, miss_count=1)
        assert slot.hit_rate == 0.75

    def test_hit_rate_zero_total(self):
        """Should return 0 for zero total requests."""
        slot = SlotState(slot_id=0)
        assert slot.hit_rate == 0.0


class TestPrefixRouter:
    """Tests for PrefixRouter class."""

    def test_initialization(self):
        """Should initialize with correct number of slots."""
        router = PrefixRouter(num_slots=4)
        assert router.num_slots == 4
        assert len(router.slots) == 4

    def test_first_request_allocates_slot(self):
        """First request should allocate a slot."""
        router = PrefixRouter(num_slots=4)
        slot_id = router.get_slot_for_prompt("Hello world")

        assert 0 <= slot_id < 4
        assert router.total_routes == 1
        assert router.cache_misses == 1

    def test_same_prefix_hits_same_slot(self):
        """Same prefix should hit same slot."""
        router = PrefixRouter(num_slots=4, prefix_length=10)
        prefix = "System: You are a helpful assistant."

        slot1 = router.get_slot_for_prompt(prefix + " First request")
        slot2 = router.get_slot_for_prompt(prefix + " Second request")

        assert slot1 == slot2
        assert router.cache_hits == 1
        assert router.cache_misses == 1

    def test_different_prefix_different_slot(self):
        """Different prefixes should get different slots."""
        router = PrefixRouter(num_slots=4, prefix_length=10)

        slot1 = router.get_slot_for_prompt("Prefix A: content")
        slot2 = router.get_slot_for_prompt("Prefix B: content")

        assert slot1 != slot2
        assert router.cache_misses == 2

    def test_lru_eviction(self):
        """Should evict LRU slot when all are full."""
        router = PrefixRouter(num_slots=2, prefix_length=8)

        # Fill both slots
        slot_a = router.get_slot_for_prompt("AAAAAAAA first")
        slot_b = router.get_slot_for_prompt("BBBBBBBB second")

        # Third prompt should evict LRU (slot_a)
        slot_c = router.get_slot_for_prompt("CCCCCCCC third")

        assert slot_c == slot_a  # Evicted slot_a

    def test_lru_order_updates_on_access(self):
        """Accessing a slot should update its LRU position."""
        router = PrefixRouter(num_slots=2, prefix_length=8)

        # Fill both slots
        slot_a = router.get_slot_for_prompt("AAAAAAAA first")
        slot_b = router.get_slot_for_prompt("BBBBBBBB second")

        # Access slot_a again (moves to end of LRU)
        router.get_slot_for_prompt("AAAAAAAA again")

        # New prompt should evict slot_b (now LRU)
        slot_c = router.get_slot_for_prompt("CCCCCCCC third")

        assert slot_c == slot_b  # Evicted slot_b

    def test_canonicalization_improves_hits(self):
        """Canonicalization should increase cache hits."""
        router = PrefixRouter(num_slots=4, prefix_length=50)

        # Same logical prompt with different timestamps
        prompt1 = "Request at 2024-01-15T10:30:00Z: Help me"
        prompt2 = "Request at 2024-01-15T11:45:00Z: Help me"

        slot1 = router.get_slot_for_prompt(prompt1, canonicalize=True)
        slot2 = router.get_slot_for_prompt(prompt2, canonicalize=True)

        assert slot1 == slot2  # Should hit same slot
        assert router.cache_hits == 1

    def test_get_stats(self):
        """Should return correct statistics."""
        router = PrefixRouter(num_slots=4)

        router.get_slot_for_prompt("first")
        router.get_slot_for_prompt("first")  # Hit
        router.get_slot_for_prompt("second")

        stats = router.get_stats()
        assert stats["total_routes"] == 3
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 2
        assert stats["hit_rate_pct"] == pytest.approx(33.33, rel=0.01)

    def test_reset_stats(self):
        """Should reset all statistics."""
        router = PrefixRouter(num_slots=4)
        router.get_slot_for_prompt("test")

        router.reset_stats()

        stats = router.get_stats()
        assert stats["total_routes"] == 0
        assert stats["cache_hits"] == 0

    def test_clear(self):
        """Should clear all slot assignments."""
        router = PrefixRouter(num_slots=4)
        router.get_slot_for_prompt("test")

        router.clear()

        assert len(router.prefix_to_slot) == 0
        assert all(s.prefix_hash == "" for s in router.slots.values())


# =============================================================================
# RadixCache Tests (Phase D)
# =============================================================================


class TestRadixNode:
    """Tests for RadixNode dataclass."""

    def test_default_values(self):
        """Should initialize with correct defaults."""
        node = RadixNode()
        assert node.children == {}
        assert node.slot_id is None
        assert node.depth == 0


class TestRadixCache:
    """Tests for RadixCache class."""

    def test_initialization(self):
        """Should initialize with correct slots."""
        cache = RadixCache(num_slots=4)
        assert cache.num_slots == 4
        assert len(cache._available_slots) == 4

    def test_insert_and_find(self):
        """Should insert and find prefixes correctly."""
        cache = RadixCache(num_slots=4, min_prefix_length=2)
        tokens = [1, 2, 3, 4, 5]

        slot = cache.insert(tokens)

        # Full match
        length, found_slot = cache.find_longest_prefix(tokens)
        assert length == 5
        assert found_slot == slot

    def test_find_partial_prefix(self):
        """Should find longest matching prefix."""
        cache = RadixCache(num_slots=4, min_prefix_length=2)

        cache.insert([1, 2, 3, 4, 5], slot_id=0)

        # Partial match - query diverges at position 3
        length, slot = cache.find_longest_prefix([1, 2, 3, 10, 20])
        assert length == 3  # Matched common prefix [1, 2, 3]
        assert slot == 0

    def test_find_no_match(self):
        """Should return None for non-matching prefix."""
        cache = RadixCache(num_slots=4, min_prefix_length=2)
        cache.insert([1, 2, 3], slot_id=0)

        length, slot = cache.find_longest_prefix([10, 20, 30])
        assert length == 0
        assert slot is None

    def test_multiple_prefixes(self):
        """Should handle multiple distinct prefixes."""
        cache = RadixCache(num_slots=4, min_prefix_length=2)

        cache.insert([1, 2, 3], slot_id=0)
        cache.insert([10, 20, 30], slot_id=1)

        len1, slot1 = cache.find_longest_prefix([1, 2, 3, 4])
        len2, slot2 = cache.find_longest_prefix([10, 20, 30, 40])

        assert slot1 == 0
        assert slot2 == 1

    def test_lru_eviction(self):
        """Should evict LRU entries when full."""
        cache = RadixCache(num_slots=2, min_prefix_length=2)

        cache.insert([1, 2, 3], slot_id=0)
        cache.insert([4, 5, 6], slot_id=1)

        # This should trigger eviction
        slot = cache.insert([7, 8, 9])

        assert slot == 0  # Evicted first slot
        assert cache.evictions == 1

    def test_find_or_allocate_hit(self):
        """find_or_allocate should return hit on match."""
        cache = RadixCache(num_slots=4, min_prefix_length=2)
        cache.insert([1, 2, 3], slot_id=0)

        length, slot, is_hit = cache.find_or_allocate([1, 2, 3, 4, 5])
        assert is_hit is True
        assert slot == 0
        assert length == 3

    def test_find_or_allocate_miss(self):
        """find_or_allocate should allocate on miss."""
        cache = RadixCache(num_slots=4, min_prefix_length=2)

        length, slot, is_hit = cache.find_or_allocate([1, 2, 3])
        assert is_hit is False
        assert slot in range(4)

    def test_get_stats(self):
        """Should return correct statistics."""
        cache = RadixCache(num_slots=4, min_prefix_length=2)

        cache.insert([1, 2, 3])
        cache.find_longest_prefix([1, 2, 3])  # Hit
        cache.find_longest_prefix([9, 9, 9])  # Miss

        stats = cache.get_stats()
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1
        assert stats["active_entries"] == 1

    def test_get_hot_prefixes(self):
        """Should return prefixes sorted by hit count."""
        cache = RadixCache(num_slots=4, min_prefix_length=2)

        cache.insert([1, 2, 3], slot_id=0)
        cache.insert([4, 5, 6], slot_id=1)

        # Generate hits
        for _ in range(5):
            cache.find_longest_prefix([4, 5, 6, 7])  # More hits
        for _ in range(2):
            cache.find_longest_prefix([1, 2, 3, 4])

        hot = cache.get_hot_prefixes(top_n=2)
        assert hot[0].slot_id == 1  # More hits
        assert hot[0].hit_count == 5

    def test_clear(self):
        """Should clear all entries."""
        cache = RadixCache(num_slots=4, min_prefix_length=2)
        cache.insert([1, 2, 3])

        cache.clear()

        assert len(cache.entries) == 0
        assert len(cache._available_slots) == 4
        assert cache.total_lookups == 0

    def test_iter_prefixes(self):
        """Should iterate over all cached prefixes."""
        cache = RadixCache(num_slots=4, min_prefix_length=2)
        cache.insert([1, 2, 3], slot_id=0)
        cache.insert([4, 5, 6], slot_id=1)

        prefixes = list(cache.iter_prefixes())
        assert len(prefixes) == 2
        assert ([1, 2, 3], 0) in prefixes
        assert ([4, 5, 6], 1) in prefixes

    def test_min_prefix_length(self):
        """Should skip prefixes shorter than minimum."""
        cache = RadixCache(num_slots=4, min_prefix_length=16)

        # Short prefix - should still allocate slot but not cache
        slot = cache.insert([1, 2, 3])

        assert slot in range(4)
        # The entry won't be in cache since it's too short
        _, found_slot = cache.find_longest_prefix([1, 2, 3])
        assert found_slot is None  # Not cached due to min_prefix_length


class TestTokenizedRadixCache:
    """Tests for TokenizedRadixCache class."""

    def test_requires_tokenizer(self):
        """Should raise error without tokenizer."""
        cache = TokenizedRadixCache(num_slots=4)

        with pytest.raises(ValueError, match="No tokenizer configured"):
            cache.insert_prompt("Hello world")

    def test_with_tokenizer(self):
        """Should work with tokenizer function."""
        # Simple tokenizer that returns byte values
        def tokenize(text):
            return list(text.encode("utf-8"))

        cache = TokenizedRadixCache(num_slots=4, min_prefix_length=2)
        cache.set_tokenizer(tokenize)

        slot = cache.insert_prompt("Hello")
        length, found_slot = cache.find_longest_prefix_for_prompt("Hello world")

        assert found_slot == slot
        assert length == len("Hello".encode("utf-8"))


# =============================================================================
# CachingBackend Tests
# =============================================================================


class TestCachingBackend:
    """Tests for CachingBackend class."""

    def test_routes_to_slot(self):
        """Should route prompts through PrefixRouter."""
        mock_backend = MagicMock()
        mock_backend.infer.return_value = MagicMock(
            role="test",
            output="response",
            tokens_generated=10,
        )
        mock_backend.get_cache_stats.return_value = MagicMock(
            hit_rate=50.0,
            token_savings_rate=30.0,
            total_prompt_tokens=100,
            cached_prompt_tokens=30,
        )

        router = PrefixRouter(num_slots=4)
        caching = CachingBackend(mock_backend, router)

        # Create request and mock role_config
        request = InferenceRequest(role="test", prompt="Test prompt")
        mock_role_config = MagicMock()

        caching.infer(mock_role_config, request)

        # Router should have tracked the request
        assert router.total_routes == 1

    def test_canonicalizes_prompt(self):
        """Should canonicalize prompts when enabled."""
        mock_backend = MagicMock()
        mock_backend.infer.return_value = MagicMock()
        mock_backend.get_cache_stats.return_value = MagicMock(
            hit_rate=0, token_savings_rate=0, total_prompt_tokens=0, cached_prompt_tokens=0
        )

        caching = CachingBackend(mock_backend, canonicalize=True)

        request = InferenceRequest(role="test", prompt="Time: 2024-01-15T10:00:00Z")
        mock_role_config = MagicMock()

        caching.infer(mock_role_config, request)

        # Canonicalization is ONLY used for cache key routing (get_slot_for_prompt),
        # NOT for mutating the prompt sent to the backend. The original prompt
        # must be preserved to avoid [DATE]/[TIMESTAMP] contamination in output.
        call_args = mock_backend.infer.call_args
        assert call_args[0][1].prompt == "Time: 2024-01-15T10:00:00Z"

    def test_get_hit_rate(self):
        """Should return correct hit rate."""
        mock_backend = MagicMock()
        router = PrefixRouter(num_slots=4)
        caching = CachingBackend(mock_backend, router)

        # Simulate some requests
        router.get_slot_for_prompt("test1")
        router.get_slot_for_prompt("test1")  # Hit

        assert caching.get_hit_rate() == 0.5

    def test_get_stats(self):
        """Should combine router and backend stats."""
        mock_backend = MagicMock()
        mock_backend.get_cache_stats.return_value = MagicMock(
            hit_rate=75.0,
            token_savings_rate=50.0,
            total_prompt_tokens=1000,
            cached_prompt_tokens=500,
        )

        router = PrefixRouter(num_slots=4)
        router.get_slot_for_prompt("test")

        caching = CachingBackend(mock_backend, router)
        stats = caching.get_stats()

        assert stats["router_total_routes"] == 1
        assert stats["backend_hit_rate"] == 0.75
        assert stats["total_prompt_tokens"] == 1000


class TestCachingBackendPersistence:
    """Tests for CachingBackend hot prefix persistence (Phase E)."""

    def test_save_hot_prefixes_no_cache_dir(self):
        """Should return 0 when no cache_dir configured."""
        mock_backend = MagicMock()
        caching = CachingBackend(mock_backend, cache_dir=None)

        saved = caching.save_hot_prefixes()
        assert saved == 0

    def test_restore_hot_prefixes_no_manifest(self, tmp_path):
        """Should return 0 when no manifest exists."""
        mock_backend = MagicMock()
        caching = CachingBackend(mock_backend, cache_dir=str(tmp_path))

        restored = caching.restore_hot_prefixes()
        assert restored == 0

    def test_save_and_restore_cycle(self, tmp_path):
        """Should save and restore hot prefixes."""
        mock_backend = MagicMock()

        # Make save_slot actually create the file
        def mock_save_slot(slot_id, filename):
            with open(filename, "wb") as f:
                f.write(b"mock kv cache data")
            return True

        mock_backend.save_slot.side_effect = mock_save_slot
        mock_backend.restore_slot.return_value = True

        router = PrefixRouter(num_slots=4)
        caching = CachingBackend(mock_backend, router, cache_dir=str(tmp_path))

        # Generate some slot usage with actual hits (same prompt)
        router.get_slot_for_prompt("prefix1 content")  # Miss, allocates slot
        router.get_slot_for_prompt("prefix1 content")  # Hit
        router.get_slot_for_prompt("prefix1 content")  # Hit

        # Save hot prefixes
        saved = caching.save_hot_prefixes()

        # Restore in new backend
        caching2 = CachingBackend(mock_backend, PrefixRouter(num_slots=4), cache_dir=str(tmp_path))
        restored = caching2.restore_hot_prefixes()

        assert saved > 0
        assert restored > 0

    def test_clear_saved_prefixes(self, tmp_path):
        """Should clear all saved cache files."""
        mock_backend = MagicMock()
        caching = CachingBackend(mock_backend, cache_dir=str(tmp_path))

        # Create some files
        (tmp_path / "slot_0_abc.bin").write_bytes(b"test")
        (tmp_path / "manifest.json").write_text("{}")

        cleared = caching.clear_saved_prefixes()
        assert cleared == 2
        assert len(list(tmp_path.iterdir())) == 0
