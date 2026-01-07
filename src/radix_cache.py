#!/usr/bin/env python3
"""Radix tree cache manager for optimal prefix matching.

This module provides a radix tree implementation for efficient prefix
matching in KV cache management. It supports:
- O(n) prefix lookup where n is the prefix length
- LRU eviction for memory management
- Integration with llama-server's slot-based caching

Usage:
    from src.radix_cache import RadixCache

    cache = RadixCache(num_slots=4)

    # Insert a prefix with slot assignment
    cache.insert([1, 2, 3, 4, 5], slot_id=0)

    # Find longest matching prefix
    length, slot = cache.find_longest_prefix([1, 2, 3, 10, 20])
    # Returns: (3, 0) - matched 3 tokens, slot 0

See research/radix_attention_handoff.md for implementation plan.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Iterator

logger = logging.getLogger(__name__)


@dataclass
class RadixNode:
    """Node in the radix tree.

    Each node represents a point in the token sequence where branching
    occurs. Leaf nodes (or nodes with slot_id set) represent cached
    prefix boundaries.

    Attributes:
        children: Mapping from token ID to child nodes.
        slot_id: Server slot holding this prefix's KV cache (None if internal node).
        last_access: Timestamp of last access (for LRU eviction).
        depth: Depth in the tree (number of tokens from root).
    """

    children: dict[int, RadixNode] = field(default_factory=dict)
    slot_id: int | None = None
    last_access: float = field(default_factory=time.time)
    depth: int = 0


@dataclass
class CacheEntry:
    """Entry tracking a cached prefix.

    Attributes:
        tokens: The token sequence for this prefix.
        slot_id: Server slot holding the KV cache.
        last_access: Timestamp of last access.
        hit_count: Number of cache hits.
    """

    tokens: list[int]
    slot_id: int
    last_access: float = field(default_factory=time.time)
    hit_count: int = 0


class RadixCache:
    """Radix tree cache manager for prefix matching.

    Implements a radix tree (compressed trie) for efficient prefix matching.
    Each path from root to a marked node represents a cached prefix with
    an associated server slot.

    The tree enables O(n) lookups where n is the query length, and supports:
    - Finding the longest cached prefix for a new token sequence
    - LRU eviction when slots are exhausted
    - Statistics tracking for cache performance

    Attributes:
        root: Root node of the radix tree.
        num_slots: Number of server slots available.
        entries: Mapping from slot_id to CacheEntry.
        total_lookups: Total number of prefix lookups.
        cache_hits: Number of successful prefix matches.
    """

    def __init__(self, num_slots: int = 4, min_prefix_length: int = 16):
        """Initialize the radix cache.

        Args:
            num_slots: Number of server slots to manage.
            min_prefix_length: Minimum prefix length to cache (avoids small prefixes).
        """
        self.root = RadixNode(depth=0)
        self.num_slots = num_slots
        self.min_prefix_length = min_prefix_length

        # Slot -> entry mapping
        self.entries: dict[int, CacheEntry] = {}

        # Available slots (not yet assigned)
        self._available_slots: set[int] = set(range(num_slots))

        # Statistics
        self.total_lookups = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.evictions = 0

    def insert(self, tokens: list[int], slot_id: int | None = None) -> int:
        """Insert a token sequence into the cache.

        Creates or updates a path in the radix tree for the given token
        sequence. If no slot_id is provided, allocates one (with LRU
        eviction if necessary).

        Args:
            tokens: Token sequence to cache.
            slot_id: Specific slot to use (allocates if None).

        Returns:
            Slot ID assigned to this prefix.
        """
        if len(tokens) < self.min_prefix_length:
            logger.debug(f"Skipping short prefix ({len(tokens)} < {self.min_prefix_length})")
            # Still need to return a slot for the request
            if slot_id is not None:
                return slot_id
            return self._allocate_slot(tokens)

        # Allocate slot if needed
        if slot_id is None:
            slot_id = self._allocate_slot(tokens)
        else:
            # Explicit slot provided - remove from available set
            self._available_slots.discard(slot_id)

        # Navigate/create path in tree, marking all nodes with slot_id
        # This enables partial prefix matching (any prefix of the cached
        # sequence can be reused)
        node = self.root
        for i, token in enumerate(tokens):
            if token not in node.children:
                node.children[token] = RadixNode(depth=i + 1)
            node = node.children[token]
            node.slot_id = slot_id  # Mark ALL nodes, not just leaf
            node.last_access = time.time()

        # Update entry tracking
        self.entries[slot_id] = CacheEntry(
            tokens=tokens.copy(),
            slot_id=slot_id,
            last_access=time.time(),
        )

        logger.debug(f"Inserted prefix of {len(tokens)} tokens into slot {slot_id}")
        return slot_id

    def find_longest_prefix(self, tokens: list[int]) -> tuple[int, int | None]:
        """Find the longest cached prefix matching the input tokens.

        Traverses the radix tree following the token sequence, tracking
        the deepest node with an assigned slot.

        Args:
            tokens: Token sequence to match.

        Returns:
            Tuple of (matched_length, slot_id). slot_id is None if no
            prefix matches.
        """
        self.total_lookups += 1

        node = self.root
        matched_length = 0
        best_slot: int | None = None
        best_length = 0

        for i, token in enumerate(tokens):
            if token not in node.children:
                break

            node = node.children[token]
            node.last_access = time.time()
            matched_length = i + 1

            # Track best match (deepest node with assigned slot)
            if node.slot_id is not None:
                best_slot = node.slot_id
                best_length = matched_length

        if best_slot is not None:
            self.cache_hits += 1
            # Update entry hit count
            if best_slot in self.entries:
                self.entries[best_slot].hit_count += 1
                self.entries[best_slot].last_access = time.time()
            logger.debug(f"Cache HIT: matched {best_length} tokens in slot {best_slot}")
        else:
            self.cache_misses += 1
            logger.debug(f"Cache MISS: no prefix match for {len(tokens)} tokens")

        return best_length, best_slot

    def find_or_allocate(self, tokens: list[int]) -> tuple[int, int, bool]:
        """Find matching prefix or allocate a new slot.

        Convenience method that combines find_longest_prefix and insert.
        Returns the matched prefix length, slot ID, and whether it was
        a cache hit.

        Args:
            tokens: Token sequence to match.

        Returns:
            Tuple of (matched_length, slot_id, is_cache_hit).
        """
        matched_length, slot_id = self.find_longest_prefix(tokens)

        if slot_id is not None:
            return matched_length, slot_id, True

        # No match - allocate new slot
        slot_id = self.insert(tokens)
        return 0, slot_id, False

    def _allocate_slot(self, tokens: list[int]) -> int:
        """Allocate a slot for a new prefix.

        Uses LRU eviction if all slots are occupied.

        Args:
            tokens: Token sequence for the new prefix.

        Returns:
            Allocated slot ID.
        """
        if self._available_slots:
            slot_id = self._available_slots.pop()
            logger.debug(f"Allocated fresh slot {slot_id}")
            return slot_id

        # All slots occupied - evict LRU
        return self._evict_lru()

    def _evict_lru(self) -> int:
        """Evict the least recently used entry.

        Removes the entry with the oldest last_access timestamp and
        clears its path in the radix tree.

        Returns:
            Evicted slot ID (now available).
        """
        if not self.entries:
            # No entries to evict - return slot 0 as fallback
            return 0

        # Find LRU entry
        lru_slot = min(self.entries.keys(), key=lambda s: self.entries[s].last_access)
        lru_entry = self.entries[lru_slot]

        logger.debug(
            f"Evicting slot {lru_slot} (last access: {lru_entry.last_access:.1f}, "
            f"hits: {lru_entry.hit_count})"
        )

        # Remove from tree
        self._remove_prefix(lru_entry.tokens)

        # Remove from entries
        del self.entries[lru_slot]

        self.evictions += 1
        return lru_slot

    def _remove_prefix(self, tokens: list[int]) -> bool:
        """Remove a prefix from the radix tree.

        Clears the slot assignment at the leaf and prunes empty branches.

        Args:
            tokens: Token sequence to remove.

        Returns:
            True if prefix was found and removed.
        """
        # Find the path to the leaf
        path: list[tuple[RadixNode, int]] = []
        node = self.root

        for token in tokens:
            if token not in node.children:
                return False
            path.append((node, token))
            node = node.children[token]

        # Clear slot assignment
        node.slot_id = None

        # Prune empty branches (nodes with no children and no slot)
        for parent, token in reversed(path):
            child = parent.children[token]
            if not child.children and child.slot_id is None:
                del parent.children[token]
            else:
                break

        return True

    def get_stats(self) -> dict[str, float | int]:
        """Get cache statistics.

        Returns:
            Dictionary with hit rate, miss count, evictions, etc.
        """
        hit_rate = self.cache_hits / self.total_lookups if self.total_lookups > 0 else 0.0
        return {
            "total_lookups": self.total_lookups,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "hit_rate_pct": hit_rate * 100,
            "evictions": self.evictions,
            "active_entries": len(self.entries),
            "available_slots": len(self._available_slots),
        }

    def get_entries(self) -> list[dict[str, int | float]]:
        """Get all cache entries with their statistics.

        Returns:
            List of entry info dictionaries.
        """
        return [
            {
                "slot_id": entry.slot_id,
                "prefix_length": len(entry.tokens),
                "hit_count": entry.hit_count,
                "last_access": entry.last_access,
            }
            for entry in self.entries.values()
        ]

    def get_hot_prefixes(self, top_n: int = 10) -> list[CacheEntry]:
        """Get the most frequently accessed prefixes.

        Args:
            top_n: Number of entries to return.

        Returns:
            List of CacheEntry sorted by hit count (descending).
        """
        sorted_entries = sorted(
            self.entries.values(),
            key=lambda e: e.hit_count,
            reverse=True,
        )
        return sorted_entries[:top_n]

    def clear(self) -> None:
        """Clear all cache entries and reset the tree."""
        self.root = RadixNode(depth=0)
        self.entries.clear()
        self._available_slots = set(range(self.num_slots))
        self.total_lookups = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.evictions = 0

    def iter_prefixes(self) -> Iterator[tuple[list[int], int]]:
        """Iterate over all cached prefixes.

        Yields:
            Tuples of (token_sequence, slot_id).
        """
        for entry in self.entries.values():
            yield entry.tokens, entry.slot_id


class TokenizedRadixCache(RadixCache):
    """RadixCache with built-in tokenization support.

    Extends RadixCache to accept string prompts and handle tokenization
    internally. Requires a tokenizer function to convert prompts to
    token sequences.

    Usage:
        from src.radix_cache import TokenizedRadixCache

        def tokenize(text):
            return model.tokenize(text)

        cache = TokenizedRadixCache(num_slots=4, tokenizer=tokenize)
        slot = cache.insert_prompt("You are a helpful assistant...")
        length, slot = cache.find_longest_prefix_for_prompt("You are a helpful...")
    """

    def __init__(
        self,
        num_slots: int = 4,
        min_prefix_length: int = 16,
        tokenizer: callable | None = None,
    ):
        """Initialize the tokenized radix cache.

        Args:
            num_slots: Number of server slots to manage.
            min_prefix_length: Minimum prefix length to cache.
            tokenizer: Function to tokenize strings to token lists.
        """
        super().__init__(num_slots=num_slots, min_prefix_length=min_prefix_length)
        self._tokenizer = tokenizer

    def set_tokenizer(self, tokenizer: callable) -> None:
        """Set the tokenizer function.

        Args:
            tokenizer: Function that converts string to list of token IDs.
        """
        self._tokenizer = tokenizer

    def _tokenize(self, prompt: str) -> list[int]:
        """Tokenize a prompt string.

        Args:
            prompt: Text to tokenize.

        Returns:
            List of token IDs.

        Raises:
            ValueError: If no tokenizer is configured.
        """
        if self._tokenizer is None:
            raise ValueError("No tokenizer configured. Call set_tokenizer() first.")
        return self._tokenizer(prompt)

    def insert_prompt(self, prompt: str, slot_id: int | None = None) -> int:
        """Insert a prompt string into the cache.

        Args:
            prompt: Prompt text to cache.
            slot_id: Specific slot to use (allocates if None).

        Returns:
            Slot ID assigned to this prefix.
        """
        tokens = self._tokenize(prompt)
        return self.insert(tokens, slot_id)

    def find_longest_prefix_for_prompt(self, prompt: str) -> tuple[int, int | None]:
        """Find the longest cached prefix matching a prompt.

        Args:
            prompt: Prompt text to match.

        Returns:
            Tuple of (matched_token_count, slot_id).
        """
        tokens = self._tokenize(prompt)
        return self.find_longest_prefix(tokens)

    def find_or_allocate_for_prompt(self, prompt: str) -> tuple[int, int, bool]:
        """Find matching prefix or allocate for a prompt.

        Args:
            prompt: Prompt text to match.

        Returns:
            Tuple of (matched_length, slot_id, is_cache_hit).
        """
        tokens = self._tokenize(prompt)
        return self.find_or_allocate(tokens)
