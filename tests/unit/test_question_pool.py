#!/usr/bin/env python3
"""Tests for question_pool — pre-extracted question pool for fast sampling."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "benchmark"))

from question_pool import (
    _HEADER_KEY,
    build_pool,
    load_pool,
    pool_header,
    sample_from_pool,
)


# ── Fixtures ──────────────────────────────────────────────────────────


class FakeAdapter:
    """Minimal adapter that produces deterministic questions."""

    suite_name = "fake_suite"
    _dataset = None

    def __init__(self, questions: list[dict]):
        self._questions = questions

    def extract_all(self) -> list[dict]:
        return list(self._questions)


@pytest.fixture
def tmp_pool_path(tmp_path):
    return tmp_path / "test_pool.jsonl"


@pytest.fixture
def sample_questions():
    """Generate sample question dicts for testing."""
    questions = []
    for suite in ("math", "coder", "thinking"):
        for i in range(5):
            questions.append({
                "id": f"{suite}_q{i:03d}",
                "suite": suite,
                "prompt": f"Test question {i} for {suite}",
                "context": "",
                "expected": f"answer_{i}",
                "image_path": "",
                "tier": (i % 3) + 1,
                "scoring_method": "exact_match",
                "scoring_config": {},
                "dataset_source": "hf_adapter",
            })
    return questions


@pytest.fixture
def pool_file_with_data(tmp_pool_path, sample_questions):
    """Write a pool file with sample data and return its path."""
    from datetime import datetime, timezone

    header = {
        _HEADER_KEY: True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "test",
        "total_questions": len(sample_questions),
        "suites": {"math": 5, "coder": 5, "thinking": 5},
    }
    with open(tmp_pool_path, "w") as f:
        f.write(json.dumps(header) + "\n")
        for q in sample_questions:
            f.write(json.dumps(q) + "\n")
    return tmp_pool_path


# ── Tests ─────────────────────────────────────────────────────────────


class TestBuildAndLoadRoundtrip:
    """Build a pool from mock adapters, load it back, verify structure."""

    def test_roundtrip(self, tmp_pool_path):
        fake_questions = [
            {
                "id": f"test_q{i:03d}",
                "suite": "general",
                "prompt": f"Question {i}",
                "expected": f"Answer {i}",
                "tier": 1,
                "scoring_method": "exact_match",
                "scoring_config": {},
            }
            for i in range(3)
        ]

        fake_adapter = FakeAdapter(fake_questions)

        with patch("dataset_adapters.ADAPTER_SUITES", {"general"}), \
             patch("dataset_adapters.YAML_ONLY_SUITES", set()), \
             patch("dataset_adapters.get_adapter", return_value=fake_adapter):
            stats = build_pool(tmp_pool_path)

        assert stats["general"] == 3

        pool = load_pool(tmp_pool_path, warn_stale=False)
        assert "general" in pool
        assert len(pool["general"]) == 3
        assert pool["general"][0]["id"] == "test_q000"
        assert pool["general"][0]["prompt"] == "Question 0"

    def test_empty_adapter(self, tmp_pool_path):
        """Adapter returning no questions produces valid pool with 0 count."""
        empty_adapter = FakeAdapter([])

        with patch("dataset_adapters.ADAPTER_SUITES", {"empty"}), \
             patch("dataset_adapters.YAML_ONLY_SUITES", set()), \
             patch("dataset_adapters.get_adapter", return_value=empty_adapter):
            stats = build_pool(tmp_pool_path)

        assert stats["empty"] == 0
        pool = load_pool(tmp_pool_path, warn_stale=False)
        assert pool.get("empty", []) == []


class TestSampleFromPoolDedup:
    """Verify seen questions are excluded from sampling."""

    def test_dedup_excludes_seen(self, pool_file_with_data):
        pool = load_pool(pool_file_with_data, warn_stale=False)

        # Mark first 3 math questions as seen
        seen = {"math_q000", "math_q001", "math_q002"}
        result = sample_from_pool(
            pool, suites=["math"], sample_per_suite=5, seed=42, seen=seen,
        )

        result_ids = {q["id"] for q in result}
        assert not result_ids.intersection(seen), "Seen questions should be excluded"

    def test_all_seen_returns_empty(self, pool_file_with_data):
        pool = load_pool(pool_file_with_data, warn_stale=False)

        # Mark ALL math questions as seen
        seen = {f"math_q{i:03d}" for i in range(5)}
        result = sample_from_pool(
            pool, suites=["math"], sample_per_suite=5, seed=42, seen=seen,
        )
        assert result == []


class TestSampleFromPoolDeterministic:
    """Same seed must produce same questions."""

    def test_deterministic(self, pool_file_with_data):
        pool = load_pool(pool_file_with_data, warn_stale=False)

        r1 = sample_from_pool(pool, suites=["math", "coder"], sample_per_suite=3, seed=123)
        r2 = sample_from_pool(pool, suites=["math", "coder"], sample_per_suite=3, seed=123)

        assert [q["id"] for q in r1] == [q["id"] for q in r2]

    def test_different_seed_different_results(self, pool_file_with_data):
        pool = load_pool(pool_file_with_data, warn_stale=False)

        r1 = sample_from_pool(pool, suites=["math", "coder"], sample_per_suite=3, seed=1)
        r2 = sample_from_pool(pool, suites=["math", "coder"], sample_per_suite=3, seed=999)

        # Different seeds should usually produce different orderings
        # (with 5 questions per suite, extremely unlikely to be identical)
        ids1 = [q["id"] for q in r1]
        ids2 = [q["id"] for q in r2]
        assert ids1 != ids2 or len(ids1) == 0


class TestPoolHeaderMetadata:
    """Verify header stores generation timestamp and stats."""

    def test_header_present(self, pool_file_with_data):
        header = pool_header(pool_file_with_data)
        assert header is not None
        assert header.get(_HEADER_KEY) is True
        assert "generated_at" in header
        assert header["total_questions"] == 15
        assert header["suites"]["math"] == 5

    def test_header_missing_file(self, tmp_path):
        result = pool_header(tmp_path / "nonexistent.jsonl")
        assert result is None

    def test_load_without_header(self, tmp_path):
        """Pool file without header line still loads questions."""
        pool_path = tmp_path / "no_header.jsonl"
        q = {"id": "q1", "suite": "test", "prompt": "hello"}
        with open(pool_path, "w") as f:
            f.write(json.dumps(q) + "\n")

        pool = load_pool(pool_path, warn_stale=False)
        assert "test" in pool
        assert len(pool["test"]) == 1


class TestInterleaving:
    """Verify round-robin interleaving across suites."""

    def test_interleave_order(self, pool_file_with_data):
        pool = load_pool(pool_file_with_data, warn_stale=False)

        result = sample_from_pool(
            pool, suites=["math", "coder", "thinking"],
            sample_per_suite=2, seed=42,
        )

        # With 2 per suite, round-robin should alternate suites
        # First 3 should each be from a different suite
        if len(result) >= 3:
            first_three_suites = [q["suite"] for q in result[:3]]
            assert len(set(first_three_suites)) == 3, \
                f"Expected 3 different suites in first 3, got: {first_three_suites}"
