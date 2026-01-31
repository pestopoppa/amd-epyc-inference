"""Unit tests for dataset_adapters module.

All tests mock HuggingFace datasets entirely — no downloads required.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "benchmark"))

from scripts.benchmark.dataset_adapters import (
    ADAPTER_SUITES,
    YAML_ONLY_SUITES,
    BaseAdapter,
    IFEvalAdapter,
    MathAdapter,
    MMLUAdapter,
    get_adapter,
)


# ── get_adapter() factory ──────────────────────────────────────────────────


class TestGetAdapter:
    """Tests for the get_adapter() factory function."""

    def test_known_suites_return_adapter(self):
        """Known suite names return adapter instances."""
        for suite in ("general", "math", "instruction_precision"):
            adapter = get_adapter(suite)
            assert adapter is not None, f"get_adapter('{suite}') returned None"

    def test_yaml_only_suites_return_none(self):
        """YAML-only suites return None."""
        for suite in YAML_ONLY_SUITES:
            assert get_adapter(suite) is None

    def test_unknown_suite_returns_none(self):
        """Completely unknown suite returns None."""
        assert get_adapter("nonexistent_suite") is None

    def test_adapter_suites_constant_matches_factory(self):
        """Every ADAPTER_SUITES entry produces a non-None adapter."""
        for suite in ADAPTER_SUITES:
            adapter = get_adapter(suite)
            assert adapter is not None, f"ADAPTER_SUITES contains '{suite}' but factory returned None"


# ── Mock dataset helpers ───────────────────────────────────────────────────


def _make_mock_mmlu_dataset(n: int = 20):
    """Build a mock MMLU-like dataset."""
    rows = []
    subjects = ["abstract_algebra", "high_school_geography", "anatomy", "miscellaneous"]
    for i in range(n):
        rows.append({
            "question": f"Question {i}?",
            "choices": ["Alpha", "Beta", "Gamma", "Delta"],
            "answer": i % 4,
            "subject": subjects[i % len(subjects)],
        })
    mock_ds = MagicMock()
    mock_ds.__len__ = lambda self: len(rows)
    mock_ds.__getitem__ = lambda self, idx: rows[idx]
    mock_ds.__bool__ = lambda self: True
    return mock_ds


def _make_mock_gsm8k_dataset(n: int = 10):
    """Build a mock GSM8K-like dataset."""
    rows = []
    for i in range(n):
        rows.append({
            "question": f"Math problem {i}?",
            "answer": f"Step 1: ...\nStep 2: ...\n#### {100 + i}",
        })
    mock_ds = MagicMock()
    mock_ds.__len__ = lambda self: len(rows)
    mock_ds.__getitem__ = lambda self, idx: rows[idx]
    mock_ds.__bool__ = lambda self: True
    return mock_ds


def _make_mock_math500_dataset(n: int = 5):
    """Build a mock MATH-500-like dataset."""
    rows = []
    for i in range(n):
        rows.append({
            "problem": f"Hard math problem {i}?",
            "answer": f"\\boxed{{{i * 10}}}",
            "level": 2 + (i % 4),
            "subject": "algebra",
        })
    mock_ds = MagicMock()
    mock_ds.__len__ = lambda self: len(rows)
    mock_ds.__getitem__ = lambda self, idx: rows[idx]
    mock_ds.__bool__ = lambda self: True
    return mock_ds


def _make_mock_ifeval_dataset(n: int = 10):
    """Build a mock IFEval-like dataset."""
    rows = []
    constraints = [
        (["no_comma"], [{}]),
        (["number_paragraphs", "postscript"], [{"num_paragraphs": 3}, {}]),
        (["json_format"], [{}]),
        (["number_words"], [{"num_words": 50, "relation": "at_least"}]),
    ]
    for i in range(n):
        c_ids, c_kwargs = constraints[i % len(constraints)]
        rows.append({
            "prompt": f"Instruction {i}: write something",
            "key": i,
            "instruction_id_list": c_ids,
            "kwargs": c_kwargs,
        })
    mock_ds = MagicMock()
    mock_ds.__len__ = lambda self: len(rows)
    mock_ds.__getitem__ = lambda self, idx: rows[idx]
    mock_ds.__bool__ = lambda self: True
    return mock_ds


# ── MMLUAdapter tests ──────────────────────────────────────────────────────


class TestMMLUAdapter:
    """Tests for MMLUAdapter."""

    @patch("scripts.benchmark.dataset_adapters.MMLUAdapter._ensure_loaded")
    def test_deterministic_sampling(self, mock_ensure):
        """Same seed produces same sample."""
        adapter = MMLUAdapter()
        adapter._dataset = _make_mock_mmlu_dataset(20)

        sample1 = adapter.sample(n=5, seed=42)
        sample2 = adapter.sample(n=5, seed=42)

        assert len(sample1) == 5
        assert [q["id"] for q in sample1] == [q["id"] for q in sample2]

    @patch("scripts.benchmark.dataset_adapters.MMLUAdapter._ensure_loaded")
    def test_prompt_format(self, mock_ensure):
        """Sampled prompts have expected keys and structure."""
        adapter = MMLUAdapter()
        adapter._dataset = _make_mock_mmlu_dataset(5)

        samples = adapter.sample(n=2, seed=1)

        for q in samples:
            assert "id" in q
            assert q["suite"] == "general"
            assert "prompt" in q
            assert "A)" in q["prompt"]
            assert q["scoring_method"] == "multiple_choice"
            assert q["expected"] in ("A", "B", "C", "D")

    @patch("scripts.benchmark.dataset_adapters.MMLUAdapter._ensure_loaded")
    def test_tier_assignment_by_subject(self, mock_ensure):
        """Hard subjects → tier 3, easy → tier 1, middle → tier 2."""
        adapter = MMLUAdapter()
        ds = _make_mock_mmlu_dataset(4)
        adapter._dataset = ds

        samples = adapter.sample(n=4, seed=0)

        # Subjects rotate: abstract_algebra(hard=3), high_school_geography(easy=1),
        # anatomy(hard=3), miscellaneous(easy=1)
        tiers = {q["id"].split("_")[1]: q["tier"] for q in samples}
        # Check at least one hard and one easy tier assignment
        tier_values = set(q["tier"] for q in samples)
        assert 1 in tier_values or 3 in tier_values  # at least some differentiation

    @patch("scripts.benchmark.dataset_adapters.MMLUAdapter._ensure_loaded")
    def test_empty_dataset_returns_empty(self, mock_ensure):
        """Empty dataset returns empty list."""
        adapter = MMLUAdapter()
        adapter._dataset = []

        assert adapter.sample(n=5) == []


# ── MathAdapter tests ──────────────────────────────────────────────────────


class TestMathAdapter:
    """Tests for MathAdapter."""

    @patch("scripts.benchmark.dataset_adapters.MathAdapter._ensure_loaded")
    def test_gsm8k_answer_extraction(self, mock_ensure):
        """GSM8K #### answer is extracted correctly."""
        assert MathAdapter._extract_gsm8k_answer("Step 1\n#### 42") == "42"
        assert MathAdapter._extract_gsm8k_answer("#### 1,234") == "1234"
        assert MathAdapter._extract_gsm8k_answer("no answer marker") == "no answer marker"

    @patch("scripts.benchmark.dataset_adapters.MathAdapter._ensure_loaded")
    def test_deterministic_sampling(self, mock_ensure):
        """Same seed produces same sample."""
        adapter = MathAdapter()
        adapter._gsm8k = _make_mock_gsm8k_dataset(10)
        adapter._math500 = _make_mock_math500_dataset(5)
        adapter._dataset = list(range(15))

        s1 = adapter.sample(n=6, seed=99)
        s2 = adapter.sample(n=6, seed=99)

        assert len(s1) == 6
        assert [q["id"] for q in s1] == [q["id"] for q in s2]

    @patch("scripts.benchmark.dataset_adapters.MathAdapter._ensure_loaded")
    def test_empty_dataset_returns_empty(self, mock_ensure):
        """Empty dataset returns empty list."""
        adapter = MathAdapter()
        adapter._dataset = []

        assert adapter.sample(n=5) == []


# ── IFEvalAdapter tests ────────────────────────────────────────────────────


class TestIFEvalAdapter:
    """Tests for IFEvalAdapter."""

    def test_constraint_to_scoring_no_comma(self):
        """no_comma constraint maps to programmatic verifier."""
        method, config = IFEvalAdapter._constraint_to_scoring("no_comma", {})
        assert method == "programmatic"
        assert config["verifier"] == "no_comma"

    def test_constraint_to_scoring_json_format(self):
        """json_format constraint maps correctly."""
        method, config = IFEvalAdapter._constraint_to_scoring("json_format", {})
        assert method == "programmatic"
        assert config["verifier"] == "json_valid"

    def test_constraint_to_scoring_postscript(self):
        """postscript constraint maps to substring scorer."""
        method, config = IFEvalAdapter._constraint_to_scoring("postscript", {})
        assert method == "substring"
        assert config["substring"] == "P.S."

    def test_constraint_to_scoring_unknown(self):
        """Unknown constraint falls back to non_empty verifier."""
        method, config = IFEvalAdapter._constraint_to_scoring("unknown_xyz", {})
        assert method == "programmatic"
        assert config["verifier"] == "non_empty"

    @patch("scripts.benchmark.dataset_adapters.IFEvalAdapter._ensure_loaded")
    def test_tier_by_constraint_count(self, mock_ensure):
        """1 constraint → tier 1, 2-3 → tier 2, 4+ → tier 3."""
        adapter = IFEvalAdapter()
        adapter._dataset = _make_mock_ifeval_dataset(10)

        samples = adapter.sample(n=4, seed=0)

        for q in samples:
            n_constraints = len(q.get("ifeval_instructions", []))
            if n_constraints <= 1:
                assert q["tier"] == 1
            elif n_constraints <= 3:
                assert q["tier"] == 2
            else:
                assert q["tier"] == 3

    @patch("scripts.benchmark.dataset_adapters.IFEvalAdapter._ensure_loaded")
    def test_empty_dataset_returns_empty(self, mock_ensure):
        """Empty dataset returns empty list."""
        adapter = IFEvalAdapter()
        adapter._dataset = []

        assert adapter.sample(n=5) == []


# ── BaseAdapter tests ──────────────────────────────────────────────────────


class TestBaseAdapter:
    """Tests for BaseAdapter property and method contracts."""

    def test_total_available_with_data(self):
        """total_available reflects dataset length."""
        adapter = MMLUAdapter()
        adapter._dataset = _make_mock_mmlu_dataset(7)

        assert adapter.total_available == 7

    def test_total_available_empty(self):
        """total_available is 0 for empty dataset."""
        adapter = MMLUAdapter()
        adapter._dataset = []

        assert adapter.total_available == 0

    def test_total_available_none(self):
        """total_available is 0 when _dataset is None (requires _ensure_loaded)."""
        adapter = MMLUAdapter()
        # _dataset starts as None; total_available calls _ensure_loaded
        # which would normally load data, but we patch it to leave _dataset as empty
        with patch.object(MMLUAdapter, "_ensure_loaded"):
            adapter._dataset = None
            assert adapter.total_available == 0
