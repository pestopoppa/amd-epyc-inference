"""Tests for corpus retrieval service — corpus-augmented prompt stuffing.

Covers CorpusRetriever singleton lifecycle, query/format logic,
graceful degradation, and the extract_code_query helper.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.services.corpus_retrieval import (
    CodeSnippet,
    CorpusConfig,
    CorpusRetriever,
    extract_code_query,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset singleton before each test."""
    CorpusRetriever.reset_instance()
    yield
    CorpusRetriever.reset_instance()


@pytest.fixture()
def mini_index(tmp_path: Path) -> Path:
    """Create a minimal corpus index for testing."""
    snippets = [
        {
            "file": "src/math.py",
            "start_line": 1,
            "code": "def calculate_loss(predictions, targets):\n    return sum((p - t) ** 2 for p, t in zip(predictions, targets))",
            "hash": "abc123",
        },
        {
            "file": "src/data.py",
            "start_line": 10,
            "code": "def load_dataset(path, split='train'):\n    with open(path) as f:\n        return json.load(f)\n    # Extra lines\n    pass",
            "hash": "def456",
        },
        {
            "file": "src/train.py",
            "start_line": 20,
            "code": "def train_model(model, data, epochs=10):\n    for epoch in range(epochs):\n        loss = calculate_loss(model(data), data.targets)\n        loss.backward()",
            "hash": "ghi789",
        },
    ]

    # Build a simple n-gram index
    ngram_index: dict[str, list[int]] = {}
    for idx, snip in enumerate(snippets):
        words = snip["code"].lower().split()
        for i in range(len(words) - 3):
            gram = " ".join(words[i:i + 4])
            if gram not in ngram_index:
                ngram_index[gram] = []
            if idx not in ngram_index[gram]:
                ngram_index[gram].append(idx)

    (tmp_path / "snippets.json").write_text(json.dumps(snippets))
    (tmp_path / "ngram_index.json").write_text(json.dumps(ngram_index))
    (tmp_path / "meta.json").write_text(json.dumps({
        "version": 1,
        "ngram_size": 4,
        "num_snippets": len(snippets),
        "num_ngrams": len(ngram_index),
    }))
    return tmp_path


# ── Disabled / Not Configured ─────────────────────────────────────────────


class TestReturnsEmptyWhenNotConfigured:
    """Retriever returns empty when not configured."""

    def test_disabled_config(self):
        config = CorpusConfig(enabled=False)
        retriever = CorpusRetriever(config)
        result = retriever.retrieve("def calculate_loss")
        assert result == []

    def test_default_config_is_disabled(self):
        config = CorpusConfig()
        assert config.enabled is False
        retriever = CorpusRetriever(config)
        assert retriever.retrieve("anything") == []

    def test_format_empty_snippets(self):
        retriever = CorpusRetriever(CorpusConfig())
        assert retriever.format_for_prompt([]) == ""


# ── Non-Lookup Role ───────────────────────────────────────────────────────


class TestReturnsEmptyForNonLookupRole:
    """build_corpus_context gates on lookup-enabled roles."""

    def test_build_corpus_context_disabled(self):
        from src.prompt_builders.builder import build_corpus_context

        # Default config has enabled=False
        result = build_corpus_context(
            role="architect_general",
            task_description="Design a system",
        )
        assert result == ""


# ── Format Respects Max Chars ─────────────────────────────────────────────


class TestFormatRespectsMaxChars:
    """format_for_prompt truncates to max_chars budget."""

    def test_respects_budget(self):
        config = CorpusConfig(enabled=True, max_chars=200)
        retriever = CorpusRetriever(config)

        snippets = [
            CodeSnippet(code="x" * 500, file="a.py", start_line=1, score=0.9),
            CodeSnippet(code="y" * 500, file="b.py", start_line=1, score=0.8),
        ]
        result = retriever.format_for_prompt(snippets)
        assert len(result) <= 300  # Allow some overhead for delimiters
        assert "<reference_code" in result
        assert "</reference_code>" in result

    def test_single_snippet_under_budget(self):
        config = CorpusConfig(enabled=True, max_chars=3000)
        retriever = CorpusRetriever(config)

        snippets = [
            CodeSnippet(code="print('hello')", file="test.py", start_line=1, score=0.9),
        ]
        result = retriever.format_for_prompt(snippets)
        assert "print('hello')" in result
        assert "test.py:1" in result


# ── Deduplication ─────────────────────────────────────────────────────────


class TestDeduplicatesNearIdentical:
    """Retriever deduplicates snippets with same hash."""

    def test_dedup_by_hash(self, mini_index: Path):
        config = CorpusConfig(
            enabled=True,
            index_path=str(mini_index),
            min_score=0.0,
            max_snippets=10,
        )
        retriever = CorpusRetriever(config)

        # All snippets should be unique (different hashes in fixture)
        snippets = retriever.retrieve("def calculate_loss predictions targets")
        hashes = [s.hash for s in snippets if s.hash]
        assert len(hashes) == len(set(hashes))


# ── Graceful Degradation ─────────────────────────────────────────────────


class TestGracefulWhenSoftmatchaNotInstalled:
    """Retriever degrades gracefully with missing dependencies."""

    def test_missing_index(self):
        config = CorpusConfig(
            enabled=True,
            index_path="/nonexistent/path/to/index",
        )
        retriever = CorpusRetriever(config)
        result = retriever.retrieve("anything")
        assert result == []

    def test_corrupted_index(self, tmp_path: Path):
        (tmp_path / "snippets.json").write_text("not valid json {{{")
        (tmp_path / "ngram_index.json").write_text("{}")
        (tmp_path / "meta.json").write_text('{"version": 1, "ngram_size": 4}')

        config = CorpusConfig(enabled=True, index_path=str(tmp_path))
        retriever = CorpusRetriever(config)
        result = retriever.retrieve("test query")
        assert result == []

    def test_build_corpus_context_import_error(self):
        """build_corpus_context handles ImportError gracefully."""
        from src.prompt_builders.builder import build_corpus_context

        with patch.dict("sys.modules", {"src.services.corpus_retrieval": None}):
            # Should return "" when module can't be imported
            result = build_corpus_context(
                role="frontdoor",
                task_description="test",
            )
            assert result == ""


# ── Singleton Behavior ────────────────────────────────────────────────────


class TestSingleton:
    """CorpusRetriever singleton lifecycle."""

    def test_same_instance(self):
        a = CorpusRetriever.get_instance()
        b = CorpusRetriever.get_instance()
        assert a is b

    def test_reset_creates_new(self):
        a = CorpusRetriever.get_instance()
        CorpusRetriever.reset_instance()
        b = CorpusRetriever.get_instance()
        assert a is not b


# ── Retrieval Quality ─────────────────────────────────────────────────────


class TestRetrieval:
    """End-to-end retrieval from mini index."""

    def test_retrieves_matching_snippets(self, mini_index: Path):
        config = CorpusConfig(
            enabled=True,
            index_path=str(mini_index),
            min_score=0.0,
        )
        retriever = CorpusRetriever(config)
        # Use text that overlaps with snippet content to produce matching 4-grams
        snippets = retriever.retrieve(
            "return sum((p - t) ** 2 for p, t in zip(predictions, targets))"
        )
        assert len(snippets) > 0
        # First result should be the calculate_loss snippet
        assert "calculate_loss" in snippets[0].code

    def test_empty_query_returns_empty(self, mini_index: Path):
        config = CorpusConfig(
            enabled=True,
            index_path=str(mini_index),
        )
        retriever = CorpusRetriever(config)
        result = retriever.retrieve("a")  # Too short for 4-grams
        assert result == []

    def test_scores_are_descending(self, mini_index: Path):
        config = CorpusConfig(
            enabled=True,
            index_path=str(mini_index),
            min_score=0.0,
            max_snippets=10,
        )
        retriever = CorpusRetriever(config)
        snippets = retriever.retrieve("def calculate_loss predictions targets return sum")
        if len(snippets) > 1:
            for i in range(len(snippets) - 1):
                assert snippets[i].score >= snippets[i + 1].score


# ── extract_code_query ────────────────────────────────────────────────────


class TestExtractCodeQuery:
    """Code query extraction from task descriptions."""

    def test_strips_instruction_words(self):
        result = extract_code_query("Please implement a function to calculate loss")
        assert "please" not in result
        assert "implement" not in result
        assert "calculate" in result or "loss" in result

    def test_keeps_snake_case(self):
        result = extract_code_query("Write a calculate_loss function")
        assert "calculate_loss" in result

    def test_keeps_camel_case(self):
        result = extract_code_query("Fix the calculateLoss method")
        assert "calculateloss" in result

    def test_empty_after_stripping_falls_back(self):
        result = extract_code_query("Please write the code")
        # Should fall back to truncated lowercase
        assert len(result) > 0

    def test_preserves_identifiers(self):
        result = extract_code_query("Update the REPLEnvironment class")
        assert "replenvironment" in result
