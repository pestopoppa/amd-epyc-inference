#!/usr/bin/env python3
"""Unit tests for orchestrator_eval.py checkpoint and dedup logic."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "benchmark"))

from scripts.benchmark.orchestrator_eval import (
    EvalResult,
    append_checkpoint,
    load_checkpoint,
    load_seen_questions,
    record_seen,
    sample_unseen_questions,
    _prompt_hash,
    _print_summary,
)


@pytest.fixture
def eval_dir(tmp_path):
    """Override EVAL_DIR and SEEN_FILE to use a temp directory."""
    import scripts.benchmark.orchestrator_eval as mod
    original_eval_dir = mod.EVAL_DIR
    original_seen_file = mod.SEEN_FILE
    mod.EVAL_DIR = tmp_path
    mod.SEEN_FILE = tmp_path / "seen_questions.jsonl"
    yield tmp_path
    mod.EVAL_DIR = original_eval_dir
    mod.SEEN_FILE = original_seen_file


def _make_result(prompt_id: str = "test_q1", suite: str = "math",
                 correct: bool = True, **kwargs) -> EvalResult:
    """Create a minimal EvalResult for testing."""
    defaults = dict(
        prompt_id=prompt_id,
        suite=suite,
        tier=1,
        dataset_source="test",
        prompt_hash="abc123",
        timestamp="2026-02-01T00:00:00+00:00",
        orchestrator_answer="42",
        routed_to="frontdoor",
        mode="direct",
        latency_ms=1000.0,
        tokens_generated=50,
        tps=18.0,
        turns=1,
        tools_used=0,
        task_id="task-001",
        scoring_method="exact_match",
        expected="42",
        correct=correct,
        reward_injected=True,
        error="",
    )
    defaults.update(kwargs)
    return EvalResult(**defaults)


# ── Checkpoint roundtrip ──────────────────────────────────────────────


class TestCheckpoint:
    def test_write_read_roundtrip(self, eval_dir):
        """JSONL checkpoint write then read preserves all fields."""
        r1 = _make_result("q1", correct=True)
        r2 = _make_result("q2", correct=False, suite="coder")

        append_checkpoint("test_session", r1)
        append_checkpoint("test_session", r2)

        loaded = load_checkpoint("test_session")
        assert len(loaded) == 2
        assert loaded[0].prompt_id == "q1"
        assert loaded[0].correct is True
        assert loaded[1].prompt_id == "q2"
        assert loaded[1].correct is False
        assert loaded[1].suite == "coder"

    def test_empty_checkpoint(self, eval_dir):
        """Loading a nonexistent checkpoint returns empty list."""
        loaded = load_checkpoint("nonexistent")
        assert loaded == []

    def test_partial_corrupt_checkpoint(self, eval_dir):
        """Corrupt lines are skipped, valid lines are loaded."""
        path = eval_dir / "corrupt_session.jsonl"
        r1 = _make_result("q1")
        with open(path, "w") as f:
            f.write(json.dumps({"prompt_id": "q1", **{k: v for k, v in r1.__dict__.items()}}) + "\n")
            f.write("NOT VALID JSON\n")
            f.write(json.dumps({"prompt_id": "q2", **{k: v for k, v in _make_result("q2").__dict__.items()}}) + "\n")

        loaded = load_checkpoint("corrupt_session")
        assert len(loaded) == 2

    def test_append_is_atomic(self, eval_dir):
        """Each append creates exactly one line."""
        for i in range(5):
            append_checkpoint("atomic_test", _make_result(f"q{i}"))

        path = eval_dir / "atomic_test.jsonl"
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 5


# ── Seen question dedup ──────────────────────────────────────────────


class TestSeenQuestions:
    def test_seen_from_checkpoints(self, eval_dir):
        """Seen set is built from all checkpoint files."""
        append_checkpoint("session_a", _make_result("q1"))
        append_checkpoint("session_a", _make_result("q2"))
        append_checkpoint("session_b", _make_result("q3"))

        seen = load_seen_questions()
        assert seen == {"q1", "q2", "q3"}

    def test_seen_from_seen_file(self, eval_dir):
        """Seen file entries are also loaded."""
        record_seen("q_seen_1", "math", "old_session")
        record_seen("q_seen_2", "coder", "old_session")

        seen = load_seen_questions()
        assert "q_seen_1" in seen
        assert "q_seen_2" in seen

    def test_seen_combines_sources(self, eval_dir):
        """Checkpoint files + seen file are merged."""
        append_checkpoint("s1", _make_result("from_checkpoint"))
        record_seen("from_seen_file", "math", "s0")

        seen = load_seen_questions()
        assert "from_checkpoint" in seen
        assert "from_seen_file" in seen

    def test_empty_seen(self, eval_dir):
        """Empty dir gives empty seen set."""
        seen = load_seen_questions()
        assert seen == set()


# ── Question sampling with dedup ──────────────────────────────────────


class TestSampling:
    def test_unseen_filtering(self, eval_dir):
        """Already-seen questions are excluded."""
        seen = {"mmlu_test_0001", "mmlu_test_0002"}

        # Mock the dataset adapter to return predictable questions
        fake_questions = [
            {"id": f"mmlu_test_{i:04d}", "suite": "general",
             "prompt": f"Question {i}", "expected": "A",
             "scoring_method": "multiple_choice", "scoring_config": {},
             "tier": 1, "dataset_source": "test"}
            for i in range(10)
        ]

        with patch("scripts.benchmark.orchestrator_eval._load_from_dataset_adapter",
                    return_value=fake_questions):
            with patch("scripts.benchmark.orchestrator_eval._load_from_yaml",
                        return_value=[]):
                result = sample_unseen_questions("general", 5, seen, seed=42)

        # Should not contain the seen IDs
        result_ids = {q["id"] for q in result}
        assert "mmlu_test_0001" not in result_ids
        assert "mmlu_test_0002" not in result_ids
        assert len(result) <= 5

    def test_sample_caps_at_requested(self, eval_dir):
        """Never returns more than requested sample count."""
        fake_questions = [
            {"id": f"q_{i}", "suite": "math", "prompt": f"Q{i}",
             "expected": "42", "scoring_method": "exact_match",
             "scoring_config": {}, "tier": 1, "dataset_source": "test"}
            for i in range(100)
        ]

        with patch("scripts.benchmark.orchestrator_eval._load_from_dataset_adapter",
                    return_value=fake_questions):
            with patch("scripts.benchmark.orchestrator_eval._load_from_yaml",
                        return_value=[]):
                result = sample_unseen_questions("math", 10, set(), seed=42)

        assert len(result) == 10


# ── Resume from partial checkpoint ────────────────────────────────────


class TestResume:
    def test_resume_skips_completed(self, eval_dir):
        """Resuming a session skips already-completed prompt IDs."""
        # Simulate a partial session
        append_checkpoint("resume_test", _make_result("done_1"))
        append_checkpoint("resume_test", _make_result("done_2"))

        completed = load_checkpoint("resume_test")
        completed_ids = {r.prompt_id for r in completed}

        assert "done_1" in completed_ids
        assert "done_2" in completed_ids
        assert len(completed) == 2

    def test_resume_preserves_order(self, eval_dir):
        """Results come back in append order."""
        for i in range(5):
            append_checkpoint("order_test", _make_result(f"q_{i}"))

        loaded = load_checkpoint("order_test")
        ids = [r.prompt_id for r in loaded]
        assert ids == ["q_0", "q_1", "q_2", "q_3", "q_4"]


# ── Prompt hash ──────────────────────────────────────────────────────


class TestPromptHash:
    def test_deterministic(self):
        """Same text gives same hash."""
        h1 = _prompt_hash("What is 2+2?")
        h2 = _prompt_hash("What is 2+2?")
        assert h1 == h2

    def test_different_for_different_text(self):
        """Different text gives different hash."""
        h1 = _prompt_hash("What is 2+2?")
        h2 = _prompt_hash("What is 3+3?")
        assert h1 != h2

    def test_is_12_chars(self):
        """Hash is truncated to 12 hex chars."""
        h = _prompt_hash("test")
        assert len(h) == 12


# ── Summary printing ─────────────────────────────────────────────────


class TestSummary:
    def test_summary_with_results(self, eval_dir, capsys):
        """Summary prints without errors."""
        results = [
            _make_result("q1", correct=True, suite="math"),
            _make_result("q2", correct=False, suite="math"),
            _make_result("q3", correct=True, suite="coder"),
        ]
        _print_summary(results, "test_session")
        captured = capsys.readouterr()
        assert "test_session" in captured.out
        assert "66.7%" in captured.out  # 2/3

    def test_summary_empty(self, eval_dir, capsys):
        """Empty results handled gracefully."""
        _print_summary([], "empty")
        captured = capsys.readouterr()
        assert "No results" in captured.out


# ── RewardRequest model ──────────────────────────────────────────────


class TestRewardRequest:
    def test_valid_request(self):
        """RewardRequest validates properly."""
        from src.api.models.requests import RewardRequest
        req = RewardRequest(
            task_description="math/q1: What is 2+2?",
            action="frontdoor:direct",
            reward=1.0,
        )
        assert req.reward == 1.0
        assert req.context is None

    def test_reward_bounds(self):
        """Reward must be between -1 and 1."""
        from src.api.models.requests import RewardRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RewardRequest(
                task_description="test",
                action="test",
                reward=2.0,  # Out of bounds
            )

    def test_reward_with_context(self):
        """Context dict is optional and passed through."""
        from src.api.models.requests import RewardRequest
        req = RewardRequest(
            task_description="test",
            action="test",
            reward=-0.5,
            context={"suite": "math", "tier": 2},
        )
        assert req.context["suite"] == "math"


# ── /chat/reward endpoint ────────────────────────────────────────────


class TestRewardEndpoint:
    def test_reward_endpoint_exists(self):
        """The /chat/reward endpoint is registered."""
        from src.api import app
        from src.api.state import reset_state
        reset_state()

        from fastapi.testclient import TestClient
        with TestClient(app) as client:
            # Should not return 404
            resp = client.post("/chat/reward", json={
                "task_description": "test",
                "action": "frontdoor:direct",
                "reward": 0.5,
            })
            # In mock mode without MemRL, should return success=False
            assert resp.status_code == 200
            assert "success" in resp.json()
