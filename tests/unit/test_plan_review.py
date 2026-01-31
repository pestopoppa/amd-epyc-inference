"""Tests for the architect-in-the-loop plan review feature.

Tests bypass logic, parsing, phases, timeout handling, and MemRL integration
without requiring real LLM inference (mock mode only).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from src.features import Features, set_features, reset_features
from src.proactive_delegation import (
    ArchitectReviewService,
    PlanReviewResult,
    TaskComplexity,
    classify_task_complexity,
)
from src.prompt_builders import build_plan_review_prompt


# ─── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_features():
    """Reset feature flags after each test."""
    yield
    reset_features()


@pytest.fixture
def mock_state():
    """Create a mock AppState with plan review fields."""
    state = MagicMock()
    state.plan_review_phase = "A"
    state._plan_review_stats = {
        "total_reviews": 0,
        "approved": 0,
        "corrected": 0,
        "task_class_q_values": {},
    }
    state.hybrid_router = None
    state.progress_logger = None
    state.q_scorer = None
    return state


@pytest.fixture
def mock_primitives():
    """Create mock LLM primitives."""
    primitives = MagicMock()
    primitives.llm_call.return_value = '{"d":"ok","s":0.9,"f":"plan looks good","p":[]}'
    return primitives


@pytest.fixture
def sample_task_ir():
    """Create a sample TaskIR for testing."""
    return {
        "task_type": "code",
        "objective": "Implement error handling for the registry loader with retry logic",
        "priority": "interactive",
        "plan": {
            "steps": [
                {"id": "S1", "actor": "coder", "action": "Implement error handler", "outputs": ["error.py"]},
                {"id": "S2", "actor": "worker", "action": "Write unit tests", "outputs": ["test_error.py"]},
            ]
        },
    }


# ─── PlanReviewResult Tests ──────────────────────────────────────────────


class TestPlanReviewResult:
    def test_is_ok_true(self):
        result = PlanReviewResult(decision="ok", score=0.9)
        assert result.is_ok is True

    def test_is_ok_false(self):
        result = PlanReviewResult(decision="reroute", score=0.7)
        assert result.is_ok is False

    def test_to_dict(self):
        result = PlanReviewResult(
            decision="reroute",
            score=0.7,
            feedback="use architect for design",
            patches=[{"step": "S1", "op": "reroute", "v": "architect_general"}],
        )
        d = result.to_dict()
        assert d["decision"] == "reroute"
        assert d["score"] == 0.7
        assert len(d["patches"]) == 1

    def test_default_values(self):
        result = PlanReviewResult()
        assert result.decision == "ok"
        assert result.score == 1.0
        assert result.is_ok is True
        assert result.patches == []


# ─── Bypass Logic Tests ──────────────────────────────────────────────────


class TestNeedsPlanReview:
    """Test _needs_plan_review bypass conditions."""

    def test_bypass_trivial_complexity(self, mock_state):
        """TRIVIAL tasks should bypass review."""
        from src.api.routes.chat import _needs_plan_review
        task_ir = {"objective": "What is Python?", "task_type": "chat"}
        assert _needs_plan_review(task_ir, ["frontdoor"], mock_state) is False

    def test_bypass_simple_complexity(self, mock_state):
        """SIMPLE tasks should bypass review."""
        from src.api.routes.chat import _needs_plan_review
        task_ir = {"objective": "Fix the typo in main.py", "task_type": "code"}
        assert _needs_plan_review(task_ir, ["coder_primary"], mock_state) is False

    def test_bypass_complex_complexity(self, mock_state):
        """COMPLEX tasks bypass review (architect already owns plan)."""
        from src.api.routes.chat import _needs_plan_review
        task_ir = {
            "objective": "Design a distributed database schema with fault-tolerant consensus protocol",
            "task_type": "code",
        }
        assert _needs_plan_review(task_ir, ["architect_general"], mock_state) is False

    def test_moderate_triggers_review(self, mock_state):
        """MODERATE complexity tasks should trigger review in Phase A."""
        from src.api.routes.chat import _needs_plan_review
        task_ir = {
            "objective": "Implement error handling and then write unit tests for the module",
            "task_type": "code",
        }
        assert _needs_plan_review(task_ir, ["coder_primary"], mock_state) is True

    def test_bypass_architect_role(self, mock_state):
        """Architect routing should not self-review."""
        from src.api.routes.chat import _needs_plan_review
        task_ir = {
            "objective": "Implement error handling and then write unit tests for the module",
            "task_type": "code",
        }
        assert _needs_plan_review(task_ir, ["architect_general"], mock_state) is False

    def test_phase_c_stochastic_skip(self, mock_state):
        """Phase C should skip 90% of reviews stochastically."""
        from src.api.routes.chat import _needs_plan_review
        mock_state.plan_review_phase = "C"
        task_ir = {
            "objective": "Implement error handling and then write unit tests for the module",
            "task_type": "code",
        }

        # Run many times and check that ~90% are skipped
        random.seed(42)
        results = [
            _needs_plan_review(task_ir, ["coder_primary"], mock_state)
            for _ in range(200)
        ]
        review_rate = sum(results) / len(results)
        # Should be ~10% (allow 3-20% range for random variance)
        assert 0.03 <= review_rate <= 0.20, f"Review rate {review_rate:.2f} not ~10%"

    def test_phase_b_q_value_bypass(self, mock_state):
        """Phase B should bypass review when Q-value >= 0.6."""
        from src.api.routes.chat import _needs_plan_review

        mock_state.plan_review_phase = "B"
        mock_router = MagicMock()
        mock_memory = MagicMock()
        mock_memory.q_value = 0.8
        mock_router.retriever.retrieve_for_routing.return_value = [mock_memory]
        mock_state.hybrid_router = mock_router

        task_ir = {
            "objective": "Implement error handling and then write unit tests for the module",
            "task_type": "code",
        }
        assert _needs_plan_review(task_ir, ["coder_primary"], mock_state) is False

    def test_phase_b_low_q_triggers_review(self, mock_state):
        """Phase B should trigger review when Q-value < 0.6."""
        from src.api.routes.chat import _needs_plan_review

        mock_state.plan_review_phase = "B"
        mock_router = MagicMock()
        mock_memory = MagicMock()
        mock_memory.q_value = 0.4
        mock_router.retriever.retrieve_for_routing.return_value = [mock_memory]
        mock_state.hybrid_router = mock_router

        task_ir = {
            "objective": "Implement error handling and then write unit tests for the module",
            "task_type": "code",
        }
        assert _needs_plan_review(task_ir, ["coder_primary"], mock_state) is True


# ─── Prompt Building Tests ───────────────────────────────────────────────


class TestBuildPlanReviewPrompt:
    def test_basic_prompt(self):
        prompt = build_plan_review_prompt(
            objective="Implement error handler",
            task_type="code",
            plan_steps=[
                {"id": "S1", "actor": "coder", "action": "Write handler", "outputs": ["handler.py"]},
            ],
        )
        assert "Review plan" in prompt
        assert "Verdict:" in prompt
        assert "S1:coder:Write handler" in prompt
        assert "handler.py" in prompt

    def test_multi_step_prompt(self):
        prompt = build_plan_review_prompt(
            objective="Build API",
            task_type="code",
            plan_steps=[
                {"id": "S1", "actor": "coder", "action": "Create routes"},
                {"id": "S2", "actor": "worker", "action": "Write tests", "deps": ["S1"]},
            ],
        )
        assert "S1:coder:Create routes" in prompt
        assert "S2:worker:Write tests" in prompt
        assert "(S1)" in prompt  # Dependency shown

    def test_truncates_long_objective(self):
        long_obj = "x" * 500
        prompt = build_plan_review_prompt(
            objective=long_obj,
            task_type="code",
            plan_steps=[{"id": "S1", "actor": "coder", "action": "Do work"}],
        )
        # Objective should be truncated to 200 chars
        assert len(prompt) < 600


# ─── ArchitectReviewService.review_plan Tests ────────────────────────────


class TestReviewPlan:
    def test_successful_review_ok(self, mock_primitives):
        service = ArchitectReviewService(mock_primitives)
        result = service.review_plan(
            objective="Implement handler",
            task_type="code",
            plan_steps=[{"id": "S1", "actor": "coder", "action": "Write code"}],
        )
        assert result is not None
        assert result.decision == "ok"
        assert result.score == 0.9

    def test_reroute_decision(self, mock_primitives):
        mock_primitives.llm_call.return_value = json.dumps({
            "d": "reroute",
            "s": 0.7,
            "f": "use architect for design first",
            "p": [{"step": "S1", "op": "reroute", "v": "architect_general"}],
        })
        service = ArchitectReviewService(mock_primitives)
        result = service.review_plan(
            objective="Design system",
            task_type="code",
            plan_steps=[{"id": "S1", "actor": "coder", "action": "Design API"}],
        )
        assert result is not None
        assert result.decision == "reroute"
        assert len(result.patches) == 1
        assert result.patches[0]["v"] == "architect_general"

    def test_returns_none_on_exception(self, mock_primitives):
        mock_primitives.llm_call.side_effect = TimeoutError("timed out")
        service = ArchitectReviewService(mock_primitives)
        result = service.review_plan(
            objective="Do something",
            task_type="code",
            plan_steps=[{"id": "S1", "actor": "coder", "action": "Work"}],
        )
        assert result is None

    def test_normalizes_invalid_decision(self, mock_primitives):
        mock_primitives.llm_call.return_value = '{"d":"invalid_decision","s":0.5,"f":"hmm"}'
        service = ArchitectReviewService(mock_primitives)
        result = service.review_plan(
            objective="Do work",
            task_type="code",
            plan_steps=[{"id": "S1", "actor": "coder", "action": "Code"}],
        )
        assert result is not None
        assert result.decision == "ok"  # Normalized to ok


# ─── Apply Plan Review Tests ─────────────────────────────────────────────


class TestApplyPlanReview:
    def test_reroute_first_step(self):
        from src.api.routes.chat import _apply_plan_review
        review = PlanReviewResult(
            decision="reroute",
            patches=[{"step": "S1", "op": "reroute", "v": "architect_general"}],
        )
        result = _apply_plan_review(["coder_primary"], review)
        assert result == ["architect_general"]

    def test_reroute_second_step(self):
        from src.api.routes.chat import _apply_plan_review
        review = PlanReviewResult(
            decision="reroute",
            patches=[{"step": "S2", "op": "reroute", "v": "worker_math"}],
        )
        result = _apply_plan_review(["coder_primary", "worker_general"], review)
        assert result == ["coder_primary", "worker_math"]

    def test_no_patches_returns_unchanged(self):
        from src.api.routes.chat import _apply_plan_review
        review = PlanReviewResult(decision="ok", patches=[])
        result = _apply_plan_review(["coder_primary"], review)
        assert result == ["coder_primary"]


# ─── Phase Transition Tests ──────────────────────────────────────────────


class TestComputePlanReviewPhase:
    def test_phase_a_low_reviews(self):
        from src.api.routes.chat import _compute_plan_review_phase
        stats = {"total_reviews": 10, "task_class_q_values": {}}
        assert _compute_plan_review_phase(stats) == "A"

    def test_phase_a_no_q_values(self):
        from src.api.routes.chat import _compute_plan_review_phase
        stats = {"total_reviews": 60, "task_class_q_values": {}}
        assert _compute_plan_review_phase(stats) == "A"

    def test_phase_b_moderate_q(self):
        from src.api.routes.chat import _compute_plan_review_phase
        stats = {
            "total_reviews": 60,
            "task_class_q_values": {"code": 0.8, "chat": 0.6},
        }
        assert _compute_plan_review_phase(stats) == "B"

    def test_phase_c_high_q(self):
        from src.api.routes.chat import _compute_plan_review_phase
        stats = {
            "total_reviews": 150,
            "task_class_q_values": {"code": 0.9, "chat": 0.8},
        }
        assert _compute_plan_review_phase(stats) == "C"

    def test_phase_a_when_min_q_too_low(self):
        from src.api.routes.chat import _compute_plan_review_phase
        stats = {
            "total_reviews": 60,
            "task_class_q_values": {"code": 0.9, "chat": 0.3},
        }
        assert _compute_plan_review_phase(stats) == "A"


# ─── MemRL Integration Tests ────────────────────────────────────────────


class TestStorePlanReviewEpisode:
    def test_stores_progress_log_entry(self, mock_state):
        from src.api.routes.chat import _store_plan_review_episode

        mock_state.progress_logger = MagicMock()
        mock_state.q_scorer = None

        review = PlanReviewResult(
            decision="ok", score=0.9, feedback="looks good"
        )
        task_ir = {"objective": "Build API", "task_type": "code"}

        _store_plan_review_episode(mock_state, "test-123", task_ir, review)

        mock_state.progress_logger.log.assert_called_once()
        entry = mock_state.progress_logger.log.call_args[0][0]
        assert entry.data["decision"] == "ok"
        assert entry.data["score"] == 0.9

    def test_updates_stats(self, mock_state):
        from src.api.routes.chat import _store_plan_review_episode

        review = PlanReviewResult(decision="reroute", score=0.6, feedback="fix routing")
        task_ir = {"objective": "Build API", "task_type": "code"}

        _store_plan_review_episode(mock_state, "test-123", task_ir, review)

        assert mock_state._plan_review_stats["total_reviews"] == 1
        assert mock_state._plan_review_stats["corrected"] == 1
        assert mock_state._plan_review_stats["approved"] == 0

    def test_approved_increments_approved(self, mock_state):
        from src.api.routes.chat import _store_plan_review_episode

        review = PlanReviewResult(decision="ok", score=0.9)
        task_ir = {"objective": "Fix bug", "task_type": "code"}

        _store_plan_review_episode(mock_state, "test-456", task_ir, review)

        assert mock_state._plan_review_stats["approved"] == 1
        assert mock_state._plan_review_stats["corrected"] == 0

    def test_stores_memrl_episode(self, mock_state):
        from src.api.routes.chat import _store_plan_review_episode

        mock_state.q_scorer = MagicMock()
        mock_state.hybrid_router = MagicMock()

        review = PlanReviewResult(decision="reroute", score=0.4, feedback="wrong role")
        task_ir = {"objective": "Build API", "task_type": "code"}

        _store_plan_review_episode(mock_state, "test-789", task_ir, review)

        mock_state.q_scorer.score_external_result.assert_called_once()
        call_kwargs = mock_state.q_scorer.score_external_result.call_args[1]
        assert call_kwargs["action"] == "plan_review:reroute"
        # reward = 0.4 * 2 - 1 = -0.2
        assert abs(call_kwargs["reward"] - (-0.2)) < 0.01


# ─── Q-Scorer Integration Tests ─────────────────────────────────────────


class TestQScorerPlanReview:
    def test_compute_reward_with_plan_review_bonus(self):
        """Approved plan should give +0.1 bonus."""
        from orchestration.repl_memory.q_scorer import QScorer
        from orchestration.repl_memory.progress_logger import ProgressEntry, EventType

        scorer = MagicMock(spec=QScorer)
        scorer.config = MagicMock()
        scorer.config.success_reward = 1.0
        scorer.config.failure_reward = -0.5
        scorer.config.partial_reward = 0.3

        # Call the real method
        task_outcome = ProgressEntry(
            event_type=EventType.TASK_COMPLETED,
            task_id="test",
            outcome="success",
        )
        approved_review = ProgressEntry(
            event_type=EventType.PLAN_REVIEWED,
            task_id="test",
            data={"decision": "ok"},
        )

        reward = QScorer._compute_reward(
            scorer, task_outcome, [], [], [approved_review]
        )
        assert abs(reward - 1.0) < 0.01  # 1.0 + 0.1 = 1.1, clamped to 1.0

    def test_compute_reward_with_plan_review_penalty(self):
        """Corrected plan should give -0.2 penalty."""
        from orchestration.repl_memory.q_scorer import QScorer
        from orchestration.repl_memory.progress_logger import ProgressEntry, EventType

        scorer = MagicMock(spec=QScorer)
        scorer.config = MagicMock()
        scorer.config.success_reward = 1.0
        scorer.config.failure_reward = -0.5
        scorer.config.partial_reward = 0.3

        task_outcome = ProgressEntry(
            event_type=EventType.TASK_COMPLETED,
            task_id="test",
            outcome="success",
        )
        corrected_review = ProgressEntry(
            event_type=EventType.PLAN_REVIEWED,
            task_id="test",
            data={"decision": "reroute"},
        )

        reward = QScorer._compute_reward(
            scorer, task_outcome, [], [], [corrected_review]
        )
        assert abs(reward - 0.8) < 0.01  # 1.0 - 0.2 = 0.8


# ─── Feature Flag Tests ─────────────────────────────────────────────────


class TestFeatureFlag:
    def test_plan_review_default_disabled(self):
        f = Features()
        assert f.plan_review is False

    def test_plan_review_in_summary(self):
        f = Features(plan_review=True)
        summary = f.summary()
        assert "plan_review" in summary
        assert summary["plan_review"] is True

    def test_plan_review_env_var(self):
        import os
        os.environ["ORCHESTRATOR_PLAN_REVIEW"] = "1"
        try:
            from src.features import get_features
            f = get_features()
            assert f.plan_review is True
        finally:
            del os.environ["ORCHESTRATOR_PLAN_REVIEW"]
            reset_features()
