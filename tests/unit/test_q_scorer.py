"""Tests for QScorer cost-aware reward computation."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass, field
from typing import Any

# Minimal stubs so we can test _compute_reward without real dependencies.
# The actual ProgressEntry uses more fields; we only need outcome + event_type + data.


class _EventType:
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    GATE_PASSED = "gate_passed"
    GATE_FAILED = "gate_failed"
    ESCALATION_TRIGGERED = "escalation_triggered"
    PLAN_REVIEWED = "plan_reviewed"


@dataclass
class _FakeEntry:
    event_type: str
    outcome: str = ""
    data: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Import the real ScoringConfig and QScorer
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from orchestration.repl_memory.q_scorer import ScoringConfig, QScorer
from orchestration.repl_memory.progress_logger import EventType


def _make_outcome(outcome: str = "success") -> _FakeEntry:
    """Create a fake task outcome entry."""
    evt = EventType.TASK_COMPLETED if outcome != "failure" else EventType.TASK_FAILED
    return _FakeEntry(event_type=evt, outcome=outcome)


def _make_gate_fail() -> _FakeEntry:
    return _FakeEntry(event_type=EventType.GATE_FAILED)


def _make_escalation() -> _FakeEntry:
    return _FakeEntry(event_type=EventType.ESCALATION_TRIGGERED)


def _make_plan_review(decision: str = "ok") -> _FakeEntry:
    return _FakeEntry(event_type=EventType.PLAN_REVIEWED, data={"decision": decision})


def _scorer(config: ScoringConfig | None = None) -> QScorer:
    """Build a QScorer with mocked dependencies (we only test _compute_reward)."""
    return QScorer(
        store=MagicMock(),
        embedder=MagicMock(),
        logger=MagicMock(),
        reader=MagicMock(),
        config=config or ScoringConfig(),
    )


# ===== ScoringConfig defaults =====


class TestScoringConfigDefaults:
    def test_cost_penalty_lambda_default(self):
        cfg = ScoringConfig()
        assert cfg.cost_penalty_lambda == 0.15

    def test_baseline_tps_has_all_production_roles(self):
        cfg = ScoringConfig()
        expected_roles = {
            "frontdoor",
            "coder_primary",
            "coder_escalation",
            "architect_general",
            "architect_coding",
            "ingest_long_context",
            "worker_explore",
            "worker_math",
            "worker_vision",
            "vision_escalation",
        }
        assert set(cfg.baseline_tps_by_role.keys()) == expected_roles

    def test_baseline_tps_values_positive(self):
        cfg = ScoringConfig()
        for role, tps in cfg.baseline_tps_by_role.items():
            assert tps > 0, f"{role} has non-positive tps: {tps}"

    def test_config_override(self):
        cfg = ScoringConfig(cost_penalty_lambda=0.5)
        assert cfg.cost_penalty_lambda == 0.5


# ===== _compute_reward without cost metrics (backward compat) =====


class TestComputeRewardNoCost:
    def test_success_no_cost(self):
        s = _scorer()
        r = s._compute_reward(_make_outcome("success"), [], [])
        assert r == 1.0

    def test_failure_no_cost(self):
        s = _scorer()
        r = s._compute_reward(_make_outcome("failure"), [], [])
        assert r == -0.5

    def test_partial_no_cost(self):
        s = _scorer()
        r = s._compute_reward(_make_outcome("partial"), [], [])
        assert r == 0.3

    def test_gate_failures_penalize(self):
        s = _scorer()
        r = s._compute_reward(_make_outcome("success"), [_make_gate_fail(), _make_gate_fail()], [])
        assert r == pytest.approx(0.8)  # 1.0 - 2*0.1

    def test_escalation_penalizes(self):
        s = _scorer()
        r = s._compute_reward(_make_outcome("success"), [], [_make_escalation()])
        assert r == pytest.approx(0.85)  # 1.0 - 0.15

    def test_plan_review_approved(self):
        s = _scorer()
        r = s._compute_reward(_make_outcome("success"), [], [], [_make_plan_review("ok")])
        assert r == pytest.approx(1.0)  # 1.0 + 0.1 clamped to 1.0

    def test_plan_review_corrected(self):
        s = _scorer()
        r = s._compute_reward(_make_outcome("success"), [], [], [_make_plan_review("corrected")])
        assert r == pytest.approx(0.8)  # 1.0 - 0.2


# ===== _compute_reward with cost metrics =====


class TestComputeRewardWithCost:
    def test_at_expected_speed_no_penalty(self):
        """Running at exactly baseline speed → cost_ratio=1.0 → no penalty."""
        s = _scorer()
        # frontdoor at 18.3 t/s, 183 tokens in 10s → exactly expected
        cost = {"tokens_generated": 183, "elapsed_seconds": 10.0, "role": "frontdoor"}
        r = s._compute_reward(_make_outcome("success"), [], [], cost_metrics=cost)
        assert r == 1.0

    def test_faster_than_expected_no_penalty(self):
        """Running faster than baseline → cost_ratio < 1.0 → no penalty."""
        s = _scorer()
        # frontdoor at 18.3 t/s, 183 tokens in 5s → 2x faster
        cost = {"tokens_generated": 183, "elapsed_seconds": 5.0, "role": "frontdoor"}
        r = s._compute_reward(_make_outcome("success"), [], [], cost_metrics=cost)
        assert r == 1.0

    def test_slower_than_expected_penalized(self):
        """Running 2x slower → cost_ratio=2.0 → penalty = 0.15 * (2.0 - 1.0) = 0.15."""
        s = _scorer()
        # frontdoor at 18.3 t/s, 183 tokens in 20s → 2x slower
        cost = {"tokens_generated": 183, "elapsed_seconds": 20.0, "role": "frontdoor"}
        r = s._compute_reward(_make_outcome("success"), [], [], cost_metrics=cost)
        assert r == pytest.approx(0.85)  # 1.0 - 0.15

    def test_much_slower_higher_penalty(self):
        """Running 5x slower → penalty = 0.15 * 4.0 = 0.60."""
        s = _scorer()
        cost = {"tokens_generated": 183, "elapsed_seconds": 50.0, "role": "frontdoor"}
        r = s._compute_reward(_make_outcome("success"), [], [], cost_metrics=cost)
        assert r == pytest.approx(0.4)  # 1.0 - 0.60

    def test_incorrect_no_cost_penalty(self):
        """Failed tasks get failure_reward regardless of cost."""
        s = _scorer()
        cost = {"tokens_generated": 183, "elapsed_seconds": 100.0, "role": "frontdoor"}
        r = s._compute_reward(_make_outcome("failure"), [], [], cost_metrics=cost)
        assert r == -0.5  # No cost penalty applied (reward <= 0)

    def test_unknown_role_no_penalty(self):
        """Unknown role has no baseline → no cost penalty."""
        s = _scorer()
        cost = {"tokens_generated": 100, "elapsed_seconds": 100.0, "role": "unknown_role"}
        r = s._compute_reward(_make_outcome("success"), [], [], cost_metrics=cost)
        assert r == 1.0

    def test_zero_tokens_no_penalty(self):
        """Zero tokens_generated → skip cost computation."""
        s = _scorer()
        cost = {"tokens_generated": 0, "elapsed_seconds": 10.0, "role": "frontdoor"}
        r = s._compute_reward(_make_outcome("success"), [], [], cost_metrics=cost)
        assert r == 1.0

    def test_zero_elapsed_no_penalty(self):
        """Zero elapsed → skip cost computation (avoid division by zero)."""
        s = _scorer()
        cost = {"tokens_generated": 100, "elapsed_seconds": 0.0, "role": "frontdoor"}
        r = s._compute_reward(_make_outcome("success"), [], [], cost_metrics=cost)
        assert r == 1.0

    def test_cost_plus_gate_penalties_stack(self):
        """Cost penalty stacks with gate failure penalties."""
        s = _scorer()
        # 2x slower + 1 gate failure
        cost = {"tokens_generated": 183, "elapsed_seconds": 20.0, "role": "frontdoor"}
        r = s._compute_reward(_make_outcome("success"), [_make_gate_fail()], [], cost_metrics=cost)
        # 1.0 - 0.1 (gate) - 0.15 (cost) = 0.75
        assert r == pytest.approx(0.75)

    def test_clamp_lower_bound(self):
        """Extreme cost penalty clamped to -1.0."""
        cfg = ScoringConfig(cost_penalty_lambda=10.0)  # Very aggressive
        s = _scorer(cfg)
        cost = {"tokens_generated": 183, "elapsed_seconds": 50.0, "role": "frontdoor"}
        r = s._compute_reward(_make_outcome("success"), [], [], cost_metrics=cost)
        assert r == -1.0

    def test_custom_lambda(self):
        """Custom lambda changes penalty magnitude."""
        cfg = ScoringConfig(cost_penalty_lambda=0.5)
        s = _scorer(cfg)
        # 2x slower with lambda=0.5 → penalty = 0.5 * 1.0 = 0.5
        cost = {"tokens_generated": 183, "elapsed_seconds": 20.0, "role": "frontdoor"}
        r = s._compute_reward(_make_outcome("success"), [], [], cost_metrics=cost)
        assert r == pytest.approx(0.5)

    def test_architect_role_slower_baseline(self):
        """Architect (6.75 t/s) at expected speed → no penalty."""
        s = _scorer()
        # 675 tokens at 6.75 t/s → 100s expected; actual 100s
        cost = {"tokens_generated": 675, "elapsed_seconds": 100.0, "role": "architect_general"}
        r = s._compute_reward(_make_outcome("success"), [], [], cost_metrics=cost)
        assert r == 1.0

    def test_architect_role_2x_slower(self):
        """Architect at 2x slower → penalty = 0.15 * 1.0 = 0.15."""
        s = _scorer()
        cost = {"tokens_generated": 675, "elapsed_seconds": 200.0, "role": "architect_general"}
        r = s._compute_reward(_make_outcome("success"), [], [], cost_metrics=cost)
        assert r == pytest.approx(0.85)


# ===== Comparative rewards (seeding script) =====


class TestComparativeRewardsCostAware:
    """Test the updated compute_comparative_rewards from seed_specialist_routing.py."""

    def _import_fn(self):
        """Import the function (adds scripts/benchmark to path)."""
        import sys

        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "benchmark"))
        from seed_specialist_routing import compute_comparative_rewards, RoleResult

        return compute_comparative_rewards, RoleResult

    def test_both_correct_cost_aware(self):
        fn, RR = self._import_fn()
        results = {
            "frontdoor:direct": RR(
                role="frontdoor",
                mode="direct",
                answer="ok",
                passed=True,
                elapsed_seconds=10.0,
                tokens_generated=183,
            ),
            "coder_primary:direct": RR(
                role="coder_primary",
                mode="direct",
                answer="ok",
                passed=True,
                elapsed_seconds=10.0,
                tokens_generated=183,
            ),
        }
        rewards = fn(results)
        # Both at expected speed → base 0.5, no penalty
        assert rewards["coder_primary:direct"] == pytest.approx(0.5)

    def test_both_correct_specialist_slow(self):
        fn, RR = self._import_fn()
        results = {
            "frontdoor:direct": RR(
                role="frontdoor",
                mode="direct",
                answer="ok",
                passed=True,
                elapsed_seconds=10.0,
                tokens_generated=183,
            ),
            "architect_general:direct": RR(
                role="architect_general",
                mode="direct",
                answer="ok",
                passed=True,
                elapsed_seconds=200.0,
                tokens_generated=675,
                # Expected: 675/6.75 = 100s, actual 200s → cost_ratio=2.0
            ),
        }
        rewards = fn(results)
        # penalty = 0.15 * (2.0 - 1.0) = 0.15; reward = 0.5 - 0.15 = 0.35
        assert rewards["architect_general:direct"] == pytest.approx(0.35)

    def test_both_correct_no_tokens_fallback(self):
        fn, RR = self._import_fn()
        results = {
            "frontdoor:direct": RR(
                role="frontdoor",
                mode="direct",
                answer="ok",
                passed=True,
                elapsed_seconds=10.0,
                tokens_generated=0,  # No token data
            ),
            "coder_primary:direct": RR(
                role="coder_primary",
                mode="direct",
                answer="ok",
                passed=True,
                elapsed_seconds=10.0,
                tokens_generated=0,
            ),
        }
        rewards = fn(results)
        # No cost data → fallback to 0.3
        assert rewards["coder_primary:direct"] == 0.3

    def test_specialist_correct_baseline_wrong(self):
        fn, RR = self._import_fn()
        results = {
            "frontdoor:direct": RR(
                role="frontdoor",
                mode="direct",
                answer="wrong",
                passed=False,
                elapsed_seconds=10.0,
            ),
            "coder_primary:direct": RR(
                role="coder_primary",
                mode="direct",
                answer="ok",
                passed=True,
                elapsed_seconds=10.0,
                tokens_generated=183,
            ),
        }
        rewards = fn(results)
        assert rewards["coder_primary:direct"] == 1.0  # Clear win

    def test_specialist_wrong_baseline_correct(self):
        fn, RR = self._import_fn()
        results = {
            "frontdoor:direct": RR(
                role="frontdoor",
                mode="direct",
                answer="ok",
                passed=True,
                elapsed_seconds=10.0,
                tokens_generated=183,
            ),
            "coder_primary:direct": RR(
                role="coder_primary",
                mode="direct",
                answer="wrong",
                passed=False,
                elapsed_seconds=10.0,
            ),
        }
        rewards = fn(results)
        assert rewards["coder_primary:direct"] == -0.5

    def test_both_wrong(self):
        fn, RR = self._import_fn()
        results = {
            "frontdoor:direct": RR(
                role="frontdoor",
                mode="direct",
                answer="wrong",
                passed=False,
                elapsed_seconds=10.0,
            ),
            "coder_primary:direct": RR(
                role="coder_primary",
                mode="direct",
                answer="wrong",
                passed=False,
                elapsed_seconds=10.0,
            ),
        }
        rewards = fn(results)
        assert rewards["coder_primary:direct"] == -0.3

    def test_baseline_incorrect_gets_zero(self):
        """Baseline wrong → reward=0.0 (xRouter: incorrect=zero)."""
        fn, RR = self._import_fn()
        results = {
            "frontdoor:direct": RR(
                role="frontdoor",
                mode="direct",
                answer="wrong",
                passed=False,
                elapsed_seconds=10.0,
            ),
        }
        rewards = fn(results)
        assert rewards["frontdoor:direct"] == 0.0

    def test_custom_cost_config(self):
        fn, RR = self._import_fn()
        results = {
            "frontdoor:direct": RR(
                role="frontdoor",
                mode="direct",
                answer="ok",
                passed=True,
                elapsed_seconds=10.0,
                tokens_generated=183,
            ),
            "coder_primary:direct": RR(
                role="coder_primary",
                mode="direct",
                answer="ok",
                passed=True,
                elapsed_seconds=20.0,
                tokens_generated=183,
                # 2x slower → penalty = 0.5 * 1.0 = 0.5 with custom lambda
            ),
        }
        config = {"lambda": 0.5, "baseline_tps_by_role": {"frontdoor": 18.3, "coder_primary": 18.3}}
        rewards = fn(results, cost_config=config)
        # base=0.5, penalty=0.5 → reward=0.0, but floor is 0.1
        assert rewards["coder_primary:direct"] == pytest.approx(0.1)

    def test_floor_at_0_1_for_correct(self):
        """Even very slow correct specialists get at least 0.1."""
        fn, RR = self._import_fn()
        results = {
            "frontdoor:direct": RR(
                role="frontdoor",
                mode="direct",
                answer="ok",
                passed=True,
                elapsed_seconds=10.0,
                tokens_generated=183,
            ),
            "architect_general:direct": RR(
                role="architect_general",
                mode="direct",
                answer="ok",
                passed=True,
                elapsed_seconds=1000.0,
                tokens_generated=675,
                # Expected: 100s, actual 1000s → 10x slower → penalty = 0.15*9 = 1.35
            ),
        }
        rewards = fn(results)
        assert rewards["architect_general:direct"] == 0.1  # Floored
