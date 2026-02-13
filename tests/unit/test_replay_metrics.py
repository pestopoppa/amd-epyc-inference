"""Tests for ReplayMetrics serialization and comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from orchestration.repl_memory.replay.metrics import ReplayMetrics


def _make_metrics(**overrides) -> ReplayMetrics:
    defaults = dict(
        candidate_id="test-001",
        num_trajectories=100,
        num_complete=95,
        routing_accuracy=0.75,
        routing_accuracy_by_type={"code": 0.80, "ingest": 0.65},
        escalation_precision=0.70,
        escalation_recall=0.60,
        q_convergence_step=50,
        cumulative_reward=42.0,
        avg_reward=0.42,
        cost_efficiency=0.85,
        tier_usage={"coder": 60, "worker": 30, "architect": 10},
        replay_duration_seconds=12.5,
    )
    defaults.update(overrides)
    return ReplayMetrics(**defaults)


class TestReplayMetrics:
    def test_to_dict_round_trip(self):
        m = _make_metrics()
        d = m.to_dict()
        m2 = ReplayMetrics.from_dict(d)

        assert m2.candidate_id == m.candidate_id
        assert m2.num_trajectories == m.num_trajectories
        assert m2.routing_accuracy == m.routing_accuracy
        assert m2.routing_accuracy_by_type == m.routing_accuracy_by_type
        assert m2.escalation_precision == m.escalation_precision
        assert m2.escalation_recall == m.escalation_recall
        assert m2.q_convergence_step == m.q_convergence_step
        assert m2.cumulative_reward == m.cumulative_reward
        assert m2.avg_reward == m.avg_reward
        assert m2.cost_efficiency == m.cost_efficiency
        assert m2.tier_usage == m.tier_usage
        assert m2.replay_duration_seconds == m.replay_duration_seconds

    def test_from_dict_defaults(self):
        """Missing optional fields get defaults."""
        minimal = {
            "candidate_id": "x",
            "num_trajectories": 10,
            "num_complete": 8,
        }
        m = ReplayMetrics.from_dict(minimal)
        assert m.routing_accuracy == 0.0
        assert m.tier_usage == {}

    def test_compare_positive_improvement(self):
        baseline = _make_metrics(routing_accuracy=0.60, cumulative_reward=30.0)
        improved = _make_metrics(routing_accuracy=0.75, cumulative_reward=42.0)
        diff = improved.compare(baseline)

        assert diff["routing_accuracy"]["delta"] == pytest.approx(0.15, abs=0.001)
        assert diff["routing_accuracy"]["pct_change"] == pytest.approx(25.0, abs=0.1)
        assert diff["cumulative_reward"]["delta"] == pytest.approx(12.0, abs=0.001)
        assert diff["cumulative_reward"]["pct_change"] == pytest.approx(40.0, abs=0.1)

    def test_compare_negative_regression(self):
        baseline = _make_metrics(routing_accuracy=0.80)
        worse = _make_metrics(routing_accuracy=0.60)
        diff = worse.compare(baseline)

        assert diff["routing_accuracy"]["delta"] == pytest.approx(-0.20, abs=0.001)
        assert diff["routing_accuracy"]["pct_change"] == pytest.approx(-25.0, abs=0.1)

    def test_compare_zero_baseline(self):
        """Zero baseline should not raise ZeroDivisionError."""
        baseline = _make_metrics(cumulative_reward=0.0, escalation_precision=0.0)
        better = _make_metrics(cumulative_reward=10.0, escalation_precision=0.5)
        diff = better.compare(baseline)

        assert diff["cumulative_reward"]["delta"] == pytest.approx(10.0)
        assert diff["cumulative_reward"]["pct_change"] == 0.0  # 0 base → 0%

    def test_compare_identical(self):
        m = _make_metrics()
        diff = m.compare(m)
        for metric_info in diff.values():
            assert metric_info["delta"] == pytest.approx(0.0)
            assert metric_info["pct_change"] == pytest.approx(0.0)
