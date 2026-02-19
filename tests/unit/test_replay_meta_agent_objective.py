"""Tests for regret-objective promotion logic in MetaAgentWorkflow."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from orchestration.repl_memory.replay.candidates import DesignCandidate
from orchestration.repl_memory.replay.meta_agent import MetaAgentWorkflow
from orchestration.repl_memory.replay.metrics import ReplayMetrics


class _StubArchive:
    def __init__(self, baseline=None):
        self._baseline = baseline

    def get_baseline(self):
        return self._baseline

    def store_result(self, candidate, metrics):
        return None

    def sample_for_reflection(self, n=5):
        return []


def _candidate(cid: str, notes: str = "") -> DesignCandidate:
    c = DesignCandidate.default()
    c.candidate_id = cid
    c.notes = notes
    c.created_at = datetime.now(timezone.utc)
    return c


def _metrics(cid: str, *, cumulative_reward: float, rm_softmax_score: float) -> ReplayMetrics:
    return ReplayMetrics(
        candidate_id=cid,
        num_trajectories=10,
        num_complete=10,
        cumulative_reward=cumulative_reward,
        avg_reward=cumulative_reward / 10.0,
        rm_softmax_score=rm_softmax_score,
        utility_score=rm_softmax_score,
    )


def test_recommend_promotion_prefers_rm_softmax_without_baseline():
    workflow = MetaAgentWorkflow(archive=_StubArchive())
    c1 = _candidate("c1")
    c2 = _candidate("c2")
    # c1 has higher cumulative reward, c2 has higher RM-softmax objective.
    results = [
        (c1, _metrics("c1", cumulative_reward=100.0, rm_softmax_score=0.40)),
        (c2, _metrics("c2", cumulative_reward=90.0, rm_softmax_score=0.55)),
    ]
    rec = workflow.recommend_promotion(results)
    assert rec is not None
    assert rec.candidate_id == "c2"


def test_recommend_promotion_uses_rm_softmax_delta_vs_baseline():
    baseline_candidate = _candidate("base", notes="production baseline")
    baseline_metrics = _metrics("base", cumulative_reward=200.0, rm_softmax_score=0.50)
    workflow = MetaAgentWorkflow(archive=_StubArchive((baseline_candidate, baseline_metrics)))

    c1 = _candidate("c1")
    c2 = _candidate("c2")
    results = [
        (c1, _metrics("c1", cumulative_reward=260.0, rm_softmax_score=0.52)),  # +4%
        (c2, _metrics("c2", cumulative_reward=210.0, rm_softmax_score=0.56)),  # +12%
    ]
    rec = workflow.recommend_promotion(results)
    assert rec is not None
    assert rec.candidate_id == "c2"

