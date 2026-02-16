"""Tests for the replay evaluation engine."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from orchestration.repl_memory.progress_logger import EventType, ProgressEntry
from orchestration.repl_memory.q_scorer import ScoringConfig
from orchestration.repl_memory.retriever import RetrievalConfig
from orchestration.repl_memory.replay.engine import (
    NullEmbedder,
    ReplayEngine,
    _make_fake_outcome,
)
from orchestration.repl_memory.replay.trajectory import Trajectory

def _ts(offset_hours: int = 0) -> datetime:
    base = datetime(2026, 2, 1, 12, 0, 0)
    return base + timedelta(hours=offset_hours)


def _make_trajectory(
    task_id: str = "t1",
    task_type: str = "code",
    objective: str = "Write a function",
    routing_decision: str = "coder_primary",
    outcome: str = "success",
    embedding: Optional[np.ndarray] = None,
    offset_hours: int = 0,
    escalations: Optional[List[str]] = None,
    cost_metrics_override: Optional[dict] = None,
) -> Trajectory:
    if embedding is None:
        embedding = np.random.default_rng(42).standard_normal(1024).astype(np.float32)
        embedding /= np.linalg.norm(embedding) + 1e-8

    outcome_entry = ProgressEntry(
        event_type=EventType.TASK_COMPLETED if outcome != "failure" else EventType.TASK_FAILED,
        task_id=task_id,
        outcome=outcome,
        data={"tokens_generated": 100, "elapsed_seconds": 2.0, "role": routing_decision},
    )

    cost_metrics = {"tokens_generated": 100, "elapsed_seconds": 2.0, "role": routing_decision}
    if cost_metrics_override:
        cost_metrics.update(cost_metrics_override)

    return Trajectory(
        task_id=task_id,
        task_type=task_type,
        objective=objective,
        routing_decision=routing_decision,
        outcome=outcome,
        cost_metrics=cost_metrics,
        escalations=escalations or [],
        gate_results=[],
        embedding=embedding,
        started_at=_ts(offset_hours),
        completed_at=_ts(offset_hours + 1),
        outcome_entry=outcome_entry,
        gate_entries=[],
        escalation_entries=[],
        plan_review_entries=[],
    )


@pytest.fixture()
def engine_tmp(tmp_path):
    """Per-test unique temp dir for replay engine."""
    d = tmp_path / "replay"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# NullEmbedder tests
# ---------------------------------------------------------------------------

class TestNullEmbedder:
    def test_raises_on_any_call(self):
        embedder = NullEmbedder()
        with pytest.raises(RuntimeError, match="NullEmbedder"):
            embedder.embed_text("hello")

    def test_raises_on_embed_batch(self):
        embedder = NullEmbedder()
        with pytest.raises(RuntimeError, match="NullEmbedder"):
            embedder.embed_batch(["hello"])


# ---------------------------------------------------------------------------
# ReplayEngine tests
# ---------------------------------------------------------------------------

class TestReplayEngine:
    def test_single_trajectory_replay(self, engine_tmp):
        engine = ReplayEngine(tmp_dir=engine_tmp)
        trajectories = [_make_trajectory()]
        results = engine.run(
            RetrievalConfig(),
            ScoringConfig(),
            trajectories,
            candidate_id="test1",
        )

        assert len(results) == 1
        r = results[0]
        assert r.trajectory_id == "t1"
        assert r.actual_action == "coder_primary"
        assert isinstance(r.reward, float)
        assert isinstance(r.q_value_after, float)
        assert 0.0 <= r.q_value_after <= 1.0

    def test_success_reward_positive(self, engine_tmp):
        engine = ReplayEngine(tmp_dir=engine_tmp)
        trajectories = [_make_trajectory(outcome="success")]
        results = engine.run(RetrievalConfig(), ScoringConfig(), trajectories, "test2")

        assert results[0].reward > 0

    def test_failure_reward_negative(self, engine_tmp):
        engine = ReplayEngine(tmp_dir=engine_tmp)
        trajectories = [_make_trajectory(outcome="failure")]
        results = engine.run(RetrievalConfig(), ScoringConfig(), trajectories, "test3")

        assert results[0].reward < 0

    def test_multiple_trajectories(self, engine_tmp):
        engine = ReplayEngine(tmp_dir=engine_tmp)
        rng = np.random.default_rng(123)
        trajectories = [
            _make_trajectory("t1", "code", "func", "coder", "success",
                             rng.standard_normal(1024).astype(np.float32), 0),
            _make_trajectory("t2", "ingest", "parse", "ingest", "failure",
                             rng.standard_normal(1024).astype(np.float32), 10),
            _make_trajectory("t3", "code", "refactor", "coder", "success",
                             rng.standard_normal(1024).astype(np.float32), 20),
        ]
        results = engine.run(RetrievalConfig(), ScoringConfig(), trajectories, "test4")

        assert len(results) == 3
        assert results[0].trajectory_id == "t1"
        assert results[2].trajectory_id == "t3"

    def test_second_trajectory_can_see_first(self, engine_tmp):
        """After replaying t1, the store should have an entry that t2 can retrieve."""
        engine = ReplayEngine(tmp_dir=engine_tmp)

        # Use the SAME embedding for both so t2 retrieves t1's memory
        same_emb = np.random.default_rng(42).standard_normal(1024).astype(np.float32)
        same_emb /= np.linalg.norm(same_emb) + 1e-8
        trajectories = [
            _make_trajectory("t1", "code", "func", "coder_primary", "success", same_emb.copy(), 0),
            _make_trajectory("t2", "code", "func", "coder_primary", "success", same_emb.copy(), 1),
        ]
        results = engine.run(RetrievalConfig(), ScoringConfig(), trajectories, "test5")

        # t1: no prior memory → candidate_action is None
        assert results[0].candidate_action is None
        # t2: should find t1's memory → candidate_action should be "coder_primary"
        assert results[1].candidate_action == "coder_primary"
        assert results[1].routing_match is True

    def test_cleanup_after_run(self, engine_tmp):
        engine = ReplayEngine(tmp_dir=engine_tmp)
        trajectories = [_make_trajectory()]
        cid = "cleanup_test"
        engine.run(RetrievalConfig(), ScoringConfig(), trajectories, cid)

        # The candidate's temp dir should be cleaned up
        assert not (engine_tmp / cid).exists()

    def test_trajectory_without_embedding(self, engine_tmp):
        """Trajectories with no embedding should still produce results."""
        engine = ReplayEngine(tmp_dir=engine_tmp)
        t = _make_trajectory()
        t.embedding = None
        results = engine.run(RetrievalConfig(), ScoringConfig(), [t], "test_no_emb")

        assert len(results) == 1
        assert results[0].candidate_action is None
        assert results[0].routing_match is False


# ---------------------------------------------------------------------------
# run_with_metrics tests
# ---------------------------------------------------------------------------

class TestReplayEngineMetrics:
    def test_metrics_basic(self, engine_tmp):
        engine = ReplayEngine(tmp_dir=engine_tmp)
        rng = np.random.default_rng(42)
        trajectories = [
            _make_trajectory("t1", "code", "f1", "coder", "success",
                             rng.standard_normal(1024).astype(np.float32), 0),
            _make_trajectory("t2", "code", "f2", "coder", "success",
                             rng.standard_normal(1024).astype(np.float32), 5),
            _make_trajectory("t3", "ingest", "p1", "ingest", "failure",
                             rng.standard_normal(1024).astype(np.float32), 10),
        ]
        metrics = engine.run_with_metrics(
            RetrievalConfig(), ScoringConfig(), trajectories, "metrics_test",
        )

        assert metrics.candidate_id == "metrics_test"
        assert metrics.num_trajectories == 3
        assert metrics.num_complete == 3
        assert 0.0 <= metrics.routing_accuracy <= 1.0
        assert metrics.replay_duration_seconds > 0
        assert "code" in metrics.routing_accuracy_by_type
        assert "ingest" in metrics.routing_accuracy_by_type

    def test_metrics_empty_trajectories(self, engine_tmp):
        engine = ReplayEngine(tmp_dir=engine_tmp)
        metrics = engine.run_with_metrics(
            RetrievalConfig(), ScoringConfig(), [], "empty_test",
        )
        assert metrics.num_trajectories == 0
        assert metrics.routing_accuracy == 0.0

    def test_metrics_escalation(self, engine_tmp):
        engine = ReplayEngine(tmp_dir=engine_tmp)
        trajectories = [
            _make_trajectory("t1", "code", "f1", "coder", "failure",
                             escalations=["architect_general"]),
        ]
        metrics = engine.run_with_metrics(
            RetrievalConfig(), ScoringConfig(), trajectories, "esc_test",
        )
        assert metrics.num_trajectories == 1

    def test_metrics_q_convergence(self, engine_tmp):
        """Q convergence should be computed from Q-value series."""
        engine = ReplayEngine(tmp_dir=engine_tmp)
        rng = np.random.default_rng(99)
        # Generate enough trajectories for convergence calculation
        trajectories = [
            _make_trajectory(f"t{i}", "code", f"f{i}", "coder", "success",
                             rng.standard_normal(1024).astype(np.float32), i)
            for i in range(20)
        ]
        metrics = engine.run_with_metrics(
            RetrievalConfig(), ScoringConfig(), trajectories, "conv_test",
        )
        assert metrics.q_convergence_step >= 0
        assert metrics.q_convergence_step <= 20

    def test_metrics_include_regret_objective_fields(self, engine_tmp):
        engine = ReplayEngine(tmp_dir=engine_tmp)
        trajectories = [
            _make_trajectory("t1", outcome="success"),
            _make_trajectory(
                "t2",
                outcome="success",
                cost_metrics_override={
                    "pass_teacher": 1,
                    "pass_chosen": 0,
                    "regret": 1,
                    "speedup_vs_teacher": 1.8,
                    "elapsed_seconds": 4.0,
                },
                offset_hours=2,
            ),
        ]
        metrics = engine.run_with_metrics(
            RetrievalConfig(cost_lambda=0.3),
            ScoringConfig(teacher_regret_penalty=0.25),
            trajectories,
            "obj_test",
        )
        assert isinstance(metrics.utility_score, float)
        assert isinstance(metrics.rm_softmax_score, float)
        assert metrics.regret_mean >= 0.0
        assert metrics.regret_p95 >= metrics.regret_mean
        assert metrics.speedup_vs_teacher_mean >= 0.0

    def test_regret_objective_decreases_when_regret_increases(self, engine_tmp):
        engine = ReplayEngine(tmp_dir=engine_tmp)
        common = dict(task_type="code", objective="same", routing_decision="coder_primary")

        low_regret = [
            _make_trajectory(
                "a1",
                outcome="success",
                offset_hours=0,
                cost_metrics_override={
                    "elapsed_seconds": 2.0,
                    "pass_teacher": 1,
                    "pass_chosen": 1,
                    "regret": 0,
                },
                **common,
            )
        ]
        high_regret = [
            _make_trajectory(
                "b1",
                outcome="success",
                offset_hours=0,
                cost_metrics_override={
                    "elapsed_seconds": 2.0,
                    "pass_teacher": 1,
                    "pass_chosen": 0,
                    "regret": 1,
                },
                **common,
            )
        ]

        retrieval = RetrievalConfig(cost_lambda=0.2)
        scoring = ScoringConfig(teacher_regret_penalty=0.4)
        m_low = engine.run_with_metrics(retrieval, scoring, low_regret, "low_regret")
        m_high = engine.run_with_metrics(retrieval, scoring, high_regret, "high_regret")

        assert m_low.utility_score > m_high.utility_score
        assert m_low.rm_softmax_score > m_high.rm_softmax_score


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_make_fake_outcome_success(self):
        entry = _make_fake_outcome("success")
        assert entry.event_type == EventType.TASK_COMPLETED
        assert entry.outcome == "success"

    def test_make_fake_outcome_failure(self):
        entry = _make_fake_outcome("failure")
        assert entry.event_type == EventType.TASK_FAILED
        assert entry.outcome == "failure"

    def test_find_convergence_step(self):
        # Constant values → immediate convergence
        values = [0.75] * 20
        step = ReplayEngine._find_convergence_step(values, threshold=0.05, window=10)
        assert step == 10  # First window

    def test_find_convergence_step_never(self):
        # Highly variable → no convergence
        rng = np.random.default_rng(42)
        values = list(rng.uniform(0.0, 1.0, 50))
        step = ReplayEngine._find_convergence_step(values, threshold=0.001, window=10)
        assert step == 50  # Never converged

    def test_find_convergence_step_short_list(self):
        values = [0.5, 0.6]
        step = ReplayEngine._find_convergence_step(values, threshold=0.05, window=10)
        assert step == 2  # Shorter than window
