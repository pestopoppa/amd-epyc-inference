"""Tests for trajectory extraction from progress logs."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from orchestration.repl_memory.progress_logger import EventType, ProgressEntry
from orchestration.repl_memory.replay.trajectory import Trajectory, TrajectoryExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(offset_hours: int = 0) -> datetime:
    """Create a timestamp offset from a fixed base."""
    base = datetime(2026, 2, 1, 12, 0, 0)
    return base + timedelta(hours=offset_hours)


def _make_entry(
    event_type: EventType,
    task_id: str = "task-1",
    data: Optional[Dict[str, Any]] = None,
    outcome: Optional[str] = None,
    offset_hours: int = 0,
    agent_role: Optional[str] = None,
) -> ProgressEntry:
    return ProgressEntry(
        event_type=event_type,
        task_id=task_id,
        timestamp=_ts(offset_hours),
        agent_role=agent_role,
        data=data or {},
        outcome=outcome,
    )


def _complete_task_entries(
    task_id: str = "task-1",
    task_type: str = "code",
    objective: str = "Write a function",
    routing_action: str = "coder_primary",
    outcome: str = "success",
    base_offset: int = 0,
) -> List[ProgressEntry]:
    """Create a minimal complete set of entries for one task."""
    return [
        _make_entry(
            EventType.TASK_STARTED, task_id,
            data={"task_type": task_type, "objective": objective},
            offset_hours=base_offset,
        ),
        _make_entry(
            EventType.ROUTING_DECISION, task_id,
            data={"action": routing_action, "role": routing_action},
            offset_hours=base_offset + 1,
        ),
        _make_entry(
            EventType.TASK_COMPLETED, task_id,
            outcome=outcome,
            data={"tokens_generated": 150, "elapsed_seconds": 3.5, "role": routing_action},
            offset_hours=base_offset + 2,
        ),
    ]


class FakeReader:
    """Fake ProgressReader that returns pre-set entries."""

    def __init__(self, entries: List[ProgressEntry]):
        self._entries = entries

    def read_recent(self, days: int = 7) -> List[ProgressEntry]:
        return list(self._entries)


# ---------------------------------------------------------------------------
# Trajectory dataclass tests
# ---------------------------------------------------------------------------

class TestTrajectory:
    def test_to_dict_round_trip(self):
        t = Trajectory(
            task_id="t1",
            task_type="code",
            objective="Write tests",
            routing_decision="coder_primary",
            outcome="success",
            cost_metrics={"tokens_generated": 100},
            escalations=["architect_general"],
            gate_results=[{"gate": "lint", "passed": True}],
            started_at=_ts(0),
            completed_at=_ts(1),
        )
        d = t.to_dict()
        assert d["task_id"] == "t1"
        assert d["outcome"] == "success"

        t2 = Trajectory.from_dict(d)
        assert t2.task_id == t.task_id
        assert t2.task_type == t.task_type
        assert t2.routing_decision == t.routing_decision
        assert t2.outcome == t.outcome
        assert t2.cost_metrics == t.cost_metrics

    def test_to_dict_none_timestamps(self):
        t = Trajectory(
            task_id="t1", task_type="code", objective="x",
            routing_decision="coder", outcome="success",
        )
        d = t.to_dict()
        assert d["started_at"] is None
        assert d["completed_at"] is None

        t2 = Trajectory.from_dict(d)
        assert t2.started_at is None


# ---------------------------------------------------------------------------
# TrajectoryExtractor tests
# ---------------------------------------------------------------------------

class TestTrajectoryExtractor:
    def test_extract_single_complete_trajectory(self):
        entries = _complete_task_entries()
        extractor = TrajectoryExtractor(reader=FakeReader(entries))
        result = extractor.extract_complete(days=7, max_trajectories=0)

        assert len(result) == 1
        t = result[0]
        assert t.task_id == "task-1"
        assert t.task_type == "code"
        assert t.objective == "Write a function"
        assert t.routing_decision == "coder_primary"
        assert t.outcome == "success"
        assert t.cost_metrics["tokens_generated"] == 150

    def test_skip_incomplete_no_routing(self):
        """Trajectories without routing_decision are excluded."""
        entries = [
            _make_entry(EventType.TASK_STARTED, "task-1", data={"task_type": "code"}),
            _make_entry(EventType.TASK_COMPLETED, "task-1", outcome="success", offset_hours=1),
        ]
        extractor = TrajectoryExtractor(reader=FakeReader(entries))
        result = extractor.extract_complete(days=7, max_trajectories=0)
        assert len(result) == 0

    def test_skip_incomplete_no_completion(self):
        """Trajectories without task_completed are excluded."""
        entries = [
            _make_entry(EventType.TASK_STARTED, "task-1", data={"task_type": "code"}),
            _make_entry(EventType.ROUTING_DECISION, "task-1", data={"action": "coder"}),
        ]
        extractor = TrajectoryExtractor(reader=FakeReader(entries))
        result = extractor.extract_complete(days=7, max_trajectories=0)
        assert len(result) == 0

    def test_skip_incomplete_no_start(self):
        """Trajectories without task_started are excluded."""
        entries = [
            _make_entry(EventType.ROUTING_DECISION, "task-1", data={"action": "coder"}),
            _make_entry(EventType.TASK_COMPLETED, "task-1", outcome="success"),
        ]
        extractor = TrajectoryExtractor(reader=FakeReader(entries))
        result = extractor.extract_complete(days=7, max_trajectories=0)
        assert len(result) == 0

    def test_multiple_tasks(self):
        entries = (
            _complete_task_entries("t1", "code", "Write func", "coder", "success", 0)
            + _complete_task_entries("t2", "ingest", "Parse doc", "ingest", "failure", 10)
            + _complete_task_entries("t3", "explore", "Search code", "worker", "partial", 20)
        )
        extractor = TrajectoryExtractor(reader=FakeReader(entries))
        result = extractor.extract_complete(days=7, max_trajectories=0)

        assert len(result) == 3
        # Sorted by started_at
        assert result[0].task_id == "t1"
        assert result[1].task_id == "t2"
        assert result[2].task_id == "t3"

    def test_failure_outcome(self):
        entries = _complete_task_entries(outcome="failure")
        # Use TASK_FAILED instead
        entries[-1] = _make_entry(
            EventType.TASK_FAILED, "task-1", outcome="failure",
            data={"error": "timeout"}, offset_hours=2,
        )
        extractor = TrajectoryExtractor(reader=FakeReader(entries))
        result = extractor.extract_complete(days=7, max_trajectories=0)

        assert len(result) == 1
        assert result[0].outcome == "failure"

    def test_escalations_captured(self):
        entries = _complete_task_entries()
        entries.insert(2, _make_entry(
            EventType.ESCALATION_TRIGGERED, "task-1",
            data={"target_role": "architect_general"},
            offset_hours=1,
        ))
        extractor = TrajectoryExtractor(reader=FakeReader(entries))
        result = extractor.extract_complete(days=7, max_trajectories=0)

        assert len(result) == 1
        assert "architect_general" in result[0].escalations

    def test_gate_results_captured(self):
        entries = _complete_task_entries()
        entries.insert(2, _make_entry(
            EventType.GATE_PASSED, "task-1",
            data={"gate_name": "lint"},
            offset_hours=1,
        ))
        entries.insert(3, _make_entry(
            EventType.GATE_FAILED, "task-1",
            data={"gate_name": "type_check"},
            offset_hours=1,
        ))
        extractor = TrajectoryExtractor(reader=FakeReader(entries))
        result = extractor.extract_complete(days=7, max_trajectories=0)

        assert len(result) == 1
        gates = result[0].gate_results
        assert len(gates) == 2
        assert gates[0] == {"gate": "lint", "passed": True}
        assert gates[1] == {"gate": "type_check", "passed": False}

    def test_plan_reviews_captured(self):
        entries = _complete_task_entries()
        entries.insert(2, _make_entry(
            EventType.PLAN_REVIEWED, "task-1",
            data={"decision": "ok"},
            offset_hours=1,
        ))
        extractor = TrajectoryExtractor(reader=FakeReader(entries))
        result = extractor.extract_complete(days=7, max_trajectories=0)

        assert len(result) == 1
        assert len(result[0].plan_review_entries) == 1

    def test_empty_logs(self):
        extractor = TrajectoryExtractor(reader=FakeReader([]))
        result = extractor.extract_complete(days=7, max_trajectories=0)
        assert len(result) == 0

    def test_entries_without_task_id_skipped(self):
        entries = _complete_task_entries()
        # Add an entry with no task_id
        bad = _make_entry(EventType.TASK_STARTED, "", data={"task_type": "x"})
        entries.append(bad)
        extractor = TrajectoryExtractor(reader=FakeReader(entries))
        result = extractor.extract_complete(days=7, max_trajectories=0)
        # Only the complete task should appear
        assert len(result) == 1

    def test_cost_metrics_extracted(self):
        entries = _complete_task_entries()
        extractor = TrajectoryExtractor(reader=FakeReader(entries))
        result = extractor.extract_complete(days=7, max_trajectories=0)

        t = result[0]
        assert t.cost_metrics["tokens_generated"] == 150
        assert t.cost_metrics["elapsed_seconds"] == 3.5
        assert t.cost_metrics["role"] == "coder_primary"


# ---------------------------------------------------------------------------
# Stratified sampling tests
# ---------------------------------------------------------------------------

class TestStratifiedSampling:
    def test_sampling_reduces_count(self):
        """Sampling with max_trajectories limits output."""
        entries = []
        for i in range(20):
            entries.extend(_complete_task_entries(
                f"t{i}", "code", f"task {i}", "coder", "success", i * 10,
            ))
        extractor = TrajectoryExtractor(reader=FakeReader(entries))
        result = extractor.extract_complete(days=30, max_trajectories=5)
        assert len(result) == 5

    def test_sampling_preserves_types(self):
        """Stratified sampling includes all task_types."""
        entries = []
        # 15 code tasks, 5 ingest tasks
        for i in range(15):
            entries.extend(_complete_task_entries(
                f"code-{i}", "code", f"code task {i}", "coder", "success", i * 5,
            ))
        for i in range(5):
            entries.extend(_complete_task_entries(
                f"ingest-{i}", "ingest", f"ingest task {i}", "ingest", "success", 100 + i * 5,
            ))
        extractor = TrajectoryExtractor(reader=FakeReader(entries))
        result = extractor.extract_complete(days=30, max_trajectories=10, seed=42)

        types = {t.task_type for t in result}
        assert "code" in types
        assert "ingest" in types
        assert len(result) == 10

    def test_sampling_is_deterministic(self):
        """Same seed produces same result."""
        entries = []
        for i in range(50):
            entries.extend(_complete_task_entries(
                f"t{i}", "code", f"task {i}", "coder", "success", i * 3,
            ))
        extractor = TrajectoryExtractor(reader=FakeReader(entries))
        r1 = extractor.extract_complete(days=30, max_trajectories=10, seed=42)
        r2 = extractor.extract_complete(days=30, max_trajectories=10, seed=42)
        assert [t.task_id for t in r1] == [t.task_id for t in r2]

    def test_no_sampling_when_under_limit(self):
        entries = _complete_task_entries()
        extractor = TrajectoryExtractor(reader=FakeReader(entries))
        result = extractor.extract_complete(days=7, max_trajectories=100)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Embedding pre-computation tests
# ---------------------------------------------------------------------------

class TestEmbeddingPrecomputation:
    def test_precompute_with_batch_embedder(self):
        entries = _complete_task_entries()
        mock_embedder = MagicMock()
        mock_embedder.embed_batch.return_value = np.random.randn(1, 1024).astype(np.float32)

        extractor = TrajectoryExtractor(
            reader=FakeReader(entries),
            embedder=mock_embedder,
            cache_dir=Path("/mnt/raid0/llm/tmp/test_cache"),
        )
        trajectories = extractor.extract_complete(days=7, max_trajectories=0)
        result = extractor.precompute_embeddings(trajectories, cache_key="test")

        assert len(result) == 1
        assert result[0].embedding is not None
        assert result[0].embedding.shape == (1024,)
        mock_embedder.embed_batch.assert_called_once()

    def test_precompute_without_embedder(self):
        entries = _complete_task_entries()
        extractor = TrajectoryExtractor(reader=FakeReader(entries))
        trajectories = extractor.extract_complete(days=7, max_trajectories=0)
        result = extractor.precompute_embeddings(trajectories)

        assert len(result) == 1
        assert result[0].embedding is None

    def test_precompute_with_text_embedder_fallback(self):
        entries = _complete_task_entries()
        mock_embedder = MagicMock(spec=[])  # No embed_batch
        mock_embedder.embed_text = MagicMock(return_value=np.random.randn(1024).astype(np.float32))

        extractor = TrajectoryExtractor(
            reader=FakeReader(entries),
            embedder=mock_embedder,
            cache_dir=Path("/mnt/raid0/llm/tmp/test_cache2"),
        )
        trajectories = extractor.extract_complete(days=7, max_trajectories=0)
        result = extractor.precompute_embeddings(trajectories, cache_key="test2")

        assert result[0].embedding is not None
        mock_embedder.embed_text.assert_called_once()
