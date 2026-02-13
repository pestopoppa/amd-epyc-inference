"""Tests for DesignCandidate and DesignArchive."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from orchestration.repl_memory.q_scorer import ScoringConfig
from orchestration.repl_memory.retriever import RetrievalConfig
from orchestration.repl_memory.staged_scorer import StagedConfig
from orchestration.repl_memory.replay.candidates import DesignCandidate, DesignArchive
from orchestration.repl_memory.replay.metrics import ReplayMetrics

TEST_ARCHIVE_BASE = Path("/mnt/raid0/llm/tmp/test_design_archive")


@pytest.fixture()
def archive_dir(tmp_path):
    """Per-test unique directory for archive DB."""
    d = tmp_path / "archive"
    d.mkdir()
    return d


def _make_metrics(candidate_id: str, reward: float = 10.0) -> ReplayMetrics:
    return ReplayMetrics(
        candidate_id=candidate_id,
        num_trajectories=100,
        num_complete=95,
        routing_accuracy=0.75,
        routing_accuracy_by_type={"code": 0.80},
        cumulative_reward=reward,
        avg_reward=reward / 100,
        cost_efficiency=0.85,
        replay_duration_seconds=5.0,
    )


# ---------------------------------------------------------------------------
# DesignCandidate tests
# ---------------------------------------------------------------------------

class TestDesignCandidate:
    def test_default_creates_baseline(self):
        c = DesignCandidate.default()
        assert c.parent_id is None
        assert c.notes == "production baseline"
        assert c.retrieval_config.semantic_k == 20
        assert c.scoring_config.learning_rate == 0.1
        assert c.staged_config is not None

    def test_json_round_trip(self):
        c = DesignCandidate.default()
        json_str = c.to_json()
        c2 = DesignCandidate.from_json(json_str)

        assert c2.candidate_id == c.candidate_id
        assert c2.parent_id == c.parent_id
        assert c2.notes == c.notes
        assert c2.retrieval_config.semantic_k == c.retrieval_config.semantic_k
        assert c2.retrieval_config.q_weight == c.retrieval_config.q_weight
        assert c2.scoring_config.learning_rate == c.scoring_config.learning_rate
        assert c2.scoring_config.success_reward == c.scoring_config.success_reward
        assert c2.scoring_config.cost_penalty_lambda == c.scoring_config.cost_penalty_lambda

    def test_json_round_trip_custom_config(self):
        c = DesignCandidate(
            candidate_id="custom-1",
            parent_id="parent-1",
            retrieval_config=RetrievalConfig(semantic_k=30, q_weight=0.8),
            scoring_config=ScoringConfig(learning_rate=0.2, failure_reward=-0.8),
            staged_config=StagedConfig(initial_lambda=0.5, anneal_steps=100),
            role_overrides={"coder": {"q_weight": 0.9}},
            notes="experiment 1",
        )
        json_str = c.to_json()
        c2 = DesignCandidate.from_json(json_str)

        assert c2.candidate_id == "custom-1"
        assert c2.parent_id == "parent-1"
        assert c2.retrieval_config.semantic_k == 30
        assert c2.retrieval_config.q_weight == 0.8
        assert c2.scoring_config.learning_rate == 0.2
        assert c2.scoring_config.failure_reward == -0.8
        assert c2.staged_config.initial_lambda == 0.5
        assert c2.role_overrides == {"coder": {"q_weight": 0.9}}

    def test_json_round_trip_no_staged(self):
        c = DesignCandidate(
            candidate_id="no-staged",
            parent_id=None,
            retrieval_config=RetrievalConfig(),
            scoring_config=ScoringConfig(),
            staged_config=None,
            notes="no staged config",
        )
        c2 = DesignCandidate.from_json(c.to_json())
        assert c2.staged_config is None


# ---------------------------------------------------------------------------
# DesignArchive tests
# ---------------------------------------------------------------------------

class TestDesignArchive:
    def test_store_and_retrieve(self, archive_dir):
        archive = DesignArchive(db_path=archive_dir / "test.db")
        c = DesignCandidate.default()
        m = _make_metrics(c.candidate_id, reward=42.0)
        archive.store_result(c, m)

        results = archive.get_top_candidates(limit=10)
        assert len(results) == 1
        loaded_c, loaded_m = results[0]
        assert loaded_c.candidate_id == c.candidate_id
        assert loaded_m.cumulative_reward == 42.0

    def test_store_without_metrics(self, archive_dir):
        archive = DesignArchive(db_path=archive_dir / "test.db")
        c = DesignCandidate.default()
        archive.store_result(c, metrics=None)

        # Should not appear in get_top_candidates (no metrics)
        results = archive.get_top_candidates(limit=10)
        assert len(results) == 0

        # But count includes it
        assert archive.count() == 1

    def test_upsert_overwrites(self, archive_dir):
        archive = DesignArchive(db_path=archive_dir / "test.db")
        c = DesignCandidate.default()
        m1 = _make_metrics(c.candidate_id, reward=10.0)
        archive.store_result(c, m1)

        m2 = _make_metrics(c.candidate_id, reward=50.0)
        archive.store_result(c, m2)

        results = archive.get_top_candidates(limit=10)
        assert len(results) == 1
        assert results[0][1].cumulative_reward == 50.0

    def test_top_candidates_sorted(self, archive_dir):
        archive = DesignArchive(db_path=archive_dir / "test.db")
        for i, reward in enumerate([10.0, 50.0, 30.0, 20.0]):
            c = DesignCandidate(
                candidate_id=f"c{i}",
                parent_id=None,
                retrieval_config=RetrievalConfig(),
                scoring_config=ScoringConfig(),
                notes=f"candidate {i}",
            )
            archive.store_result(c, _make_metrics(f"c{i}", reward))

        results = archive.get_top_candidates(metric="cumulative_reward", limit=3)
        assert len(results) == 3
        assert results[0][1].cumulative_reward == 50.0
        assert results[1][1].cumulative_reward == 30.0
        assert results[2][1].cumulative_reward == 20.0

    def test_get_lineage(self, archive_dir):
        archive = DesignArchive(db_path=archive_dir / "test.db")

        # Create a lineage: grandparent → parent → child
        grandparent = DesignCandidate(
            candidate_id="gp", parent_id=None,
            retrieval_config=RetrievalConfig(), scoring_config=ScoringConfig(),
            notes="grandparent",
        )
        parent = DesignCandidate(
            candidate_id="p", parent_id="gp",
            retrieval_config=RetrievalConfig(), scoring_config=ScoringConfig(),
            notes="parent",
        )
        child = DesignCandidate(
            candidate_id="c", parent_id="p",
            retrieval_config=RetrievalConfig(), scoring_config=ScoringConfig(),
            notes="child",
        )
        for c in [grandparent, parent, child]:
            archive.store_result(c)

        lineage = archive.get_lineage("c")
        assert len(lineage) == 3
        assert lineage[0].candidate_id == "c"
        assert lineage[1].candidate_id == "p"
        assert lineage[2].candidate_id == "gp"

    def test_get_lineage_single(self, archive_dir):
        archive = DesignArchive(db_path=archive_dir / "test.db")
        c = DesignCandidate.default()
        archive.store_result(c)

        lineage = archive.get_lineage(c.candidate_id)
        assert len(lineage) == 1
        assert lineage[0].candidate_id == c.candidate_id

    def test_get_lineage_missing(self, archive_dir):
        archive = DesignArchive(db_path=archive_dir / "test.db")
        lineage = archive.get_lineage("nonexistent")
        assert len(lineage) == 0

    def test_get_baseline(self, archive_dir):
        archive = DesignArchive(db_path=archive_dir / "test.db")
        c = DesignCandidate.default()  # notes="production baseline"
        m = _make_metrics(c.candidate_id, 42.0)
        archive.store_result(c, m)

        result = archive.get_baseline()
        assert result is not None
        assert result[0].notes == "production baseline"
        assert result[1].cumulative_reward == 42.0

    def test_get_baseline_none(self, archive_dir):
        archive = DesignArchive(db_path=archive_dir / "test.db")
        assert archive.get_baseline() is None

    def test_sample_for_reflection(self, archive_dir):
        archive = DesignArchive(db_path=archive_dir / "test.db")
        for i in range(10):
            c = DesignCandidate(
                candidate_id=f"c{i}", parent_id=None,
                retrieval_config=RetrievalConfig(), scoring_config=ScoringConfig(),
                notes=f"candidate {i}",
            )
            archive.store_result(c, _make_metrics(f"c{i}", float(i * 5)))

        sample = archive.sample_for_reflection(n=5)
        assert len(sample) == 5

        # Top 2 should be highest rewards
        rewards = [s[1].cumulative_reward for s in sample]
        assert 45.0 in rewards  # Top 1
        assert 40.0 in rewards  # Top 2
        # Worst should be present
        assert 0.0 in rewards

    def test_sample_for_reflection_small_archive(self, archive_dir):
        archive = DesignArchive(db_path=archive_dir / "test.db")
        c = DesignCandidate.default()
        archive.store_result(c, _make_metrics(c.candidate_id))

        sample = archive.sample_for_reflection(n=5)
        assert len(sample) == 1

    def test_count(self, archive_dir):
        archive = DesignArchive(db_path=archive_dir / "test.db")
        assert archive.count() == 0
        c = DesignCandidate.default()
        archive.store_result(c)
        assert archive.count() == 1
