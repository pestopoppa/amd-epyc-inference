"""
Unit tests for skill-aware replay engine and SkillBankConfig integration.

All tests use in-memory stores — no live inference, no API calls.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def skill_bank(temp_dir):
    from orchestration.repl_memory.skill_bank import SkillBank

    return SkillBank(
        db_path=temp_dir / "skills.db",
        faiss_path=temp_dir,
        embedding_dim=128,
    )


def _make_trajectory(task_id="t1", task_type="code", outcome="success",
                     routing_decision="frontdoor", dim=128):
    """Create a minimal Trajectory for testing."""
    from orchestration.repl_memory.replay.trajectory import Trajectory

    return Trajectory(
        task_id=task_id,
        task_type=task_type,
        objective=f"Test task {task_id}",
        routing_decision=routing_decision,
        outcome=outcome,
        escalations=[],
        embedding=np.random.randn(dim).astype(np.float32),
    )


class TestSkillBankConfig:
    """Tests for SkillBankConfig dataclass."""

    def test_defaults(self):
        from orchestration.repl_memory.replay.skill_replay import SkillBankConfig

        cfg = SkillBankConfig()
        assert cfg.enabled is True
        assert cfg.general_skills_max == 6
        assert cfg.min_similarity == 0.4

    def test_to_retrieval_config(self):
        from orchestration.repl_memory.replay.skill_replay import SkillBankConfig

        cfg = SkillBankConfig(
            general_skills_max=3,
            task_specific_k=4,
            min_similarity=0.5,
        )
        rc = cfg.to_retrieval_config()
        assert rc.general_skills_max == 3
        assert rc.task_specific_k == 4
        assert rc.min_similarity == 0.5

    def test_disabled(self):
        from orchestration.repl_memory.replay.skill_replay import SkillBankConfig

        cfg = SkillBankConfig(enabled=False)
        assert cfg.enabled is False


class TestDesignCandidateSkillConfig:
    """Tests for SkillBankConfig on DesignCandidate."""

    def test_default_candidate_has_skill_config(self):
        from orchestration.repl_memory.replay.candidates import DesignCandidate

        c = DesignCandidate.default()
        assert c.skill_config is not None
        assert c.skill_config.enabled is True

    def test_serialization_roundtrip_with_skill_config(self):
        from orchestration.repl_memory.replay.candidates import DesignCandidate
        from orchestration.repl_memory.replay.skill_replay import SkillBankConfig

        c = DesignCandidate.default()
        c.skill_config = SkillBankConfig(
            enabled=True,
            general_skills_max=4,
            min_similarity=0.5,
        )

        json_str = c.to_json()
        c2 = DesignCandidate.from_json(json_str)

        assert c2.skill_config is not None
        assert c2.skill_config.general_skills_max == 4
        assert c2.skill_config.min_similarity == 0.5

    def test_serialization_without_skill_config(self):
        from orchestration.repl_memory.replay.candidates import DesignCandidate

        c = DesignCandidate.default()
        c.skill_config = None

        json_str = c.to_json()
        c2 = DesignCandidate.from_json(json_str)
        assert c2.skill_config is None

    def test_backward_compat_old_json(self):
        """Old JSON without skill_config field should deserialize fine."""
        import json
        from orchestration.repl_memory.replay.candidates import DesignCandidate

        c = DesignCandidate.default()
        j = json.loads(c.to_json())
        del j["skill_config"]  # Simulate old format
        c2 = DesignCandidate.from_json(json.dumps(j))
        assert c2.skill_config is None  # Gracefully missing


class TestSkillAwareReplayEngine:
    """Tests for SkillAwareReplayEngine."""

    def test_no_skill_bank(self, temp_dir):
        from orchestration.repl_memory.replay.skill_replay import (
            SkillAwareReplayEngine,
            SkillBankConfig,
        )
        from orchestration.repl_memory.retriever import RetrievalConfig
        from orchestration.repl_memory.q_scorer import ScoringConfig

        engine = SkillAwareReplayEngine(
            tmp_dir=temp_dir / "replay",
            embedding_dim=128,
        )

        trajectories = [_make_trajectory(f"t{i}") for i in range(3)]
        metrics = engine.run_with_skill_metrics(
            retrieval_config=RetrievalConfig(),
            scoring_config=ScoringConfig(),
            skill_config=SkillBankConfig(enabled=True),
            trajectories=trajectories,
        )

        assert metrics.base_metrics.num_trajectories == 3
        assert metrics.total_skills_retrieved == 0
        assert metrics.skill_coverage == 0.0

    def test_with_skill_bank(self, temp_dir, skill_bank):
        from orchestration.repl_memory.replay.skill_replay import (
            SkillAwareReplayEngine,
            SkillBankConfig,
        )
        from orchestration.repl_memory.skill_bank import Skill, SkillBank
        from orchestration.repl_memory.retriever import RetrievalConfig
        from orchestration.repl_memory.q_scorer import ScoringConfig

        # Add a general skill
        skill = Skill(
            id=SkillBank.generate_id("general"),
            title="Always Test First",
            skill_type="general",
            principle="Run tests before deploying.",
            when_to_apply="always",
            task_types=["*"],
            source_trajectory_ids=["t0"],
            source_outcome="success",
            confidence=0.8,
        )
        skill_bank.store(skill)

        engine = SkillAwareReplayEngine(
            skill_bank=skill_bank,
            tmp_dir=temp_dir / "replay",
            embedding_dim=128,
        )

        trajectories = [_make_trajectory(f"t{i}") for i in range(3)]
        metrics = engine.run_with_skill_metrics(
            retrieval_config=RetrievalConfig(),
            scoring_config=ScoringConfig(),
            skill_config=SkillBankConfig(),
            trajectories=trajectories,
        )

        assert metrics.base_metrics.num_trajectories == 3
        # Should have retrieved the general skill for each step
        assert metrics.total_skills_retrieved >= 3
        assert metrics.skill_coverage > 0.0
        assert metrics.avg_skills_per_step >= 1.0

    def test_disabled_skill_config(self, temp_dir, skill_bank):
        from orchestration.repl_memory.replay.skill_replay import (
            SkillAwareReplayEngine,
            SkillBankConfig,
        )
        from orchestration.repl_memory.skill_bank import Skill, SkillBank
        from orchestration.repl_memory.retriever import RetrievalConfig
        from orchestration.repl_memory.q_scorer import ScoringConfig

        # Add skill
        skill = Skill(
            id=SkillBank.generate_id("general"),
            title="Disabled Test",
            skill_type="general",
            principle="Should not be retrieved.",
            when_to_apply="always",
            task_types=["*"],
            source_trajectory_ids=["t0"],
            source_outcome="success",
            confidence=0.8,
        )
        skill_bank.store(skill)

        engine = SkillAwareReplayEngine(
            skill_bank=skill_bank,
            tmp_dir=temp_dir / "replay",
            embedding_dim=128,
        )

        trajectories = [_make_trajectory("t1")]
        metrics = engine.run_with_skill_metrics(
            retrieval_config=RetrievalConfig(),
            scoring_config=ScoringConfig(),
            skill_config=SkillBankConfig(enabled=False),
            trajectories=trajectories,
        )

        assert metrics.total_skills_retrieved == 0
        assert metrics.skill_coverage == 0.0


class TestSkillReplayMetrics:
    """Tests for SkillReplayMetrics serialization."""

    def test_to_dict(self):
        from orchestration.repl_memory.replay.skill_replay import SkillReplayMetrics
        from orchestration.repl_memory.replay.metrics import ReplayMetrics

        base = ReplayMetrics(
            candidate_id="test",
            num_trajectories=10,
            num_complete=10,
            routing_accuracy=0.8,
        )
        metrics = SkillReplayMetrics(
            base_metrics=base,
            avg_skills_per_step=2.5,
            total_skills_retrieved=25,
            skill_coverage=0.9,
            avg_skill_context_tokens=150.0,
            skills_by_type={"routing": 15, "failure_lesson": 10},
        )
        d = metrics.to_dict()

        assert d["routing_accuracy"] == 0.8
        assert d["skill_metrics"]["avg_skills_per_step"] == 2.5
        assert d["skill_metrics"]["total_skills_retrieved"] == 25
        assert d["skill_metrics"]["skills_by_type"]["routing"] == 15
