"""
Unit tests for skill evolution monitor.

All tests use in-memory SkillBank — no live inference.
"""

import tempfile
from pathlib import Path

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


def _make_skill(skill_bank, title, confidence=0.5, retrieval_count=10,
                effectiveness_score=0.0, skill_type="routing", task_types=None):
    from orchestration.repl_memory.skill_bank import Skill, SkillBank

    skill = Skill(
        id=SkillBank.generate_id(skill_type),
        title=title,
        skill_type=skill_type,
        principle=f"Principle for {title}",
        when_to_apply="always",
        task_types=task_types or ["*"],
        source_trajectory_ids=["t1"],
        source_outcome="success",
        confidence=confidence,
        retrieval_count=retrieval_count,
        effectiveness_score=effectiveness_score,
    )
    skill_bank.store(skill)
    return skill


class TestEvolutionMonitor:
    """Tests for EvolutionMonitor."""

    def test_empty_bank(self, skill_bank):
        from orchestration.repl_memory.skill_evolution import EvolutionMonitor

        monitor = EvolutionMonitor(skill_bank)
        report = monitor.run_evolution_cycle()

        assert report.skills_evaluated == 0
        assert report.skills_promoted == 0
        assert report.skills_deprecated == 0

    def test_promotes_high_effectiveness(self, skill_bank):
        from orchestration.repl_memory.skill_evolution import (
            EvolutionMonitor,
            OutcomeTracker,
        )

        skill = _make_skill(skill_bank, "Good Skill", confidence=0.7, retrieval_count=10)

        tracker = OutcomeTracker()
        for _ in range(10):
            tracker.record_outcome(skill.id, f"t{_}", success=True)

        monitor = EvolutionMonitor(skill_bank)
        report = monitor.run_evolution_cycle(outcome_tracker=tracker)

        assert report.skills_promoted == 1
        updated = skill_bank.get_by_id(skill.id)
        assert updated.confidence > 0.7

    def test_decays_low_effectiveness(self, skill_bank):
        from orchestration.repl_memory.skill_evolution import (
            EvolutionMonitor,
            OutcomeTracker,
        )

        skill = _make_skill(skill_bank, "Bad Skill", confidence=0.5, retrieval_count=10)

        tracker = OutcomeTracker()
        for i in range(10):
            tracker.record_outcome(skill.id, f"t{i}", success=False)

        monitor = EvolutionMonitor(skill_bank)
        report = monitor.run_evolution_cycle(outcome_tracker=tracker)

        assert report.skills_decayed == 1
        updated = skill_bank.get_by_id(skill.id)
        assert updated.confidence < 0.5

    def test_deprecates_very_low_confidence(self, skill_bank):
        from orchestration.repl_memory.skill_evolution import (
            EvolutionMonitor,
            OutcomeTracker,
        )

        skill = _make_skill(skill_bank, "Terrible Skill", confidence=0.08, retrieval_count=10)

        tracker = OutcomeTracker()
        for i in range(10):
            tracker.record_outcome(skill.id, f"t{i}", success=False)

        monitor = EvolutionMonitor(skill_bank)
        report = monitor.run_evolution_cycle(outcome_tracker=tracker)

        assert report.skills_deprecated == 1
        updated = skill_bank.get_by_id(skill.id)
        assert updated.deprecated is True

    def test_redistillation_candidates(self, skill_bank):
        from orchestration.repl_memory.skill_evolution import (
            EvolutionMonitor,
            OutcomeTracker,
        )

        skill = _make_skill(
            skill_bank, "Dying Skill",
            confidence=0.08, retrieval_count=10,
            task_types=["code_generation", "debugging"],
        )

        tracker = OutcomeTracker()
        for i in range(10):
            tracker.record_outcome(skill.id, f"t{i}", success=False)

        monitor = EvolutionMonitor(skill_bank)
        report = monitor.run_evolution_cycle(outcome_tracker=tracker)

        assert "code_generation" in report.redistillation_candidates
        assert "debugging" in report.redistillation_candidates

    def test_skips_under_retrieved_skills(self, skill_bank):
        from orchestration.repl_memory.skill_evolution import (
            EvolutionMonitor,
            OutcomeTracker,
        )

        _make_skill(skill_bank, "New Skill", confidence=0.5, retrieval_count=2)

        tracker = OutcomeTracker()
        monitor = EvolutionMonitor(skill_bank)
        report = monitor.run_evolution_cycle(outcome_tracker=tracker)

        assert report.skills_evaluated == 1
        assert report.skills_promoted == 0
        assert report.skills_decayed == 0

    def test_heuristic_effectiveness_with_score(self, skill_bank):
        from orchestration.repl_memory.skill_evolution import EvolutionMonitor

        _make_skill(
            skill_bank, "Scored Skill",
            confidence=0.5, retrieval_count=20,
            effectiveness_score=0.9,
        )

        monitor = EvolutionMonitor(skill_bank)
        report = monitor.run_evolution_cycle()

        assert report.skills_promoted == 1

    def test_heuristic_effectiveness_without_score(self, skill_bank):
        from orchestration.repl_memory.skill_evolution import EvolutionMonitor

        # High confidence + high retrieval = heuristic effective
        _make_skill(
            skill_bank, "Popular Skill",
            confidence=0.9, retrieval_count=30,
        )

        monitor = EvolutionMonitor(skill_bank)
        report = monitor.run_evolution_cycle()

        assert report.skills_promoted == 1

    def test_max_confidence_cap(self, skill_bank):
        from orchestration.repl_memory.skill_evolution import (
            EvolutionMonitor,
            EvolutionConfig,
            OutcomeTracker,
        )

        skill = _make_skill(skill_bank, "Max Skill", confidence=0.94, retrieval_count=10)

        tracker = OutcomeTracker()
        for i in range(10):
            tracker.record_outcome(skill.id, f"t{i}", success=True)

        config = EvolutionConfig(max_confidence=0.95, promotion_boost=0.1)
        monitor = EvolutionMonitor(skill_bank, config=config)
        monitor.run_evolution_cycle(outcome_tracker=tracker)

        updated = skill_bank.get_by_id(skill.id)
        assert updated.confidence <= 0.95

    def test_report_to_dict(self):
        from orchestration.repl_memory.skill_evolution import EvolutionReport

        report = EvolutionReport(
            skills_evaluated=10,
            skills_promoted=3,
            skills_decayed=2,
            skills_deprecated=1,
            redistillation_candidates=["code"],
        )
        d = report.to_dict()
        assert d["skills_evaluated"] == 10
        assert d["redistillation_candidates"] == ["code"]


class TestOutcomeTracker:
    """Tests for OutcomeTracker."""

    def test_empty_returns_neutral(self):
        from orchestration.repl_memory.skill_evolution import OutcomeTracker

        tracker = OutcomeTracker()
        assert tracker.get_skill_effectiveness("nonexistent") == 0.5

    def test_all_success(self):
        from orchestration.repl_memory.skill_evolution import OutcomeTracker

        tracker = OutcomeTracker()
        for i in range(5):
            tracker.record_outcome("sk_1", f"t{i}", success=True)
        assert tracker.get_skill_effectiveness("sk_1") == 1.0

    def test_all_failure(self):
        from orchestration.repl_memory.skill_evolution import OutcomeTracker

        tracker = OutcomeTracker()
        for i in range(5):
            tracker.record_outcome("sk_1", f"t{i}", success=False)
        assert tracker.get_skill_effectiveness("sk_1") == 0.0

    def test_mixed_outcomes(self):
        from orchestration.repl_memory.skill_evolution import OutcomeTracker

        tracker = OutcomeTracker()
        tracker.record_outcome("sk_1", "t1", success=True)
        tracker.record_outcome("sk_1", "t2", success=True)
        tracker.record_outcome("sk_1", "t3", success=False)
        assert abs(tracker.get_skill_effectiveness("sk_1") - 0.6667) < 0.01

    def test_get_all_effectiveness(self):
        from orchestration.repl_memory.skill_evolution import OutcomeTracker

        tracker = OutcomeTracker()
        tracker.record_outcome("sk_1", "t1", success=True)
        tracker.record_outcome("sk_2", "t2", success=False)

        all_eff = tracker.get_all_effectiveness()
        assert all_eff["sk_1"] == 1.0
        assert all_eff["sk_2"] == 0.0


class TestEvolutionSummary:
    """Tests for get_evolution_summary()."""

    def test_empty_bank(self, skill_bank):
        from orchestration.repl_memory.skill_evolution import EvolutionMonitor

        monitor = EvolutionMonitor(skill_bank)
        summary = monitor.get_evolution_summary()

        assert summary["total_active"] == 0
        assert summary["avg_confidence"] == 0.0

    def test_populated_bank(self, skill_bank):
        from orchestration.repl_memory.skill_evolution import EvolutionMonitor

        _make_skill(skill_bank, "S1", confidence=0.8, retrieval_count=10, skill_type="routing")
        _make_skill(skill_bank, "S2", confidence=0.6, retrieval_count=5, skill_type="routing")
        _make_skill(skill_bank, "S3", confidence=0.9, retrieval_count=20, skill_type="failure_lesson")

        monitor = EvolutionMonitor(skill_bank)
        summary = monitor.get_evolution_summary()

        assert summary["total_active"] == 3
        assert 0.7 < summary["avg_confidence"] < 0.8
        assert "routing" in summary["by_type"]
        assert "failure_lesson" in summary["by_type"]
        assert summary["by_type"]["routing"]["count"] == 2
