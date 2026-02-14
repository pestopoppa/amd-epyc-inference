"""
Unit tests for FailureBridge (FailureGraph ↔ SkillBank bridge).

All tests use mock FailureGraph — no Kuzu dependency, no live inference.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class MockFailureGraphConn:
    """Mock Kuzu connection for testing."""

    def __init__(self, query_results=None):
        self._results = query_results or {}

    def execute(self, query, params=None):
        mock_result = MagicMock()
        # Return empty DataFrame by default
        import pandas as pd

        mock_result.get_as_df.return_value = pd.DataFrame()
        return mock_result


class MockFailureGraph:
    """Mock FailureGraph for testing without Kuzu."""

    def __init__(
        self,
        stats=None,
        effective_mitigations=None,
        all_mitigations=None,
    ):
        self._stats = stats or {
            "failuremode_count": 0,
            "mitigation_count": 0,
            "symptom_count": 0,
            "memorylink_count": 0,
        }
        self._effective_mitigations = effective_mitigations or []
        self._all_mitigations = all_mitigations or []
        self.conn = MockFailureGraphConn()

    def get_stats(self):
        return self._stats

    def get_effective_mitigations(self, symptoms):
        return self._effective_mitigations

    def get_failure_risk(self, action):
        return 0.0


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


class TestFailureContextGeneration:
    """Tests for get_failure_context_for_distillation()."""

    def test_empty_graph_returns_default(self, skill_bank):
        from orchestration.repl_memory.distillation.failure_bridge import FailureBridge

        fg = MockFailureGraph()
        bridge = FailureBridge(fg, skill_bank)
        summary = bridge.get_failure_context_for_distillation()
        assert "No FailureGraph context available" in summary

    def test_populated_graph_returns_summary(self, skill_bank):
        from orchestration.repl_memory.distillation.failure_bridge import FailureBridge

        fg = MockFailureGraph(stats={
            "failuremode_count": 5,
            "mitigation_count": 3,
            "symptom_count": 8,
            "memorylink_count": 10,
        })
        bridge = FailureBridge(fg, skill_bank)
        summary = bridge.get_failure_context_for_distillation()

        assert "5 known failure modes" in summary
        assert "3 mitigations" in summary
        assert "8 symptoms" in summary
        assert "Cross-reference" in summary

    def test_includes_existing_failure_lessons(self, skill_bank):
        from orchestration.repl_memory.distillation.failure_bridge import FailureBridge
        from orchestration.repl_memory.skill_bank import Skill, SkillBank

        # Add a failure_lesson to skill bank
        skill = Skill(
            id=SkillBank.generate_id("failure_lesson"),
            title="Avoid Prompt Lookup for Novel Code",
            skill_type="failure_lesson",
            principle="FAILURE POINT: 0% acceptance. PREVENTION: Use spec decode.",
            when_to_apply="code_generation without context",
            task_types=["code_generation"],
            source_trajectory_ids=["t1"],
            source_outcome="failure",
            confidence=0.85,
        )
        skill_bank.store(skill)

        fg = MockFailureGraph(stats={
            "failuremode_count": 2,
            "mitigation_count": 1,
            "symptom_count": 3,
        })
        bridge = FailureBridge(fg, skill_bank)
        summary = bridge.get_failure_context_for_distillation()

        assert "Avoid Prompt Lookup" in summary
        assert "confidence=0.85" in summary

    def test_graph_error_returns_default(self, skill_bank):
        from orchestration.repl_memory.distillation.failure_bridge import FailureBridge

        fg = MockFailureGraph()
        fg.get_stats = MagicMock(side_effect=RuntimeError("DB error"))
        bridge = FailureBridge(fg, skill_bank)

        summary = bridge.get_failure_context_for_distillation()
        assert "No FailureGraph context available" in summary


class TestCheckSkillAgainstGraph:
    """Tests for cross-referencing proposed skills against FailureGraph."""

    def test_no_symptoms_returns_not_duplicate(self, skill_bank):
        from orchestration.repl_memory.distillation.failure_bridge import FailureBridge

        fg = MockFailureGraph()
        bridge = FailureBridge(fg, skill_bank)

        result = bridge.check_skill_against_graph("Some principle", skill_symptoms=None)
        assert result["is_duplicate"] is False

    def test_no_matching_mitigation(self, skill_bank):
        from orchestration.repl_memory.distillation.failure_bridge import FailureBridge

        fg = MockFailureGraph(effective_mitigations=[])
        bridge = FailureBridge(fg, skill_bank)

        result = bridge.check_skill_against_graph(
            "New principle", skill_symptoms=["timeout"]
        )
        assert result["is_duplicate"] is False

    def test_high_success_mitigation_is_duplicate(self, skill_bank):
        from orchestration.repl_memory.distillation.failure_bridge import FailureBridge

        fg = MockFailureGraph(effective_mitigations=[
            {"action": "Use spec decode", "success_rate": 0.9},
        ])
        bridge = FailureBridge(fg, skill_bank)

        result = bridge.check_skill_against_graph(
            "Avoid prompt lookup", skill_symptoms=["0% acceptance"]
        )
        assert result["is_duplicate"] is True
        assert result["existing_mitigation"] == "Use spec decode"
        assert result["coverage"] == 0.9

    def test_low_success_mitigation_not_duplicate(self, skill_bank):
        from orchestration.repl_memory.distillation.failure_bridge import FailureBridge

        fg = MockFailureGraph(effective_mitigations=[
            {"action": "Retry with different model", "success_rate": 0.4},
        ])
        bridge = FailureBridge(fg, skill_bank)

        result = bridge.check_skill_against_graph(
            "Retry principle", skill_symptoms=["error"]
        )
        assert result["is_duplicate"] is False

    def test_graph_error_graceful(self, skill_bank):
        from orchestration.repl_memory.distillation.failure_bridge import FailureBridge

        fg = MockFailureGraph()
        fg.get_effective_mitigations = MagicMock(side_effect=RuntimeError("DB error"))
        bridge = FailureBridge(fg, skill_bank)

        result = bridge.check_skill_against_graph(
            "Some principle", skill_symptoms=["error"]
        )
        assert result["is_duplicate"] is False


class TestSyncMitigationsToSkills:
    """Tests for syncing FailureGraph mitigations to SkillBank."""

    def _make_bridge_with_mitigations(self, skill_bank, mitigations):
        """Create bridge with mocked _get_all_mitigations."""
        from orchestration.repl_memory.distillation.failure_bridge import FailureBridge

        fg = MockFailureGraph()
        bridge = FailureBridge(fg, skill_bank)
        bridge._get_all_mitigations = lambda: mitigations
        return bridge

    def test_empty_mitigations(self, skill_bank):
        bridge = self._make_bridge_with_mitigations(skill_bank, [])
        stats = bridge.sync_mitigations_to_skills()

        assert stats["checked"] == 0
        assert stats["created"] == 0
        assert skill_bank.count() == 0

    def test_creates_skills_from_mitigations(self, skill_bank):
        mitigations = [
            {
                "failure_id": "f1",
                "failure_description": "Prompt lookup returns 0 tokens",
                "action": "Use speculative decode instead",
                "success_rate": 0.85,
                "symptoms": ["0% acceptance", "empty output"],
            },
        ]
        bridge = self._make_bridge_with_mitigations(skill_bank, mitigations)
        stats = bridge.sync_mitigations_to_skills()

        assert stats["checked"] == 1
        assert stats["created"] == 1
        assert skill_bank.count() == 1

        skills = skill_bank.get_skills(skill_type="failure_lesson")
        assert len(skills) == 1
        assert "Prompt lookup returns 0 tokens" in skills[0].title
        assert "FAILURE POINT" in skills[0].principle
        assert "PREVENTION" in skills[0].principle

    def test_skips_low_success_rate(self, skill_bank):
        mitigations = [
            {
                "failure_id": "f1",
                "failure_description": "Random failure",
                "action": "Flaky workaround",
                "success_rate": 0.3,
                "symptoms": ["random"],
            },
        ]
        bridge = self._make_bridge_with_mitigations(skill_bank, mitigations)
        stats = bridge.sync_mitigations_to_skills()

        assert stats["checked"] == 1
        assert stats["skipped_low_quality"] == 1
        assert stats["created"] == 0

    def test_skips_duplicates(self, skill_bank):
        from orchestration.repl_memory.skill_bank import Skill, SkillBank

        # Pre-populate with matching skill
        existing = Skill(
            id=SkillBank.generate_id("failure_lesson"),
            title="Avoid: Prompt lookup returns 0 tokens",
            skill_type="failure_lesson",
            principle="Already known.",
            when_to_apply="always",
            task_types=["*"],
            source_trajectory_ids=["t1"],
            source_outcome="failure",
            confidence=0.8,
        )
        skill_bank.store(existing)

        mitigations = [
            {
                "failure_id": "f1",
                "failure_description": "Prompt lookup returns 0 tokens",
                "action": "Use spec decode",
                "success_rate": 0.9,
                "symptoms": ["0% acceptance"],
            },
        ]
        bridge = self._make_bridge_with_mitigations(skill_bank, mitigations)
        stats = bridge.sync_mitigations_to_skills()

        assert stats["skipped_existing"] == 1
        assert stats["created"] == 0
        assert skill_bank.count() == 1  # Only the pre-existing one

    def test_multiple_mitigations_mixed(self, skill_bank):
        mitigations = [
            {
                "failure_id": "f1",
                "failure_description": "Timeout on large context",
                "action": "Chunk input",
                "success_rate": 0.9,
                "symptoms": ["timeout"],
            },
            {
                "failure_id": "f2",
                "failure_description": "Model OOM",
                "action": "Reduce batch",
                "success_rate": 0.5,  # Below threshold
                "symptoms": ["oom"],
            },
            {
                "failure_id": "f3",
                "failure_description": "Wrong format output",
                "action": "Add format constraint",
                "success_rate": 0.8,
                "symptoms": ["format_error"],
            },
        ]
        bridge = self._make_bridge_with_mitigations(skill_bank, mitigations)
        stats = bridge.sync_mitigations_to_skills()

        assert stats["checked"] == 3
        assert stats["created"] == 2
        assert stats["skipped_low_quality"] == 1

    def test_confidence_capped_at_09(self, skill_bank):
        mitigations = [
            {
                "failure_id": "f1",
                "failure_description": "Known issue",
                "action": "Fix it",
                "success_rate": 0.99,
                "symptoms": ["bug"],
            },
        ]
        bridge = self._make_bridge_with_mitigations(skill_bank, mitigations)
        bridge.sync_mitigations_to_skills()

        skills = skill_bank.get_skills(skill_type="failure_lesson")
        assert skills[0].confidence == 0.9  # Capped
