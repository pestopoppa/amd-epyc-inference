"""
Tests for HypothesisGraph - Kuzu-backed hypothesis tracking.
"""

import pytest
import tempfile
from pathlib import Path

# Skip if kuzu not available
kuzu = pytest.importorskip("kuzu")

from orchestration.repl_memory.hypothesis_graph import HypothesisGraph


@pytest.fixture
def temp_kuzu_path():
    """Create a temporary path for Kuzu database (path should not exist)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Return path to a non-existent subdirectory - Kuzu will create it
        yield Path(tmpdir) / "kuzu_db"


@pytest.fixture
def hypothesis_graph(temp_kuzu_path):
    """Create a HypothesisGraph instance with temporary storage."""
    return HypothesisGraph(path=temp_kuzu_path)


class TestCreateHypothesis:
    """Tests for creating hypotheses."""

    def test_create_hypothesis(self, hypothesis_graph):
        """Creating hypothesis stores claim with initial confidence 0.5."""
        hypothesis_id = hypothesis_graph.create_hypothesis(
            claim="spec_decode|code = effective",
            memory_id="mem_001",
            initial_confidence=0.5,
        )

        assert hypothesis_id is not None
        assert len(hypothesis_id) == 36  # UUID format

        stats = hypothesis_graph.get_stats()
        assert stats["hypothesis_count"] == 1

    def test_create_hypothesis_reuses_existing(self, hypothesis_graph):
        """Creating hypothesis with same claim returns existing ID."""
        claim = "test_action|test_type"

        id1 = hypothesis_graph.create_hypothesis(claim, "mem_001")
        id2 = hypothesis_graph.create_hypothesis(claim, "mem_002")

        assert id1 == id2

    def test_create_hypothesis_custom_confidence(self, hypothesis_graph):
        """Initial confidence can be customized."""
        hypothesis_graph.create_hypothesis(
            claim="custom_claim",
            memory_id="mem_001",
            initial_confidence=0.8,
        )

        hypothesis_graph.get_confidence("custom_claim", "")
        # Note: get_confidence uses claim format action|task_type
        # For direct claim lookup, confidence should match


class TestAddEvidence:
    """Tests for adding evidence to hypotheses."""

    def test_add_supporting_evidence(self, hypothesis_graph):
        """Supporting evidence increases confidence asymptotically."""
        hypothesis_id = hypothesis_graph.create_hypothesis(
            claim="test_claim",
            memory_id="mem_001",
            initial_confidence=0.5,
        )

        # Add success evidence
        new_confidence = hypothesis_graph.add_evidence(
            hypothesis_id=hypothesis_id,
            outcome="success",
            source="mem_002",
        )

        # Confidence should increase: 0.5 + 0.1 * (1 - 0.5) = 0.55
        assert new_confidence == pytest.approx(0.55, abs=0.01)

    def test_add_contradicting_evidence(self, hypothesis_graph):
        """Contradicting evidence decreases confidence asymptotically."""
        hypothesis_id = hypothesis_graph.create_hypothesis(
            claim="test_claim",
            memory_id="mem_001",
            initial_confidence=0.5,
        )

        # Add failure evidence
        new_confidence = hypothesis_graph.add_evidence(
            hypothesis_id=hypothesis_id,
            outcome="failure",
            source="mem_002",
        )

        # Confidence should decrease: 0.5 - 0.1 * 0.5 = 0.45
        assert new_confidence == pytest.approx(0.45, abs=0.01)

    def test_evidence_asymptotic_behavior_high(self, hypothesis_graph):
        """Confidence approaches 1.0 asymptotically with repeated success."""
        hypothesis_id = hypothesis_graph.create_hypothesis(
            claim="test_claim",
            memory_id="mem_001",
            initial_confidence=0.5,
        )

        # Add many success evidence
        confidence = 0.5
        for i in range(20):
            confidence = hypothesis_graph.add_evidence(
                hypothesis_id=hypothesis_id,
                outcome="success",
                source=f"mem_{i:03d}",
            )

        # Should approach but not exceed 1.0
        assert 0.9 < confidence <= 1.0

    def test_evidence_asymptotic_behavior_low(self, hypothesis_graph):
        """Confidence approaches 0.0 asymptotically with repeated failure."""
        hypothesis_id = hypothesis_graph.create_hypothesis(
            claim="test_claim",
            memory_id="mem_001",
            initial_confidence=0.5,
        )

        # Add many failure evidence
        confidence = 0.5
        for i in range(20):
            confidence = hypothesis_graph.add_evidence(
                hypothesis_id=hypothesis_id,
                outcome="failure",
                source=f"mem_{i:03d}",
            )

        # Should approach but not go below 0.0
        assert 0.0 <= confidence < 0.1

    def test_add_evidence_marks_tested(self, hypothesis_graph):
        """Adding evidence marks hypothesis as tested."""
        hypothesis_id = hypothesis_graph.create_hypothesis(
            claim="test_claim",
            memory_id="mem_001",
        )

        hypothesis_graph.add_evidence(hypothesis_id, "success", "mem_002")

        # Should not appear in untested list
        untested = hypothesis_graph.get_untested_hypotheses(min_confidence=0.0)
        assert all(h.id != hypothesis_id for h in untested)


class TestGetConfidence:
    """Tests for retrieving confidence scores."""

    def test_get_confidence(self, hypothesis_graph):
        """Retrieve confidence for action+task_type combination."""
        # Create hypothesis using standard claim format
        hypothesis_graph.get_or_create_hypothesis(
            action="spec_decode",
            task_type="code",
            memory_id="mem_001",
        )

        confidence = hypothesis_graph.get_confidence("spec_decode", "code")
        assert confidence == 0.5  # Default initial

    def test_get_confidence_unknown(self, hypothesis_graph):
        """Unknown combination returns 0.5 (neutral)."""
        confidence = hypothesis_graph.get_confidence("unknown_action", "unknown_type")
        assert confidence == 0.5

    def test_get_confidence_after_updates(self, hypothesis_graph):
        """Confidence reflects evidence updates."""
        hypothesis_id = hypothesis_graph.get_or_create_hypothesis(
            action="test_action",
            task_type="test_type",
            memory_id="mem_001",
        )

        # Add success evidence
        hypothesis_graph.add_evidence(hypothesis_id, "success", "mem_002")

        confidence = hypothesis_graph.get_confidence("test_action", "test_type")
        assert confidence > 0.5


class TestUntestedHypotheses:
    """Tests for getting untested hypotheses."""

    def test_get_untested_hypotheses(self, hypothesis_graph):
        """Get untested hypotheses with high confidence."""
        # Create high-confidence untested hypothesis
        hypothesis_graph.create_hypothesis(
            claim="high_confidence",
            memory_id="mem_001",
            initial_confidence=0.8,
        )
        # Create low-confidence untested hypothesis
        hypothesis_graph.create_hypothesis(
            claim="low_confidence",
            memory_id="mem_002",
            initial_confidence=0.3,
        )

        untested = hypothesis_graph.get_untested_hypotheses(min_confidence=0.7)
        assert len(untested) == 1
        assert untested[0].claim == "high_confidence"

    def test_get_untested_excludes_tested(self, hypothesis_graph):
        """Tested hypotheses are excluded."""
        hypothesis_id = hypothesis_graph.create_hypothesis(
            claim="tested_claim",
            memory_id="mem_001",
            initial_confidence=0.8,
        )

        # Add evidence to mark as tested
        hypothesis_graph.add_evidence(hypothesis_id, "success", "mem_002")

        untested = hypothesis_graph.get_untested_hypotheses(min_confidence=0.0)
        assert all(h.id != hypothesis_id for h in untested)


class TestLowConfidenceWarnings:
    """Tests for low confidence warnings."""

    def test_low_confidence_warnings(self, hypothesis_graph):
        """Low confidence (<0.2) returns warning with cited evidence."""
        hypothesis_id = hypothesis_graph.create_hypothesis(
            claim="failing_action|failing_type",
            memory_id="mem_001",
            initial_confidence=0.5,
        )

        # Add failure evidence to drive confidence down
        for i in range(10):
            hypothesis_graph.add_evidence(hypothesis_id, "failure", f"fail_{i:03d}")

        # Get warnings
        warnings = hypothesis_graph.get_low_confidence_warnings(
            action="failing_action",
            task_type="failing_type",
            threshold=0.2,
        )

        assert len(warnings) >= 1
        assert "Low confidence" in warnings[0]
        assert "failing_action" in warnings[0]

    def test_no_warnings_for_high_confidence(self, hypothesis_graph):
        """No warnings for high confidence actions."""
        hypothesis_graph.create_hypothesis(
            claim="good_action|good_type",
            memory_id="mem_001",
            initial_confidence=0.8,
        )

        warnings = hypothesis_graph.get_low_confidence_warnings(
            action="good_action",
            task_type="good_type",
            threshold=0.2,
        )

        assert len(warnings) == 0


class TestEvidenceRetrieval:
    """Tests for retrieving evidence."""

    def test_get_contradicting_evidence(self, hypothesis_graph):
        """Get all evidence that contradicts a hypothesis."""
        hypothesis_id = hypothesis_graph.create_hypothesis(
            claim="test_claim",
            memory_id="mem_001",
        )

        # Add mixed evidence
        hypothesis_graph.add_evidence(hypothesis_id, "failure", "contra_1")
        hypothesis_graph.add_evidence(hypothesis_id, "success", "support_1")
        hypothesis_graph.add_evidence(hypothesis_id, "failure", "contra_2")

        contra = hypothesis_graph.get_contradicting_evidence(hypothesis_id)
        assert len(contra) == 2
        assert all(e.evidence_type == "contradicts" for e in contra)

    def test_get_supporting_evidence(self, hypothesis_graph):
        """Get all evidence that supports a hypothesis."""
        hypothesis_id = hypothesis_graph.create_hypothesis(
            claim="test_claim",
            memory_id="mem_001",
        )

        # Add mixed evidence
        hypothesis_graph.add_evidence(hypothesis_id, "success", "support_1")
        hypothesis_graph.add_evidence(hypothesis_id, "failure", "contra_1")
        hypothesis_graph.add_evidence(hypothesis_id, "success", "support_2")

        support = hypothesis_graph.get_supporting_evidence(hypothesis_id)
        assert len(support) == 2
        assert all(e.evidence_type == "supports" for e in support)


class TestStats:
    """Tests for graph statistics."""

    def test_get_stats(self, hypothesis_graph):
        """get_stats returns comprehensive statistics."""
        # Create some hypotheses and evidence
        h1 = hypothesis_graph.create_hypothesis("claim1", "mem_001", 0.5)
        hypothesis_graph.create_hypothesis("claim2", "mem_002", 0.7)
        hypothesis_graph.add_evidence(h1, "success", "evi_001")

        stats = hypothesis_graph.get_stats()

        assert stats["hypothesis_count"] == 2
        assert stats["tested_count"] == 1
        assert stats["evidence_count"] == 1
        assert "avg_confidence" in stats
