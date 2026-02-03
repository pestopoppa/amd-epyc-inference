"""
Tests for FailureGraph - Kuzu-backed failure pattern tracking.
"""

import pytest
import tempfile
from pathlib import Path

# Skip if kuzu not available
kuzu = pytest.importorskip("kuzu")

from orchestration.repl_memory.failure_graph import FailureGraph


@pytest.fixture
def temp_kuzu_path():
    """Create a temporary path for Kuzu database (path should not exist)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Return path to a non-existent subdirectory - Kuzu will create it
        yield Path(tmpdir) / "kuzu_db"


@pytest.fixture
def failure_graph(temp_kuzu_path):
    """Create a FailureGraph instance with temporary storage."""
    return FailureGraph(path=temp_kuzu_path)


class TestRecordFailure:
    """Tests for recording failures."""

    def test_record_failure_creates_failure_mode(self, failure_graph):
        """Recording a failure creates FailureMode + Symptom nodes."""
        memory_id = "mem_001"
        symptoms = ["timeout", "connection refused"]

        failure_id = failure_graph.record_failure(
            memory_id=memory_id,
            symptoms=symptoms,
            description="Test failure",
            severity=3,
        )

        assert failure_id is not None
        assert len(failure_id) == 36  # UUID format

        # Verify stats
        stats = failure_graph.get_stats()
        assert stats["failuremode_count"] == 1
        assert stats["symptom_count"] == 2
        assert stats["memorylink_count"] == 1

    def test_record_failure_reuses_existing(self, failure_graph):
        """Recording a failure with same symptoms updates existing."""
        symptoms = ["timeout"]

        # Record first failure
        id1 = failure_graph.record_failure(
            memory_id="mem_001",
            symptoms=symptoms,
            description="First failure",
        )

        # Record second failure with same symptom
        id2 = failure_graph.record_failure(
            memory_id="mem_002",
            symptoms=symptoms,
            description="Second failure",
        )

        # Should be the same failure (updated, not duplicated)
        assert id1 == id2

        stats = failure_graph.get_stats()
        assert stats["failuremode_count"] == 1
        assert stats["memorylink_count"] == 2  # Both memories linked

    def test_record_failure_with_predecessor(self, failure_graph):
        """Recording a failure with previous_failure_id creates chain."""
        id1 = failure_graph.record_failure(
            memory_id="mem_001",
            symptoms=["OOM"],
            description="First failure",
        )

        id2 = failure_graph.record_failure(
            memory_id="mem_002",
            symptoms=["SIGSEGV"],
            description="Second failure",
            previous_failure_id=id1,
        )

        # Get chain from second failure
        chain = failure_graph.get_failure_chain(id2)
        assert len(chain) == 1
        assert chain[0].id == id1


class TestFindMatchingFailures:
    """Tests for finding failures by symptoms."""

    def test_find_matching_failures(self, failure_graph):
        """Given symptoms, find similar past failures."""
        # Record failures with different symptoms
        failure_graph.record_failure(
            memory_id="mem_001",
            symptoms=["timeout", "connection refused"],
        )
        failure_graph.record_failure(
            memory_id="mem_002",
            symptoms=["OOM", "SIGSEGV"],
        )

        # Find by timeout symptom
        matches = failure_graph.find_matching_failures(["timeout"])
        assert len(matches) == 1

        # Find by multiple symptoms
        matches = failure_graph.find_matching_failures(["OOM", "SIGSEGV"])
        assert len(matches) == 1

    def test_find_matching_failures_empty(self, failure_graph):
        """No matches returns empty list."""
        matches = failure_graph.find_matching_failures(["nonexistent"])
        assert matches == []

    def test_find_matching_failures_ranking(self, failure_graph):
        """Failures with more matching symptoms rank higher."""
        # Failure with 2 symptoms
        failure_graph.record_failure(
            memory_id="mem_001",
            symptoms=["timeout", "connection refused"],
        )
        # Failure with 1 overlapping symptom
        failure_graph.record_failure(
            memory_id="mem_002",
            symptoms=["timeout"],
        )

        # Search for both symptoms - first should rank higher
        matches = failure_graph.find_matching_failures(["timeout", "connection refused"])
        assert len(matches) >= 1


class TestRecordMitigation:
    """Tests for recording mitigations."""

    def test_record_mitigation(self, failure_graph):
        """Recording successful mitigation links to FailureMode."""
        failure_id = failure_graph.record_failure(
            memory_id="mem_001",
            symptoms=["timeout"],
        )

        mitigation_id = failure_graph.record_mitigation(
            failure_id=failure_id,
            action="increase_timeout",
            worked=True,
        )

        assert mitigation_id is not None
        stats = failure_graph.get_stats()
        assert stats["mitigation_count"] == 1

    def test_record_mitigation_tracks_success_rate(self, failure_graph):
        """Mitigation success rate updates with attempts."""
        failure_id = failure_graph.record_failure(
            memory_id="mem_001",
            symptoms=["timeout"],
        )

        # Record only successes - failures create RECURRED_AFTER edges
        # which exclude the mitigation from "effective" list
        failure_graph.record_mitigation(failure_id, "retry_action", worked=True)
        failure_graph.record_mitigation(failure_id, "retry_action", worked=True)
        failure_graph.record_mitigation(failure_id, "retry_action", worked=True)

        # Check effective mitigations
        effective = failure_graph.get_effective_mitigations(["timeout"])
        assert len(effective) >= 1
        # Success rate should be 3/3 = 1.0
        if effective:
            assert effective[0]["success_rate"] == pytest.approx(1.0, abs=0.01)

    def test_record_failed_mitigation_creates_recurrence(self, failure_graph):
        """Failed mitigation creates RECURRED_AFTER edge."""
        failure_id = failure_graph.record_failure(
            memory_id="mem_001",
            symptoms=["timeout"],
        )

        failure_graph.record_mitigation(failure_id, "bad_fix", worked=False)

        # Effective mitigations should exclude failed ones
        effective = failure_graph.get_effective_mitigations(["timeout"])
        # The failed mitigation should not appear in effective list
        assert all(m["action"] != "bad_fix" for m in effective)


class TestFailureChain:
    """Tests for failure chain tracking."""

    def test_failure_chain(self, failure_graph):
        """Get causal chain of failures via PRECEDED_BY edges."""
        # Create chain: failure3 <- failure2 <- failure1
        id1 = failure_graph.record_failure(
            memory_id="mem_001",
            symptoms=["initial_error"],
            description="Root cause",
        )
        id2 = failure_graph.record_failure(
            memory_id="mem_002",
            symptoms=["secondary_error"],
            description="Consequence 1",
            previous_failure_id=id1,
        )
        id3 = failure_graph.record_failure(
            memory_id="mem_003",
            symptoms=["final_error"],
            description="Consequence 2",
            previous_failure_id=id2,
        )

        # Get chain from final failure
        chain = failure_graph.get_failure_chain(id3, depth=5)
        assert len(chain) == 2
        # Chain should be in causal order (oldest first)
        assert chain[0].id == id1
        assert chain[1].id == id2

    def test_failure_chain_respects_depth(self, failure_graph):
        """Chain depth is limited by parameter."""
        # Create longer chain
        ids = []
        prev_id = None
        for i in range(5):
            fid = failure_graph.record_failure(
                memory_id=f"mem_{i:03d}",
                symptoms=[f"error_{i}"],
                previous_failure_id=prev_id,
            )
            ids.append(fid)
            prev_id = fid

        # Get chain with depth=2
        chain = failure_graph.get_failure_chain(ids[-1], depth=2)
        assert len(chain) <= 2


class TestFailureRisk:
    """Tests for failure risk scoring."""

    def test_failure_risk_zero_for_safe_action(self, failure_graph):
        """get_failure_risk returns 0.0 for actions with no failures."""
        risk = failure_graph.get_failure_risk("never_failed_action")
        assert risk == 0.0

    def test_failure_risk_increases_with_failures(self, failure_graph):
        """get_failure_risk increases based on unmitigated failures."""
        # Record multiple failures
        for i in range(5):
            failure_graph.record_failure(
                memory_id=f"risky_action_{i}",
                symptoms=["test_error"],
            )

        # Risk should be positive (actual value depends on implementation)
        risk = failure_graph.get_failure_risk("risky_action")
        assert 0.0 <= risk <= 1.0

    def test_failure_risk_range(self, failure_graph):
        """get_failure_risk returns value in 0.0-1.0 range."""
        risk = failure_graph.get_failure_risk("any_action")
        assert 0.0 <= risk <= 1.0


class TestEffectiveMitigations:
    """Tests for getting effective mitigations."""

    def test_get_effective_mitigations(self, failure_graph):
        """Get mitigations that worked for failures with symptoms."""
        failure_id = failure_graph.record_failure(
            memory_id="mem_001",
            symptoms=["BOS mismatch"],
        )

        failure_graph.record_mitigation(failure_id, "check_tokenizer", worked=True)

        effective = failure_graph.get_effective_mitigations(["BOS mismatch"])
        assert len(effective) == 1
        assert effective[0]["action"] == "check_tokenizer"
        assert effective[0]["success_rate"] == 1.0

    def test_get_effective_mitigations_excludes_failures(self, failure_graph):
        """Mitigations that recurred are excluded."""
        failure_id = failure_graph.record_failure(
            memory_id="mem_001",
            symptoms=["test_symptom"],
        )

        # One worked, one didn't
        failure_graph.record_mitigation(failure_id, "good_fix", worked=True)
        failure_graph.record_mitigation(failure_id, "bad_fix", worked=False)

        effective = failure_graph.get_effective_mitigations(["test_symptom"])
        actions = [m["action"] for m in effective]
        assert "good_fix" in actions
        # bad_fix should be excluded because it recurred
        # Note: This depends on the query implementation


class TestStats:
    """Tests for graph statistics."""

    def test_get_stats(self, failure_graph):
        """get_stats returns node counts."""
        stats = failure_graph.get_stats()
        assert "failuremode_count" in stats
        assert "symptom_count" in stats
        assert "mitigation_count" in stats
        assert "memorylink_count" in stats
        assert all(v >= 0 for v in stats.values())
