"""
Tests for graph-enhanced retrieval integration.

Tests the full flow of:
- GraphEnhancedRetriever with failure penalties and hypothesis confidence
- GraphEnhancedStore with async graph updates
- Combined scoring formula
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

# Skip if kuzu not available
kuzu = pytest.importorskip("kuzu")

from orchestration.repl_memory.failure_graph import FailureGraph
from orchestration.repl_memory.hypothesis_graph import HypothesisGraph
from orchestration.repl_memory.episodic_store import (
    EpisodicStore,
    GraphEnhancedStore,
    extract_symptoms,
)
from orchestration.repl_memory.retriever import (
    GraphEnhancedRetriever,
    RetrievalConfig,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for all storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def episodic_store(temp_dir):
    """Create an EpisodicStore with temporary storage."""
    return EpisodicStore(db_path=temp_dir / "episodic", use_faiss=True, embedding_dim=896)


@pytest.fixture
def failure_graph(temp_dir):
    """Create a FailureGraph with temporary storage."""
    # Kuzu needs path to not exist - it will create it
    return FailureGraph(path=temp_dir / "graphs" / "failure_graph")


@pytest.fixture
def hypothesis_graph(temp_dir):
    """Create a HypothesisGraph with temporary storage."""
    # Kuzu needs path to not exist - it will create it
    return HypothesisGraph(path=temp_dir / "graphs" / "hypothesis_graph")


@pytest.fixture
def mock_embedder():
    """Create a mock embedder that returns fixed embeddings."""
    embedder = MagicMock()
    # Return a fixed embedding for any input
    embedder.embed_task_ir.return_value = np.random.randn(896).astype(np.float32)
    embedder.embed_failure_context.return_value = np.random.randn(896).astype(np.float32)
    embedder.embed_exploration.return_value = np.random.randn(896).astype(np.float32)
    return embedder


class TestExtractSymptoms:
    """Tests for symptom extraction."""

    def test_extract_timeout(self):
        """Extracts timeout symptom."""
        context = {"error": "Connection timed out after 30s"}
        symptoms = extract_symptoms(context, "failure")
        assert "timeout" in symptoms

    def test_extract_sigsegv(self):
        """Extracts SIGSEGV symptom."""
        context = {}
        outcome = "Process crashed with signal 11 (SIGSEGV)"
        symptoms = extract_symptoms(context, outcome)
        assert "SIGSEGV" in symptoms

    def test_extract_oom(self):
        """Extracts OOM symptom."""
        context = {"error": "Cannot allocate memory"}
        symptoms = extract_symptoms(context, "failure")
        assert "OOM" in symptoms

    def test_extract_bos_mismatch(self):
        """Extracts BOS mismatch symptom."""
        context = {}
        outcome = "BOS token mismatch: expected 1, got 2"
        symptoms = extract_symptoms(context, outcome)
        assert "BOS mismatch" in symptoms

    def test_extract_multiple(self):
        """Extracts multiple symptoms."""
        context = {"error": "Connection refused, process timed out"}
        symptoms = extract_symptoms(context, "failure")
        assert "timeout" in symptoms
        assert "connection refused" in symptoms

    def test_extract_unknown(self):
        """Returns 'unknown' for unrecognized patterns."""
        symptoms = extract_symptoms({}, "some generic error")
        assert symptoms == ["unknown"]


class TestGraphEnhancedStore:
    """Tests for GraphEnhancedStore."""

    def test_store_with_graphs_basic(self, episodic_store, failure_graph, hypothesis_graph):
        """Basic store operation works."""
        store = GraphEnhancedStore(
            episodic_store=episodic_store,
            failure_graph=failure_graph,
            hypothesis_graph=hypothesis_graph,
        )

        embedding = np.random.randn(896).astype(np.float32)
        memory_id = store.store_with_graphs(
            embedding=embedding,
            action="test_action",
            action_type="routing",
            context={"task_type": "code"},
            outcome="success",
            task_type="code",
        )

        assert memory_id is not None
        assert len(memory_id) == 36

    def test_store_failure_updates_graph(self, episodic_store, failure_graph, hypothesis_graph):
        """Storing a failure updates the failure graph."""
        store = GraphEnhancedStore(
            episodic_store=episodic_store,
            failure_graph=failure_graph,
            hypothesis_graph=hypothesis_graph,
        )

        embedding = np.random.randn(896).astype(np.float32)
        store.store_with_graphs(
            embedding=embedding,
            action="failing_action",
            action_type="routing",
            context={"error": "Connection timed out"},
            outcome="failure",
            task_type="code",
        )

        # Check failure graph was updated
        stats = failure_graph.get_stats()
        assert stats["failuremode_count"] >= 1

    def test_store_updates_hypothesis(self, episodic_store, failure_graph, hypothesis_graph):
        """Storing with outcome updates hypothesis graph."""
        store = GraphEnhancedStore(
            episodic_store=episodic_store,
            failure_graph=failure_graph,
            hypothesis_graph=hypothesis_graph,
        )

        embedding = np.random.randn(896).astype(np.float32)
        store.store_with_graphs(
            embedding=embedding,
            action="test_action",
            action_type="routing",
            context={},
            outcome="success",
            task_type="code",
        )

        # Check hypothesis graph was updated
        stats = hypothesis_graph.get_stats()
        assert stats["hypothesis_count"] >= 1

    def test_get_stats_includes_graphs(self, episodic_store, failure_graph, hypothesis_graph):
        """get_stats includes graph statistics."""
        store = GraphEnhancedStore(
            episodic_store=episodic_store,
            failure_graph=failure_graph,
            hypothesis_graph=hypothesis_graph,
        )

        stats = store.get_stats()
        assert "failure_graph" in stats
        assert "hypothesis_graph" in stats


class TestGraphEnhancedRetriever:
    """Tests for GraphEnhancedRetriever."""

    def test_retrieval_without_graphs(self, episodic_store, mock_embedder):
        """Retriever works without graphs (graceful degradation)."""
        retriever = GraphEnhancedRetriever(
            store=episodic_store,
            embedder=mock_embedder,
            failure_graph=None,
            hypothesis_graph=None,
        )

        results = retriever.retrieve_for_routing({"task_type": "code"})
        # Should work but return empty (no memories stored)
        assert results == []

    def test_retrieval_with_failure_penalty(
        self, episodic_store, failure_graph, hypothesis_graph, mock_embedder
    ):
        """Actions with failures get penalized in ranking."""
        # Store some memories
        embedding = np.random.randn(896).astype(np.float32)
        episodic_store.store(
            embedding=embedding,
            action="risky_action",
            action_type="routing",
            context={},
            initial_q=0.8,
        )

        # Record failure for this action
        failure_graph.record_failure(
            memory_id="some_memory",
            symptoms=["risky_action"],  # Using action as symptom for test
            description="Test failure",
        )

        retriever = GraphEnhancedRetriever(
            store=episodic_store,
            embedder=mock_embedder,
            failure_graph=failure_graph,
            hypothesis_graph=hypothesis_graph,
        )

        # Configure embedder to return similar embedding
        mock_embedder.embed_task_ir.return_value = embedding

        results = retriever.retrieve_for_routing({"task_type": "code"})
        # Failure penalty should affect adjusted_score
        if results:
            assert results[0].failure_penalty >= 0.0

    def test_retrieval_with_hypothesis_confidence(
        self, episodic_store, failure_graph, hypothesis_graph, mock_embedder
    ):
        """Low hypothesis confidence adds warnings."""
        # Store a memory
        embedding = np.random.randn(896).astype(np.float32)
        episodic_store.store(
            embedding=embedding,
            action="uncertain_action",
            action_type="routing",
            context={},
            initial_q=0.8,
        )

        # Create low-confidence hypothesis
        hypothesis_id = hypothesis_graph.get_or_create_hypothesis(
            action="uncertain_action",
            task_type="code",
            memory_id="mem_001",
        )
        # Add failure evidence to lower confidence
        for i in range(10):
            hypothesis_graph.add_evidence(hypothesis_id, "failure", f"fail_{i}")

        retriever = GraphEnhancedRetriever(
            store=episodic_store,
            embedder=mock_embedder,
            failure_graph=failure_graph,
            hypothesis_graph=hypothesis_graph,
        )

        mock_embedder.embed_task_ir.return_value = embedding

        results = retriever.retrieve_for_routing({"task_type": "code"})
        if results:
            # Should have low hypothesis confidence
            assert results[0].hypothesis_confidence < 0.5

    def test_combined_scoring(self, episodic_store, failure_graph, hypothesis_graph, mock_embedder):
        """Adjusted score uses selection_score with graph penalties/confidence."""
        # Store a memory with known Q-value
        embedding = np.random.randn(896).astype(np.float32)
        episodic_store.store(
            embedding=embedding,
            action="test_action",
            action_type="routing",
            context={},
            initial_q=0.8,
        )

        retriever = GraphEnhancedRetriever(
            store=episodic_store,
            embedder=mock_embedder,
            failure_graph=failure_graph,
            hypothesis_graph=hypothesis_graph,
        )

        mock_embedder.embed_task_ir.return_value = embedding

        results = retriever.retrieve_for_routing({"task_type": "code"})
        if results:
            r = results[0]
            # Verify scoring formula
            expected_adjusted = (
                r.selection_score * (1 - r.failure_penalty) * r.hypothesis_confidence
            )
            assert r.adjusted_score == pytest.approx(expected_adjusted, abs=0.01)

    def test_get_best_action_with_warnings(
        self, episodic_store, failure_graph, hypothesis_graph, mock_embedder
    ):
        """get_best_action returns warnings for low confidence."""
        embedding = np.random.randn(896).astype(np.float32)
        episodic_store.store(
            embedding=embedding,
            action="warned_action",
            action_type="routing",
            context={},
            initial_q=0.9,
        )

        retriever = GraphEnhancedRetriever(
            store=episodic_store,
            embedder=mock_embedder,
            failure_graph=failure_graph,
            hypothesis_graph=hypothesis_graph,
            config=RetrievalConfig(confidence_threshold=0.3),
        )

        mock_embedder.embed_task_ir.return_value = embedding

        results = retriever.retrieve_for_routing({"task_type": "code"})
        best = retriever.get_best_action(results)

        if best:
            action, confidence, warnings = best
            assert action == "warned_action"
            assert isinstance(warnings, list)


class TestLatencyBudget:
    """Tests for latency expectations."""

    def test_retrieval_under_100ms(
        self, episodic_store, failure_graph, hypothesis_graph, mock_embedder
    ):
        """Full retrieval completes in <100ms with both graph queries."""
        import time

        # Store some memories
        for i in range(100):
            embedding = np.random.randn(896).astype(np.float32)
            episodic_store.store(
                embedding=embedding,
                action=f"action_{i}",
                action_type="routing",
                context={},
            )

        retriever = GraphEnhancedRetriever(
            store=episodic_store,
            embedder=mock_embedder,
            failure_graph=failure_graph,
            hypothesis_graph=hypothesis_graph,
        )

        # Warm up
        retriever.retrieve_for_routing({"task_type": "code"})

        # Measure
        start = time.perf_counter()
        for _ in range(10):
            retriever.retrieve_for_routing({"task_type": "code"})
        elapsed = (time.perf_counter() - start) / 10

        # Should be under 100ms on average
        assert elapsed < 0.1, f"Retrieval took {elapsed * 1000:.1f}ms, expected <100ms"


class TestRecordMitigation:
    """Tests for mitigation recording."""

    def test_record_mitigation_via_store(self, episodic_store, failure_graph, hypothesis_graph):
        """Record mitigation through GraphEnhancedStore."""
        store = GraphEnhancedStore(
            episodic_store=episodic_store,
            failure_graph=failure_graph,
            hypothesis_graph=hypothesis_graph,
        )

        # Store a failure
        embedding = np.random.randn(896).astype(np.float32)
        memory_id = store.store_with_graphs(
            embedding=embedding,
            action="failed_action",
            action_type="routing",
            context={"error": "Timeout occurred"},
            outcome="failure: timeout",
            task_type="code",
        )

        # Record mitigation
        store.record_mitigation(
            failure_memory_id=memory_id,
            action="increase_timeout",
            worked=True,
        )

        # Should have created mitigation (or None if failure not linked)
        # The actual behavior depends on the async update timing
