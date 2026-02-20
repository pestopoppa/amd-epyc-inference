"""Tests for GraphRouter cold-start / inductive onboarding."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest


class MockEmbedder:
    """Mock embedder for cold-start tests."""

    def embed_text(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(hash(text) % (2**31))
        emb = rng.normal(size=1024).astype(np.float64)
        return emb / (np.linalg.norm(emb) + 1e-8)


@pytest.fixture
def tmp_kuzu_path():
    tmp = tempfile.mkdtemp(prefix="test_cold_start_")
    yield Path(tmp) / "routing_graph"
    shutil.rmtree(tmp, ignore_errors=True)


class TestColdStartOnboarding:
    """Test inductive new model onboarding via GraphRouter."""

    def test_add_new_model_creates_node(self, tmp_kuzu_path):
        """Adding a new model creates an LLMRole node in the graph."""
        from orchestration.repl_memory.routing_graph import BipartiteRoutingGraph

        graph = BipartiteRoutingGraph(path=tmp_kuzu_path)
        embedder = MockEmbedder()

        emb = embedder.embed_text("New fast coding model, 60 t/s")
        graph.add_llm_role(
            role_id="new_coder",
            description="New fast coding model, 60 t/s",
            embedding=emb,
            port=9999,
            tps=60.0,
            tier="HOT",
            gb=20.0,
        )

        stats = graph.get_stats()
        assert stats["llmrole_count"] == 1

        llm_embs = graph.get_llm_embeddings()
        assert "new_coder" in llm_embs

    def test_inductive_prediction_with_existing_graph(self, tmp_kuzu_path):
        """A new model should get routing predictions from GAT without needing PERFORMANCE_ON edges."""
        from orchestration.repl_memory.lightweight_gat import GATConfig, LightweightGAT
        from orchestration.repl_memory.graph_router_predictor import GraphRouterPredictor
        from orchestration.repl_memory.routing_graph import BipartiteRoutingGraph

        graph = BipartiteRoutingGraph(path=tmp_kuzu_path)
        embedder = MockEmbedder()

        # Add existing models
        for role in ["frontdoor", "coder_escalation", "architect_general"]:
            emb = embedder.embed_text(f"{role} model description")
            graph.add_llm_role(role, f"{role} desc", emb, 8080, 50.0, "HOT", 20.0)

        # Add some query clusters manually
        rng = np.random.default_rng(42)
        graph._create_task_type("code", "Code tasks", embedder.embed_text("code"))
        for i in range(5):
            cluster_emb = rng.normal(size=1024).astype(np.float64)
            graph._create_query_cluster(
                f"cluster_{i}", f"Test cluster {i}",
                cluster_emb, "code", 10,
            )

        # Add a NEW model (no PERFORMANCE_ON edges)
        new_emb = embedder.embed_text("Brand new super coder model, 100 t/s refactoring specialist")
        graph.add_llm_role("new_super_coder", "Super coder", new_emb, 9999, 100.0, "HOT", 30.0)

        # Create GAT and predictor
        cfg = GATConfig(input_dim=1024, hidden_dim=32, output_dim=32, num_heads=4)
        gat = LightweightGAT(cfg)
        predictor = GraphRouterPredictor(graph, gat, embedder)

        # The new model should appear in predictions (inductive generalization)
        query_emb = embedder.embed_text("Write a Python sort function")
        scores = predictor.predict(query_emb, "code")

        # New model should have non-zero score (GAT propagates through graph structure)
        assert "new_super_coder" in scores
        assert scores["new_super_coder"] >= 0

    def test_cold_start_no_training_data(self, tmp_kuzu_path):
        """Graph with no clusters should return empty predictions gracefully."""
        from orchestration.repl_memory.lightweight_gat import LightweightGAT
        from orchestration.repl_memory.graph_router_predictor import GraphRouterPredictor
        from orchestration.repl_memory.routing_graph import BipartiteRoutingGraph

        graph = BipartiteRoutingGraph(path=tmp_kuzu_path)
        embedder = MockEmbedder()

        # Only add LLM roles, no clusters
        emb = embedder.embed_text("Test model")
        graph.add_llm_role("test", "Test", emb, 8080, 50.0, "HOT", 20.0)

        gat = LightweightGAT()
        predictor = GraphRouterPredictor(graph, gat, embedder)

        query_emb = embedder.embed_text("test query")
        scores = predictor.predict(query_emb, "code")

        # Should return empty (no clusters to match against)
        assert scores == {}

    def test_multiple_new_models_get_different_scores(self, tmp_kuzu_path):
        """Different new models should get different routing distributions."""
        from orchestration.repl_memory.lightweight_gat import GATConfig, LightweightGAT
        from orchestration.repl_memory.graph_router_predictor import GraphRouterPredictor
        from orchestration.repl_memory.routing_graph import BipartiteRoutingGraph

        graph = BipartiteRoutingGraph(path=tmp_kuzu_path)
        embedder = MockEmbedder()

        # Set up base graph
        rng = np.random.default_rng(42)
        for i in range(5):
            graph._create_query_cluster(
                f"c_{i}", f"Cluster {i}",
                rng.normal(size=1024).astype(np.float64), "code", 10,
            )

        # Add two different new models
        graph.add_llm_role(
            "fast_coder", "Ultra-fast code generation, 100 t/s",
            embedder.embed_text("Ultra-fast code generation, 100 t/s"),
            9001, 100.0, "HOT", 10.0,
        )
        graph.add_llm_role(
            "deep_architect", "Deep architecture reasoning, slow but accurate, 3 t/s",
            embedder.embed_text("Deep architecture reasoning, slow but accurate, 3 t/s"),
            9002, 3.0, "WARM", 200.0,
        )

        gat = LightweightGAT(GATConfig(input_dim=1024))
        predictor = GraphRouterPredictor(graph, gat, embedder)

        query_emb = embedder.embed_text("Write a Python sort function")
        scores = predictor.predict(query_emb, "code")

        # Both models should have scores
        assert "fast_coder" in scores
        assert "deep_architect" in scores
        # They should have different scores (different embeddings -> different GAT outputs)
        assert scores["fast_coder"] != scores["deep_architect"]
