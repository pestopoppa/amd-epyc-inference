"""Tests for BipartiteRoutingGraph Kuzu schema and sync."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest


@dataclass
class MockMemory:
    """Mock episodic memory entry."""

    task_description: str = "test task"
    action: str = "frontdoor:direct"
    q_value: float = 0.8
    embedding: Optional[np.ndarray] = None
    context: Dict[str, Any] = field(default_factory=dict)
    update_count: int = 1

    def __post_init__(self):
        if self.embedding is None:
            self.embedding = np.random.default_rng(42).normal(size=1024).astype(np.float64)


class MockEpisodicStore:
    """Mock episodic store for testing graph sync."""

    def __init__(self, memories: List[MockMemory]):
        self._memories = memories

    def get_all_memories(self) -> List[MockMemory]:
        return self._memories

    def count(self) -> int:
        return len(self._memories)


class MockEmbedder:
    """Mock embedder that returns deterministic embeddings."""

    def embed_text(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(hash(text) % (2**31))
        emb = rng.normal(size=1024).astype(np.float64)
        return emb / (np.linalg.norm(emb) + 1e-8)


@pytest.fixture
def tmp_kuzu_path():
    """Create a temporary directory for Kuzu DB."""
    tmp = tempfile.mkdtemp(prefix="test_routing_graph_")
    yield Path(tmp) / "routing_graph"
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def graph(tmp_kuzu_path):
    """Create a BipartiteRoutingGraph for testing."""
    from orchestration.repl_memory.routing_graph import BipartiteRoutingGraph

    return BipartiteRoutingGraph(path=tmp_kuzu_path)


class TestBipartiteRoutingGraphSchema:
    """Test schema creation and basic node operations."""

    def test_schema_created(self, graph):
        stats = graph.get_stats()
        assert "tasktype_count" in stats
        assert "querycluster_count" in stats
        assert "llmrole_count" in stats

    def test_add_llm_role(self, graph):
        emb = np.random.default_rng(42).normal(size=1024)
        graph.add_llm_role(
            role_id="test_role",
            description="A test model",
            embedding=emb,
            port=8080,
            tps=50.0,
            tier="HOT",
            gb=20.0,
        )
        stats = graph.get_stats()
        assert stats["llmrole_count"] == 1

    def test_add_llm_role_upsert(self, graph):
        """Adding same role twice should upsert."""
        emb = np.random.default_rng(42).normal(size=1024)
        for _ in range(2):
            graph.add_llm_role(
                role_id="test_role",
                description="A test model",
                embedding=emb,
                port=8080,
                tps=50.0,
                tier="HOT",
                gb=20.0,
            )
        stats = graph.get_stats()
        assert stats["llmrole_count"] == 1

    def test_get_llm_embeddings(self, graph):
        emb = np.random.default_rng(42).normal(size=1024)
        graph.add_llm_role("r1", "desc", emb, 8080, 50.0, "HOT", 20.0)
        result = graph.get_llm_embeddings()
        assert "r1" in result
        np.testing.assert_allclose(result["r1"], emb, rtol=1e-5)


class TestBipartiteRoutingGraphSync:
    """Test sync_from_episodic_store."""

    def _make_memories(self, n: int = 50, task_types=None) -> List[MockMemory]:
        """Generate test memories."""
        if task_types is None:
            task_types = ["code", "chat", "architecture"]
        rng = np.random.default_rng(42)
        memories = []
        roles = ["frontdoor", "coder_escalation", "architect_general"]
        for i in range(n):
            tt = task_types[i % len(task_types)]
            role = roles[i % len(roles)]
            memories.append(MockMemory(
                task_description=f"Task {i} of type {tt}",
                action=f"{role}:direct",
                q_value=rng.uniform(0.3, 1.0),
                embedding=rng.normal(size=1024).astype(np.float64),
                context={"task_type": tt, "role": role, "elapsed_seconds": rng.uniform(1, 30)},
            ))
        return memories

    def test_sync_populates_graph(self, graph):
        memories = self._make_memories(60)
        store = MockEpisodicStore(memories)
        embedder = MockEmbedder()

        # Add LLM roles first
        for role in ["frontdoor", "coder_escalation", "architect_general"]:
            graph.add_llm_role(role, f"{role} desc", embedder.embed_text(role), 8080, 50.0, "HOT", 20.0)

        result = graph.sync_from_episodic_store(store, embedder)
        assert result["task_types"] == 3
        assert result["clusters"] > 0

    def test_sync_empty_store(self, graph):
        store = MockEpisodicStore([])
        embedder = MockEmbedder()
        result = graph.sync_from_episodic_store(store, embedder)
        assert result["task_types"] == 0
        assert result["clusters"] == 0

    def test_get_node_features(self, graph):
        memories = self._make_memories(30)
        store = MockEpisodicStore(memories)
        embedder = MockEmbedder()
        graph.sync_from_episodic_store(store, embedder)

        feats = graph.get_node_features()
        assert "task_type" in feats
        assert "query_cluster" in feats
        assert "llm_role" in feats
        assert feats["task_type"].shape[1] == 1024

    def test_get_edge_index(self, graph):
        memories = self._make_memories(30)
        store = MockEpisodicStore(memories)
        embedder = MockEmbedder()

        for role in ["frontdoor", "coder_escalation", "architect_general"]:
            graph.add_llm_role(role, f"{role} desc", embedder.embed_text(role), 8080, 50.0, "HOT", 20.0)

        graph.sync_from_episodic_store(store, embedder)
        edges = graph.get_edge_index()

        assert "belongs_to" in edges
        assert "performance_on" in edges
        assert edges["belongs_to"].shape[0] == 2

    def test_get_query_cluster_for_embedding(self, graph):
        memories = self._make_memories(30)
        store = MockEpisodicStore(memories)
        embedder = MockEmbedder()
        graph.sync_from_episodic_store(store, embedder)

        # Use one of the memory embeddings
        test_emb = memories[0].embedding
        cluster = graph.get_query_cluster_for_embedding(test_emb, "code")
        # Should find a cluster (if code type memories exist)
        assert cluster is not None or graph.get_stats()["querycluster_count"] == 0
