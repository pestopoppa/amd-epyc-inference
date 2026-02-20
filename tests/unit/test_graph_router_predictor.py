"""Tests for GraphRouterPredictor inference wrapper."""

from __future__ import annotations


import numpy as np
import pytest

from orchestration.repl_memory.graph_router_predictor import GraphRouterPredictor


class MockGraph:
    """Mock BipartiteRoutingGraph."""

    def __init__(self, n_clusters=10, n_llm=4):
        self.n_clusters = n_clusters
        self.n_llm = n_llm
        self._rng = np.random.default_rng(42)
        self._qc_ids = [f"qc_{i}" for i in range(n_clusters)]
        self._llm_ids = [f"llm_{i}" for i in range(n_llm)]

    def get_stats(self):
        return {
            "tasktype_count": 3,
            "querycluster_count": self.n_clusters,
            "llmrole_count": self.n_llm,
            "performance_edge_count": self.n_clusters * self.n_llm,
        }

    def get_node_features(self):
        return {
            "task_type": self._rng.normal(size=(3, 1024)).astype(np.float64),
            "query_cluster": self._rng.normal(size=(self.n_clusters, 1024)).astype(np.float64),
            "llm_role": self._rng.normal(size=(self.n_llm, 1024)).astype(np.float64),
            "task_type_ids": ["code", "chat", "architecture"],
            "query_cluster_ids": self._qc_ids,
            "llm_role_ids": self._llm_ids,
        }

    def get_edge_index(self):
        return {
            "belongs_to": np.array(
                [[i for i in range(self.n_clusters)], [i % 3 for i in range(self.n_clusters)]],
                dtype=np.int64,
            ),
            "performance_on": np.array(
                [
                    [i % self.n_llm for i in range(self.n_clusters)],
                    list(range(self.n_clusters)),
                ],
                dtype=np.int64,
            ),
        }

    def get_query_cluster_for_embedding(self, embed, task_type):
        # Return first cluster of matching type
        return self._qc_ids[0]


class MockGAT:
    """Mock LightweightGAT."""

    def __init__(self):
        self._weights = {"l1_W_task_type": np.zeros(1)}

    def forward(self, node_features, edge_index, **kwargs):
        return {
            nt: np.random.default_rng(42).normal(size=(n, 32)).astype(np.float32)
            for nt, n in [
                ("task_type", node_features["task_type"].shape[0]),
                ("query_cluster", node_features["query_cluster"].shape[0]),
                ("llm_role", node_features["llm_role"].shape[0]),
            ]
        }

    def predict_edges(self, q_emb, l_emb):
        scores = q_emb @ l_emb.T
        return 1.0 / (1.0 + np.exp(-scores))


class MockEmbedder:
    def embed_text(self, text):
        return np.random.default_rng(hash(text) % (2**31)).normal(size=1024).astype(np.float64)

    def embed_task_ir(self, task_ir):
        return self.embed_text(str(task_ir))


class TestGraphRouterPredictor:
    """Test GraphRouterPredictor."""

    @pytest.fixture
    def predictor(self):
        graph = MockGraph()
        gat = MockGAT()
        embedder = MockEmbedder()
        return GraphRouterPredictor(graph, gat, embedder, cache_ttl=60)

    def test_is_ready(self, predictor):
        assert predictor.is_ready

    def test_is_ready_empty_graph(self):
        graph = MockGraph(n_clusters=0, n_llm=0)
        gat = MockGAT()
        predictor = GraphRouterPredictor(graph, gat, MockEmbedder())
        assert not predictor.is_ready

    def test_predict_returns_scores(self, predictor):
        emb = np.random.default_rng(42).normal(size=1024)
        scores = predictor.predict(emb, "code")
        assert isinstance(scores, dict)
        assert len(scores) == 4  # n_llm
        for role, score in scores.items():
            assert 0 <= score <= 1

    def test_predict_sums_to_one(self, predictor):
        emb = np.random.default_rng(42).normal(size=1024)
        scores = predictor.predict(emb, "code")
        total = sum(scores.values())
        np.testing.assert_almost_equal(total, 1.0, decimal=3)

    def test_predict_caching(self, predictor):
        emb = np.random.default_rng(42).normal(size=1024)
        scores1 = predictor.predict(emb, "code")
        scores2 = predictor.predict(emb, "code")
        # Should get same results from cache
        for k in scores1:
            np.testing.assert_almost_equal(scores1[k], scores2[k])

    def test_invalidate_cache(self, predictor):
        emb = np.random.default_rng(42).normal(size=1024)
        predictor.predict(emb, "code")
        assert predictor._cache_valid
        predictor.invalidate_cache()
        assert not predictor._cache_valid

    def test_predict_no_cluster(self):
        graph = MockGraph()
        graph.get_query_cluster_for_embedding = lambda e, t: None
        gat = MockGAT()
        predictor = GraphRouterPredictor(graph, gat, MockEmbedder())
        emb = np.random.default_rng(42).normal(size=1024)
        scores = predictor.predict(emb, "unknown_type")
        assert scores == {}
