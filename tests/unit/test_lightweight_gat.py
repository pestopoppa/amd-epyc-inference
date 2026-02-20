"""Tests for LightweightGAT pure numpy implementation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from orchestration.repl_memory.lightweight_gat import (
    GATConfig,
    LightweightGAT,
    _elu,
    _leaky_relu,
    _softmax_by_group,
)


class TestActivations:
    """Test activation function implementations."""

    def test_elu_positive(self):
        x = np.array([1.0, 2.0, 3.0])
        result = _elu(x)
        np.testing.assert_array_equal(result, x)

    def test_elu_negative(self):
        x = np.array([-1.0])
        result = _elu(x)
        assert result[0] < 0
        assert result[0] > -1.0  # ELU is bounded above -alpha

    def test_leaky_relu_positive(self):
        x = np.array([1.0, 2.0])
        result = _leaky_relu(x, 0.2)
        np.testing.assert_array_equal(result, x)

    def test_leaky_relu_negative(self):
        x = np.array([-1.0])
        result = _leaky_relu(x, 0.2)
        np.testing.assert_almost_equal(result[0], -0.2)

    def test_softmax_by_group(self):
        values = np.array([1.0, 2.0, 3.0, 1.0])
        groups = np.array([0, 0, 1, 1])
        result = _softmax_by_group(values, groups, 2)
        # Within group 0: softmax of [1, 2]
        assert result.shape == (4,)
        # Sum within each group should be ~1
        np.testing.assert_almost_equal(result[0] + result[1], 1.0, decimal=5)
        np.testing.assert_almost_equal(result[2] + result[3], 1.0, decimal=5)

    def test_softmax_empty(self):
        result = _softmax_by_group(np.array([]), np.array([]), 0)
        assert len(result) == 0


class TestLightweightGATInit:
    """Test GAT initialization."""

    def test_default_config(self):
        gat = LightweightGAT()
        assert gat.config.input_dim == 1024
        assert gat.config.hidden_dim == 32
        assert gat.config.num_heads == 4

    def test_custom_config(self):
        cfg = GATConfig(input_dim=64, hidden_dim=16, num_heads=2)
        gat = LightweightGAT(cfg)
        assert gat.config.input_dim == 64

    def test_weights_initialized(self):
        gat = LightweightGAT()
        assert len(gat._weights) > 0
        # Check per-node-type layer 1 weights exist
        assert "l1_W_task_type" in gat._weights
        assert "l1_W_query_cluster" in gat._weights
        assert "l1_W_llm_role" in gat._weights

    def test_param_count(self):
        gat = LightweightGAT()
        assert gat.param_count > 0


class TestLightweightGATForward:
    """Test GAT forward pass."""

    @pytest.fixture
    def small_gat(self):
        """Create a small GAT for testing."""
        cfg = GATConfig(input_dim=16, hidden_dim=4, output_dim=4, num_heads=2)
        return LightweightGAT(cfg)

    @pytest.fixture
    def small_graph(self):
        """Create a small test graph."""
        rng = np.random.default_rng(42)
        node_features = {
            "task_type": rng.normal(size=(3, 16)).astype(np.float32),
            "query_cluster": rng.normal(size=(10, 16)).astype(np.float32),
            "llm_role": rng.normal(size=(4, 16)).astype(np.float32),
        }
        edge_index = {
            "belongs_to": np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                     [0, 0, 0, 1, 1, 1, 2, 2, 2, 0]], dtype=np.int64),
            "performance_on": np.array([[0, 1, 2, 3, 0, 1],
                                         [0, 2, 5, 7, 3, 8]], dtype=np.int64),
        }
        return node_features, edge_index

    def test_forward_shapes(self, small_gat, small_graph):
        node_features, edge_index = small_graph
        out = small_gat.forward(node_features, edge_index)

        assert "task_type" in out
        assert "query_cluster" in out
        assert "llm_role" in out
        assert out["task_type"].shape == (3, 4)  # output_dim
        assert out["query_cluster"].shape == (10, 4)
        assert out["llm_role"].shape == (4, 4)

    def test_forward_no_nan(self, small_gat, small_graph):
        node_features, edge_index = small_graph
        out = small_gat.forward(node_features, edge_index)
        for key, val in out.items():
            assert not np.any(np.isnan(val)), f"NaN in {key}"
            assert not np.any(np.isinf(val)), f"Inf in {key}"

    def test_forward_empty_nodes(self, small_gat):
        node_features = {
            "task_type": np.zeros((0, 16), dtype=np.float32),
            "query_cluster": np.zeros((0, 16), dtype=np.float32),
            "llm_role": np.zeros((0, 16), dtype=np.float32),
        }
        edge_index = {
            "belongs_to": np.zeros((2, 0), dtype=np.int64),
            "performance_on": np.zeros((2, 0), dtype=np.int64),
        }
        out = small_gat.forward(node_features, edge_index)
        assert out["task_type"].shape[0] == 0

    def test_forward_training_mode(self, small_gat, small_graph):
        """Training mode should not crash (dropout applied)."""
        node_features, edge_index = small_graph
        out = small_gat.forward(node_features, edge_index, training=True)
        assert out["query_cluster"].shape == (10, 4)


class TestLightweightGATEdgePrediction:
    """Test edge prediction."""

    def test_predict_edges_shape(self):
        cfg = GATConfig(input_dim=16, hidden_dim=4, output_dim=4, num_heads=2)
        gat = LightweightGAT(cfg)
        q_emb = np.random.default_rng(42).normal(size=(5, 4)).astype(np.float32)
        l_emb = np.random.default_rng(43).normal(size=(3, 4)).astype(np.float32)
        scores = gat.predict_edges(q_emb, l_emb)
        assert scores.shape == (5, 3)

    def test_predict_edges_range(self):
        cfg = GATConfig(input_dim=16, hidden_dim=4, output_dim=4, num_heads=2)
        gat = LightweightGAT(cfg)
        q_emb = np.random.default_rng(42).normal(size=(5, 4)).astype(np.float32)
        l_emb = np.random.default_rng(43).normal(size=(3, 4)).astype(np.float32)
        scores = gat.predict_edges(q_emb, l_emb)
        assert np.all(scores >= 0) and np.all(scores <= 1)

    def test_predict_edges_empty(self):
        gat = LightweightGAT(GATConfig(input_dim=16, hidden_dim=4, output_dim=4, num_heads=2))
        scores = gat.predict_edges(np.zeros((0, 4)), np.zeros((3, 4)))
        assert scores.shape == (0, 3)


class TestLightweightGATLoss:
    """Test loss computation."""

    def test_bce_loss_perfect(self):
        gat = LightweightGAT()
        preds = np.array([[0.99, 0.01], [0.01, 0.99]])
        targets = np.array([[1.0, 0.0], [0.0, 1.0]])
        loss = gat.compute_loss(preds, targets)
        assert loss < 0.1

    def test_bce_loss_bad(self):
        gat = LightweightGAT()
        preds = np.array([[0.01, 0.99], [0.99, 0.01]])
        targets = np.array([[1.0, 0.0], [0.0, 1.0]])
        loss = gat.compute_loss(preds, targets)
        assert loss > 1.0

    def test_bce_loss_masked(self):
        gat = LightweightGAT()
        preds = np.array([[0.5, 0.5], [0.5, 0.5]])
        targets = np.array([[1.0, 0.0], [0.0, 1.0]])
        mask = np.array([[1.0, 0.0], [0.0, 1.0]])
        loss = gat.compute_loss(preds, targets, mask=mask)
        assert loss > 0


class TestLightweightGATSaveLoad:
    """Test weight persistence."""

    def test_save_load_roundtrip(self):
        cfg = GATConfig(input_dim=16, hidden_dim=4, output_dim=4, num_heads=2)
        gat1 = LightweightGAT(cfg)
        original_weight = gat1._weights["l1_W_task_type"].copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weights.npz"
            gat1.save(path)

            gat2 = LightweightGAT(cfg)
            gat2.load(path)

            np.testing.assert_array_equal(
                gat2._weights["l1_W_task_type"],
                original_weight,
            )

    def test_load_nonexistent(self):
        gat = LightweightGAT()
        with pytest.raises(FileNotFoundError):
            gat.load(Path("/nonexistent/weights.npz"))
