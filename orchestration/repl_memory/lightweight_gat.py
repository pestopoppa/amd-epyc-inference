"""LightweightGAT: Pure numpy 2-layer heterogeneous Graph Attention Network.

Implements a simplified GAT for routing prediction on small graphs (~130 nodes).
No PyG/DGL dependency — pure numpy with scatter aggregation via np.add.at.

Architecture:
    Layer 1: MultiHeadGAT(1024 -> 32, heads=4) + ELU -> 128-dim (concat)
    Layer 2: MultiHeadGAT(128 -> 32, heads=1) + ELU -> 32-dim
    Edge prediction: sigmoid(dot(query_emb, llm_emb))

Separate weight matrices per node type (TaskType, QueryCluster, LLMRole).
Shared attention weights per edge type (BELONGS_TO, PERFORMANCE_ON).

Reference: GraphRouter (ICLR 2025, arxiv 2410.03834)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GATConfig:
    """Configuration for the lightweight GAT."""

    input_dim: int = 1024  # BGE-large embedding dimension
    hidden_dim: int = 32  # Hidden dimension per head
    output_dim: int = 32  # Final output dimension
    num_heads: int = 4  # Multi-head attention (layer 1)
    dropout: float = 0.1
    leaky_relu_slope: float = 0.2
    clip_logits: float = 10.0  # Clip attention logits for numerical stability
    epsilon: float = 1e-8  # Numerical stability constant


# Node types in the heterogeneous graph
NODE_TYPES = ["task_type", "query_cluster", "llm_role"]

# Edge types: (src_type, rel_name, dst_type)
EDGE_TYPES = [
    ("query_cluster", "belongs_to", "task_type"),
    ("llm_role", "performance_on", "query_cluster"),
]


def _elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """ELU activation function."""
    return np.where(x > 0, x, alpha * (np.exp(np.clip(x, -20, 0)) - 1))


def _leaky_relu(x: np.ndarray, slope: float = 0.2) -> np.ndarray:
    """Leaky ReLU activation function."""
    return np.where(x > 0, x, slope * x)


def _softmax_by_group(values: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    """Compute softmax within each group of indices.

    Args:
        values: (E,) attention logits
        groups: (E,) group assignment for each value (destination node indices)
        n_groups: Total number of groups

    Returns:
        (E,) softmax probabilities per group
    """
    if len(values) == 0:
        return values

    # Subtract max per group for numerical stability
    max_per_group = np.full(n_groups, -np.inf)
    np.maximum.at(max_per_group, groups, values)
    shifted = values - max_per_group[groups]

    # Exp and sum per group
    exp_vals = np.exp(np.clip(shifted, -20, 20))
    sum_per_group = np.zeros(n_groups)
    np.add.at(sum_per_group, groups, exp_vals)

    return exp_vals / (sum_per_group[groups] + 1e-8)


class LightweightGAT:
    """2-layer heterogeneous GAT, pure numpy.

    Layer 1: MultiHeadGAT(input_dim -> hidden_dim, heads=num_heads) + ELU
             Output: (num_heads * hidden_dim)-dim concatenated
    Layer 2: MultiHeadGAT(num_heads * hidden_dim -> output_dim, heads=1) + ELU
             Output: output_dim

    Edge prediction: sigmoid(dot(query_emb, llm_emb))
    """

    def __init__(self, config: Optional[GATConfig] = None):
        self.config = config or GATConfig()
        self._weights: Dict[str, np.ndarray] = {}
        self._init_weights(np.random.default_rng(42))

    def _init_weights(self, rng: np.random.Generator) -> None:
        """Initialize weight matrices with Xavier/Glorot initialization."""
        cfg = self.config
        concat_dim = cfg.num_heads * cfg.hidden_dim  # Layer 1 output

        # Layer 1: per-node-type projection (input_dim -> hidden_dim per head)
        for nt in NODE_TYPES:
            fan_in, fan_out = cfg.input_dim, cfg.hidden_dim
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self._weights[f"l1_W_{nt}"] = rng.normal(0, scale, (cfg.input_dim, cfg.hidden_dim)).astype(np.float32)

        # Layer 1: per-edge-type attention vectors (2 * hidden_dim -> 1 per head)
        for _, rel, _ in EDGE_TYPES:
            for h in range(cfg.num_heads):
                scale = np.sqrt(2.0 / (2 * cfg.hidden_dim))
                self._weights[f"l1_a_{rel}_h{h}"] = rng.normal(0, scale, (2 * cfg.hidden_dim,)).astype(np.float32)

        # Layer 2: per-node-type projection (concat_dim -> output_dim)
        for nt in NODE_TYPES:
            fan_in, fan_out = concat_dim, cfg.output_dim
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self._weights[f"l2_W_{nt}"] = rng.normal(0, scale, (concat_dim, cfg.output_dim)).astype(np.float32)

        # Layer 2: per-edge-type attention vectors (single head)
        for _, rel, _ in EDGE_TYPES:
            scale = np.sqrt(2.0 / (2 * cfg.output_dim))
            self._weights[f"l2_a_{rel}"] = rng.normal(0, scale, (2 * cfg.output_dim,)).astype(np.float32)

    def forward(
        self,
        node_features: Dict[str, np.ndarray],
        edge_index: Dict[str, np.ndarray],
        edge_features: Optional[Dict[str, np.ndarray]] = None,
        training: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Forward pass through 2-layer GAT.

        Args:
            node_features: {node_type: (N, input_dim)} initial embeddings
            edge_index: {edge_type: (2, E)} adjacency indices
            edge_features: Optional {edge_type: (E, F)} edge features (unused in base GAT)
            training: If True, apply dropout

        Returns:
            {node_type: (N, output_dim)} updated embeddings
        """
        cfg = self.config

        # === Layer 1: Multi-head attention ===
        # Project all node types: (N, input_dim) -> (N, hidden_dim)
        projected_l1 = {}
        for nt in NODE_TYPES:
            feat = node_features.get(nt, np.zeros((0, cfg.input_dim), dtype=np.float32))
            if feat.shape[0] == 0:
                projected_l1[nt] = np.zeros((0, cfg.hidden_dim), dtype=np.float32)
            else:
                projected_l1[nt] = feat.astype(np.float32) @ self._weights[f"l1_W_{nt}"]

        # Multi-head attention aggregation
        layer1_out = {}
        for nt in NODE_TYPES:
            n = projected_l1[nt].shape[0]
            if n == 0:
                layer1_out[nt] = np.zeros((0, cfg.num_heads * cfg.hidden_dim), dtype=np.float32)
                continue
            head_outputs = []
            for h in range(cfg.num_heads):
                agg = self._gat_aggregate(
                    projected_l1, nt, edge_index,
                    prefix="l1", head=h, n_nodes=n, training=training,
                )
                head_outputs.append(agg)
            # Concatenate heads: (N, num_heads * hidden_dim)
            layer1_out[nt] = _elu(np.concatenate(head_outputs, axis=1))

        # === Layer 2: Single-head attention ===
        projected_l2 = {}
        concat_dim = cfg.num_heads * cfg.hidden_dim
        for nt in NODE_TYPES:
            feat = layer1_out[nt]
            if feat.shape[0] == 0:
                projected_l2[nt] = np.zeros((0, cfg.output_dim), dtype=np.float32)
            else:
                projected_l2[nt] = feat @ self._weights[f"l2_W_{nt}"]

        layer2_out = {}
        for nt in NODE_TYPES:
            n = projected_l2[nt].shape[0]
            if n == 0:
                layer2_out[nt] = np.zeros((0, cfg.output_dim), dtype=np.float32)
                continue
            agg = self._gat_aggregate(
                projected_l2, nt, edge_index,
                prefix="l2", head=None, n_nodes=n, training=training,
            )
            layer2_out[nt] = _elu(agg)

        return layer2_out

    def _gat_aggregate(
        self,
        projected: Dict[str, np.ndarray],
        target_type: str,
        edge_index: Dict[str, np.ndarray],
        prefix: str,
        head: Optional[int],
        n_nodes: int,
        training: bool = False,
    ) -> np.ndarray:
        """Single-head GAT aggregation for a target node type.

        For each edge type where target_type is the destination,
        compute attention-weighted message passing.

        Args:
            projected: {node_type: (N, dim)} projected features
            target_type: Which node type to aggregate into
            edge_index: {rel_name: (2, E)} edge indices
            prefix: "l1" or "l2" for weight lookup
            head: Head index (None for single-head)
            n_nodes: Number of target nodes
            training: Apply dropout if True

        Returns:
            (n_nodes, dim) aggregated features
        """
        cfg = self.config
        dim = projected[target_type].shape[1] if projected[target_type].shape[0] > 0 else (
            cfg.hidden_dim if prefix == "l1" else cfg.output_dim
        )

        # Start with self-loop (identity)
        result = projected[target_type].copy() if projected[target_type].shape[0] > 0 else np.zeros((n_nodes, dim), dtype=np.float32)

        for src_type, rel, dst_type in EDGE_TYPES:
            if dst_type != target_type:
                continue

            # Get edge index for this relation
            ei = edge_index.get(rel, np.zeros((2, 0), dtype=np.int64))
            if ei.shape[1] == 0:
                continue

            src_idx = ei[0]  # Source node indices
            dst_idx = ei[1]  # Destination node indices

            src_feat = projected[src_type]
            dst_feat = projected[dst_type]

            if src_feat.shape[0] == 0 or dst_feat.shape[0] == 0:
                continue

            # Attention: a^T [W_dst*h_i || W_src*h_j]
            h_src = src_feat[src_idx]  # (E, dim)
            h_dst = dst_feat[dst_idx]  # (E, dim)

            # Get attention vector
            if head is not None:
                a_key = f"{prefix}_a_{rel}_h{head}"
            else:
                a_key = f"{prefix}_a_{rel}"
            a_vec = self._weights[a_key]

            concat = np.concatenate([h_dst, h_src], axis=1)  # (E, 2*dim)
            logits = concat @ a_vec  # (E,)
            logits = _leaky_relu(logits, cfg.leaky_relu_slope)
            logits = np.clip(logits, -cfg.clip_logits, cfg.clip_logits)

            # Softmax per destination node
            attn = _softmax_by_group(logits, dst_idx, n_nodes)

            if training and cfg.dropout > 0:
                mask = np.random.random(attn.shape) > cfg.dropout
                attn = attn * mask

            # Weighted aggregation: scatter add
            messages = h_src * attn[:, None]  # (E, dim)
            np.add.at(result, dst_idx, messages)

        return result

    def predict_edges(
        self,
        query_embeddings: np.ndarray,
        llm_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Predict edge scores between query clusters and LLM roles.

        Args:
            query_embeddings: (N_q, output_dim) GAT-transformed query embeddings
            llm_embeddings: (N_llm, output_dim) GAT-transformed LLM embeddings

        Returns:
            (N_q, N_llm) sigmoid scores
        """
        if query_embeddings.shape[0] == 0 or llm_embeddings.shape[0] == 0:
            return np.zeros((query_embeddings.shape[0], llm_embeddings.shape[0]), dtype=np.float32)

        # Normalize for cosine-like scoring
        q_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
        l_norm = llm_embeddings / (np.linalg.norm(llm_embeddings, axis=1, keepdims=True) + 1e-8)

        # Dot product + sigmoid
        scores = q_norm @ l_norm.T  # (N_q, N_llm)
        return 1.0 / (1.0 + np.exp(-np.clip(scores, -20, 20)))

    def compute_loss(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> float:
        """Binary cross-entropy loss for edge prediction.

        Args:
            predictions: (N_q, N_llm) predicted edge scores
            targets: (N_q, N_llm) binary targets
            mask: Optional (N_q, N_llm) mask for valid edges

        Returns:
            Scalar loss value
        """
        eps = self.config.epsilon
        preds = np.clip(predictions, eps, 1.0 - eps)
        bce = -(targets * np.log(preds) + (1 - targets) * np.log(1 - preds))
        if mask is not None:
            bce = bce * mask
            return float(bce.sum() / (mask.sum() + eps))
        return float(bce.mean())

    def get_gradients(
        self,
        node_features: Dict[str, np.ndarray],
        edge_index: Dict[str, np.ndarray],
        targets: np.ndarray,
        qc_to_pred_idx: np.ndarray,
        llm_to_pred_idx: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Compute gradients via numerical differentiation (finite differences).

        For a ~130-node graph with small weight matrices, numerical gradients
        are fast enough (~10-30s per epoch) and much simpler than implementing
        full backprop through heterogeneous attention layers.

        Args:
            node_features: Input features
            edge_index: Graph structure
            targets: Target edge labels
            qc_to_pred_idx: Query cluster indices in prediction matrix
            llm_to_pred_idx: LLM role indices in prediction matrix

        Returns:
            {weight_name: gradient_array}
        """
        delta = 1e-4
        gradients = {}

        # Compute base loss
        out = self.forward(node_features, edge_index)
        preds = self.predict_edges(
            out["query_cluster"][qc_to_pred_idx] if len(qc_to_pred_idx) > 0 else out["query_cluster"],
            out["llm_role"][llm_to_pred_idx] if len(llm_to_pred_idx) > 0 else out["llm_role"],
        )
        base_loss = self.compute_loss(preds, targets)

        for name, W in self._weights.items():
            grad = np.zeros_like(W)
            flat_W = W.ravel()
            flat_grad = grad.ravel()

            # Sample a subset of parameters for efficiency
            n_params = len(flat_W)
            if n_params > 500:
                # Stochastic gradient: sample 500 parameters
                sample_idx = np.random.choice(n_params, 500, replace=False)
            else:
                sample_idx = np.arange(n_params)

            for idx in sample_idx:
                old_val = flat_W[idx]
                flat_W[idx] = old_val + delta
                W_reshaped = flat_W.reshape(W.shape)
                self._weights[name] = W_reshaped

                out_p = self.forward(node_features, edge_index)
                preds_p = self.predict_edges(
                    out_p["query_cluster"][qc_to_pred_idx] if len(qc_to_pred_idx) > 0 else out_p["query_cluster"],
                    out_p["llm_role"][llm_to_pred_idx] if len(llm_to_pred_idx) > 0 else out_p["llm_role"],
                )
                loss_p = self.compute_loss(preds_p, targets)

                flat_grad[idx] = (loss_p - base_loss) / delta
                flat_W[idx] = old_val

            # Restore and scale stochastic gradient
            self._weights[name] = flat_W.reshape(W.shape)
            if n_params > 500:
                flat_grad *= n_params / 500.0
            gradients[name] = flat_grad.reshape(W.shape)

        return gradients

    def update_weights(self, gradients: Dict[str, np.ndarray], lr: float) -> None:
        """SGD weight update.

        Args:
            gradients: {weight_name: gradient_array}
            lr: Learning rate
        """
        for name, grad in gradients.items():
            if name in self._weights:
                self._weights[name] -= lr * grad

    def save(self, path: Path) -> None:
        """Save weights to .npz file.

        Args:
            path: Output path (should end in .npz)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save config alongside weights
        config_dict = {
            "config_input_dim": np.array(self.config.input_dim),
            "config_hidden_dim": np.array(self.config.hidden_dim),
            "config_output_dim": np.array(self.config.output_dim),
            "config_num_heads": np.array(self.config.num_heads),
        }
        np.savez(str(path), **self._weights, **config_dict)
        logger.info("GAT weights saved to %s (%d parameters)", path, len(self._weights))

    def load(self, path: Path) -> None:
        """Load weights from .npz file.

        Args:
            path: Input path
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"GAT weights not found: {path}")

        data = np.load(str(path))
        loaded = 0
        for name in data.files:
            if name.startswith("config_"):
                continue
            if name in self._weights:
                self._weights[name] = data[name].astype(np.float32)
                loaded += 1
            else:
                logger.debug("Skipping unknown weight: %s", name)

        logger.info("GAT weights loaded from %s (%d/%d parameters)", path, loaded, len(self._weights))

    @property
    def param_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(w.size for w in self._weights.values())
