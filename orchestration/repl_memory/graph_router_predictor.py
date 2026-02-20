"""GraphRouterPredictor: Inference wrapper for GNN-based routing signal.

Combines BipartiteRoutingGraph + LightweightGAT to produce per-role routing
scores for a given query embedding. Uses TTL caching for sub-millisecond
latency on warm cache.

Latency budget: <5ms total
- GAT forward on ~130 nodes x 32-dim: <0.1ms
- Nearest-cluster lookup: ~2-3ms (cached to <0.1ms)
- Total with warm cache: ~0.5ms
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS_PATH = Path(
    "/mnt/raid0/llm/claude/orchestration/repl_memory/graph_router_weights.npz"
)


class GraphRouterPredictor:
    """Cached inference wrapper for the GraphRouter GNN.

    Caches:
    - GAT-transformed node embeddings (changes only on weight reload or graph sync)
    - LLM role embeddings (changes only on model fleet changes)
    - Graph structure (changes only on sync)
    """

    def __init__(
        self,
        graph: Any,  # BipartiteRoutingGraph
        gat: Any,  # LightweightGAT
        embedder: Any,  # TaskEmbedder
        cache_ttl: int = 60,
    ):
        self.graph = graph
        self.gat = gat
        self.embedder = embedder
        self._cache_ttl = cache_ttl

        # Cached state
        self._transformed_embeddings: Optional[Dict[str, np.ndarray]] = None
        self._llm_ids: Optional[list] = None
        self._cache_valid = False

        try:
            from cachetools import TTLCache
            self._cluster_cache: Optional[Any] = TTLCache(maxsize=500, ttl=cache_ttl)
        except ImportError:
            self._cluster_cache = None

    def _ensure_cache(self) -> bool:
        """Refresh cached GAT embeddings if stale.

        Returns:
            True if cache is valid
        """
        if self._cache_valid:
            return True

        try:
            node_feats = self.graph.get_node_features()
            edge_idx = self.graph.get_edge_index()

            # Check minimum graph size
            n_qc = node_feats["query_cluster"].shape[0]
            n_llm = node_feats["llm_role"].shape[0]
            if n_qc == 0 or n_llm == 0:
                logger.debug("Graph too sparse for prediction (qc=%d, llm=%d)", n_qc, n_llm)
                return False

            # Map edge_index keys to match GAT expectations
            gat_edge_index = {
                "belongs_to": edge_idx["belongs_to"],
                "performance_on": edge_idx["performance_on"],
            }

            # Run GAT forward
            out = self.gat.forward(node_feats, gat_edge_index)
            self._transformed_embeddings = out
            self._llm_ids = node_feats["llm_role_ids"]
            self._qc_ids = node_feats["query_cluster_ids"]
            self._cache_valid = True
            return True
        except Exception as e:
            logger.warning("GAT forward cache refresh failed: %s", e)
            return False

    def invalidate_cache(self) -> None:
        """Force cache refresh on next predict call."""
        self._cache_valid = False
        if self._cluster_cache is not None:
            self._cluster_cache.clear()

    def predict(
        self,
        query_embedding: np.ndarray,
        task_type: str,
    ) -> Dict[str, float]:
        """Predict routing scores for all LLM roles.

        Args:
            query_embedding: BGE-large 1024-dim embedding of the query
            task_type: Task type string for cluster lookup

        Returns:
            {role_id: probability} for all LLM roles
        """
        if not self._ensure_cache():
            return {}

        # Find nearest query cluster
        cluster_id = self._find_nearest_cluster(query_embedding, task_type)
        if cluster_id is None:
            return {}

        # Get cluster index in GAT output
        try:
            qc_idx = self._qc_ids.index(cluster_id)
        except (ValueError, AttributeError):
            return {}

        # Get GAT-transformed embeddings
        qc_emb = self._transformed_embeddings["query_cluster"][qc_idx:qc_idx + 1]  # (1, dim)
        llm_emb = self._transformed_embeddings["llm_role"]  # (N_llm, dim)

        if llm_emb.shape[0] == 0:
            return {}

        # Edge prediction: sigmoid(dot product)
        scores = self.gat.predict_edges(qc_emb, llm_emb)  # (1, N_llm)
        scores = scores[0]  # (N_llm,)

        # Softmax to get probabilities
        exp_scores = np.exp(scores - scores.max())
        probs = exp_scores / (exp_scores.sum() + 1e-8)

        return {
            role_id: float(probs[i])
            for i, role_id in enumerate(self._llm_ids)
        }

    def _find_nearest_cluster(self, embed: np.ndarray, task_type: str) -> Optional[str]:
        """Find nearest query cluster with caching."""
        # Create a cache key from task_type + embedding hash
        cache_key = f"{task_type}:{hash(embed.tobytes())}"

        if self._cluster_cache is not None and cache_key in self._cluster_cache:
            return self._cluster_cache[cache_key]

        cluster_id = self.graph.get_query_cluster_for_embedding(embed, task_type)

        if self._cluster_cache is not None and cluster_id is not None:
            self._cluster_cache[cache_key] = cluster_id

        return cluster_id

    @property
    def is_ready(self) -> bool:
        """True if weights are loaded AND graph has LLM nodes."""
        try:
            stats = self.graph.get_stats()
            has_llm = stats.get("llmrole_count", 0) > 0
            has_clusters = stats.get("querycluster_count", 0) > 0
            has_weights = len(self.gat._weights) > 0
            return has_llm and has_clusters and has_weights
        except Exception:
            return False
