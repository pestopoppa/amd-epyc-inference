"""BipartiteRoutingGraph: Kuzu-backed bipartite graph for GNN-based routing.

Implements a heterogeneous graph with three node types:
- TaskType: High-level task categories (code, chat, ingest, architecture)
- QueryCluster: Clustered query embeddings per task type (MiniBatchKMeans centroids)
- LLMRole: Model fleet roles with capability embeddings

Edge types:
- BELONGS_TO: QueryCluster -> TaskType
- PERFORMANCE_ON: LLMRole -> QueryCluster (success_rate, avg_q_value, avg_latency)

Used by GraphRouterPredictor to provide a parallel routing signal alongside
TwoPhaseRetriever. The GAT learns from the graph structure to generalize routing
predictions for new models (cold-start optimization).

Reference: GraphRouter (ICLR 2025, arxiv 2410.03834)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_KUZU_PATH = Path(
    "/mnt/raid0/llm/claude/orchestration/repl_memory/kuzu_db/routing_graph"
)


class BipartiteRoutingGraph:
    """Kuzu-backed bipartite graph for routing signal generation.

    Schema mirrors GraphRouter paper with richer edge features:
    - TaskType nodes: task taxonomy (code, chat, ingest, architecture, etc.)
    - QueryCluster nodes: MiniBatchKMeans centroids of episodic memory embeddings
    - LLMRole nodes: model fleet roles with capability embeddings
    - PERFORMANCE_ON edges: (role, cluster) with success_rate, avg_q, avg_latency
    """

    def __init__(self, path: Path = DEFAULT_KUZU_PATH):
        try:
            import kuzu
        except ImportError as e:
            raise ImportError("kuzu not installed. Run: pip install kuzu") from e

        self._kuzu = kuzu
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.db = kuzu.Database(str(self.path))
        self.conn = kuzu.Connection(self.db)
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize graph schema if not exists."""
        node_schemas = [
            """CREATE NODE TABLE IF NOT EXISTS TaskType(
                id STRING,
                description STRING,
                embedding DOUBLE[1024],
                PRIMARY KEY(id)
            )""",
            """CREATE NODE TABLE IF NOT EXISTS QueryCluster(
                id STRING,
                representative_text STRING,
                embedding DOUBLE[1024],
                task_type_id STRING,
                sample_count INT64,
                PRIMARY KEY(id)
            )""",
            """CREATE NODE TABLE IF NOT EXISTS LLMRole(
                id STRING,
                description STRING,
                embedding DOUBLE[1024],
                port INT64,
                tokens_per_second DOUBLE,
                memory_tier STRING,
                memory_gb DOUBLE,
                PRIMARY KEY(id)
            )""",
        ]

        rel_schemas = [
            """CREATE REL TABLE IF NOT EXISTS BELONGS_TO(
                FROM QueryCluster TO TaskType
            )""",
            """CREATE REL TABLE IF NOT EXISTS PERFORMANCE_ON(
                FROM LLMRole TO QueryCluster,
                success_rate DOUBLE,
                avg_q_value DOUBLE,
                avg_latency_s DOUBLE,
                sample_count INT64,
                last_updated TIMESTAMP
            )""",
        ]

        for schema in node_schemas + rel_schemas:
            try:
                self.conn.execute(schema)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning("Schema creation warning: %s", e)

    def sync_from_episodic_store(
        self,
        store: Any,
        embedder: Any,
        n_clusters_per_type: int = 20,
    ) -> Dict[str, int]:
        """Full rebuild: cluster memories by task_type, create graph structure.

        Steps:
        1. Export all memories from episodic store grouped by task_type
        2. Cluster each group via MiniBatchKMeans
        3. Create QueryCluster nodes with centroid embeddings
        4. Compute PERFORMANCE_ON edges from (role, cluster) outcome aggregates

        Args:
            store: EpisodicStore instance
            embedder: TaskEmbedder instance for embedding task type descriptions
            n_clusters_per_type: Max clusters per task type (dynamic: min 5)

        Returns:
            {"task_types": N, "clusters": M, "edges": E}
        """
        from sklearn.cluster import MiniBatchKMeans

        # Export all memories with embeddings
        memories = store.get_all_memories()
        if not memories:
            logger.warning("No memories in episodic store, skipping graph sync")
            return {"task_types": 0, "clusters": 0, "edges": 0}

        # Group by task_type
        by_type: Dict[str, List] = {}
        for mem in memories:
            ctx = mem.context or {} if hasattr(mem, "context") else {}
            task_type = ctx.get("task_type", "general")
            by_type.setdefault(task_type, []).append(mem)

        # Clear existing data
        self._clear_graph()

        # Create TaskType nodes
        task_types_created = 0
        for task_type in by_type:
            desc = f"Task type: {task_type}"
            try:
                emb = embedder.embed_text(desc)
            except Exception:
                emb = np.zeros(1024, dtype=np.float64)
            self._create_task_type(task_type, desc, emb)
            task_types_created += 1

        # Cluster memories per type and create QueryCluster nodes
        clusters_created = 0
        for task_type, mems in by_type.items():
            # Collect embeddings
            embeddings = []
            valid_mems = []
            for m in mems:
                emb = getattr(m, "embedding", None)
                if emb is not None and len(emb) > 0:
                    embeddings.append(np.asarray(emb, dtype=np.float64))
                    valid_mems.append(m)

            if not embeddings:
                continue

            X = np.stack(embeddings)
            # Dynamic cluster count
            k = max(5, min(n_clusters_per_type, len(X) // 10))
            k = min(k, len(X))  # Can't have more clusters than samples

            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=min(256, len(X)))
            labels = kmeans.fit_predict(X)
            centroids = kmeans.cluster_centers_

            for ci in range(k):
                cluster_id = str(uuid.uuid4())
                mask = labels == ci
                count = int(mask.sum())
                if count == 0:
                    continue

                # Pick representative text from closest to centroid
                rep_idx = np.argmin(np.linalg.norm(X[mask] - centroids[ci], axis=1))
                rep_mem = [m for m, keep in zip(valid_mems, mask) if keep][rep_idx]
                rep_text = getattr(rep_mem, "task_description", "")[:200]

                self._create_query_cluster(
                    cluster_id, rep_text, centroids[ci], task_type, count
                )

                # Compute PERFORMANCE_ON edges for this cluster
                cluster_mems = [m for m, keep in zip(valid_mems, mask) if keep]
                self._compute_performance_edges(cluster_id, cluster_mems)
                clusters_created += 1

        # Count edges
        edge_count = self._count_edges()

        logger.info(
            "Graph sync complete: %d task_types, %d clusters, %d edges",
            task_types_created, clusters_created, edge_count,
        )
        return {
            "task_types": task_types_created,
            "clusters": clusters_created,
            "edges": edge_count,
        }

    def _clear_graph(self) -> None:
        """Remove all nodes and edges."""
        for table in ["PERFORMANCE_ON", "BELONGS_TO"]:
            try:
                self.conn.execute(f"MATCH ()-[r:{table}]->() DELETE r")
            except Exception:
                pass
        for table in ["QueryCluster", "TaskType", "LLMRole"]:
            try:
                self.conn.execute(f"MATCH (n:{table}) DELETE n")
            except Exception:
                pass

    def _create_task_type(self, id: str, description: str, embedding: np.ndarray) -> None:
        self.conn.execute(
            "CREATE (t:TaskType {id: $id, description: $d, embedding: $emb})",
            {"id": id, "d": description, "emb": embedding.tolist()},
        )

    def _create_query_cluster(
        self, id: str, rep_text: str, embedding: np.ndarray,
        task_type_id: str, sample_count: int,
    ) -> None:
        self.conn.execute(
            "CREATE (q:QueryCluster {id: $id, representative_text: $text, embedding: $emb, task_type_id: $tt, sample_count: $cnt})",
            {"id": id, "text": rep_text, "emb": embedding.tolist(), "tt": task_type_id, "cnt": sample_count},
        )
        # Link to TaskType
        try:
            self.conn.execute(
                """MATCH (q:QueryCluster {id: $qid}), (t:TaskType {id: $tid})
                CREATE (q)-[:BELONGS_TO]->(t)""",
                {"qid": id, "tid": task_type_id},
            )
        except Exception as e:
            logger.debug("BELONGS_TO link error: %s", e)

    def _compute_performance_edges(self, cluster_id: str, memories: List) -> None:
        """Aggregate outcomes by role and create PERFORMANCE_ON edges."""
        role_stats: Dict[str, Dict[str, Any]] = {}
        for mem in memories:
            ctx = mem.context or {} if hasattr(mem, "context") else {}
            role = ctx.get("role", "")
            if not role:
                action = getattr(mem, "action", "")
                role = action.split(",")[0].split(":")[0].strip() if action else ""
            if not role:
                continue

            stats = role_stats.setdefault(role, {
                "successes": 0, "total": 0, "q_sum": 0.0,
                "latency_sum": 0.0, "latency_count": 0,
            })
            stats["total"] += 1
            q = getattr(mem, "q_value", 0.5)
            stats["q_sum"] += q
            if q >= 0.7:
                stats["successes"] += 1

            elapsed = ctx.get("elapsed_seconds", 0.0)
            if elapsed > 0:
                stats["latency_sum"] += elapsed
                stats["latency_count"] += 1

        now = datetime.now(timezone.utc)
        for role, stats in role_stats.items():
            if stats["total"] == 0:
                continue
            success_rate = stats["successes"] / stats["total"]
            avg_q = stats["q_sum"] / stats["total"]
            avg_latency = (
                stats["latency_sum"] / stats["latency_count"]
                if stats["latency_count"] > 0 else 0.0
            )
            try:
                self.conn.execute(
                    "MATCH (r:LLMRole {id: $rid}), (q:QueryCluster {id: $qid}) CREATE (r)-[:PERFORMANCE_ON {success_rate: $sr, avg_q_value: $aq, avg_latency_s: $al, sample_count: $sc, last_updated: $lu}]->(q)",
                    {"rid": role, "qid": cluster_id, "sr": success_rate, "aq": avg_q, "al": avg_latency, "sc": stats["total"], "lu": now},
                )
            except Exception as e:
                # LLMRole node may not exist yet — that's fine, edges will be
                # created after add_llm_role is called
                logger.debug("PERFORMANCE_ON edge skipped (role=%s): %s", role, e)

    def _count_edges(self) -> int:
        try:
            result = self.conn.execute(
                "MATCH ()-[r:PERFORMANCE_ON]->() RETURN COUNT(r) as cnt"
            )
            rows = result.get_as_df()
            return int(rows.iloc[0]["cnt"]) if len(rows) > 0 else 0
        except Exception:
            return 0

    def add_llm_role(
        self,
        role_id: str,
        description: str,
        embedding: np.ndarray,
        port: int,
        tps: float,
        tier: str,
        gb: float,
    ) -> None:
        """Add or update an LLMRole node.

        Args:
            role_id: Role identifier (e.g., "frontdoor", "coder_escalation")
            description: Capability description text
            embedding: BGE-large 1024-dim embedding of description
            port: Server port number
            tps: Tokens per second
            tier: Memory tier ("HOT" or "WARM")
            gb: VRAM usage in GB
        """
        # Upsert: delete if exists, then create
        try:
            self.conn.execute(
                "MATCH (r:LLMRole {id: $id}) DELETE r",
                {"id": role_id},
            )
        except Exception:
            pass

        self.conn.execute(
            "CREATE (r:LLMRole {id: $id, description: $d, embedding: $emb, port: $port, tokens_per_second: $tps, memory_tier: $tier, memory_gb: $gb})",
            {"id": role_id, "d": description, "emb": embedding.tolist(), "port": port, "tps": tps, "tier": tier, "gb": gb},
        )

    def get_llm_embeddings(self) -> Dict[str, np.ndarray]:
        """Get embeddings for all LLM role nodes.

        Returns:
            {role_id: embedding_array}
        """
        result = self.conn.execute(
            "MATCH (r:LLMRole) RETURN r.id, r.embedding"
        )
        rows = result.get_as_df()
        out = {}
        for _, row in rows.iterrows():
            out[row["r.id"]] = np.array(row["r.embedding"], dtype=np.float64)
        return out

    def get_query_cluster_for_embedding(
        self, embed: np.ndarray, task_type: str,
    ) -> Optional[str]:
        """Find nearest QueryCluster by cosine similarity within a task type.

        Args:
            embed: Query embedding (1024-dim)
            task_type: Task type filter

        Returns:
            Cluster ID or None
        """
        result = self.conn.execute(
            """MATCH (q:QueryCluster)
            WHERE q.task_type_id = $tt
            RETURN q.id, q.embedding""",
            {"tt": task_type},
        )
        rows = result.get_as_df()
        if len(rows) == 0:
            return None

        query_norm = embed / (np.linalg.norm(embed) + 1e-8)
        best_id = None
        best_sim = -1.0
        for _, row in rows.iterrows():
            c_emb = np.array(row["q.embedding"], dtype=np.float64)
            c_norm = c_emb / (np.linalg.norm(c_emb) + 1e-8)
            sim = float(np.dot(query_norm, c_norm))
            if sim > best_sim:
                best_sim = sim
                best_id = row["q.id"]
        return best_id

    def get_performance_edges(self) -> List[Dict]:
        """Export all PERFORMANCE_ON edges for training.

        Returns:
            List of {from_role, to_cluster, success_rate, avg_q_value, avg_latency_s, sample_count}
        """
        result = self.conn.execute(
            """MATCH (r:LLMRole)-[p:PERFORMANCE_ON]->(q:QueryCluster)
            RETURN r.id, q.id, p.success_rate, p.avg_q_value,
                   p.avg_latency_s, p.sample_count"""
        )
        rows = result.get_as_df()
        edges = []
        for _, row in rows.iterrows():
            edges.append({
                "from_role": row["r.id"],
                "to_cluster": row["q.id"],
                "success_rate": float(row["p.success_rate"]),
                "avg_q_value": float(row["p.avg_q_value"]),
                "avg_latency_s": float(row["p.avg_latency_s"]),
                "sample_count": int(row["p.sample_count"]),
            })
        return edges

    def get_node_features(self) -> Dict[str, np.ndarray]:
        """Export node feature matrices for GAT input.

        Returns:
            {
                "task_type": (N_tt, 1024),
                "query_cluster": (N_qc, 1024),
                "llm_role": (N_llm, 1024),
                "task_type_ids": [id1, ...],
                "query_cluster_ids": [id1, ...],
                "llm_role_ids": [id1, ...],
            }
        """
        features = {}
        for node_type, table in [
            ("task_type", "TaskType"),
            ("query_cluster", "QueryCluster"),
            ("llm_role", "LLMRole"),
        ]:
            result = self.conn.execute(
                f"MATCH (n:{table}) RETURN n.id, n.embedding ORDER BY n.id"
            )
            rows = result.get_as_df()
            if len(rows) == 0:
                features[node_type] = np.zeros((0, 1024), dtype=np.float64)
                features[f"{node_type}_ids"] = []
            else:
                embs = []
                ids = []
                for _, row in rows.iterrows():
                    embs.append(np.array(row["n.embedding"], dtype=np.float64))
                    ids.append(row["n.id"])
                features[node_type] = np.stack(embs)
                features[f"{node_type}_ids"] = ids
        return features

    def get_edge_index(self) -> Dict[str, np.ndarray]:
        """Export edge indices for GAT adjacency.

        Returns:
            {
                "belongs_to": (2, E_bt) — [query_cluster_idx, task_type_idx],
                "performance_on": (2, E_po) — [llm_role_idx, query_cluster_idx],
                "performance_features": (E_po, 2) — [success_rate, norm_latency],
            }
        """
        node_feats = self.get_node_features()
        tt_ids = node_feats["task_type_ids"]
        qc_ids = node_feats["query_cluster_ids"]
        llm_ids = node_feats["llm_role_ids"]

        tt_idx = {id: i for i, id in enumerate(tt_ids)}
        qc_idx = {id: i for i, id in enumerate(qc_ids)}
        llm_idx = {id: i for i, id in enumerate(llm_ids)}

        # BELONGS_TO edges
        result = self.conn.execute(
            "MATCH (q:QueryCluster)-[:BELONGS_TO]->(t:TaskType) RETURN q.id, t.id"
        )
        rows = result.get_as_df()
        bt_src, bt_dst = [], []
        for _, row in rows.iterrows():
            qid, tid = row["q.id"], row["t.id"]
            if qid in qc_idx and tid in tt_idx:
                bt_src.append(qc_idx[qid])
                bt_dst.append(tt_idx[tid])

        # PERFORMANCE_ON edges
        perf_edges = self.get_performance_edges()
        po_src, po_dst = [], []
        po_features = []
        for e in perf_edges:
            rid, cid = e["from_role"], e["to_cluster"]
            if rid in llm_idx and cid in qc_idx:
                po_src.append(llm_idx[rid])
                po_dst.append(qc_idx[cid])
                # Normalize latency to [0,1] range (cap at 120s)
                norm_lat = min(e["avg_latency_s"] / 120.0, 1.0)
                po_features.append([e["success_rate"], norm_lat])

        return {
            "belongs_to": np.array([bt_src, bt_dst], dtype=np.int64) if bt_src else np.zeros((2, 0), dtype=np.int64),
            "performance_on": np.array([po_src, po_dst], dtype=np.int64) if po_src else np.zeros((2, 0), dtype=np.int64),
            "performance_features": np.array(po_features, dtype=np.float64) if po_features else np.zeros((0, 2), dtype=np.float64),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        stats = {}
        for table in ["TaskType", "QueryCluster", "LLMRole"]:
            try:
                result = self.conn.execute(f"MATCH (n:{table}) RETURN COUNT(n) as count")
                rows = result.get_as_df()
                stats[f"{table.lower()}_count"] = int(rows.iloc[0]["count"]) if len(rows) > 0 else 0
            except Exception:
                stats[f"{table.lower()}_count"] = 0

        stats["performance_edge_count"] = self._count_edges()
        return stats

    def close(self) -> None:
        """Close the database connection."""
        pass
