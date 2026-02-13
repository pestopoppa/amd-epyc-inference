# Chapter 15: MemRL System

## Introduction

The Memory-augmented Reinforcement Learning (MemRL) system enables the orchestrator to learn from experience. Episodic memory stores task contexts, actions, and outcomes. A two-phase retriever finds similar past situations. Q-value scoring learns which actions work best for which tasks. The system progresses through 7 phases, from manual routing to full learned orchestration.

As of 2026-01-31, the system contains 2,714 memories (1,213 canonical seeds + 1,501 non-seed) with FAISS-accelerated retrieval providing 35x speedup over NumPy baseline (70ms → ~2ms at 500K scale). A contamination cleanup on 2026-01-31 surgically removed 6,506 entries from buggy validation script runs while preserving the original Jan 28 seed corpus.

As of 2026-02-02, the seeding pipeline was enriched with 90 mode-advantage tasks (see [Chapter 24](24-benchmark-suite-construction.md)) specifically designed to produce strong comparative rewards. Prior to this, the debug suite (327 single-turn QA) produced weak routing signal — all tasks were solvable by direct inference, so MemRL learned cost-awareness but not routing quality. The mode-advantage tasks shift +1.0 rewards from ~5% to ~25-35% of episodes. Three external HuggingFace dataset adapters (GAIA 165q, CRUXEval 1600q, BigCodeBench 1140q) further expand the evaluation pool.

## Episodic Memory Architecture

> **Scope**: The MemRL episodic store handles *routing memories* (task→action→outcome with Q-values).
> For *codebase retrieval* (finding source code and documentation passages), see
> the NextPLAID integration in [Ch11: REPL Environment](11-repl-environment.md).
> These are complementary systems: BGE 1024-dim single-vector vs ColBERT 128-dim multi-vector.

### Storage Layout

```
/mnt/raid0/llm/claude/orchestration/repl_memory/sessions/
├── episodic.db           # SQLite metadata (action, context, q_value, timestamps)
├── embeddings.faiss      # FAISS index (L2-normalized inner product)
└── id_map.npy           # memory_id → faiss_idx mapping
```

**Design Rationale**: SQLite for rich queries (filter by action_type, q_value), FAISS for O(log n) similarity search.

### Memory Schema

```sql
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    embedding_idx INTEGER NOT NULL,
    action TEXT NOT NULL,
    action_type TEXT NOT NULL,  -- "routing", "escalation", "exploration"
    context TEXT NOT NULL,       -- JSON task context
    outcome TEXT,                -- "success", "failure"
    q_value REAL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    update_count INTEGER DEFAULT 0
);

CREATE INDEX idx_action_type ON memories(action_type);
CREATE INDEX idx_q_value ON memories(q_value DESC);
CREATE INDEX idx_type_q ON memories(action_type, q_value DESC);
```

**Indexes Optimized For**:
- Two-phase retrieval (filter by action_type + Q-value)
- Top-k Q-value queries for graph seeding
- Temporal queries (created_at DESC)

### FAISS Backend

```python
class FAISSEmbeddingStore:
    """FAISS IndexFlatIP with L2 normalization for cosine similarity."""

    def __init__(self, path: Path, dim: int = 1024):
        # BGE-large embedding dim
        self.index = faiss.IndexFlatIP(dim)  # Inner product

    def add(self, memory_id: str, embedding: np.ndarray) -> int:
        # L2 normalize for cosine similarity
        faiss.normalize_L2(embedding)
        self.index.add(embedding)
        self.id_map.append(memory_id)

    def search(self, query: np.ndarray, k: int = 20) -> list[tuple[str, float]]:
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, k)
        return [(self.id_map[idx], score) for idx, score in zip(indices[0], scores[0])]
```

**Performance Expectations**:

| Memory Count | FAISS Search Time | NumPy Baseline | Speedup |
|--------------|-------------------|----------------|---------|
| 5K | 0.5ms | 15ms | 30x |
| 50K | 1ms | 150ms | 150x |
| 500K | 2ms | 1500ms | 750x |
| 1M | 3ms | 3000ms | 1000x |

At 2714 memories (current), FAISS overhead is negligible vs NumPy.

## Task Embedding

### TaskEmbedder Architecture

```python
class TaskEmbedder:
    """Generate embeddings via HTTP server (2-5ms) or subprocess (50-200ms)."""

    def __init__(self):
        self.model_path = "bge-large-en-v1.5-f16.gguf"
        self.server_url = "http://127.0.0.1:8090"
        self.embedding_dim = 1024  # BGE-large embedding dim

    def embed_task_ir(self, task_ir: Dict[str, Any]) -> np.ndarray:
        # Serialize to focus on semantic fields
        text = self._serialize_task_ir(task_ir)
        return self._generate_embedding(text)
```

### Serialization Strategy

```python
def _serialize_task_ir(self, task_ir: Dict[str, Any]) -> str:
    parts = [
        f"type:{task_ir['task_type']}",
        f"objective:{task_ir['objective']}",
        f"priority:{task_ir['priority']}",
        f"constraints:{','.join(task_ir['constraints'][:5])}",
        f"input_types:{','.join(inp['type'] for inp in task_ir['inputs'])}",
    ]
    return " | ".join(parts)
```

**Why This Format**: Focuses on task semantics, not content. Similar tasks have similar embeddings even with different input data.

### Backend Fallback Chain

1. **HTTP Server** (8090): 2-5ms via `/embedding` endpoint (40x faster than subprocess)
2. **Subprocess** (`llama-embedding`): 50-200ms with `--embd-output-format json`
3. **Hash-Based Pseudo-Embeddings**: Deterministic SHA-256 expansion (fallback when model unavailable)

Hash fallback preserves identity (same input = same embedding) but NOT similarity. Used only in dev environments without model access.

## Two-Phase Retrieval

### Phase 1: Semantic Filtering

```python
def retrieve_by_similarity(
    self,
    query_embedding: np.ndarray,
    k: int = 20,
    action_type: Optional[str] = None,
    min_q_value: float = 0.0,
) -> List[MemoryEntry]:
    # FAISS search (O(log n))
    candidates = self._embedding_store.search(query_embedding, k=k * 2)

    # Phase 2: SQLite filter and enrich
    memory_ids = [memory_id for memory_id, score in candidates]
    placeholders = ",".join("?" * len(memory_ids))
    query = f"""
        SELECT id, embedding_idx, action, action_type, context, outcome, q_value
        FROM memories
        WHERE id IN ({placeholders})
    """
    if action_type:
        query += " AND action_type = ?"
    if min_q_value > 0:
        query += " AND q_value >= ?"

    # Return top k sorted by similarity
```

**Over-Fetching Strategy**: Retrieve 2k candidates from FAISS, then filter in SQL. Accounts for action_type/q_value filters without separate FAISS indexes.

### Phase 2: Q-Value Ranking

```python
@dataclass
class RetrievalConfig:
    semantic_k: int = 20           # Candidates from Phase 1
    min_similarity: float = 0.3    # Cosine similarity threshold
    q_weight: float = 0.7          # Emphasize learned utility
    top_n: int = 5                 # Final results
    confidence_threshold: float = 0.6  # Min combined score to trust
```

**Combined Score**: `0.7 * q_value + 0.3 * similarity`

Emphasizes learned utility (Q-value) over pure semantic match. A high-Q dissimilar memory is preferred over a low-Q similar one.

## Q-Value Learning

### TD-Learning Update

```python
def update_q_value(self, memory_id: str, reward: float, learning_rate: float = 0.1) -> float:
    """Q(m) ← Q(m) + α(r - Q(m))"""
    old_q = self.get_q_value(memory_id)
    new_q = old_q + learning_rate * (reward - old_q)
    new_q = max(0.0, min(1.0, new_q))  # Clamp to [0, 1]
    self.store.update(memory_id, q_value=new_q, update_count=old_q_count + 1)
    return new_q
```

**Reward Signal**:

```python
def _compute_reward(outcome, gate_failures, escalations) -> float:
    if outcome == "success":
        base_reward = 1.0
    elif outcome == "partial":
        base_reward = 0.3
    else:
        base_reward = -0.5

    penalty = gate_failures * 0.1 + escalations * 0.15
    return max(-1.0, min(1.0, base_reward - penalty))
```

**Interpretation**:
- Q=0.9+: Highly successful pattern
- Q=0.5: Neutral (default)
- Q=0.3-: Likely to fail or escalate

### Async Scoring via QScorer

```python
class QScorer:
    """Async Q-value update agent (runs every 5 min)."""

    def score_pending_tasks(self) -> Dict[str, Any]:
        unscored_task_ids = self.reader.get_unscored_tasks()

        for task_id in unscored_task_ids[:batch_size]:
            trajectory = self.reader.get_task_trajectory(task_id)
            reward = self._compute_reward(trajectory)

            # Update routing memory
            routing_memory_id = find_routing_memory(trajectory)
            self.store.update_q_value(routing_memory_id, reward)

            # Update escalation memories
            for escalation in trajectory.escalations:
                self.store.update_q_value(escalation.memory_id, reward)
```

Keeps Q-updates off the critical inference path. Runs periodically via cron or on-demand trigger.

### Multi-Dimensional Cost Model

QScorer penalizes cost across 3 independent dimensions, each with its own lambda:

```python
# Dimension 1: Latency cost (original)
latency_penalty = cost_penalty_lambda * cost_ratio
# cost_ratio = actual_elapsed / expected_elapsed

# Dimension 2: Quality gap penalty (new)
quality_gap_penalty = cost_lambda_quality_gap * max(0, model_quality - 0.75)
# Applied only when answer is correct. Penalizes using expensive models
# when cheaper ones would suffice.

# Dimension 3: Memory tier penalty (new)
memory_tier_penalty = cost_lambda_memory * (mem_cost - 1.0)
# Applied only for WARM tier models (loaded on demand).
# mem_cost normalized: HOT=1.0, architect_general=3.0, architect_coding=5.0

total_cost_penalty = latency_penalty + quality_gap_penalty + memory_tier_penalty
reward = base_reward - total_cost_penalty
```

**Quality gap baseline scores** (from benchmark suite, `baseline_quality_by_role`):

| Model | Role | Baseline Quality |
|-------|------|-----------------|
| Qwen3-235B-A22B | architect_general | 0.94 |
| Qwen2.5-Coder-32B | coder | 0.915 |
| Qwen3-Coder-30B-A3B | orchestrator | 0.895 |
| Qwen2.5-7B | worker_explore | 0.745 |

**Interpretation**: If a task is answered correctly by the 235B architect (quality=0.94), dimension 2 penalizes with `lambda * (0.94 - 0.75) = lambda * 0.19`. The same correct answer from 7B (quality=0.745) receives zero quality gap penalty. This teaches the system to prefer cheap models when they can solve the task.

### Try-Cheap-First Q-Value Convergence

The cost model drives a "try cheap first" routing strategy through Q-value convergence:

```
Q(task_class, "worker_explore") learns from:
  - Success → high reward (correct + zero quality gap penalty + HOT tier)
  - Failure → low reward → system escalates to coder/architect
```

During orchestration, Phase B/C nodes check `Q(task_class, "worker_explore") > threshold` to decide whether to attempt the cheap model first. As Q-values converge, the system learns which task classes the 7B worker can handle — routing those directly — and which require immediate escalation, avoiding wasted cheap attempts.

## MemRL Phases

| Phase | Capability | Status (2026-01) |
|-------|------------|------------------|
| 1 | Manual routing via `model_registry.yaml` | Production |
| 2 | Episodic store with embeddings | Production (2714 memories) |
| 3 | Two-phase retrieval (semantic + Q-value) | Production |
| 4 | Learned routing (HybridRouter) | Production |
| 5 | Proactive delegation (complexity-aware) | Production |
| 6 | Graph-enhanced retrieval (failure anti-memory) | Production |
| 7 | FAISS migration (O(log n) embedding search) | Production |
| 8 | Model self-routing (REPL tools + routing context) | Production |

**Current Focus**: Phase 8 (model self-routing) is production-ready. Models can query MemRL Q-values directly via REPL tools and make informed escalation/delegation decisions.

## MemRL Quality Review Gate

A two-phase quality review triggered when the MemRL Q-value for a role+task combination falls below 0.6:

**Phase 1 — Architect Verdict** (6.75 t/s, ~40 tokens, ~6s):
- Receives question + answer (TOON-encoded if worker digests available)
- Outputs: `OK` (return unchanged) or `WRONG: <concise corrections>` (trigger Phase 2)

**Phase 2 — Worker Revision** (44 t/s, ~500 tokens, ~11s, only on WRONG):
- Receives: question + original answer + architect corrections
- Outputs: revised answer incorporating corrections

**Performance Impact**:
- Trigger rate: ~20% of requests (Q < 0.6)
- WRONG rate: ~30% of reviews
- Net: ~1.9s average added latency (20% × (6s + 30% × 11s))
- This is 3x more efficient than full architect review (~6s avg vs ~18s)

**Implementation**: `src/api/routes/chat.py` (`_should_review`, `_architect_verdict`, `_fast_revise`)

## Model Self-Routing (Phase 8)

Models now have agency in routing decisions via 5 REPL functions:

| Function | Purpose |
|----------|---------|
| `my_role()` | Self-awareness: role, tier, capabilities |
| `route_advice(task)` | MemRL Q-values + recommended role |
| `delegate(prompt, role, reason)` | Tracked delegation with outcome logging |
| `escalate(reason, target_role)` | Request escalation to specific target |
| `recall(query)` | Episodic memory search with Q-values |

**Routing context** injected on turn 0: compact MemRL Q-values for similar tasks (TOON-encoded when ≥2 results). Models use this to make informed routing decisions without explicit REPL calls.

## Performance Metrics

### Memory Statistics (2026-01-28)

```
Total memories: 2714
├── routing: 1205 (avg Q=0.62)
├── escalation: 892 (avg Q=0.51)
└── exploration: 617 (avg Q=0.68)

Overall avg Q: 0.607
Backend: faiss
Embeddings count: 2714
```

**Graph Stats** (when enabled):
- Failure graph: Links memories to symptom patterns
- Hypothesis graph: Tracks action-task confidence

**Graph Wiring** (as of 2026-02-07, pydantic-graph migration):
The following MemRL functions are now called from `src/graph/nodes.py`:
- `failure_graph.record_failure()` — called on every error in `_handle_error()`
- `failure_graph.record_mitigation()` — called when an escalated role resolves a failure
- `hypothesis_graph.add_evidence()` — called on task success/failure outcomes
- `retriever.retrieve_for_escalation()` — called during `_check_memrl_suggestion()`

These were previously dead code (declared but never invoked) in the old `repl_executor.py` manual loop.

### Retrieval Latency

| Operation | Time | Notes |
|-----------|------|-------|
| Embed query | 2-5ms | HTTP server (8090) |
| FAISS search (2714 entries) | <1ms | O(log n) |
| SQL filter + enrich | 3-8ms | Indexed queries |
| **Total retrieval** | **5-13ms** | Fast enough for interactive |

With 500K memories (projected), FAISS search would be ~2ms, total ~10-20ms.

## References

### Core Concepts

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. (TD-learning, Q-values)

2. Johnson, J., Douze, M., & Jégou, H. (2019). *Billion-scale similarity search with GPUs*. IEEE Transactions on Big Data. (FAISS architecture)

### Implementation

3. `orchestration/repl_memory/episodic_store.py`: Memory storage (861 lines)
4. `orchestration/repl_memory/faiss_store.py`: FAISS backend (343 lines)
5. `orchestration/repl_memory/embedder.py`: Task embedding (393 lines)
6. `orchestration/repl_memory/retriever.py`: Two-phase retrieval (608 lines)
7. `orchestration/repl_memory/q_scorer.py`: Async Q-learning (502 lines)

### Related Systems

8. Prioritized Experience Replay (Schaul et al., 2015): https://arxiv.org/abs/1511.05952
9. Episodic Memory in Lifelong Learning (Kemker et al., 2018): https://arxiv.org/abs/1802.07569

---

*Previous: [Chapter 14: TOON Encoding](14-toon-encoding.md)* | *Next: [Chapter 16: Graph-Based Reasoning](16-graph-reasoning.md)*
