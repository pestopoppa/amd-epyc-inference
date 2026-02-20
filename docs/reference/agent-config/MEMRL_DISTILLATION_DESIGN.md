# MemRL Routing Distillation Architecture

**Status**: DESIGN (no code yet)
**Created**: 2026-02-20
**Motivation**: ColBERT-Zero (arXiv:2602.16609) — supervised fine-tuning before distillation bridges performance gap
**Handoff**: `handoffs/active/colbert-zero-research-integration.md` (Track 2)

---

## Problem

The current routing pipeline (`TwoPhaseRetriever`) performs:

1. FAISS cosine search over ~500K episodic entries (1024-dim BGE embeddings)
2. Q-weighted cost-aware ranking with 7-dimensional reward signal
3. Prior injection + graph adjustments (FailureGraph, HypothesisGraph)
4. Confidence thresholding for go/no-go

This works but is heavy for routine decisions. Many routing choices are highly predictable from surface features alone (task_type + a few keywords → action). A small classifier could handle the easy 60-70% of requests in <1ms, with the full retrieval pipeline as fallback.

## Proposed Architecture

### 3-Stage Pipeline (Mirrors ColBERT-Zero)

```
Stage 1: Unsupervised         Stage 2: Supervised           Stage 3: Distillation
─────────────────────────    ──────────────────────────    ─────────────────────────
Episodic store entries        (task, best_action) pairs     HybridRouter decisions
→ contrastive task            weighted by Q-value           → compressed classifier
  embedding learning          → supervised classifier       → fast first-pass router
```

### Why 3 Stages (Not Direct Distillation)

Our current `DistillationPipeline` goes directly from raw trajectories to skills, skipping supervised fine-tuning. The ColBERT-Zero finding: the supervised stage anchors representations before compression, bridging a measurable performance gap. Applying this:

- **Stage 1** teaches the model what tasks look like (clustering)
- **Stage 2** teaches it what good routing decisions look like (discrimination)
- **Stage 3** compresses the full pipeline into a fast approximation

---

## Stage 1: Task Embedding Learning (Unsupervised)

**Goal**: Learn a compact task representation from episodic store entries.

### Training Data Extraction

```sql
-- Extract (task_type, objective, action, outcome) tuples
SELECT
    json_extract(context, '$.task_type') AS task_type,
    json_extract(context, '$.objective') AS objective,
    action,
    outcome,
    q_value
FROM memories
WHERE action_type = 'routing'
AND outcome IS NOT NULL
ORDER BY created_at DESC
LIMIT 100000
```

### Feature Engineering

| Feature | Type | Source | Encoding |
|---------|------|--------|----------|
| `task_type` | categorical | `context.task_type` | One-hot (12 categories) |
| `objective_keywords` | multi-hot | Top-100 TF-IDF terms from `context.objective` | Binary vector |
| `context_length` | float | `len(context.objective)` | Log-scaled, normalized |
| `has_images` | bool | `context.has_images` | Binary |
| `has_code` | bool | Regex detect in objective | Binary |
| `has_tool_request` | bool | Tool keywords in objective | Binary |
| `prior_action_count` | int | `count_by_combo(action)` per action | Log-scaled |

Total feature vector: ~120 dimensions (12 task_type + 100 keywords + 8 numeric).

### Method

Contrastive learning on (task_A, task_B) pairs:

- **Positive pairs**: Same `action` + same `outcome` (both successful routing to same role)
- **Negative pairs**: Different `action` or different `outcome`
- **Loss**: InfoNCE with temperature 0.07

Output: Task encoder producing 32-dim compact embeddings (down from 1024-dim BGE).

---

## Stage 2: Supervised Routing Classifier

**Goal**: Train a classifier on (task_features, best_action) pairs weighted by Q-value.

### Training Data

```sql
-- Best action per task_type with Q-value weighting
SELECT
    json_extract(context, '$.task_type') AS task_type,
    json_extract(context, '$.objective') AS objective,
    action,
    q_value,
    outcome
FROM memories
WHERE action_type = 'routing'
AND outcome = 'success'
AND q_value > 0.6
ORDER BY q_value DESC
```

### Hard Negative Mining

For each positive (task, best_action) pair, sample hard negatives:

```sql
-- Hard negatives: same task_type, different action, lower Q
SELECT action, q_value
FROM memories
WHERE action_type = 'routing'
AND json_extract(context, '$.task_type') = ?
AND action != ?
AND outcome = 'failure'
ORDER BY q_value ASC
LIMIT 3
```

### Model Architecture

```
Input: task_features (120-dim)
  → Linear(120, 64) + ReLU + Dropout(0.1)
  → Linear(64, 32) + ReLU
  → Linear(32, N_actions) + Softmax

N_actions ≈ 9 (frontdoor:direct, frontdoor:repl, coder_escalation,
               architect_general, architect_coding, worker_explore,
               worker_math, worker_vision, ingest_long_context)
```

Total parameters: ~120×64 + 64×32 + 32×9 = **10,016** (~40KB).

### Loss Function

```python
loss = CrossEntropy(pred, target)
     + λ_q * Q_weight * CrossEntropy(pred, target)   # Q-value importance weighting
     + λ_cost * CostPenalty(pred, role_costs)          # memory tier penalty
```

Where:
- `Q_weight = q_value / max_q` (normalized importance, range [0,1])
- `CostPenalty = sum(pred_prob[a] * memory_cost_by_role[a])` (penalize expensive routes)
- `λ_q = 0.5`, `λ_cost = 0.1` (hyperparameters to tune)

### Constants from Q-Scorer

```python
memory_cost_by_role = {
    "frontdoor":           1.0,   # 19GB HOT
    "coder_escalation":    1.05,  # 20GB HOT
    "worker_explore":      0.5,   # 4.4GB HOT
    "worker_math":         0.5,   # 4.4GB HOT
    "architect_general":   3.0,   # 133GB WARM
    "architect_coding":    5.0,   # 271GB WARM
    "ingest_long_context": 1.5,   # 46GB WARM
}
```

---

## Stage 3: HybridRouter Distillation

**Goal**: Compress the full `TwoPhaseRetriever` decision pipeline into the Stage 2 classifier.

### Teacher Signal

For each routing request, record the full pipeline output:

```python
# Teacher: run full TwoPhaseRetriever pipeline
results = retriever.retrieve_for_routing(task_ir)
teacher_probs = softmax([r.posterior_score for r in results])
teacher_action = results[0].memory.action
teacher_confidence = results[0].q_confidence
```

### Knowledge Distillation Loss

```python
loss_kd = KL_div(student_logits / T, teacher_probs / T) * T²
loss_hard = CrossEntropy(student_logits, teacher_action)
loss = α * loss_kd + (1 - α) * loss_hard
```

Where `T = 2.0` (temperature), `α = 0.7` (KD weight).

### Confidence Calibration

The classifier must output calibrated confidence scores for the fallback decision:

```python
if classifier_confidence >= 0.6:
    return classifier_action        # fast path (~0.1ms)
else:
    return retriever.retrieve_for_routing(task_ir)  # full pipeline (~2-5ms)
```

Calibrate via temperature scaling on held-out validation set.

---

## Integration Point

### In `HybridRouter.route()`

```python
class HybridRouter:
    def route(self, task_ir: Dict) -> RoutingDecision:
        # Stage 1: Fast classifier (if loaded)
        if self.fast_classifier:
            features = self._extract_features(task_ir)
            action, confidence = self.fast_classifier.predict(features)
            if confidence >= self.fast_confidence_threshold:  # default 0.6
                return RoutingDecision(
                    action=action,
                    confidence=confidence,
                    source="fast_classifier",
                )

        # Stage 2: Full retrieval pipeline (fallback)
        results = self.retriever.retrieve_for_routing(task_ir)
        # ... existing logic ...
```

### Feature Extraction Helper

```python
def _extract_features(self, task_ir: Dict) -> np.ndarray:
    task_type = task_ir.get("task_type", "unknown")
    objective = task_ir.get("objective", "")

    # One-hot task_type (12 categories)
    type_vec = np.zeros(12)
    type_idx = TASK_TYPE_INDEX.get(task_type, 11)  # 11 = "other"
    type_vec[type_idx] = 1.0

    # TF-IDF keyword features (100-dim, pre-fitted vocabulary)
    kw_vec = self._tfidf.transform([objective]).toarray()[0]

    # Numeric features
    num_vec = np.array([
        np.log1p(len(objective)),           # context_length
        float("image" in objective.lower()), # has_images
        float(bool(CODE_PATTERN.search(objective))),  # has_code
        float(bool(TOOL_PATTERN.search(objective))),  # has_tool_request
    ])

    return np.concatenate([type_vec, kw_vec[:100], num_vec])
```

---

## Evaluation Protocol

### Offline (Replay Harness)

Use existing `replay/engine.py` infrastructure:

1. Extract 10K held-out trajectories (stratified by task_type)
2. For each trajectory, compare:
   - `classifier_action` vs `retriever_action` vs `actual_best_action`
3. Metrics:
   - **Routing accuracy**: % match with retriever's top choice
   - **Q-regret**: mean Q-value difference between classifier choice and optimal
   - **Coverage**: % of requests where classifier confidence >= 0.6
   - **Latency**: classifier inference time (target: <0.5ms)

### Online (Shadow Mode)

1. Run classifier in shadow alongside full pipeline
2. Log disagreements: `(task_ir, classifier_action, retriever_action, classifier_confidence)`
3. Analyze disagreement patterns to identify systematic errors
4. Promote to production when:
   - Routing accuracy >= 90% on shadow traffic
   - Coverage >= 60% (most requests handled by fast path)
   - No Q-regret > 0.1 on average

---

## Data Requirements

| Metric | Current | Needed |
|--------|---------|--------|
| Episodic entries | ~500K | ~500K (sufficient) |
| Routing entries with outcome | ~350K (est.) | ~100K minimum |
| Distinct task_types | ~12 | ~12 (sufficient) |
| Distinct actions | ~9 | ~9 (sufficient) |
| Q-value distribution | ~0.5 mean, [0,1] | Need sufficient high-Q (>0.8) examples |

### Data Quality Checks

```sql
-- Verify sufficient high-Q training data per action
SELECT action, COUNT(*) as n, AVG(q_value) as mean_q
FROM memories
WHERE action_type = 'routing'
AND outcome = 'success'
AND q_value > 0.6
GROUP BY action
HAVING n >= 100
ORDER BY mean_q DESC
```

Minimum: 100 high-Q examples per action class for supervised training.

---

## Implementation Phases

| Phase | Description | Effort | Dependencies |
|-------|-------------|--------|--------------|
| 1 | This design document | Done | ColBERT-Zero review |
| 2 | Training data extraction script | Small | None |
| 3 | Stage 2 classifier prototype (skip Stage 1) | Medium | Phase 2 |
| 4 | Shadow mode integration in HybridRouter | Medium | Phase 3 |
| 5 | Stage 1 contrastive pre-training | Medium | Phase 3 results |
| 6 | Stage 3 KD from full pipeline | Medium | Phase 4 evidence |
| 7 | Production promotion | Small | Phase 6 evaluation |

**Shortcut**: Phase 3 can start without Phase 5 (Stage 1) — train directly on features. Add contrastive pre-training later if feature-only accuracy is insufficient.

---

## Files Reference

| File | Role in Architecture |
|------|---------------------|
| `orchestration/repl_memory/episodic_store.py` | Training data source (SQLite + FAISS) |
| `orchestration/repl_memory/q_scorer.py` | Reward signals, cost baselines |
| `orchestration/repl_memory/retriever.py` | Teacher model (TwoPhaseRetriever) |
| `orchestration/repl_memory/failure_graph.py` | Anti-memory features (optional) |
| `orchestration/repl_memory/distillation/pipeline.py` | Existing distillation (to extend) |
| `orchestration/repl_memory/skill_evolution.py` | OutcomeTracker for online feedback |
