# Handoff: Replay Evaluation Harness (Meta-Learned Memory Design)

- **Created**: 2026-02-13
- **Status**: IMPLEMENTATION COMPLETE (all 8 phases done, 75 tests passing)
- **Priority**: High
- **Blocked by**: None (all prerequisites — MemRL Phases 1-8, FAISS, QScorer — are complete)

---

## Context

ALMA (Xiong et al., Feb 2026) demonstrates meta-learned memory designs consistently outperform hand-crafted ones. Our orchestration stack has a sophisticated but **static** memory layer (EpisodicStore + TwoPhaseRetriever + QScorer). The replay harness is the **prerequisite** for:

- Using Claude as meta-agent to evolve memory configs (RetrievalConfig, ScoringConfig)
- Role-specific memory schemas (different configs per tier)
- Model swap resilience (warm-start protocol when swapping e.g. Qwen2.5-Coder-32B → Qwen3-Coder-30B)

Without a replay harness, we cannot compare design variants offline. With it, Claude can propose config mutations, evaluate them against ~2500 historical trajectories, and recommend promotions — all without touching production.

---

## Dependency Graph

```
Phase 1 (trajectory extraction)
    ↓
Phase 2 (replay engine)  ←── Phase 3 (metrics)
    ↓
Phase 4 (design archive)
    ↓
Phase 5 (warm_start)  [parallel with Phase 4]
    ↓
Phase 6 (meta_agent)  [depends on 1-4]
    ↓
Phase 7 (packaging)
    ↓
Phase 8 (verification)
```

---

## Phase Checklist

| Phase | Description | Status | Est. LOC |
|-------|-------------|--------|----------|
| 1 | Trajectory Extraction | ✅ DONE | 400-500 |
| 2 | Replay Engine | ✅ DONE | 350-450 |
| 3 | Replay Metrics | ✅ DONE | 150-200 |
| 4 | Design Candidates + Archive | ✅ DONE | 300-400 |
| 5 | Role Configs + Model Swap Warm-Start | ✅ DONE | 200-300 |
| 6 | Claude Meta-Agent Integration | ✅ DONE | 250-350 |
| 7 | Package Init | ✅ DONE | 20 |
| 8 | Verification + Gates | ✅ DONE | — |

**Grand total**: ~2800-3600 LOC across 15 new + 3 modified files (incl. tests)

---

## Phase 1: Trajectory Extraction

**File**: `orchestration/repl_memory/replay/trajectory.py`
**Tests**: `tests/unit/test_trajectory_extractor.py`

### Dataclass

```python
@dataclass
class Trajectory:
    task_id: str
    task_type: str                      # from ProgressEntry.data
    objective: str                      # from task_started data
    routing_decision: str               # from routing_decision event
    outcome: str                        # "success" | "failure" | "partial"
    cost_metrics: Dict[str, Any]        # token counts, tier costs
    escalations: List[str]              # escalation chain
    gate_results: List[Dict[str, Any]]  # gate pass/fail
    embedding: np.ndarray               # pre-computed 1024-dim
    started_at: datetime
    completed_at: datetime
```

### TrajectoryExtractor class

- Reads progress logs via `ProgressReader.read_recent(days)` (progress_logger.py:595)
- Groups events by task_id using `ProgressReader.get_task_trajectory(task_id)` (progress_logger.py:638)
- Builds `Trajectory` objects from complete event sequences
- Pre-computes embeddings via `TaskEmbedder.embed_text()` (embedder.py:369), caches as `{cache_dir}/embeddings.npz` (~10MB for 2500 trajectories)

### Design decisions

- **Single-pass grouping**: iterate all entries → `dict[task_id, list[ProgressEntry]]` → construct `Trajectory` per group
- **Completeness filter**: only trajectories with both `task_started` + `task_completed`/`task_failed` + `routing_decision` events → yields ~2500 from 25 days of logs
- **Default stratified sample of 1000**: proportional by task_type, reproducible via fixed seed. `--max-trajectories 1000` (default), `--max-trajectories 0` for all
- **Incomplete trajectories**: logged as warnings, excluded from replay
- **Embedding cache**: `np.savez_compressed` keyed by task_id; invalidated by date range change

### Modification needed

Add `ProgressReader.read_all(days: int = 31) -> List[ProgressEntry]` — just wraps `read_recent` with configurable range (trivial one-liner).

### Key interfaces (verified)

| Interface | Location | Signature |
|-----------|----------|-----------|
| `ProgressEntry` | progress_logger.py:87-106 | dataclass: event_type, task_id, timestamp, agent_tier, agent_role, data, memory_id, outcome, outcome_details |
| `ProgressReader.read_recent()` | progress_logger.py:595-605 | `(days: int = 7) -> List[ProgressEntry]` |
| `ProgressReader.get_task_trajectory()` | progress_logger.py:638-642 | `(task_id: str) -> List[ProgressEntry]` |
| `TaskEmbedder.embed_text()` | embedder.py:369-379 | `(text: str) -> np.ndarray` |
| `TaskEmbedder.embed_batch()` | embedder.py:427-440 | `(texts: List[str]) -> np.ndarray` |

### Resume commands

```bash
# Verify ProgressReader works
python3 -c "
from orchestration.repl_memory.progress_logger import ProgressReader
reader = ProgressReader()
entries = reader.read_recent(days=7)
print(f'{len(entries)} entries from last 7 days')
task_ids = set(e.task_id for e in entries if e.task_id)
print(f'{len(task_ids)} unique task IDs')
"

# After implementing Phase 1:
python3 -c "
from orchestration.repl_memory.replay.trajectory import TrajectoryExtractor
from orchestration.repl_memory.progress_logger import ProgressReader
t = TrajectoryExtractor(ProgressReader())
trajectories = t.extract_complete(days=7)
print(f'{len(trajectories)} complete trajectories')
"
```

---

## Phase 2: Replay Engine

**File**: `orchestration/repl_memory/replay/engine.py`
**Tests**: `tests/unit/test_replay_engine.py`

### Dataclass

```python
@dataclass
class ReplayStepResult:
    trajectory_id: str
    candidate_action: Optional[str]     # what candidate's config would have routed
    actual_action: str                   # what actually happened
    routing_match: bool                  # candidate == actual
    q_value_after: float                # Q-value after this step
    reward: float                       # computed reward
    escalation_predicted: bool          # candidate predicted escalation
```

### ReplayEngine

Takes `DesignCandidate` + `list[Trajectory]`, creates isolated store, replays chronologically:

```
for each trajectory (sorted by started_at):
    1. embed = trajectory.embedding (pre-computed, NO embedder calls)
    2. results = retriever.retrieve_for_routing(embed)  # candidate's config
    3. candidate_action = results[0].memory.action if results else None
    4. routing_match = (candidate_action == trajectory.routing_decision)
    5. reward = scorer._compute_reward(trajectory.outcome, ...)
    6. store.store(embed, trajectory.routing_decision, "routing", context, outcome, q=0.5+reward*0.5)
    7. if existing similar memory: store.update_q_value(memory_id, reward)
    8. collect ReplayStepResult
```

### Isolation

- Fresh `EpisodicStore(db_path=tmp_dir, embedding_dim=1024, use_faiss=True)` per candidate
- tmp_dir: `/mnt/raid0/llm/tmp/replay/{candidate_id}/` — cleaned up after run
- Deterministic: fixed trajectory order, no randomness in retriever/scorer
- **No graph integration in v1** — FailureGraph/HypothesisGraph deferred (Kuzu per candidate too expensive)

### NullEmbedder safety guard

Replay engine does NOT call the embedder — uses pre-computed embeddings from Phase 1. Pass a `NullEmbedder` stub that raises `RuntimeError("Replay engine must use pre-computed embeddings")` if called.

### Key interfaces (verified)

| Interface | Location | Signature |
|-----------|----------|-----------|
| `EpisodicStore.__init__()` | episodic_store.py:109-144 | `(db_path, embeddings_path, embedding_dim=1024, use_faiss=True, flush_interval=10.0)` |
| `EpisodicStore.store()` | episodic_store.py:212-267 | `(embedding, action, action_type, context, outcome=None, initial_q=0.5) -> str` |
| `TwoPhaseRetriever.__init__()` | retriever.py:85-94 | `(store, embedder, config=None)` |
| `QScorer._compute_reward()` | q_scorer.py:271-364 | `(task_outcome, gate_results, escalations, plan_reviews=None, cost_metrics=None) -> float` |

### Resume commands

```bash
# After implementing Phase 2:
pytest tests/unit/test_replay_engine.py -v

# Smoke test with 10 trajectories:
python3 -c "
from orchestration.repl_memory.replay.engine import ReplayEngine
from orchestration.repl_memory.replay.trajectory import TrajectoryExtractor
from orchestration.repl_memory.replay.candidates import DesignCandidate
from orchestration.repl_memory.progress_logger import ProgressReader
t = TrajectoryExtractor(ProgressReader())
trajectories = t.extract_complete(days=7)[:10]
engine = ReplayEngine()
results = engine.run(DesignCandidate.default(), trajectories)
print(f'Routing accuracy: {sum(r.routing_match for r in results)/len(results):.1%}')
"
```

---

## Phase 3: Replay Metrics

**File**: `orchestration/repl_memory/replay/metrics.py`
**Tests**: `tests/unit/test_replay_metrics.py`

### ReplayMetrics dataclass

```python
@dataclass
class ReplayMetrics:
    candidate_id: str
    num_trajectories: int
    num_complete: int
    routing_accuracy: float              # % match with successful actual route
    routing_accuracy_by_type: Dict[str, float]  # per task_type
    escalation_precision: float          # low-confidence predicted actual escalation
    escalation_recall: float             # fraction of actual escalations predicted
    q_convergence_step: int              # step where Q std < 0.05
    cumulative_reward: float
    avg_reward: float
    cost_efficiency: float               # reward / weighted tier cost
    tier_usage: Dict[str, int]
    replay_duration_seconds: float
```

### Methods

- `to_dict() -> dict` / `from_dict(d: dict) -> ReplayMetrics` — JSON round-trip for archive storage
- `compare(baseline: ReplayMetrics) -> dict` — per-metric deltas and % change

---

## Phase 4: Design Candidates + Archive

**File**: `orchestration/repl_memory/replay/candidates.py`
**Tests**: `tests/unit/test_design_archive.py`

### DesignCandidate dataclass

```python
@dataclass
class DesignCandidate:
    candidate_id: str                    # UUID
    parent_id: Optional[str]             # lineage
    retrieval_config: RetrievalConfig    # from retriever.py:47-64
    scoring_config: ScoringConfig        # from q_scorer.py:34-112
    staged_config: Optional[StagedConfig]
    role_overrides: Optional[Dict[str, Dict]]  # per-role config overrides (Phase 5)
    notes: str
    created_at: datetime
```

- `default()` classmethod: returns current production config as baseline candidate
- `to_json()` / `from_json()` for serialization

### DesignArchive (SQLite)

- **Location**: `/mnt/raid0/llm/claude/orchestration/repl_memory/meta_archive/archive.db`
- Schema: `candidates(id TEXT PK, config_json TEXT, metrics_json TEXT, created_at TEXT, parent_id TEXT, notes TEXT)`
- Methods:
  - `store_result(candidate, metrics)` — upsert candidate + metrics
  - `get_top_candidates(metric="cumulative_reward", limit=10)` — ranked query
  - `get_lineage(candidate_id)` — ancestor chain via parent_id
  - `sample_for_reflection(n=5)` — top 2 + worst 1 + 2 random (diverse sampling for Claude)
  - `get_baseline()` — production default candidate metrics (or None if never run)

### Key interfaces (verified)

| Interface | Location | Fields |
|-----------|----------|--------|
| `RetrievalConfig` | retriever.py:47-64 | semantic_k=20, min_similarity=0.3, min_q_value=0.3, q_weight=0.7, top_n=5, confidence_threshold=0.6 |
| `ScoringConfig` | q_scorer.py:34-112 | 17 fields incl. learning_rate=0.1, success_reward=1.0, failure_reward=-0.5, cost_penalty_lambda=0.15, baseline_tps_by_role, baseline_quality_by_role, memory_cost_by_role |

### Resume commands

```bash
# After implementing Phase 4:
pytest tests/unit/test_design_archive.py -v

# Archive round-trip test:
python3 -c "
from orchestration.repl_memory.replay.candidates import DesignCandidate, DesignArchive
archive = DesignArchive()
baseline = DesignCandidate.default()
print(f'Baseline candidate: {baseline.candidate_id}')
print(f'RetrievalConfig: semantic_k={baseline.retrieval_config.semantic_k}')
"
```

---

## Phase 5: Role-Specific Configs + Model Swap Resilience

**File**: `orchestration/repl_memory/replay/warm_start.py`
**Tests**: `tests/unit/test_warm_start.py`

### MemoryEntry modification — model_id field

Add `model_id: Optional[str] = None` to `MemoryEntry` in episodic_store.py:43-87.

**Schema migration** (backward-compatible):
```sql
ALTER TABLE memories ADD COLUMN model_id TEXT;
```
In `_init_db()` (episodic_store.py:146-182) — SQLite ADD COLUMN is safe, default NULL, no migration tool needed. Add `model_id` to `to_dict()` / `from_dict()` (default None).

### When model_id gets SET (write path)

- **`EpisodicStore.store()`**: add optional `model_id` param; caller passes current model ID from `model_registry.yaml` server_mode entry
- **`QScorer._update_routing_memory()`**: passes `model_id` from `ProgressEntry.data["model_id"]`
- **`ProgressLogger.log_routing_decision()`**: add `model_id` to data payload
- **Backfill**: existing memories get `model_id=NULL` (treated as "unknown/legacy" — no affinity bonus, no penalty)

### How model_id is USED (read path)

In `TwoPhaseRetriever` Phase 2 scoring (extends existing `CACHE_AFFINITY_BONUS` at retriever.py:81-83):

```python
if memory.model_id == current_model_id:
    combined_score *= 1.15   # Same model → trust this experience more
elif memory.model_id is not None:
    combined_score *= 0.90   # Different model → discount (not discard)
# model_id=None → no adjustment (legacy memories)
```

### RoleConfig dataclass

```python
@dataclass
class RoleConfig:
    role: str
    model_id: str                       # e.g. "qwen2.5-coder-32b-q4km"
    retrieval_config: RetrievalConfig   # role-specific retrieval params
    scoring_config: ScoringConfig       # role-specific scoring params
```

Loaded from `model_registry.yaml` under each server_mode entry. Falls back to global defaults if no per-role override exists.

### WarmStartProtocol

- `detect_model_swap(role, current_model_id, store)` — queries memories for role, checks if majority have different model_id
- `execute_warm_start(role, new_model_id, store)`:
  1. Reset Q-values to 0.5 for memories where `context.role == role` AND `model_id != new_model_id`
  2. Update `model_id` to `new_model_id` on reset memories
  3. Set `hypothesis.tested = False` for role-specific hypotheses in HypothesisGraph
  4. Keep FailureGraph intact (failure patterns often transfer)
  5. Double `learning_rate` for this role's ScoringConfig during warmup (first 50 tasks)
  6. Return stats: `{memories_reset, hypotheses_invalidated, warmup_tasks_remaining}`
- `is_warmup_active(role, store)` → True if fewer than 50 tasks scored since last warm-start

### Resume commands

```bash
# After implementing Phase 5:
pytest tests/unit/test_warm_start.py -v

# Warm-start smoke test:
python3 -c "
from orchestration.repl_memory.replay.warm_start import WarmStartProtocol
from orchestration.repl_memory.episodic_store import EpisodicStore
store = EpisodicStore()
stats = store.get_stats()
print(f'Total memories: {stats}')
swap = WarmStartProtocol.detect_model_swap('coder', 'qwen2.5-coder-32b-q4km', store)
print(f'Model swap detected: {swap}')
"
```

---

## Phase 6: Claude Meta-Agent Integration

**File**: `orchestration/repl_memory/replay/meta_agent.py`
**Prompt**: `orchestration/prompts/meta_agent_reflect.md`

### MetaAgentWorkflow

- `build_reflection_prompt()` — assembles:
  - Current production config (from `DesignCandidate.default()`)
  - Archive summary (top 5 + worst 2 + trends)
  - Recent trajectory stats (routing accuracy, escalation rate, failure types)
  - Request: propose 2-3 new `DesignCandidate` configs as JSON
- `parse_candidates(claude_response)` — extract JSON candidates, validate ranges, assign UUIDs + parent_ids
- `evaluate_candidates(candidates, trajectories)` — run replay engine per candidate (sequential), archive results
- `recommend_promotion(results)` → best candidate if beats baseline by >5%, else None
- `generate_report(results)` — markdown comparison table for human review

### Promotion pathway: **human-in-the-loop**

Meta-agent generates report + recommendation. Human reviews, approves, manually updates `model_registry.yaml` retrieval/scoring sections. No auto-promotion in v1 (safety).

### Dual interface: CLI + library

**CLI** (standalone/cron):
```bash
python3 -m orchestration.repl_memory.replay.meta_agent \
    --days 14 \
    --max-trajectories 1000 \
    --archive-path /mnt/raid0/llm/claude/orchestration/repl_memory/meta_archive/archive.db
```

**Library** (Claude Code sessions):
```python
from orchestration.repl_memory.replay.meta_agent import MetaAgentWorkflow
from orchestration.repl_memory.replay.candidates import DesignArchive
workflow = MetaAgentWorkflow(archive=DesignArchive(), ...)
prompt = workflow.build_reflection_prompt()     # Claude reads this
candidates = workflow.parse_candidates(response) # Claude's output parsed
results = workflow.evaluate_candidates(candidates, trajectories)
report = workflow.generate_report(results)       # Markdown comparison
```

CLI outputs: report to stdout, candidates + metrics to archive. Library mode: Claude calls methods directly, reflects on results in-session, proposes next iteration.

### Prompt template (`meta_agent_reflect.md`)

~50 lines. Provides:
- Current config snapshot
- Historical performance trends
- Constraint ranges (e.g. learning_rate ∈ [0.01, 0.5], q_weight ∈ [0.3, 1.0])
- Request: 2-3 candidates as JSON with `notes` explaining rationale

---

## Phase 7: Package Init

**Files**:
- `orchestration/repl_memory/replay/__init__.py` — exports TrajectoryExtractor, ReplayEngine, ReplayMetrics, DesignCandidate, DesignArchive, WarmStartProtocol, MetaAgentWorkflow
- Update `orchestration/repl_memory/__init__.py` — add replay subpackage imports

---

## Phase 8: Verification

### Unit tests

```bash
pytest tests/unit/test_trajectory_extractor.py \
       tests/unit/test_replay_engine.py \
       tests/unit/test_replay_metrics.py \
       tests/unit/test_design_archive.py \
       tests/unit/test_warm_start.py -v
```

### Smoke tests

```bash
# Trajectory extraction
python3 -c "
from orchestration.repl_memory.replay.trajectory import TrajectoryExtractor
from orchestration.repl_memory.progress_logger import ProgressReader
t = TrajectoryExtractor(ProgressReader())
print(len(t.extract_complete(days=7)))
"

# Replay baseline
python3 -c "
from orchestration.repl_memory.replay.engine import ReplayEngine
from orchestration.repl_memory.replay.trajectory import TrajectoryExtractor
from orchestration.repl_memory.replay.candidates import DesignCandidate
from orchestration.repl_memory.progress_logger import ProgressReader
t = TrajectoryExtractor(ProgressReader())
trajectories = t.extract_complete(days=7)
engine = ReplayEngine()
metrics = engine.run_with_metrics(DesignCandidate.default(), trajectories)
print(f'Routing accuracy: {metrics.routing_accuracy:.1%}')
print(f'Cumulative reward: {metrics.cumulative_reward:.2f}')
"

# Archive round-trip
python3 -c "
from orchestration.repl_memory.replay.candidates import DesignCandidate, DesignArchive
from orchestration.repl_memory.replay.metrics import ReplayMetrics
archive = DesignArchive()
c = DesignCandidate.default()
m = ReplayMetrics(candidate_id=c.candidate_id, num_trajectories=100, num_complete=95,
    routing_accuracy=0.75, routing_accuracy_by_type={}, escalation_precision=0.8,
    escalation_recall=0.6, q_convergence_step=50, cumulative_reward=42.0,
    avg_reward=0.42, cost_efficiency=0.85, tier_usage={}, replay_duration_seconds=12.3)
archive.store_result(c, m)
loaded = archive.get_top_candidates(limit=1)
assert loaded[0].candidate_id == c.candidate_id
print('Archive round-trip OK')
"

# Warm-start verification
python3 -c "
from orchestration.repl_memory.replay.warm_start import WarmStartProtocol
from orchestration.repl_memory.episodic_store import EpisodicStore
import tempfile, numpy as np
from pathlib import Path
tmp = Path('/mnt/raid0/llm/tmp/test_warm_start')
tmp.mkdir(exist_ok=True)
store = EpisodicStore(db_path=tmp / 'test.db', embedding_dim=1024, use_faiss=True)
emb = np.random.randn(1024).astype(np.float32)
store.store(emb, 'coder', 'routing', {'role': 'coder'}, 'success', initial_q=0.8)
swap = WarmStartProtocol.detect_model_swap('coder', 'new-model-id', store)
print(f'Model swap detected: {swap}')
"
```

### Gates

```bash
cd /mnt/raid0/llm/claude && make gates
```

---

## Files Summary

### New files (10 production + 5 test)

| File | LOC est. |
|------|----------|
| `orchestration/repl_memory/replay/__init__.py` | 20 |
| `orchestration/repl_memory/replay/trajectory.py` | 400-500 |
| `orchestration/repl_memory/replay/engine.py` | 350-450 |
| `orchestration/repl_memory/replay/metrics.py` | 150-200 |
| `orchestration/repl_memory/replay/candidates.py` | 300-400 |
| `orchestration/repl_memory/replay/warm_start.py` | 200-300 |
| `orchestration/repl_memory/replay/meta_agent.py` | 250-350 |
| `orchestration/prompts/meta_agent_reflect.md` | 50 |
| `tests/unit/test_trajectory_extractor.py` | 200-300 |
| `tests/unit/test_replay_engine.py` | 300-400 |
| `tests/unit/test_replay_metrics.py` | 100-150 |
| `tests/unit/test_design_archive.py` | 200-250 |
| `tests/unit/test_warm_start.py` | 150-200 |

### Modified files (3)

| File | Change |
|------|--------|
| `orchestration/repl_memory/episodic_store.py:43-87` | Add `model_id: Optional[str] = None` to MemoryEntry + ALTER TABLE in `_init_db()` |
| `orchestration/repl_memory/progress_logger.py:595+` | Add `read_all(days=31)` to ProgressReader |
| `orchestration/BLOCKED_TASKS.md` | Add replay harness entry |

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Progress log format drift | Medium | TrajectoryExtractor skips malformed entries (existing ProgressReader pattern) |
| Embedding server down during pre-compute | Medium | TaskEmbedder has hash-based fallback; pre-compute is one-time |
| OOM on full trajectory set | Low | ~2500 trajectories x 1024-dim x 4B = ~10MB; isolated store is tiny |
| model_id ALTER TABLE on live DB | Low | SQLite ADD COLUMN is safe, default NULL, no migration needed |
| Replay not representative (offline ≠ online) | Medium | Accepted limitation; replay tests config sensitivity, not absolute performance |

---

## Completion Checklist

- [x] All 8 phases implemented
- [x] All 5 test files passing (75/75 tests green)
- [x] Full unit suite passing (3386 passed, 53 skipped, 0 failed)
- [ ] Smoke tests from Phase 8 passing (requires live orchestration, not seeding)
- [x] `make gates` passes (shellcheck fix applied, shfmt + mdlint pass)
- [x] Baseline replay run completed (1000 trajectories, 0.18s, routing 0% expected with mock data)
- [ ] Archive has at least one stored result (pending live data)
- [x] Technical findings → `docs/chapters/15-memrl-system.md` (Replay Evaluation Harness section)
- [ ] Key metrics → `docs/reference/benchmarks/RESULTS.md` (pending meaningful routing accuracy data)
- [x] Progress summary → `progress/2026-02/2026-02-13.md`
- [ ] This handoff file deleted from `handoffs/active/`
- [x] `BLOCKED_TASKS.md` updated (IMPLEMENTATION COMPLETE)
