# Chapter 17: Memory Seeding & Bootstrap

## Introduction

The MemRL system requires bootstrap data to function effectively. Without seed memories, the episodic store is empty and retrieval returns nothing. This chapter covers the seeding infrastructure that provides canonical examples, diverse exploration patterns, and graph-backed failure/hypothesis knowledge.

**Seeding philosophy:** Provide high-quality, high-Q-value canonical examples that enable immediate retrieval on common tasks, while also seeding diverse exploration patterns to prevent overfitting to simple cases.

## Seed Loader Architecture

### Core Components

| Component | Purpose | Location |
|-----------|---------|----------|
| `seed_loader.py` | Main seeding script | `orchestration/repl_memory/seed_loader.py` |
| `seed_examples.json` | 56 canonical REPL examples | `orchestration/repl_memory/seed_examples.json` |
| `graph_seeds.yaml` | Failure modes & hypotheses | `orchestration/repl_memory/graph_seeds.yaml` |
| Seeding scripts | Diverse seeding strategies | `scripts/seed_*.py` (9 scripts) |

### Seed Examples JSON Format

Each canonical example has the following structure:

```json
{
  "task": "List files in a directory",
  "code": "result = list_dir('/path/to/dir')\nFINAL(result)",
  "tools_used": ["list_dir", "FINAL"],
  "category": "filesystem"
}
```

Categories include:
- `filesystem` - Directory listing, file info, peeking
- `search` - Grep patterns, function definitions, TODO comments
- `document` - OCR, PDF extraction, figure extraction
- `analysis` - Data parsing, log analysis
- `computation` - Math, statistics, aggregations

### Loading Canonical Seeds

**Basic usage:**

```bash
# First-time seeding
python orchestration/repl_memory/seed_loader.py

# Force reload (clears existing memories)
python orchestration/repl_memory/seed_loader.py --force
```

**What it does:**

1. Loads 56 examples from `seed_examples.json`
2. Generates embeddings using `TaskEmbedder` (BGE-large, 1024-dim)
3. Stores in episodic memory with:
   - `action`: The code snippet
   - `action_type`: "exploration"
   - `outcome`: "success"
   - `initial_q`: 0.9 (high Q-value for canonical examples)
   - `context`: `{"is_seed": True, "category": "...", "tools_used": [...]}`

**Output:**

```
Loading 56 seed examples...
  Loaded 10/56 examples...
  Loaded 20/56 examples...
  ...
Seeding complete!
  Loaded: 56
  Failed: 0
  By category: {'filesystem': 20, 'search': 12, 'document': 15, 'analysis': 6, 'computation': 3}

Memory stats:
  Total memories: 56
  FAISS embeddings: 56
  Average Q-value: 0.90
```

## Seeding Strategies

The system provides 9 specialized seeding scripts, each with a distinct strategy:

### 1. Diverse Memories (`seed_diverse_memories.py`)

**Purpose:** Prevent overfitting to simple cases by seeding complex multi-step tasks.

**Strategy:**
- Generate 100-500 memories with increasing complexity
- Combine multiple tools in sequence
- Include conditional logic and error handling
- Q-values: 0.7-0.9 (high but not canonical)

**Example generated task:**

```python
# Complex multi-step: OCR → grep → analysis
doc = json.loads(ocr_document('/path/to/paper.pdf'))
matches = grep('algorithm', file_path='/tmp/extracted.txt')
result = analyze_pattern(matches)
FINAL(result)
```

### 2. Failure Memories (`seed_failure_memories.py`)

**Purpose:** Seed common failure patterns with Q-values < 0.5 to teach the system what NOT to do.

**Strategy:**
- Generate 50-100 failed attempts
- Include common mistakes (wrong tool, missing imports, path errors)
- Outcome: "failure"
- Q-values: 0.1-0.4 (low, to discourage repetition)

**Example failure:**

```python
# BAD: Uses Python imports instead of REPL tools
import os
files = os.listdir('/tmp')  # Will trigger security error
FINAL(files)
```

### 3. Diverse Failures (`seed_diverse_failures.py`)

**Purpose:** Seed complex failure chains and recovery patterns.

**Strategy:**
- Generate multi-step failures (A fails → B attempted → B also fails)
- Include partial successes (step 1 works, step 2 fails)
- Link failures to graph via `failure_graph.record_failure()`

### 4. Probabilistic Memories (`seed_probabilistic_memories.py`)

**Purpose:** Seed exploration with randomized Q-values to model uncertainty.

**Strategy:**
- Same tasks as canonical examples
- Q-values drawn from Beta(2, 5) distribution (mean ~0.3, variance)
- Outcome randomized: 70% success, 30% failure

**Why this matters:** Prevents overconfidence in untested scenarios.

### 5. Decomposition Memories (`seed_decomposition_memories.py`)

**Purpose:** Seed task decomposition patterns (high-level goal → sub-tasks).

**Strategy:**
- Generate 50 examples of task hierarchies
- Include planning, subtask execution, aggregation
- Q-values: 0.8 (high, decomposition is valuable)

**Example:**

```python
# High-level: Analyze all PDFs in directory
files = list_dir('/docs/')
results = []
for f in filter(lambda x: x.endswith('.pdf'), files):
    doc = ocr_document(f)
    results.append(summarize(doc))
FINAL(aggregate(results))
```

### 6. Memory from Logs (`seed_memory_from_logs.py`)

**Purpose:** Bootstrap from real agent activity logs.

**Strategy:**
- Parse `logs/agent_audit.log`
- Extract task-action pairs from successful executions
- Assign Q-values based on outcome (1.0 for success, 0.2 for failure)
- Filter out low-quality entries (truncated, error messages)

**Usage:**

```bash
python scripts/seed_memory_from_logs.py --log-file logs/agent_audit.log --min-quality 0.5
```

### 7. Success Patterns (`seed_success_patterns.py`)

**Purpose:** Seed known-good patterns from benchmark results.

**Strategy:**
- Extract successful action sequences from benchmark JSON
- Focus on high-scoring runs (Claude-as-Judge score ≥ 3)
- Assign Q-values based on benchmark score (0.8-1.0)

### 8. Graph Seeds (`seed_graphs.py`)

**Purpose:** Load failure modes and hypotheses into graph databases.

**Strategy:**
- Parse `graph_seeds.yaml`
- Create FailureMode, Symptom, Mitigation nodes
- Create Hypothesis nodes with initial confidence
- Link to episodic memory where applicable

**Usage:**

```bash
python scripts/seed_graphs.py --force
```

**Output:**

```
Loading failure modes...
  Created 14 failure modes
  Created 45 symptom patterns
  Created 16 mitigations

Loading hypotheses...
  Created 15 hypotheses
  Average initial confidence: 0.78

Graph stats:
  Failure graph: 14MB (75 nodes, 120 edges)
  Hypothesis graph: 4.6MB (15 nodes, 30 edges)
```

### 9. Remaining Phase B (`seed_remaining_phase_b.py`)

**Purpose:** Seed incomplete Phase B implementation tasks (specialist workflows).

**Strategy:**
- Generate placeholder memories for unimplemented features
- Q-values: 0.3 (uncertain, needs validation)
- Mark with `{"phase": "B", "status": "pending"}`

### 10. 3-Way Routing Evaluation (`seed_specialist_routing.py --3way`)

**Purpose:** Train frontdoor for faithful probability estimation via 3-way comparative testing.

**Strategy:**
- Runs each question through 4 configurations:
  - `SELF:direct` - Frontdoor, no tools
  - `SELF:repl` - Frontdoor/vision worker with tools, delegation disabled
  - `ARCHITECT` - Dual-architect evaluation (architect_general + architect_coding; best-of-two)
  - `WORKER` - Scored indirectly via delegation chains
- Binary rewards (1.0 for pass, 0.0 for fail)
- Cost metrics stored separately for later Optuna optimization
- Infrastructure errors (timeouts, connection failures) produce **no reward** — action is skipped and retried next batch
- For VL questions, `SELF:repl` is `worker_vision:repl` (legacy `worker_vision:react` is backward-compatible in historical reward parsing).

**Usage:**

```bash
# Full 3-way seeding run
python scripts/benchmark/seed_specialist_routing.py --3way --suites thinking coder --sample-size 20

# Dry run (no reward injection)
python scripts/benchmark/seed_specialist_routing.py --3way --dry-run --suites thinking --sample-size 5
```

**Key difference from comparative seeding:**
- Comparative seeding uses cost-weighted rewards
- 3-way seeding uses binary rewards for faithful P(success) estimation
- Cost is stored in metadata, not incorporated into Q-values

### Question Pool (Pre-extracted)

All ~53K questions from 18 HF dataset adapters + YAML suites are pre-extracted into `benchmarks/prompts/question_pool.jsonl`. Runtime sampling reads this file (~100ms) instead of loading 16 Arrow/Parquet datasets (~30s).

- **Sampling**: Full shuffle per suite, take first N unseen. Guarantees coverage of entire pool.
- **Seen tracking**: `benchmarks/results/eval/seen_questions.jsonl` — questions marked seen only when rewards are injected.
- **Debug mode** (`--debug`): When a suite is exhausted, backfills with seen questions (via `allow_reseen`). Normal mode skips exhausted suites.
- **Reset**: `scripts/session/reset_episodic_memory.sh` clears episodic DB + FAISS + seen set.
- **Rebuild**: `--rebuild-pool` re-extracts from all adapters.

### Claude-in-the-Loop Debugger

The `--debug` flag (requires `--3way`) enables automatic pipeline debugging via a persistent Claude Code session. See [Chapter 26](26-claude-debugger.md) for full documentation: 17 anomaly signals, hot-swap/code fixes, 3-phase regression suite (verify/generalize/regress), MemRL interaction (TD-learning on retried questions), auto-discovery of new failure patterns, and audit trail.

**Anomaly signals (17):** repetition_loop, comment_only, template_echo, self_doubt_loop, format_violation, think_tag_leak, near_empty, excessive_tokens, delegation_format_error, self_escalation, vision_blindness, silent_execution, repl_no_tools, slow_delegation, function_repr_leak, status_phrase_final, misrouted_to_coder.

**Auto-discovery:** The debugger instructs Claude to propose new anomaly detectors via structured `NEW_SIGNAL:` output. Proposals are persisted to `logs/proposed_signals.jsonl` for human review and optional inclusion in `anomaly.py`.

**Retry persistence:** Retry queue survives script crashes via JSONL persistence (`logs/retry_queue.jsonl`). Previous sessions' pending retries are loaded on startup.

```bash
# Live debugging (Claude analyzes every 5 answers)
python scripts/benchmark/seed_specialist_routing.py --3way --continuous --debug

# With auto-commit of debugger fixes
python scripts/benchmark/seed_specialist_routing.py --3way --continuous --debug --debug-auto-commit

# Dry run (log diagnostics without invoking Claude)
python scripts/benchmark/seed_specialist_routing.py --3way --debug --debug-dry-run
```

## Seeding Order & Dependencies

**Recommended seeding order:**

1. **Canonical examples first** - High-quality baseline
   ```bash
   python orchestration/repl_memory/seed_loader.py --force
   ```

2. **Graph seeds** - Failure/hypothesis knowledge
   ```bash
   python scripts/seed_graphs.py --force
   ```

3. **Diverse patterns** - Prevent overfitting
   ```bash
   python scripts/seed_diverse_memories.py --count 200
   ```

4. **Failure patterns** - Learn what NOT to do
   ```bash
   python scripts/seed_failure_memories.py --count 50
   python scripts/seed_diverse_failures.py --count 50
   ```

5. **Real logs** - Bootstrap from production
   ```bash
   python scripts/seed_memory_from_logs.py --min-quality 0.6
   ```

6. **Success patterns** - Benchmark-driven
   ```bash
   python scripts/seed_success_patterns.py --min-score 3
   ```

## Memory Distribution After Seeding

Typical distribution after full seeding:

| Source | Count | Avg Q-value | Purpose |
|--------|-------|-------------|---------|
| Canonical examples | 56 | 0.90 | High-confidence patterns |
| Diverse memories | 200 | 0.75 | Exploration variety |
| Failure memories | 100 | 0.25 | Anti-patterns |
| Log-based | 50-500 | 0.60 | Real usage patterns |
| Benchmark-driven | 100-200 | 0.85 | Proven solutions |
| **Total** | **500-1000** | **0.65** | Balanced coverage |

## Verification

**Check seeding status:**

```python
from orchestration.repl_memory.episodic_store import EpisodicStore

store = EpisodicStore()
stats = store.get_stats()

print(f"Total memories: {stats['total_memories']}")
print(f"Average Q-value: {stats['overall_avg_q']:.2f}")
print(f"Recent successes: {stats.get('recent_success_rate', 0):.0%}")
```

**Expected output:**

```
Total memories: 856
Average Q-value: 0.67
Recent successes: 78%
```

## 3-Way Action Keys (February 2026)

The 3-way evaluation mode uses a distinct action vocabulary:

| Action Key | What It Represents | Source Role | Mode |
|------------|-------------------|-------------|------|
| `SELF:direct` | Frontdoor without tools | frontdoor | direct |
| `SELF:repl` | Frontdoor with tools | frontdoor | repl |
| `ARCHITECT` | Architect with delegation | architect_general + architect_coding (best-of-two) | delegated |
| `WORKER` | Worker models | via delegation | — |

These action keys are stored in episodic memory and used for routing decisions. The HybridRouter's `route_3way()` method retrieves memories by these action keys.

## Infra Safeguards (2026-02-07)

Recent seeding regressions showed that a single stalled heavy-model request can
block the orchestrator event loop and cascade into 600s timeouts. The infra
plan now includes:

- **CPU-exclusive inference lock**: heavy models acquire an exclusive lock; workers/embedders acquire a shared lock and only run when no heavy model is active.
- **Async safety**: all blocking LLM calls are offloaded from the event loop.
- **3-way timeout cleanup**: slot erasure on infra timeouts to prevent stuck backends.
- **Backend probes in /health**: detect hung backends even when circuit state is stale.

## Architect Delegation in 3-Way Eval (2026-02-09)

The 3-way ARCHITECT evaluation runs `architect_general` and `architect_coding` in
delegated mode. The architect decides via TOON whether to answer directly (`D|answer`)
or delegate to a specialist (`I|brief:<spec>|to:coder_escalation`).

**Known issue (fixed):** The original architect prompt presented `D|` and `I|` as
side-by-side template examples. Qwen3-235B echoed both, causing `_extract_toon_decision`
to find `D|Answer` first and parse it as a direct answer. The delegation chain to
`coder_escalation` (port 8081, Qwen2.5-Coder-32B) was never exercised.

**Fix:** Prompt restructured as bullet-list alternatives with "EXACTLY ONE line" guard.
Architect now correctly delegates code tasks and provides architectural design briefs
(approach, data structures, algorithm, complexity) for the coding specialist.

**Slot-erase for stuck backends:** When the seeding script's HTTP client times out, the
llama-server may still be generating. `_erase_slots(port)` sends
`POST /slots/{id}?action=erase` to cancel in-progress inference. If the server is stuck
in prompt eval, the erase request itself may hang. The `_SLOT_ERASE_CAPABILITY` cache
tracks which ports support slot erasure and disables erase attempts on ports that return
404/405/501.

## Timeout + Telemetry Updates (2026-02-08)

- **Adaptive per-call timeout budget** in 3-way eval: timeout is selected by role/mode/modality and capped by CLI `--timeout` (hard ceiling). This reduces worst-case stall wait while preserving headroom for slow architect paths.
- **Observed-runtime timeout bumping**: REPL and architect calls can be raised using earlier per-question observed latency (direct/repl), reducing false `INFRA` on hard long-generation tasks while still respecting the hard ceiling.
- **Structured error normalization**: benchmark caller now maps `error_code`/`error_detail` into a unified `error` field.
- **Telemetry consistency invariant**: `tools_used`, `tools_called`, and `tool_timings` are normalized and kept internally consistent for debugging and post-hoc analysis.
- **Slot-erase capability guard**: 3-way cleanup now detects unsupported `/slots/{id}?action=erase` behavior on llama-server builds and disables repeated failing erase attempts instead of logging false success.
- **Live slot progress polling (2026-02-09)**: forced 3-way calls poll backend `/slots` during execution and emit `[slot-progress]` logs with task id + decoded token counters.
- **INFRA token estimate (2026-02-09)**: when API returns `0 tok` under timeout/disconnect, seeding records `tokens_generated_estimate` from slot counters and surfaces it in logs (`0 tok, est N tok`).

## References

- **Seed loader**: `orchestration/repl_memory/seed_loader.py`
- **Canonical examples**: `orchestration/repl_memory/seed_examples.json`
- **Graph seeds**: `orchestration/repl_memory/graph_seeds.yaml`
- **Seeding scripts**: `scripts/seed_*.py` (9 scripts)
- **3-way seeding**: `scripts/benchmark/seed_specialist_routing.py --3way`
- **Seeding types**: `scripts/benchmark/seeding_types.py` (action keys, cost tiers)
- **Seeding rewards**: `scripts/benchmark/seeding_rewards.py` (binary rewards)
- **EpisodicStore**: `orchestration/repl_memory/episodic_store.py`

---

*Previous: [Chapter 16: Graph-Based Reasoning](16-graph-reasoning.md)* | *Next: [Chapter 18: Escalation & Routing](18-escalation-and-routing.md)*
