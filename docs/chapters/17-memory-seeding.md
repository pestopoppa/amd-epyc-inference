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
2. Generates embeddings using `TaskEmbedder` (Qwen2.5-Coder-0.5B)
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

## References

- **Seed loader**: `orchestration/repl_memory/seed_loader.py`
- **Canonical examples**: `orchestration/repl_memory/seed_examples.json`
- **Graph seeds**: `orchestration/repl_memory/graph_seeds.yaml`
- **Seeding scripts**: `scripts/seed_*.py` (9 scripts)
- **EpisodicStore**: `orchestration/repl_memory/episodic_store.py`

---

*Previous: [Chapter 16: Graph-Based Reasoning](16-graph-reasoning.md)* | *Next: [Chapter 18: Escalation & Routing](18-escalation-and-routing.md)*
