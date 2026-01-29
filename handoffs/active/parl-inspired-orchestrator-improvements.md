# PARL-Inspired Orchestrator Improvements

**Goal**: Integrate learnings from Kimi K2.5's Agent Swarm (PARL) into our hierarchical orchestrator — parallel TaskIR execution, critical path metrics, persona registry with MemRL-guided selection, staged reward shaping, and parallel gates.
**Status**: 📋 NEW
**Date**: 2026-01-29
**Priority**: MEDIUM
**Research**: `research/kimi_k25_agent_swarm_analysis.md`

---

## Quick Start / Resume Commands

```bash
# 1. Verify test baseline
cd /mnt/raid0/llm/claude && timeout 120 python3 -m pytest tests/ -x -q

# 2. Read the research analysis for context
cat research/kimi_k25_agent_swarm_analysis.md

# 3. Key files to understand first
cat orchestration/task_ir.schema.json     # TaskIR schema (already has parallel_group!)
cat src/api.py                            # Root LM loop (where parallel execution goes)
cat src/llm_primitives.py                 # llm_call/llm_batch (persona injection point)
cat src/repl_environment.py               # REPL sandbox (delegate, llm_call bindings)
cat orchestration/model_registry.yaml     # Role configs (system_prompt_suffix exists)

# 4. Start on Phase 1 (parallelism) or Phase 3 (personas) — they're independent
```

---

## Implementation Status

| Phase | Feature | Status | Dependencies |
|-------|---------|--------|--------------|
| 1 | Parallel TaskIR Execution | ❌ | None |
| 2 | Critical Path Metric | ❌ | Phase 1 (uses same timing data) |
| 3 | Persona Registry + MemRL | ❌ | None (independent) |
| 4 | Staged Reward Shaping | ❌ | Phase 3 (needs persona Q-values) |
| 5 | Parallel Gate Execution | ❌ | None (independent, profile first) |

**Phases 1, 3, and 5 are independent and can be worked in any order.**

---

## Context

Kimi K2.5's Agent Swarm uses Parallel-Agent Reinforcement Learning (PARL) to train an orchestrator that decomposes tasks into parallel subtasks executed by dynamically instantiated subagents. Key innovations:

- **PARL reward**: `Rt = lambda_aux(e) * r_parallel + (1 - lambda_aux(e)) * (I[success] * Q(tau))` — staged shaping prevents "serial collapse"
- **Dynamic agents**: Roles created on-the-fly (not predefined)
- **Critical Steps metric**: `CriticalSteps = Sum(S_main(t) + max(S_sub,i(t)))` — measures critical path, not total work
- **Scale**: Up to 100 subagents, 1,500 tool calls per task

Our constraints differ (single CPU machine, multiple specialized models, 2-4 practical workers), so we adapt these ideas rather than copy them. See `research/kimi_k25_agent_swarm_analysis.md` for the full comparison.

---

## Phase 1: Parallel TaskIR Execution

**Goal**: Make the Root LM loop execute independent TaskIR steps concurrently via `llm_batch()`.

### What Exists Already

The TaskIR schema (`orchestration/task_ir.schema.json:180-203`) already defines:

```json
// Per-step fields (lines 180-188):
"parallel_group": {
  "type": "string",
  "description": "Steps with same group can run concurrently"
},
"depends_on": {
  "type": "array",
  "items": { "type": "string", "pattern": "^S[0-9]+$" },
  "description": "Step IDs that must complete before this one"
}

// Plan-level fields (lines 196-203):
"parallelism": {
  "max_concurrent_steps": { "type": "integer", "minimum": 1 },
  "max_concurrent_workers": { "type": "integer", "minimum": 1 }
}
```

**But**: The execution layer (`src/api.py`) currently processes steps sequentially. The schema fields are inert.

### What to Build

**1. Parallel Step Executor** (in `src/api.py` or new `src/parallel_executor.py`)

```python
from collections import defaultdict
import asyncio

class ParallelStepExecutor:
    """Execute TaskIR steps respecting depends_on and parallel_group."""

    def __init__(self, worker_pool, max_concurrent: int = 4):
        self.worker_pool = worker_pool
        self.max_concurrent = max_concurrent
        self.step_results: dict[str, StepResult] = {}
        self.step_timings: dict[str, float] = {}  # For Phase 2

    def compute_execution_waves(self, steps: list[dict]) -> list[list[dict]]:
        """Group steps into waves where all steps in a wave can run concurrently.

        A step can run when all its depends_on steps have completed.
        Steps with the same parallel_group are also grouped together.
        """
        completed = set()
        waves = []
        remaining = list(steps)

        while remaining:
            # Find steps whose dependencies are all completed
            ready = [s for s in remaining
                     if all(dep in completed for dep in s.get("depends_on", []))]

            if not ready:
                raise CyclicDependencyError(remaining)

            # Respect max concurrency
            wave = ready[:self.max_concurrent]
            waves.append(wave)
            completed.update(s["id"] for s in wave)
            remaining = [s for s in remaining if s["id"] not in completed]

        return waves

    async def execute_plan(self, plan: dict) -> dict:
        """Execute a TaskIR plan with parallelism."""
        steps = plan["steps"]
        max_conc = plan.get("parallelism", {}).get("max_concurrent_steps", self.max_concurrent)

        waves = self.compute_execution_waves(steps)

        for wave in waves:
            if len(wave) == 1:
                # Sequential: just execute directly
                result = await self._execute_step(wave[0])
                self.step_results[wave[0]["id"]] = result
            else:
                # Parallel: use worker pool batch
                results = await asyncio.gather(
                    *[self._execute_step(s) for s in wave]
                )
                for step, result in zip(wave, results):
                    self.step_results[step["id"]] = result

        return self.step_results
```

**2. Dual execution paths** (`src/api.py`)

Two paths coexist, routed by whether the TaskIR has explicit parallelism:

```python
# In the Root LM loop (api.py):
if task_ir and task_ir.get("plan", {}).get("parallelism"):
    # Path A: Structured parallel execution via ParallelStepExecutor
    # For tasks with explicit parallel_group / depends_on in TaskIR
    executor = ParallelStepExecutor(worker_pool, max_concurrent=4)
    results = await executor.execute_plan(task_ir["plan"])
else:
    # Path B: REPL code generation (existing behavior)
    # Frontdoor can still use llm_batch() in generated code for ad-hoc parallelism
    ...
```

**When to use which**:
- **Path A (executor)**: TaskIR has `parallelism` hints — multi-step plans with clear dependency structure (e.g., "explore 5 files then summarize")
- **Path B (REPL)**: Freeform tasks, single-step tasks, or tasks where the frontdoor uses `llm_batch()` directly in generated code

Both paths feed into Phase 2's critical path metric.

**3. Frontdoor parallel_group assignment**

Teach the frontdoor system prompt to assign `parallel_group` and `depends_on` when emitting TaskIR. For REPL-native parallelism, also teach it to generate `llm_batch()` calls when appropriate. Add to `src/prompt_builders.py` system prompt:

```
When emitting TaskIR with multi-step plans:
- Steps that read different files → same parallel_group, no depends_on overlap
- Steps that write different outputs with no shared inputs → same parallel_group
- Steps that depend on a prior step's output → add depends_on: ["S1"]
- If parallelism is identified, add plan.parallelism.max_concurrent_steps
- Default: sequential (omit parallel_group)

When generating REPL code for parallel work:
- Use llm_batch([prompt1, prompt2, ...], role='worker') for parallel sub-calls
- Use llm_call() for sequential single calls
```

### Files to Modify

| File | Change |
|------|--------|
| `src/api.py` | Add parallel execution fast path in Root LM loop |
| `src/prompt_builders.py` | Update frontdoor system prompt for parallel_group assignment |
| `src/services/worker_pool.py` | Ensure `batch()` supports mixed-role parallel execution |

### Files to Create

| File | Purpose |
|------|---------|
| `src/parallel_executor.py` | ParallelStepExecutor class (~150 lines) |
| `tests/unit/test_parallel_executor.py` | Unit tests for wave computation, dependency resolution |

### Verification

```bash
# Unit tests for wave computation
pytest tests/unit/test_parallel_executor.py -v

# Verify TaskIR schema still validates
python3 orchestration/validate_ir.py task orchestration/last_task_ir.json

# Integration: run a multi-file explore task and verify parallel execution
# (requires orchestrator stack running)
```

### Don't Touch

- `orchestration/task_ir.schema.json` — already has the fields we need
- `src/services/worker_pool.py` internals — use existing `batch()` API

---

## Phase 2: Critical Path Metric

**Goal**: Track wall-clock time per TaskIR step and compute critical path length. Identify tasks that benefit most from parallelization.

### Design

```python
@dataclass
class StepTiming:
    step_id: str
    start_time: float
    end_time: float
    wall_clock_seconds: float
    role: str
    parallel_group: str | None = None
    depends_on: list[str] = field(default_factory=list)

@dataclass
class CriticalPathReport:
    """Inspired by K2.5's CriticalSteps metric."""
    total_sequential_time: float   # Sum of all step durations (if run sequentially)
    critical_path_time: float      # Longest dependency chain duration
    parallelism_ratio: float       # total_sequential / critical_path (>1 = parallelism helped)
    bottleneck_steps: list[str]    # Steps on the critical path
    step_timings: list[StepTiming]
```

**Critical path computation**: Standard DAG longest-path algorithm on the step dependency graph, weighted by wall-clock duration.

```python
def compute_critical_path(timings: list[StepTiming]) -> CriticalPathReport:
    """Compute critical path through step dependency DAG.

    Uses topological sort + dynamic programming:
    - For each step, critical_time[s] = duration[s] + max(critical_time[dep] for dep in depends_on)
    - Critical path = chain of steps with max critical_time
    """
    # Build adjacency
    by_id = {t.step_id: t for t in timings}
    critical_time = {}

    # Topological order (already guaranteed by wave execution)
    for t in timings:
        dep_max = max((critical_time.get(d, 0) for d in t.depends_on), default=0)
        critical_time[t.step_id] = t.wall_clock_seconds + dep_max

    cp_time = max(critical_time.values())
    total_time = sum(t.wall_clock_seconds for t in timings)

    # Trace back critical path
    bottleneck = []
    current = max(critical_time, key=critical_time.get)
    while current:
        bottleneck.append(current)
        deps = by_id[current].depends_on
        if not deps:
            break
        current = max(deps, key=lambda d: critical_time.get(d, 0))

    return CriticalPathReport(
        total_sequential_time=total_time,
        critical_path_time=cp_time,
        parallelism_ratio=total_time / cp_time if cp_time > 0 else 1.0,
        bottleneck_steps=list(reversed(bottleneck)),
        step_timings=timings,
    )
```

### Integration Points

**Path A (executor)**: `ParallelStepExecutor` records `StepTiming` per step natively.

**Path B (REPL-native)**: Instrument `llm_batch()` in `src/llm_primitives.py` to record per-call timing:

```python
# In LLMPrimitives.llm_batch() (llm_primitives.py:479):
def llm_batch(self, prompts, role="worker"):
    start_time = time.perf_counter()
    # ... existing batch logic ...
    elapsed = time.perf_counter() - start_time

    # Record batch timing for critical path analysis
    self._batch_timings.append(BatchTiming(
        n_prompts=len(prompts),
        role=role,
        wall_clock_seconds=elapsed,
        per_prompt_seconds=[...],  # If available from worker pool
        timestamp=time.time(),
    ))
    return results
```

This gives `_tracked_llm_batch` in the REPL visibility into parallel execution timing without changing the REPL wrapper (it already forwards via `**kwargs`).

**Both paths**:
- After task completion, compute `CriticalPathReport`
- Log to `orchestration/progress/` or append to agent audit log
- Over time, build dataset of (task_type → parallelism_ratio) to guide when to use parallel execution

### Files to Modify

| File | Change |
|------|--------|
| `src/parallel_executor.py` | Add timing collection + critical path computation |
| `src/llm_primitives.py` | Add `_batch_timings` list, record timing in `llm_batch()` |
| `src/api.py` | Log CriticalPathReport after task completion (both paths) |

### Files to Create

| File | Purpose |
|------|---------|
| `src/metrics/critical_path.py` | CriticalPathReport, compute_critical_path(), BatchTiming (~120 lines) |
| `tests/unit/test_critical_path.py` | Tests for DAG critical path computation |

### Verification

```bash
pytest tests/unit/test_critical_path.py -v

# After running a parallel task (either path), check the report:
# Expected output: parallelism_ratio > 1.0 means parallelism helped
#
# For Path B, verify llm_batch timing is recorded:
# python3 -c "
# from src.llm_primitives import LLMPrimitives
# p = LLMPrimitives(mock_mode=True)
# p.llm_batch(['test1', 'test2'], role='worker')
# print(p._batch_timings)  # Should have 1 entry
# "
```

---

## Phase 3: Persona Registry + MemRL-Guided Selection

**Goal**: Build a persona registry — structured prompt definitions that shape worker behavior — with MemRL learning which persona works best for each task type.

### What Exists Already

**System prompt suffix** (`src/llm_primitives.py:418-430`):
```python
system_prompt_suffix = None
if self.registry:
    default_n, _default_temp, system_prompt_suffix = self.registry.get_role_defaults(role)
    if n_tokens is None:
        n_tokens = default_n
if system_prompt_suffix:
    prompt = f"{prompt}\n\n{system_prompt_suffix}"
```

This is the injection point. Personas extend the suffix mechanism.

**Delegate with MemRL tracking** (`src/repl_environment.py:1662-1742`):
```python
def _delegate(self, prompt, target_role="worker_general", reason=""):
    # Records delegation in MemRL for future routing decisions
    delegation_record = {
        "from_role": self.role,
        "to_role": target_role,
        "reason": reason,
        ...
    }
```

This tracks role selection. We extend it to also track persona selection.

**REPL globals** (`src/repl_environment.py:487-490`):
```python
globals_dict["llm_call"] = self._tracked_llm_call
globals_dict["llm_batch"] = self._tracked_llm_batch
```

### Persona Registry Design

**File**: `orchestration/persona_registry.yaml`

```yaml
# Persona Registry
# Each persona defines a behavioral profile that shapes worker output.
# The model (determined by role) stays the same — only the prompt changes.

personas:
  security_auditor:
    display_name: "Security Auditor"
    description: "Adversarial analysis focused on vulnerabilities"
    system_prompt: |
      You are a security auditor. Your task is to find vulnerabilities,
      not to praise the code. Focus on:
      - OWASP Top 10 (injection, XSS, SSRF, etc.)
      - Authentication/authorization bypasses
      - Input validation gaps
      - Secrets/credentials in code
      - Privilege escalation vectors

      Output format:
      1. FINDING: [description]
         SEVERITY: [CRITICAL/HIGH/MEDIUM/LOW]
         LOCATION: [file:line]
         REMEDIATION: [fix]
    compatible_roles: [worker_explore, coder_primary, coder_escalation]
    output_format: structured_findings
    evaluation_criteria:
      - "Identifies at least one real vulnerability"
      - "No false positives on safe patterns"

  technical_writer:
    display_name: "Technical Writer"
    description: "Clear, precise documentation with examples"
    system_prompt: |
      You are a technical writer. Produce clear, precise documentation.
      Requirements:
      - Lead with a one-sentence summary
      - Use concrete examples (not abstract descriptions)
      - Include code snippets where relevant
      - Define acronyms on first use
      - Assume the reader is a competent engineer unfamiliar with this specific codebase
    compatible_roles: [worker_explore, worker_general]
    output_format: markdown
    evaluation_criteria:
      - "Summary sentence present"
      - "At least one code example"

  performance_optimizer:
    display_name: "Performance Optimizer"
    description: "Identifies bottlenecks and proposes measurable improvements"
    system_prompt: |
      You are a performance optimization specialist. Analyze code for:
      - Algorithmic complexity (O(n^2) loops, unnecessary allocations)
      - I/O bottlenecks (synchronous where async possible)
      - Memory usage (large copies, cache misses)
      - Concurrency opportunities (independent operations)

      For each finding, provide:
      1. Current cost (estimated)
      2. Proposed change
      3. Expected improvement (with reasoning)
      4. Risk/tradeoff
    compatible_roles: [worker_explore, coder_primary, architect_general]
    output_format: structured_findings

  test_designer:
    display_name: "Test Designer"
    description: "Comprehensive test case generation"
    system_prompt: |
      You are a test design specialist. Generate thorough test cases:
      - Happy path (normal operation)
      - Edge cases (empty input, max size, unicode, etc.)
      - Error paths (invalid input, timeouts, resource exhaustion)
      - Boundary conditions (off-by-one, overflow)
      - Regression cases (based on known bug patterns)

      Output format: pytest-style test functions with docstrings explaining
      what each test verifies and why it matters.
    compatible_roles: [coder_primary, coder_escalation]
    output_format: python_code

  code_reviewer:
    display_name: "Code Reviewer"
    description: "Detailed code review with actionable feedback"
    system_prompt: |
      You are a senior code reviewer. Review for:
      - Correctness: Does the code do what it claims?
      - Clarity: Can another engineer understand this in 5 minutes?
      - Maintainability: Will this be easy to modify in 6 months?
      - Error handling: Are failure modes covered?
      - Naming: Do names convey intent?

      For each issue:
      - MUST FIX: Correctness bugs, security issues
      - SHOULD FIX: Clarity, maintainability
      - NICE TO HAVE: Style preferences

      Do NOT nitpick formatting or style that a linter handles.
    compatible_roles: [coder_primary, coder_escalation, architect_general]
    output_format: review_comments

  data_analyst:
    display_name: "Data Analyst"
    description: "Statistical analysis and data interpretation"
    system_prompt: |
      You are a data analyst. When examining data:
      - State the sample size and any limitations
      - Report central tendency AND spread (mean + std, or median + IQR)
      - Identify outliers and explain possible causes
      - Note correlations but never assume causation without evidence
      - Visualize when helpful (describe what chart you'd use)
      - Express uncertainty explicitly (confidence intervals, p-values if applicable)
    compatible_roles: [worker_explore, worker_general]
    output_format: structured_analysis

  inference_specialist:
    display_name: "Inference Optimization Specialist"
    description: "LLM inference performance analysis and optimization"
    system_prompt: |
      You are an LLM inference optimization specialist. Analyze and advise on:
      - Quantization tradeoffs (Q4_K_M vs Q8_0 vs FP16 — quality vs speed vs memory)
      - Speculative decoding: draft-target compatibility, acceptance rates, K-value tuning
      - MoE expert routing: expert reduction, load balancing, memory layout
      - Memory bandwidth analysis: tokens/s vs theoretical bandwidth limits
      - KV cache optimization: paged attention, prefix caching, radix trees
      - Prompt lookup: n-gram matching effectiveness, when it helps vs hurts
      - CPU-specific optimization: NUMA, thread pinning, AVX-512 utilization
      - Batch size vs latency tradeoffs for concurrent serving

      Always quantify: expected t/s, memory footprint, and quality impact.
      Reference specific hardware constraints (EPYC 9655, DDR5-5600, 460 GB/s bandwidth).
    compatible_roles: [coder_primary, coder_escalation, architect_general, worker_explore]
    output_format: structured_findings

  benchmark_analyst:
    display_name: "Benchmark Analyst"
    description: "Benchmark methodology, scoring, and model comparison"
    system_prompt: |
      You are a benchmark methodology specialist. When analyzing benchmarks:
      - Check for ceiling/floor effects (>90% or <10% scores)
      - Verify statistical significance (sample size, variance, confidence intervals)
      - Identify confounds (prompt sensitivity, temperature effects, output length bias)
      - Compare fairly: same prompts, same temperature, same max tokens
      - Distinguish speed benchmarks from quality benchmarks — never conflate
      - Watch for contamination: test data in training data
      - Report both absolute scores and relative rankings
      - Note when score differences are within noise margin

      Output format:
      1. METHODOLOGY: Was the benchmark setup fair?
      2. FINDINGS: Key results with confidence
      3. CAVEATS: What could invalidate these results
      4. RECOMMENDATIONS: What to test next
    compatible_roles: [worker_explore, worker_general, architect_general]
    output_format: structured_analysis

  computational_physicist:
    display_name: "Computational Physicist"
    description: "Physics-informed analysis of compute systems and numerical methods"
    system_prompt: |
      You are a computational physicist. Apply physics thinking to:
      - Memory hierarchy as energy levels (L1/L2/L3/DRAM/NVMe bandwidth tiers)
      - Roofline model analysis: compute-bound vs memory-bound regimes
      - Amdahl's law for parallelization limits
      - Queueing theory for request scheduling and batch sizing
      - Information-theoretic analysis: bits per parameter, compression limits
      - Dimensional analysis: verify units in throughput/latency calculations
      - Scaling laws: how performance changes with model size, batch size, context length
      - Thermal/power constraints on sustained vs burst performance

      Think in first principles. Derive before measuring. Predict before benchmarking.
      Express results with proper units and significant figures.
    compatible_roles: [architect_general, worker_explore, coder_primary]
    output_format: structured_analysis

  ai_engineer:
    display_name: "AI/ML Engineer"
    description: "End-to-end ML systems engineering"
    system_prompt: |
      You are a senior AI/ML engineer. Focus on:
      - Model architecture: attention mechanisms, MoE routing, SSM vs transformer tradeoffs
      - Training pipeline: data quality, curriculum learning, RL fine-tuning
      - Serving infrastructure: model sharding, pipeline parallelism, batching strategies
      - Evaluation: benchmark design, human eval correlation, automated scoring
      - MLOps: model versioning, A/B testing, rollback procedures
      - Cost optimization: compute/quality Pareto frontier, model distillation
      - Safety: alignment, output filtering, adversarial robustness

      Balance theory with practical engineering constraints.
      Prefer battle-tested solutions over novel approaches unless novelty is justified.
    compatible_roles: [architect_general, architect_coding, coder_primary, worker_explore]
    output_format: markdown

# MemRL seed Q-values for persona selection
# Format: (task_pattern, persona) → initial_q_value
# Higher Q = stronger initial preference
memrl_seeds:
  - task_pattern: "code review|review.*code|PR review"
    persona: code_reviewer
    initial_q: 0.85
  - task_pattern: "security|vulnerability|audit|CVE"
    persona: security_auditor
    initial_q: 0.90
  - task_pattern: "document|README|explain|describe"
    persona: technical_writer
    initial_q: 0.80
  - task_pattern: "test|coverage|edge case|regression"
    persona: test_designer
    initial_q: 0.85
  - task_pattern: "performance|slow|optimize|bottleneck|profile"
    persona: performance_optimizer
    initial_q: 0.90
  - task_pattern: "benchmark|results|analysis|compare|statistics"
    persona: data_analyst
    initial_q: 0.80
  - task_pattern: "inference|speed|throughput|latency|tokens per second|quantiz|spec.*decod|MoE|expert"
    persona: inference_specialist
    initial_q: 0.90
  - task_pattern: "benchmark.*method|score.*valid|ceiling|statistical.*signif|rubric"
    persona: benchmark_analyst
    initial_q: 0.85
  - task_pattern: "roofline|bandwidth|scaling law|Amdahl|memory.*bound|compute.*bound|thermal"
    persona: computational_physicist
    initial_q: 0.85
  - task_pattern: "model.*arch|training.*pipeline|serving|MLOps|distill|alignment|fine.?tun"
    persona: ai_engineer
    initial_q: 0.80

# Hybrid auto-selection behavior:
# 1. If persona is explicitly specified in delegate() or TaskIR → use it
# 2. If no persona specified → MemRL suggests highest-Q persona for task_type
#    (only if Q > 0.6 and the persona is compatible with the assigned role)
# 3. If no match or Q < 0.6 → no persona (current behavior, vanilla prompt)
auto_selection:
  enabled: true
  min_q_threshold: 0.6   # Don't auto-suggest personas with Q below this
  require_role_compat: true  # Only suggest personas compatible with assigned role
```

### Integration Changes

**1. `llm_call()` persona parameter** (`src/llm_primitives.py:372`)

Current:
```python
def llm_call(self, prompt, context_slice="", role="worker", n_tokens=None):
```

New:
```python
def llm_call(self, prompt, context_slice="", role="worker", n_tokens=None, persona=None):
```

In the body (after line 430):
```python
# Apply persona prompt if specified
if persona and self.persona_registry:
    persona_config = self.persona_registry.get(persona)
    if persona_config:
        # Check role compatibility
        if role in persona_config.get("compatible_roles", []) or not persona_config.get("compatible_roles"):
            persona_prompt = persona_config["system_prompt"]
            prompt = f"{persona_prompt}\n\n---\n\n{prompt}"
```

**2. `delegate()` persona tracking** (`src/repl_environment.py:1662`)

Current:
```python
def _delegate(self, prompt, target_role="worker_general", reason=""):
```

New:
```python
def _delegate(self, prompt, target_role="worker_general", reason="", persona=None):
```

Add persona to delegation record:
```python
delegation_record = {
    "from_role": self.role,
    "to_role": target_role,
    "persona": persona,  # NEW
    "reason": reason,
    ...
}
# Pass persona to llm_call:
result = self.llm_primitives.llm_call(prompt, role=target_role, persona=persona)
```

**3. `_tracked_llm_call` wrapper** (`src/repl_environment.py:2484`)

Pass through persona kwarg:
```python
def _tracked_llm_call(self, *args, **kwargs) -> str:
    self._exploration_calls += 1
    result = self.llm_primitives.llm_call(*args, **kwargs)
    self._exploration_log.add_event("llm_call", {"args": args, "kwargs": kwargs}, result)
    return result
```

No change needed — `**kwargs` already passes `persona` through.

**4. Hybrid auto-selection** (new logic in `src/llm_primitives.py` or `src/persona_loader.py`)

When no persona is explicitly specified, auto-suggest based on MemRL Q-values:

```python
def auto_select_persona(
    task_description: str,
    role: str,
    persona_registry: PersonaRegistry,
    q_values: dict,  # From MemRL
    min_q: float = 0.6,
) -> str | None:
    """Auto-select best persona for a task based on MemRL Q-values.

    Returns None if no persona exceeds min_q threshold or is compatible.
    """
    candidates = []
    for persona_id, config in persona_registry.personas.items():
        # Check role compatibility
        if role not in config.get("compatible_roles", []):
            continue
        # Check Q-value
        q = q_values.get((task_description_pattern, persona_id), 0.0)
        if q >= min_q:
            candidates.append((persona_id, q))

    if not candidates:
        return None

    # Return highest-Q persona
    return max(candidates, key=lambda x: x[1])[0]
```

Integration in `llm_call()`:
```python
# If no persona specified, try auto-selection
if persona is None and self.persona_registry and self.auto_selection_enabled:
    persona = auto_select_persona(prompt, role, self.persona_registry, self.q_values)
    # persona may still be None if no match — that's fine, vanilla behavior
```

**5. TaskIR persona_hint field** (`orchestration/task_ir.schema.json`)

Add to agents items:
```json
"persona_hint": {
  "type": "string",
  "description": "Suggested persona from persona_registry.yaml"
}
```

**5. MemRL seed loading** (`orchestration/repl_memory/seed_loader.py`)

Add persona seed loading from `persona_registry.yaml` `memrl_seeds` section. Seeds create initial Q-values for (task_pattern, persona) pairs.

**6. Seed examples** (`orchestration/repl_memory/seed_examples.json`)

Add persona-aware examples:
```json
{
  "task": "Review this code for security issues",
  "code": "result = delegate('Review the following code for security vulnerabilities:\\n' + peek(5000), target_role='worker_explore', persona='security_auditor', reason='Security review task matches security_auditor persona')\nFINAL(result)",
  "tools_used": ["delegate", "peek", "FINAL"],
  "category": "persona_routing"
}
```

### Files to Modify

| File | Change |
|------|--------|
| `src/llm_primitives.py` | Add `persona` param to `llm_call()`, apply persona prompt |
| `src/repl_environment.py` | Add `persona` param to `_delegate()`, pass through in `_tracked_llm_call` |
| `orchestration/task_ir.schema.json` | Add `persona_hint` to agents items |
| `orchestration/repl_memory/seed_loader.py` | Load persona Q-value seeds |
| `orchestration/repl_memory/seed_examples.json` | Add persona-aware examples |

### Files to Create

| File | Purpose |
|------|---------|
| `orchestration/persona_registry.yaml` | Persona definitions + MemRL seeds (~200 lines) |
| `src/persona_loader.py` | YAML loader for persona registry (~50 lines) |
| `tests/unit/test_persona_registry.py` | Tests for persona loading, role compatibility, prompt injection |

### Verification

```bash
# Unit tests
pytest tests/unit/test_persona_registry.py -v

# Validate persona YAML loads correctly
python3 -c "from src.persona_loader import PersonaRegistry; r = PersonaRegistry(); print(r.list_personas())"

# Verify llm_call persona injection (mock mode)
python3 -c "
from src.llm_primitives import LLMPrimitives
p = LLMPrimitives(mock_mode=True)
r = p.llm_call('Review this code', role='worker', persona='security_auditor')
print('OK:', len(r))
"

# Verify TaskIR schema still validates
python3 orchestration/validate_ir.py task orchestration/last_task_ir.json
```

---

## Phase 4: Staged Reward Shaping for MemRL

**Goal**: Apply PARL's annealing schedule to MemRL Q-value updates — explore early, exploit later.

### Background

K2.5's PARL reward formula:
```
Rt = lambda_aux(e) * r_parallel + (1 - lambda_aux(e)) * (I[success] * Q(tau))
```

Lambda anneals from 0.1 → 0.0 during training, shifting from "reward trying new things" to "reward task success."

Our adaptation: Instead of parallelism reward, we use **exploration bonus** — trying personas/roles the system hasn't used much for a given task type.

### Design

```python
class StagedQScorer:
    """Q-value scorer with PARL-inspired annealing.

    Early in a session: high exploration bonus for trying
    underexplored (task_type, role, persona) combinations.

    Later: pure exploitation of best-known combinations.
    """

    def __init__(self, anneal_steps: int = 50):
        self.anneal_steps = anneal_steps  # Steps to anneal over
        self.step_count = 0

    def lambda_aux(self) -> float:
        """Annealing schedule: starts at 0.3, decays to 0.0."""
        if self.step_count >= self.anneal_steps:
            return 0.0
        return 0.3 * (1 - self.step_count / self.anneal_steps)

    def compute_reward(
        self,
        task_success: bool,
        task_quality: float,  # 0.0-1.0 from gate results
        exploration_bonus: float,  # Higher for underexplored combos
    ) -> float:
        lam = self.lambda_aux()
        success_reward = (1.0 if task_success else 0.0) * task_quality
        return lam * exploration_bonus + (1 - lam) * success_reward

    def update(self, task_result):
        """Update Q-values with staged reward."""
        self.step_count += 1
        ...
```

**Exploration bonus computation**:
```python
def exploration_bonus(task_type: str, role: str, persona: str | None, history: dict) -> float:
    """Higher bonus for underexplored (task_type, role, persona) combinations.

    Uses count-based exploration: bonus = 1 / sqrt(N + 1)
    where N = number of times this combination has been tried.
    """
    key = (task_type, role, persona or "none")
    count = history.get(key, 0)
    return 1.0 / (count + 1) ** 0.5
```

### Files to Modify

| File | Change |
|------|--------|
| `orchestration/repl_memory/` (Q-scorer) | Add annealing schedule, exploration bonus |

### Files to Create

| File | Purpose |
|------|---------|
| `src/staged_q_scorer.py` | StagedQScorer class (~100 lines) |
| `tests/unit/test_staged_q_scorer.py` | Tests for annealing, bonus computation |

### Verification

```bash
pytest tests/unit/test_staged_q_scorer.py -v

# Verify annealing behavior:
# - At step 0: lambda=0.3, exploration weighted 30%
# - At step 25: lambda=0.15, exploration weighted 15%
# - At step 50+: lambda=0.0, pure exploitation
```

---

## Phase 5: Parallel Gate Execution (with Hardware Guardrails)

**Goal**: Run independent gates concurrently. Profile first to confirm this is worthwhile.

### Hardware Constraints

We're on a single machine running 9+ llama-server instances consuming 535GB+ RAM. Gate parallelism must NOT:
- Compete with inference for memory bandwidth
- Spawn processes that OOM the system
- Run unit/integration tests in parallel with inference (they may load models)

### Step 1: Profile Current Gate Performance (DO THIS FIRST)

```bash
# Time each gate individually
time make check-schema   2>&1 | tail -1
time make check-shell    2>&1 | tail -1
time make check-format   2>&1 | tail -1
time make check-lint     2>&1 | tail -1
time make test-unit      2>&1 | tail -1

# Record total sequential time
time make gates          2>&1 | tail -1
```

**Decision gate**: If total sequential gate time < 10 seconds, parallelism is not worth the complexity. Stop here.

### Step 2: Identify Independent Gates

| Gate | Depends On | Tool | Memory | CPU |
|------|-----------|------|--------|-----|
| schema | Nothing | python3 | Low | Low |
| shellcheck | Nothing | shellcheck | Low | Low |
| format | Nothing | shfmt, mdformat | Low | Low |
| lint | Nothing | markdownlint | Low | Low |
| unit | schema (optional) | pytest | **MEDIUM** | Medium |
| integration | unit | pytest | **HIGH** | High |

**Safe to parallelize**: schema, shellcheck, format, lint (all lightweight, independent)
**Must stay sequential**: unit after lightweight gates, integration after unit

### Step 3: Implementation (Only if Profiling Justifies)

```makefile
# Makefile: parallel lightweight gates, sequential heavy gates
.PHONY: gates-fast
gates-fast:
	$(MAKE) -j3 check-schema check-shell check-format check-lint
	$(MAKE) test-unit
	$(MAKE) test-integration
```

Or async Python wrapper:
```python
async def run_gates_parallel(artifact_paths: list[str], max_concurrent: int = 3):
    lightweight = ["schema", "shellcheck", "format", "lint"]
    heavy = ["unit", "integration"]

    # Phase 1: lightweight gates in parallel
    results = await asyncio.gather(
        *[run_gate(g, artifact_paths) for g in lightweight]
    )
    if any(r.failed for r in results):
        return GateReport(failed=True, results=results)

    # Phase 2: heavy gates sequentially
    for gate in heavy:
        result = await run_gate(gate, artifact_paths)
        if result.failed:
            return GateReport(failed=True, results=[*results, result])

    return GateReport(failed=False, results=results)
```

### Files to Modify

| File | Change |
|------|--------|
| `Makefile` | Add `gates-fast` target with `-j3` for lightweight gates |

### Verification

```bash
# Compare sequential vs parallel
time make gates
time make gates-fast

# Verify same gates pass/fail
make gates && echo "sequential OK"
make gates-fast && echo "parallel OK"
```

---

## Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Parallel TaskIR execution | >2x speedup on multi-file tasks | CriticalPathReport.parallelism_ratio |
| Critical path visibility | Reports generated for every multi-step task | Check `orchestration/progress/` logs |
| Persona quality improvement | >10% quality score improvement when persona matches task | MemRL Q-value comparison with/without persona |
| MemRL persona learning | Persona Q-values converge within 20 tasks per type | Monitor Q-value stability over sessions |
| Gate parallelism | >30% wall-clock reduction (if profiling justifies) | `time make gates` vs `time make gates-fast` |

---

## Completion Checklist

When this handoff is complete:

- [ ] Phase 1: ParallelStepExecutor tests passing
- [ ] Phase 2: CriticalPathReport generated for multi-step tasks
- [ ] Phase 3: Persona registry loads, llm_call accepts persona, MemRL seeds loaded
- [ ] Phase 4: StagedQScorer annealing verified
- [ ] Phase 5: Gate profiling done (implement only if justified)
- [ ] All tests passing: `pytest tests/ -x -q`
- [ ] Gates passing: `make gates`
- [ ] Key findings → `docs/chapters/` (if significant)
- [ ] Update `orchestration/BLOCKED_TASKS.md`
- [ ] DELETE this handoff file

---

## References

- Research analysis: `research/kimi_k25_agent_swarm_analysis.md`
- [Kimi K2.5 Technical Report](https://www.kimi.com/blog/kimi-k2-5.html)
- [Kimi K2 arxiv (2507.20534)](https://arxiv.org/abs/2507.20534)
- TaskIR schema: `orchestration/task_ir.schema.json`
- Root LM loop: `src/api.py`
- LLM primitives: `src/llm_primitives.py`
- REPL environment: `src/repl_environment.py`
- Model registry: `orchestration/model_registry.yaml`
- MemRL seeds: `orchestration/repl_memory/seed_examples.json`
