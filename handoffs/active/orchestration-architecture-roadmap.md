# Orchestration Architecture Roadmap

**Status**: Active
**Created**: 2026-01-30 (merged from `orchestration-refactoring.md` + `parl-inspired-orchestrator-improvements.md`)
**Priority**: MEDIUM
**Research**: `research/kimi_k25_agent_swarm_analysis.md`

---

## Quick Start / Resume Commands

```bash
# 1. Verify test baseline
cd /mnt/raid0/llm/claude && timeout 120 python3 -m pytest tests/ -x -q

# 2. Key files to understand
cat src/api/routes/chat.py           # _architect_delegated_answer() (line 1086)
cat src/proactive_delegation.py      # ProactiveDelegator, ArchitectReviewService
cat src/llm_primitives.py            # llm_call/llm_batch (persona injection point)
cat src/prompt_builders/__init__.py   # System prompts package (re-exports all 27 names)
cat orchestration/task_ir.schema.json # TaskIR schema (has parallel_group!)
cat orchestration/model_registry.yaml # Role configs

# 3. Pick a work item (A-G) — they're mostly independent
```

---

## Completed Work (Reference)

### Refactoring Phases 1-3 (2026-01-14/15)

- **Phase 1**: Foundation — thread-safe AppState, exception logging, AST-based REPL security, feature flags
- **Phase 2**: Structure — `src/api/` modular split (routes/models/services/state), unified escalation (`src/escalation.py`), Role enum (`src/roles.py`)
- **Phase 3**: Abstractions — `LLMBackend` protocol, unified `OrchestratorConfig`, `PromptBuilder`, RestrictedPython executor, SSE utilities
- Tests: 537 passed, 4 pre-existing failures

### Phase 4.1: OpenAI Compatibility (2026-01-16)

- `/v1/chat/completions` uses full orchestration (REPL, Root LM loop)
- Mock mode fallback, feature flag respect

### Architect Delegation (2026-01-30)

- `_architect_delegated_answer()` in `chat.py:1086` — TOON-encoded investigation briefs
- `_parse_architect_decision()` — TOON/JSON/bare-text parser
- Multi-loop capped at 3, feature flag `architect_delegation`
- `"delegated"` force mode, `read_file`/`list_directory` in `REACT_TOOL_WHITELIST`
- 884 tests passing

### Vision → Document Pipeline Integration (2026-02-01)

- `/chat` vision requests now use full document pipeline (DocumentPreprocessor → DocumentChunker → FigureAnalyzer → DocumentREPLEnvironment)
- `_execute_vision()` preprocesses, stores on `routing.document_result`, returns None (no early return)
- Mode selection forces REPL + FRONTDOOR when document results present
- `_execute_repl()` creates DocumentREPLEnvironment with sections/figures/search tools
- Base64 image input support via temp file on RAID
- 1234 tests passing

### Architecture Review Work Items (2026-02-01)

- **WI-9**: Staged reward shaping — `StagedScorer` with PARL-inspired λ annealing (see F below)
- **WI-10**: Parallel gate execution — `asyncio.gather()` for independent gates (see G below)
- **WI-11**: `prompt_builders.py` decomposition — 1,501-line monolith → `src/prompt_builders/` package with 6 sub-modules (types, constants, builder, review, code_utils, formatting). Zero downstream import changes.
- 1398 tests passing

### Post-Refactoring Architecture Cleanup (2026-02-01)

- **N1**: `repl_environment.py` decomposition — 3,511-line monolith → `src/repl_environment/` package with 8 modules (types, security, file_tools, document_tools, routing, procedure_tools, context, state, environment). Mixin-based: REPLEnvironment inherits 6 focused mixins. Zero downstream import changes.
- **N2**: Replaced all 5 `shell=True` subprocess calls with `shlex.split()` + `shell=False` (model_server, file_tools, script_registry, formalizer, gate_runner).
- **N3**: Deleted dead `src/api.py` (1,852 lines) — shadowed by `src/api/__init__.py` package.
- 1419 tests passing

### Other Infrastructure

- HTTP connection pooling (httpx, ~6x latency reduction)
- Unified orchestrator stack launcher (`scripts/server/orchestrator_stack.py`)
- ProactiveDelegator module (`src/proactive_delegation.py` — IterationContext, ArchitectReviewService, AggregationService)

---

## Active Remaining Work

### A. Structured Logging (from refactoring Phase 4.2)

**Goal**: Replace ad-hoc f-string logging with structured fields for log aggregation.

**Current state**: 38 basic `logger.info(f"...")` calls in `src/api/routes/chat.py` with no structured fields.

**Work**:
- Add `extra={"task_id": ..., "role": ..., "latency_ms": ...}` pattern to all log calls
- JSON formatter for log aggregation (optional OpenTelemetry hooks)
- Consistent structured error hierarchy

**Files**: `src/api/routes/chat.py`, logging configuration

### B. Integration Test Import Fix (from refactoring)

**Bug**: `tests/integration/test_frontend_integration.py:15` imports `_sessions` but the module exports `_session_store`.

**Fix**: Update import to match actual export name.

**Files**: `tests/integration/test_frontend_integration.py`

### C. Full ProactiveDelegator Wiring (refactoring Phase 5 + PARL Phase 1)

**Goal**: Wire complete multi-specialist TaskIR decomposition and parallel execution.

**Current state**: Sequential architect delegation works (`_architect_delegated_answer()`). `ProactiveDelegator` class exists in `src/proactive_delegation.py` but isn't wired to API routes. `ParallelStepExecutor` doesn't exist yet.

**Work**:
1. Wire `/delegate` and `/delegate/review` endpoints to `ProactiveDelegator`
2. Create `src/parallel_executor.py` — `ParallelStepExecutor` class:
   - `compute_execution_waves(steps)` — group by `depends_on`/`parallel_group`
   - `execute_plan(plan)` — wave-based concurrent execution
   - Respects `plan.parallelism.max_concurrent_steps` from TaskIR
3. Dual execution paths in Root LM loop:
   - Path A (executor): TaskIR has `parallelism` hints → ParallelStepExecutor
   - Path B (REPL): Freeform/single-step tasks → existing behavior
4. Update frontdoor system prompt for `parallel_group` assignment
5. Enable `architect_delegation` feature flag for production

**TaskIR schema already has**: `parallel_group`, `depends_on`, `parallelism.max_concurrent_steps` — these fields are inert, need execution layer.

**Files to modify**: `src/api/routes/chat.py`, `src/prompt_builders/`
**Files to create**: `src/parallel_executor.py`, `tests/unit/test_parallel_executor.py`

### D. Critical Path Metric (PARL Phase 2)

**Goal**: Track wall-clock time per TaskIR step, compute critical path length.

**Work**:
- `StepTiming` and `CriticalPathReport` dataclasses
- DAG longest-path computation (topological sort + DP)
- Instrument `llm_batch()` timing in `src/llm_primitives.py`
- Log `CriticalPathReport` after task completion

**Depends on**: Phase C (uses same timing data from executor)

**Files to create**: `src/metrics/critical_path.py`, `tests/unit/test_critical_path.py`
**Files to modify**: `src/parallel_executor.py`, `src/llm_primitives.py`

### E. Persona Registry + MemRL (PARL Phase 3)

**Goal**: Structured prompt definitions that shape worker behavior, with MemRL learning which persona works best per task type.

**Work**:
1. Create `orchestration/persona_registry.yaml` — 10 personas defined:
   - security_auditor, technical_writer, performance_optimizer, test_designer,
     code_reviewer, data_analyst, inference_specialist, benchmark_analyst,
     computational_physicist, ai_engineer
2. Create `src/persona_loader.py` — YAML loader (~50 lines)
3. Add `persona` param to `llm_call()` in `src/llm_primitives.py`
4. Add `persona` param to `_delegate()` in `src/repl_environment.py`
5. Add `persona_hint` field to TaskIR schema agents items
6. MemRL seed Q-values for persona selection (regex task_pattern matching)
7. Hybrid auto-selection: if no explicit persona, use highest-Q from MemRL (threshold 0.6)

**Independent**: Can be worked in any order relative to C/D.

**Files to create**: `orchestration/persona_registry.yaml`, `src/persona_loader.py`, `tests/unit/test_persona_registry.py`
**Files to modify**: `src/llm_primitives.py`, `src/repl_environment.py`, `orchestration/task_ir.schema.json`, `orchestration/repl_memory/seed_loader.py`

### F. Staged Reward Shaping (PARL Phase 4) — ✅ COMPLETE (WI-9)

**Goal**: PARL-inspired annealing for MemRL Q-value updates — explore early, exploit later.

**Implemented**:
- `StagedScorer` class in `orchestration/repl_memory/staged_scorer.py` (~120 lines)
- Annealing schedule: λ(step) = λ_init × max(0, 1 − step/horizon), default λ_init=0.3
- Exploration bonus: `1/√(N+1)` for underexplored combos
- Reward: `λ × exploration_bonus + (1 − λ) × success_reward`
- 8 unit tests in `tests/unit/test_staged_scorer.py`

### G. Parallel Gate Execution (PARL Phase 5) — ✅ COMPLETE (WI-10)

**Goal**: Run independent gates concurrently.

**Implemented**:
- `_run_gate_parallel()` using `asyncio.to_thread()` for subprocess gates
- `run_gates_parallel()` using `asyncio.gather()` for independent gates
- Sequential fallback preserved for dependent gates
- `parallel_gates` feature flag in `src/features.py`
- 6 unit tests in `tests/unit/test_gate_runner.py`

**Files modified**: `src/gate_runner.py`, `src/features.py`

---

## Implementation Status

| Item | Status | Dependencies |
|------|--------|--------------|
| A. Structured Logging | ✅ (task_extra + JSONFormatter + 14 pipeline calls) | None |
| B. Integration Test Fix | ✅ (already fixed in prior session) | None |
| C. ProactiveDelegator + Parallel Execution | ❌ | None |
| D. Critical Path Metric | ❌ | C |
| E. Persona Registry + MemRL | ❌ | None |
| F. Staged Reward Shaping | ✅ (WI-9: StagedScorer + 8 tests) | E (loosened — implemented independently) |
| G. Parallel Gate Execution | ✅ (WI-10: asyncio.gather + feature flag + 6 tests) | None |

**Independent items**: A, B, C, E, G can be worked in any order.

---

## Verification

```bash
# After any changes
cd /mnt/raid0/llm/claude && make gates

# Unit tests
pytest tests/ -x -q

# Validate TaskIR schema
python3 orchestration/validate_ir.py task orchestration/last_task_ir.json
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/api/routes/chat.py` | Chat endpoints, _architect_delegated_answer() |
| `src/proactive_delegation.py` | ProactiveDelegator, ArchitectReviewService |
| `src/llm_primitives.py` | llm_call/llm_batch, persona injection point |
| `src/prompt_builders/` | System prompts package (types, constants, builder, review, code_utils, formatting) |
| `src/repl_environment/` | REPL sandbox package (types, security, file_tools, document_tools, routing, procedure_tools, context, state, environment) |
| `src/features.py` | Feature flags (architect_delegation, etc.) |
| `src/api/routes/config.py` | POST /config — runtime feature flag hot-reload |
| `src/escalation.py` | Unified escalation policy |
| `src/roles.py` | Role and Tier enums |
| `orchestration/task_ir.schema.json` | TaskIR schema (parallel_group, depends_on) |
| `orchestration/model_registry.yaml` | Role configs, system_prompt_suffix |
| `research/kimi_k25_agent_swarm_analysis.md` | PARL research context |

---

## Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Parallel execution | >2x speedup on multi-file tasks | CriticalPathReport.parallelism_ratio |
| Critical path visibility | Reports for every multi-step task | Check orchestration/progress/ logs |
| Persona quality | >10% quality improvement when matched | MemRL Q-value comparison |
| MemRL persona learning | Q-values converge within 20 tasks/type | Monitor stability over sessions |
| Gate parallelism | >30% wall-clock reduction (if profiling justifies) | `time make gates` vs `time make gates-fast` |

---

## Completion Checklist

When this roadmap is complete:

- [ ] A: Structured logging in chat.py
- [ ] B: Integration test import fixed
- [ ] C: ParallelStepExecutor tests passing, /delegate endpoints wired
- [ ] D: CriticalPathReport generated for multi-step tasks
- [ ] E: Persona registry loads, llm_call accepts persona, MemRL seeds loaded
- [x] F: StagedScorer annealing verified (WI-9, 8 tests)
- [x] G: Parallel gate execution implemented (WI-10, 6 tests)
- [ ] All tests passing: `pytest tests/ -x -q`
- [ ] Gates passing: `make gates`
- [ ] Key findings → `docs/chapters/` (if significant)
- [ ] Update `orchestration/BLOCKED_TASKS.md`
- [ ] DELETE this handoff file
