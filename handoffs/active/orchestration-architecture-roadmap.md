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
cat src/proactive_delegation/__init__.py  # ProactiveDelegator package (types, complexity, review_service, delegator)
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

### Security & Thread Safety Hardening (2026-02-01)

- **Path traversal fix**: `validate_api_path()` with `os.path.realpath()` + allowlist in `documents.py`, `chat_vision.py`
- **Base64 size limits**: 100MB max for documents, 50MB max for images (HTTP 413)
- **Config endpoint auth**: `POST /config` restricted to localhost only (HTTP 403)
- **Thread-safe singletons**: Double-checked locking on 6 critical singletons (worker_pool, sessions, figure_analyzer, document_chunker, document_client, draft_cache)
- **Dead code**: Removed unused `role_enum` in routing.py, deprecation note on gradio_ui.py
- 1425 tests passing

### Observability & Remaining Hardening (2026-02-01)

- **Sync→async fix**: `httpx.get()` → `httpx.AsyncClient()` in `chat_summarization.py:153` (was blocking event loop)
- **Exception logging**: Added `log.debug()` to 17 bare `except: pass` blocks across 6 chat route files
- **Thread-safe PromptCompressor**: Double-checked locking on `PromptCompressor.get_instance()` (7th singleton fixed)
- **Module-level loggers**: Consolidated function-level logger creation to module top in 7 chat route files
- 1425 tests passing

### Final Polish (2026-02-01, Session 10)

- **chat_pipeline.py decomposition**: 1,439-line monolith → `src/api/routes/chat_pipeline/` package (4 modules: `routing.py` 228L, `stages.py` 704L, `repl_executor.py` 532L, `__init__.py` 64L). Zero downstream import changes; test mock patch targets updated.
- **Misc cleanup**: `_RestrictedREPLEnvironment` removed from `__all__`, handoff lifecycle cleanup, test count updated in docs.
- 1517 tests passing

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

### C. Full ProactiveDelegator Wiring (refactoring Phase 5 + PARL Phase 1) — ✅ COMPLETE

**Goal**: Wire complete multi-specialist TaskIR decomposition and parallel execution.

**Implemented**:
1. `_execute_proactive()` in `src/api/routes/chat_pipeline.py` (~100 lines) — complexity-gated proactive delegation:
   - Checks `features().parallel_execution and request.real_mode`
   - `classify_task_complexity()` — only COMPLEX tasks enter
   - Architect bypass (avoids double-entry with `_execute_delegated()`)
   - Architect plan generation → `_parse_plan_steps()` JSON parser (tolerant of markdown fences, trailing commas)
   - `ProactiveDelegator.delegate()` for wave-based parallel execution
2. Stage 6.5 wired in `_handle_chat()` in `src/api/routes/chat.py` — between vision (Stage 6) and mode selection (Stage 7)
3. `build_task_decomposition_prompt()` in `src/prompt_builders/builder.py` — architect prompt for plan generation
4. `src/proactive_delegation.py` decomposed into `src/proactive_delegation/` package (4 modules + `__init__.py`, zero downstream import changes)
5. 17 unit tests in `tests/unit/test_proactive_pipeline.py`

### D. Critical Path Metric (PARL Phase 2) — ✅ ALREADY DONE

Discovered during exploration that this was already fully implemented:
- `src/metrics/critical_path.py` — `StepTiming`, `CriticalPathReport`, `compute_critical_path()` (DAG longest-path via topological sort)
- `src/proactive_delegation/delegator.py:195-212` — post-hoc logging of CriticalPathReport after parallel execution
- `src/parallel_step_executor.py` — `extract_step_timings()` builds timing data
- 15 tests in `tests/unit/test_critical_path.py`

### E. Persona Registry + MemRL (PARL Phase 3) — ✅ COMPLETE

**Implemented**:
1. `persona` parameter added to `llm_batch()` and `llm_batch_async()` in `src/llm_primitives.py`
2. `_apply_persona_prefix()` helper method for shared persona injection logic
3. StepExecutor persona injection in `src/parallel_step_executor.py:_execute_step()`:
   - Reads persona from step `persona` or `persona_hint` field
   - Falls back to MemRL auto-selection via `_auto_select_persona()`
4. `_auto_select_persona()` method (~45 lines) — queries HybridRouter retriever for persona-related episodes, picks highest Q-value above 0.6 threshold
5. `hybrid_router` parameter wired through `StepExecutor` → `ProactiveDelegator`

**Previously existing** (not modified): PersonaRegistry (18 personas), `persona_loader.py`, `llm_call()` persona param, seed Q-values, 30 tests in `test_persona_registry.py`

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
| C. ProactiveDelegator + Parallel Execution | ✅ (Stage 6.5 wired, _execute_proactive, _parse_plan_steps, decomposition, 17 tests) | None |
| D. Critical Path Metric | ✅ (already implemented: compute_critical_path + extract_step_timings + 15 tests) | C |
| E. Persona Registry + MemRL | ✅ (llm_batch persona, StepExecutor injection, MemRL auto-selection, hybrid_router wiring) | None |
| F. Staged Reward Shaping | ✅ (WI-9: StagedScorer + 8 tests) | E (loosened — implemented independently) |
| G. Parallel Gate Execution | ✅ (WI-10: asyncio.gather + feature flag + 6 tests) | None |

**All 7 items complete.** Roadmap fully implemented.

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
| `src/api/routes/chat_pipeline/` | Pipeline package (routing, stages, repl_executor) |
| `src/proactive_delegation/` | ProactiveDelegator package (types, complexity, review_service, delegator) |
| `src/llm_primitives.py` | llm_call/llm_batch, persona injection point |
| `src/prompt_builders/` | System prompts package (types, constants, builder, review, code_utils, formatting) |
| `src/repl_environment/` | REPL sandbox package (types, security, file_tools, document_tools, routing, procedure_tools, context, state, environment) |
| `src/features.py` | Feature flags (architect_delegation, etc.) |
| `src/api/routes/config.py` | POST /config — runtime feature flag hot-reload (localhost-only) |
| `src/api/routes/path_validation.py` | validate_api_path() — shared path traversal prevention |
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

- [x] A: Structured logging in chat.py
- [x] B: Integration test import fixed
- [x] C: ProactiveDelegator wired (Stage 6.5, _execute_proactive, decomposition, 17 tests)
- [x] D: CriticalPathReport already implemented (compute_critical_path + 15 tests)
- [x] E: Persona registry + llm_batch persona + StepExecutor injection + MemRL auto-selection
- [x] F: StagedScorer annealing verified (WI-9, 8 tests)
- [x] G: Parallel gate execution implemented (WI-10, 6 tests)
- [x] All tests passing: 1517 passed, 25 skipped, 0 failed
- [x] Gates passing: `make gates`
- [x] Key findings → `docs/chapters/10-orchestration-architecture.md` updated
- [x] Update `orchestration/BLOCKED_TASKS.md`
- [ ] DELETE this handoff file (all items complete — ready for archival)
