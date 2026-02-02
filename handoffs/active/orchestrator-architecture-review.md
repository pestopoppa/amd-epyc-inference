# Orchestrator Architecture Review & chat.py Decomposition

**Date**: 2026-02-01
**Status**: COMPLETE (Phases 1-3 + WI-9/10/11 + C/D/E roadmap + H1-H7 hardening + T1-T4 tests + final polish + Review #2: all 7 WIs done — eval() fix, URL consolidation, llm_primitives/file_tools decomposition, DI layer, 48 new tests. +314 coverage tests (2026-02-02). 2015 tests pass, 67.48% coverage.)
**Author**: Claude Opus 4.5 (architecture review session)

## Summary

Comprehensive architecture review of the hierarchical orchestrator codebase (110 source files, 44K LOC, 51 test files). Identified 5 critical issues, 9 serious issues, 6 minor issues. Phase 1 decomposes the 3,763-line `chat.py` God Module into 8 focused modules.

## Architecture Review Findings

### Critical Issues

| # | Issue | File(s) | Impact |
|---|-------|---------|--------|
| C1 | `_handle_chat()` is 1,091 lines with ~30 code paths | `src/api/routes/chat.py` | Untestable, unsafe to modify |
| C2 | Global mutable singletons with partial thread safety | `state.py`, `features.py`, `memrl.py` | Race conditions |
| C3 | No dependency injection - manual wiring via imports | `api/__init__.py` | Hard to test |
| C4 | String command generation without shell escaping | `registry_loader.py` | Injection vector |
| C5 | 0.44x test:source ratio, heavy mocking, no CI/CD | `tests/` | False confidence |

### Serious Issues

- S1: Hardcoded `/mnt/raid0/` paths (non-portable)
- ~~S2: `prompt_builders.py` mixes all 40+ prompt types (1,501 lines)~~ → **RESOLVED (WI-11)**: Decomposed into `src/prompt_builders/` package with 6 sub-modules (types, constants, builder, review, code_utils, formatting). Zero downstream import changes.
- S3: Dead `api/services/orchestrator.py` facade (deleted in Phase 1)
- S4: 8/15 `AppState` fields typed as `Any` (no type checking)
- S5: Magic numbers scattered (token budgets, thresholds, intervals)
- S6: CORS wildcard `*` in production
- S7: Incomplete feature flag validation (missing memrl deps)
- S8: No rate limiting or circuit breakers
- S9: 19-line health check (no dependent service checks)

### What Works Well

1. Tier architecture (A/B/C/D) is clean
2. Feature flag system is well-designed
3. Backend abstraction (ModelBackend + CachingBackend decorator)
4. MemRL episodic memory with two-phase retrieval
5. AST-based REPL sandboxing
6. Generation monitor for early abort
7. TOON token compression (40-65% reduction)

## Phase 1: chat.py Decomposition

### New Module Structure

```
src/api/routes/
  chat.py              3,763 -> ~300 lines (thin orchestrator)
  chat_utils.py        ~200 lines (utilities + constants)
  chat_routing.py      ~120 lines (intent classification + mode selection)
  chat_vision.py       ~600 lines (vision pipeline)
  chat_summarization.py ~200 lines (two/three-stage pipeline)
  chat_review.py       ~500 lines (architect review + quality gates)
  chat_react.py        ~200 lines (ReAct tool loop)
  chat_delegation.py   ~400 lines (architect delegation)
```

### Function-to-Module Mapping

| Function | From Line | To Module |
|----------|-----------|-----------|
| `_estimate_tokens` | :98 | chat_utils |
| `_is_summarization_task` | :103 | chat_summarization |
| `_should_use_two_stage` | :121 | chat_summarization |
| `_run_two_stage_summarization` | :158 | chat_summarization |
| `_is_ocr_heavy_prompt` | :317 | chat_vision |
| `_needs_structured_analysis` | :336 | chat_vision |
| `_handle_vision_request` | :361 | chat_vision |
| `_execute_vision_tool` | :583 | chat_vision |
| `_vision_react_mode_answer` | :660 | chat_vision |
| `_handle_multi_file_vision` | :856 | chat_vision |
| `_is_stub_final` | :978 | chat_utils |
| `_strip_tool_outputs` | :989 | chat_utils |
| `_resolve_answer` | :1043 | chat_utils |
| `_detect_output_quality_issue` | :1067 | chat_review |
| `_truncate_looped_answer` | :1118 | chat_utils |
| `_should_review` | :1150 | chat_review |
| `_architect_verdict` | :1189 | chat_review |
| `_fast_revise` | :1230 | chat_review |
| `_parse_architect_decision` | :1272 | chat_delegation |
| `_architect_delegated_answer` | :1374 | chat_delegation |
| `_needs_plan_review` | :1599 | chat_review |
| `_architect_plan_review` | :1669 | chat_review |
| `_apply_plan_review` | :1735 | chat_review |
| `_store_plan_review_episode` | :1777 | chat_review |
| `_compute_plan_review_phase` | :1843 | chat_review |
| `_should_formalize` | :1877 | chat_utils |
| `_formalize_output` | :1895 | chat_utils |
| `_parse_react_args` | :1938 | chat_react |
| `_should_use_react_mode` | :1995 | chat_react |
| `_react_mode_answer` | :2032 | chat_react |
| `_should_use_direct_mode` | :2167 | chat_routing |
| `_select_mode` | :2214 | chat_routing |
| `_classify_and_route` | :2255 | chat_routing |

### Cross-Module Dependencies

```
chat.py (orchestrator)
  imports from: ALL new modules

chat_delegation.py
  imports from: chat_react (_react_mode_answer)

chat_summarization.py
  imports from: chat_utils (estimate_tokens, constants)

All other modules: no cross-deps to new modules
```

### Also Done in Phase 1

- Deleted `src/api/services/orchestrator.py` (dead facade)
- Updated all imports to use `src.prompt_builders` directly

## Completed Phases

- **Phase 1**: chat.py God Module decomposition (3,763→561 lines, 8 modules) — see `chat-module-decomposition.md`
- **Phase 1b**: Pipeline restructure (`_handle_chat()` 1,091→80 lines, 11 stage functions)
- **Phase 2**: State management + circuit breaker (8 Protocol interfaces, BackendHealthTracker)
- **Phase 3**: Configuration consolidation (27 files, ~185 values wired to `src/config.py`, 1012+165 tests)
- **WI-9**: Staged reward shaping — PARL-inspired annealing coefficient λ(step) 0.3→0.0, exploration bonus 1/√(N+1)
- **WI-10**: Parallel gate execution — `asyncio.to_thread()` + `asyncio.gather()` for concurrent subprocess gates, `parallel_gates` feature flag
- **WI-11**: prompt_builders.py decomposition — 1,501-line monolith → 6-module package (resolves S2)

**Test count**: 2015 passed, 0 failures (67.48% coverage)

## Final Review (2026-02-01) — Grade: A- (89/100)

### Additional completed work (post-Phase 3):
- **H1-H7**: Hardening (thread-safe singletons, bare except logging, async safety, path traversal fix, base64 limits, config auth, dead code cleanup)
- **T1-T4**: Test coverage expansion (+92 new tests, real behavior testing)
- **Roadmap items A-G**: All completed (see orchestration-architecture-roadmap.md)
- **orchestration/ shell=True**: Replaced 2/3 with `shlex.split()`, documented 1 as intentional
- **Vision timeout centralization**: 6 hardcoded values → `config.py:TimeoutsConfig`
- **chat_pipeline.py decomposition**: 1,439-line monolith → `chat_pipeline/` package (4 modules: routing.py 228L, stages.py 704L, repl_executor.py 532L, __init__.py 64L)
- **repl_environment naming**: Removed `_RestrictedREPLEnvironment` from `__all__` (internal class)

### Coverage expansion (2026-02-02):
- **+314 tests** across 18 new files (cost tracking, escalation, executor, inference mixin, LlamaServer, MemRL service, model server, parsing config, LLM primitives, REPL state, roles, session models, session protocol, SQLite store, SSE utils, tools base, worker pool)
- **Bugs fixed**: memrl patch target (local import binding), ThreadPoolExecutor ordering (lifespan shutdown leak)
- **Result**: 2015 tests, 67.48% coverage

### Remaining (low priority):
- **C3**: AppState service locator → FastAPI `Depends()` DI (18 files, high effort)
- Stale handoff cleanup (many old handoffs in active/)

## Remaining Phases (Not Yet Implemented)

- **Phase 4**: Test quality (integration tests, coverage, benchmarks) — partially addressed by T1-T4
- **Phase 5**: Infrastructure hardening (rate limiting, circuit breakers, health) — partially addressed by H1-H7

## MemRL Database Cleanup (2026-01-31)

Surgically removed 6,506 validation-run contaminated entries from episodic.db while preserving 2,714 original seed data entries. FAISS index rebuilt (9,181→2,714 embeddings, -70% file size). Database ready for clean validation run.

## How to Add New Execution Modes

After decomposition, adding a new mode (e.g., "plan" mode) requires:
1. Create `src/api/routes/chat_plan.py` with handler function
2. Add mode to `_select_mode()` in `chat_routing.py`
3. Add `elif execution_mode == "plan":` branch in `chat.py`'s `_handle_chat()`
4. Write tests in `tests/unit/test_chat_plan.py`

## Resume Commands

```bash
# Run tests after decomposition
cd /mnt/raid0/llm/claude && pytest tests/unit/test_api.py tests/integration/ -v

# Run gates
cd /mnt/raid0/llm/claude && make gates

# Check for orchestrator.py import remnants
grep -r "from src.api.services.orchestrator" src/ tests/
```
