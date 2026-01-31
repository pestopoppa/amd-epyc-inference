# chat.py God Module Decomposition — Phases 1 + 1b + 2 + 3

**Date**: 2026-01-31
**Status**: Complete (Phase 1 + Phase 1b + Phase 2 + Phase 3)
**Author**: Claude Opus 4.5 (architecture review session)
**Parent**: `handoffs/active/orchestrator-architecture-review.md`

## What Was Done

### Phase 1: Function Extraction (33 functions → 7 modules)

Decomposed the 3,763-line `src/api/routes/chat.py` God Module into 8 focused modules. The original file contained 38 functions including `_handle_chat()` (1,091 lines, ~30 code paths) which was untestable and unsafe to modify.

### Phase 1b: Pipeline Restructure + KV Cache Bug Fix

Restructured `_handle_chat()` from 1,091 lines to ~80-line thin dispatcher calling named pipeline stage functions in `chat_pipeline.py`. Simultaneously integrated KV cache pressure bug fix: explicit error_code/error_detail on ChatResponse instead of silent HTTP 200 OK with error strings.

### New Module Structure

```
src/api/routes/
├── chat.py               # Thin dispatcher: endpoints + ~80-line _handle_chat (561 lines)
├── chat_pipeline.py       # Pipeline stages: route → preprocess → init → execute → annotate (1,203 lines)
├── chat_utils.py          # Constants + utilities + RoutingResult + ROLE_TIMEOUTS (297 lines)
├── chat_vision.py         # Vision pipeline (OCR, VL routing, ReAct VL, multi-file)
├── chat_summarization.py  # Two-stage/three-stage context processing pipeline
├── chat_review.py         # Architect review, quality gates, plan review
├── chat_react.py          # ReAct tool loop (Thought/Action/Observation)
├── chat_delegation.py     # Architect delegation (TOON parsing, multi-loop dispatch)
└── chat_routing.py        # Intent classification, mode selection, specialist routing
```

### Function-to-Module Mapping

| Function | Original Line | New Module |
|----------|--------------|------------|
| `_estimate_tokens` | :98 | chat_utils |
| `_is_stub_final` | :978 | chat_utils |
| `_strip_tool_outputs` | :989 | chat_utils |
| `_resolve_answer` | :1043 | chat_utils |
| `_truncate_looped_answer` | :1118 | chat_utils |
| `_should_formalize` | :1877 | chat_utils |
| `_formalize_output` | :1895 | chat_utils |
| `_is_ocr_heavy_prompt` | :317 | chat_vision |
| `_needs_structured_analysis` | :336 | chat_vision |
| `_handle_vision_request` | :361 | chat_vision |
| `_execute_vision_tool` | :583 | chat_vision |
| `_vision_react_mode_answer` | :660 | chat_vision |
| `_handle_multi_file_vision` | :856 | chat_vision |
| `_is_summarization_task` | :103 | chat_summarization |
| `_should_use_two_stage` | :121 | chat_summarization |
| `_run_two_stage_summarization` | :158 | chat_summarization |
| `_detect_output_quality_issue` | :1067 | chat_review |
| `_should_review` | :1150 | chat_review |
| `_architect_verdict` | :1189 | chat_review |
| `_fast_revise` | :1230 | chat_review |
| `_needs_plan_review` | :1599 | chat_review |
| `_architect_plan_review` | :1669 | chat_review |
| `_apply_plan_review` | :1735 | chat_review |
| `_store_plan_review_episode` | :1777 | chat_review |
| `_compute_plan_review_phase` | :1843 | chat_review |
| `_parse_react_args` | :1938 | chat_react |
| `_should_use_react_mode` | :1995 | chat_react |
| `_react_mode_answer` | :2032 | chat_react |
| `_parse_architect_decision` | :1272 | chat_delegation |
| `_architect_delegated_answer` | :1374 | chat_delegation |
| `_should_use_direct_mode` | :2167 | chat_routing |
| `_select_mode` | :2214 | chat_routing |
| `_classify_and_route` | :2255 | chat_routing |

### Constants Moved to chat_utils.py

- `THREE_STAGE_CONFIG` — Three-stage summarization thresholds
- `TWO_STAGE_CONFIG` — Alias for THREE_STAGE_CONFIG
- `QWEN_STOP` — Qwen chat-template stop token `<|im_end|>`
- `LONG_CONTEXT_CONFIG` — Long context exploration thresholds
- `_STUB_PATTERNS` — FINAL() stub detection patterns
- `ROLE_TIMEOUTS` — Role → timeout mapping (Phase 1b)
- `DEFAULT_TIMEOUT_S` — Fallback timeout 120s (Phase 1b)

### Phase 1b: Pipeline Stage Functions (chat_pipeline.py)

| Function | Stage | Description |
|----------|-------|-------------|
| `_route_request()` | 1 | HybridRouter, failure veto, MemRL logging → RoutingResult |
| `_preprocess()` | 2 | Input formalization (MathSmith-8B gate) |
| `_init_primitives()` | 3 | LLMPrimitives setup, server URL, backend health check |
| `_execute_mock()` | 4 | Mock mode simulated response (early return) |
| `_plan_review_gate()` | 5 | Architect pre-execution plan review |
| `_execute_vision()` | 6 | Vision pipeline (OCR, VL, multi-file) |
| `_execute_delegated()` | 7 | Architect → specialist delegation |
| `_execute_react()` | 8 | ReAct tool loop with quality check |
| `_execute_direct()` | 9 | Direct LLM call + formalization + review |
| `_execute_repl()` | 10 | Multi-turn REPL orchestration, escalation, summarization |
| `_annotate_error()` | post | Detect `[ERROR:`/`[FAILED:` → set error_code (504/502/500) |

### Phase 1b: New Dataclasses

**RoutingResult** (chat_utils.py) — Encapsulates all routing decisions:
- `task_id`, `task_ir`, `use_mock`, `routing_decision`, `routing_strategy`
- `formalization_applied`, `timeout_s`
- Properties: `role` (primary role), `timeout_for_role(role)` (per-role lookup)

**ROLE_TIMEOUTS** (chat_utils.py) — Role-specific timeout mapping:
- Workers (7B): 30s
- Frontdoor/coder_primary (30B MoE): 60s
- Coder escalation/ingest (32B/80B): 120s
- Architects (235B/480B): 300s

### Phase 1b: KV Cache Bug Fix (error_code/error_detail)

**Bug**: KV cache pressure causes llama-server timeouts → `[ERROR: ...]` strings returned as HTTP 200 OK (silent failure). Benchmarks silently collected empty/error results.

**Fix**: `_annotate_error()` detects error patterns in ChatResponse.answer:
- `[ERROR: ... timed out ...]` → error_code=504, error_detail=answer
- `[ERROR: ... backend/failed ...]` → error_code=502
- `[ERROR: ...]` (other) → error_code=500
- `[FAILED: ...]` → error_code=500
- Success → error_code=None, error_detail=None

**ChatResponse** (responses.py) — Added fields:
- `error_code: int | None` — None=success, 504=timeout, 502=backend, 500=generic
- `error_detail: str | None` — Structured error description

### Cross-Module Dependencies

```
chat.py (thin dispatcher, 561 lines)
  ├── imports from: chat_pipeline (11 stage functions)
  ├── imports from: chat_utils (RoutingResult, constants)
  ├── imports from: chat_routing (_select_mode)
  └── imports from: src.prompt_builders, src.api.services.memrl, src.features

chat_pipeline.py (pipeline stages, 1,203 lines)
  ├── imports from: chat_utils (RoutingResult, ROLE_TIMEOUTS, constants)
  ├── imports from: chat_vision, chat_summarization, chat_review
  ├── imports from: chat_react, chat_delegation, chat_routing
  └── imports from: src.prompt_builders, src.features, src.api.state

chat_utils.py (leaf — no new-module deps)
  └── imports from: src.features, src.prompt_builders

chat_vision.py
  ├── imports from: chat_utils (QWEN_STOP)
  ├── imports from: chat_summarization (_run_two_stage_summarization)
  └── imports from: src.prompt_builders (VISION_REACT_EXECUTABLE_TOOLS, etc.)

chat_summarization.py
  ├── imports from: chat_utils (_estimate_tokens, TWO_STAGE_CONFIG, LONG_CONTEXT_CONFIG)
  └── no other new-module deps

chat_review.py (leaf — no new-module deps)
  └── imports from: src.prompt_builders, src.proactive_delegation

chat_react.py
  ├── imports from: chat_utils (QWEN_STOP)
  ├── imports from: src.features, src.prompt_builders
  └── no other new-module deps

chat_delegation.py
  ├── imports from: chat_react (_react_mode_answer)
  └── imports from: src.repl_environment, src.prompt_builders

chat_routing.py
  ├── imports from: chat_react (_should_use_react_mode) [lazy in _select_mode]
  └── imports from: src.features, src.roles
```

### Also Done

- **Deleted**: `src/api/services/orchestrator.py` (dead facade)
- **Updated**: All imports to use `src.prompt_builders` directly
- **Migrated**: `ESCALATION_ROLES` dict from orchestrator.py → `src/api/services/__init__.py` (inline)

## Final Test Results

| Suite | Passed | Failed | Skipped | Notes |
|-------|--------|--------|---------|-------|
| Unit tests | 891 | 11 | 13 | All 11 failures pre-existing (pdf_router=2, worker_pool=9) |
| Integration tests | 129 | 13 | 12 | All 13 failures pre-existing (document/archive pipeline) |
| Decomposition-affected tests | 121 | 0 | 0 | All pass: react, delegation, plan_review, vision, api_imports |

### Test Files Updated

| File | Changes |
|------|---------|
| `test_api_imports.py` | Phase 1: 15 tests. Phase 1b: +16 tests (pipeline imports, RoutingResult, timeouts, annotate_error) |
| `test_react_mode.py` | Import paths → `chat_react`, patch targets, 3-value unpack |
| `test_architect_delegation.py` | Import paths → `chat_delegation`, patch targets |
| `test_plan_review.py` | Import paths → `chat_review` (4 functions) |
| `test_vision_routing.py` | Import paths → `chat_utils`/`chat_vision`, `asyncio.run()` fix |

### Phase 1b New Tests (16 tests in test_api_imports.py)

| Class | Tests | What's Covered |
|-------|-------|----------------|
| `TestChatPipelineImports` | 3 | Module import, all 11 stage functions importable, RoutingResult importable |
| `TestRoutingResult` | 4 | Default values, role property (populated/empty), timeout_for_role |
| `TestRoleTimeouts` | 3 | All known roles have timeouts, workers < architects, DEFAULT_TIMEOUT_S bounds |
| `TestAnnotateError` | 6 | Success unchanged, timeout→504, backend→502, generic→500, failed→500, defaults None |

### Metrics

| Metric | Before | After Phase 1 | After Phase 1b |
|--------|--------|---------------|----------------|
| chat.py lines | 3,763 | 1,558 | **561** |
| _handle_chat() lines | 1,091 | 1,091 | **~80** |
| Functions in chat.py | 38 | 5 | 3 |
| Pipeline modules | 0 | 0 | **1** (chat_pipeline.py, 1,203 lines) |
| Test failures (decomp-related) | 45 | 0 | **0** |
| Independently testable modules | 1 | 8 | **9** |
| Dead imports removed | — | 13 | 13 |
| New tests added | — | 15 | **31** (15 + 16) |

## Phase 2: State Management + Circuit Breaker (Complete)

### Protocol Interfaces (src/api/protocols.py — NEW)

8 runtime-checkable Protocol interfaces replacing `Any` types on AppState:
- `QScorerProtocol`, `EpisodicStoreProtocol`, `HybridRouterProtocol`
- `ProgressLoggerProtocol`, `ToolRegistryProtocol`, `ScriptRegistryProtocol`
- `RegistryLoaderProtocol`, `FailureGraphProtocol`

All under `TYPE_CHECKING` guard — zero runtime import cost. Structural subtyping: existing classes satisfy protocols without inheriting.

### BackendHealthTracker (src/api/health_tracker.py — NEW)

Circuit breaker with three states: closed → open → half-open → closed.

| State | Behavior |
|-------|----------|
| closed | Normal. Failures increment counter. |
| open | Fast-fail (no HTTP sent). After cooldown, → half-open. |
| half-open | Allow one probe. Success → closed. Failure → open (double cooldown, max 300s). |

Thread-safe via `threading.Lock`. Integrated at `LLMPrimitives._call_caching_backend()` — checks `is_available()` before dispatch, records `record_success()`/`record_failure()` after. Fast-fail before HTTP request avoids 300s timeout waits.

### Health Endpoint Enriched

`/health` now returns per-backend circuit status via `backend_health` field on `HealthResponse`. Status "ok" when all circuits closed, "degraded" when any open/half-open.

### Feature Validation

Added 3 dependency rules: `specialist_routing`, `plan_review`, `architect_delegation` all require `memrl`.

### Phase 2 Files

| File | Change |
|------|--------|
| `src/api/protocols.py` | **NEW** — 8 Protocol interfaces |
| `src/api/health_tracker.py` | **NEW** — BackendHealthTracker circuit breaker |
| `src/api/state.py` | 8 `Any` → Protocol types, +health_tracker field |
| `src/features.py` | +3 dependency validations |
| `src/api/__init__.py` | Feature validation at startup, health_tracker cleanup |
| `src/llm_primitives.py` | health_tracker integration (pre-dispatch check + post-dispatch record) |
| `src/api/routes/health.py` | Per-backend health aggregation |
| `src/api/models/responses.py` | +backend_health field on HealthResponse |
| `tests/unit/test_health_tracker.py` | **NEW** — 18 circuit breaker tests |
| `tests/unit/test_api_imports.py` | +5 Protocol/health_tracker tests |

### Phase 2 Test Results

| Suite | Passed | Notes |
|-------|--------|-------|
| test_health_tracker.py | 18/18 | All circuit states, cooldown, thread safety |
| test_api_imports.py | 37/37 | +5 new Protocol/health_tracker tests |
| Full unit suite | 940/940 | Excluding 2 pre-existing pdf_router failures |

## Phase 3: Configuration Consolidation (Complete)

Wired dead `src/config.py` into runtime. ~185 hardcoded config values from 27 files migrated to centralized config with env var override support.

### config.py Expansion (1039 lines)

7 new dataclass sections: `ServerURLsConfig`, `TimeoutsConfig`, `VisionConfig`, `ChatPipelineConfig`, `DelegationConfig`, `ServicesConfig`, `WorkerPoolPathsConfig`. Each has pydantic-settings variant for `ORCHESTRATOR_*` env var overrides.

### Wiring (27 files, 4 waves)

| Wave | Files | Risk |
|------|-------|------|
| 0 | `src/config.py`, `tests/unit/test_config_consolidation.py` (NEW) | LOW |
| 1 | 10 leaf consumers (vision, services/*, session, gradio) | LOW |
| 2 | 8 mid-level consumers (backends, monitor, llm_primitives, escalation, executor, context) | MEDIUM |
| 3 | 8 top-level consumers (chat_utils, chat_vision, chat_review, api/__init__, delegation, registry, tools) | MEDIUM |

### Patterns

- `default_factory=lambda: helper_fn()` for dataclass fields
- Lazy imports inside function bodies to prevent circular deps
- `__init__` with None defaults for non-dataclass classes
- Static fallbacks preserved for backward compat
- `repl_environment.py` security paths left hardcoded (by design)

### Tests

70 config consolidation tests (52 defaults + 8 env overrides + 18 wiring verification). Full suite: 1012 unit + 165 integration pass. `reset_config()` autouse fixture in `tests/conftest.py`.

### Commit

`5290955` — feat: Phase 3 configuration consolidation — wire dead config.py into runtime

## MemRL Database Cleanup (2026-01-31)

Validation script runs on Jan 30-31 contaminated `episodic.db` with 6,506 routing entries (57 unique questions × ~114 repetitions). Surgically removed via date-based SQL delete + FAISS index rebuild. Preserved all 2,714 original Jan 28 seed entries (1,213 with `is_seed: true`). Database ready for clean validation run.

## Remaining Phases (Not Yet Implemented)

- **Phase 4**: Test quality (integration tests, coverage, benchmarks, 0.44x → 0.8x ratio)
- **Phase 5**: Infrastructure hardening (rate limiting, CORS tightening, structured logging)

## How to Add New Execution Modes

After Phase 1b pipeline restructure, adding a new mode (e.g., "plan" mode) requires:
1. Create `src/api/routes/chat_plan.py` with `_execute_plan()` returning `ChatResponse | None`
2. Add mode detection to `_select_mode()` in `chat_routing.py`
3. Add handler call in `chat.py`'s `_handle_chat()` dispatcher (~3 lines)
4. Wrap result with `_annotate_error()` for automatic error signaling
5. Write tests in `tests/unit/test_chat_plan.py`

## Resume Commands

```bash
# Run tests after decomposition
cd /mnt/raid0/llm/claude && pytest tests/unit/test_api.py tests/integration/ -v

# Run gates
cd /mnt/raid0/llm/claude && make gates

# Check for stale imports
grep -r "from src.api.services.orchestrator" src/ tests/

# Verify new modules import correctly
python3 -c "from src.api.routes.chat_utils import QWEN_STOP; print('OK')"
python3 -c "from src.api.routes.chat_routing import _classify_and_route; print('OK')"
```

## Key Design Decisions

1. **Function names preserved** — All `_` prefixed names kept identical to minimize diff in `_handle_chat()`. Callers just change `_foo()` to `chat_utils._foo()` or use explicit imports.
2. **chat_stream() included** — Decomposed alongside `_handle_chat()` (uses same module imports). Not deferred.
3. **orchestrator.py deleted** — Dead facade removed. `ESCALATION_ROLES` dict moved inline to `src/api/services/__init__.py` since it's only used by the services package.
4. **No behavior changes** — Pure extract-and-move. All logic, thresholds, and heuristics preserved exactly.
5. **Pipeline stages return ChatResponse | None** (Phase 1b) — Mode handlers return `None` to signal "fall through to next mode", preserving original cascading semantics.
6. **_annotate_error() wraps all returns** (Phase 1b) — Single post-processing step for KV cache bug fix. No mode handler code needed to change.
7. **RoutingResult is mutable dataclass** (Phase 1b) — Not frozen, because `_preprocess()` mutates `formalization_applied`. Will freeze in Phase 2 when DI passes immutable routing context.
