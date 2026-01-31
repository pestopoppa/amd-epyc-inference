# chat.py God Module Decomposition — Phase 1

**Date**: 2026-01-31
**Status**: Complete
**Author**: Claude Opus 4.5 (architecture review session)
**Parent**: `handoffs/active/orchestrator-architecture-review.md`

## What Was Done

Decomposed the 3,763-line `src/api/routes/chat.py` God Module into 8 focused modules. The original file contained 38 functions including `_handle_chat()` (1,091 lines, ~30 code paths) which was untestable and unsafe to modify.

### New Module Structure

```
src/api/routes/
├── chat.py               # Thin orchestrator: endpoints + _handle_chat pipeline (1,558 lines)
├── chat_utils.py          # Constants + utilities (token est, stub detection, formalization)
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

### Cross-Module Dependencies

```
chat.py (thin orchestrator)
  ├── imports from: ALL 7 new modules
  ├── imports from: src.prompt_builders (build_root_lm_prompt, etc.)
  └── imports from: src.api.services.memrl, src.features, etc.

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
| `test_api_imports.py` | Rewritten: 15 tests verify all 8 modules + facade deletion |
| `test_react_mode.py` | Import paths → `chat_react`, patch targets, 3-value unpack |
| `test_architect_delegation.py` | Import paths → `chat_delegation`, patch targets |
| `test_plan_review.py` | Import paths → `chat_review` (4 functions) |
| `test_vision_routing.py` | Import paths → `chat_utils`/`chat_vision`, `asyncio.run()` fix |

### Metrics

| Metric | Before | After |
|--------|--------|-------|
| chat.py lines | 3,763 | 1,558 |
| Functions in chat.py | 38 | 5 |
| Test failures (decomposition-related) | 45 | **0** |
| Independently testable modules | 1 | 8 |
| Dead imports removed | — | 13 |

## Remaining Phases (Not Yet Implemented)

- **Phase 2**: Fix state management (DI via FastAPI Depends(), Protocol types, frozen configs)
- **Phase 3**: Configuration consolidation (paths, thresholds, magic numbers → Config class)
- **Phase 4**: Test quality (integration tests, coverage, benchmarks, 0.44x → 0.8x ratio)
- **Phase 5**: Infrastructure hardening (rate limiting, circuit breakers, health checks)

## How to Add New Execution Modes

After decomposition, adding a new mode (e.g., "plan" mode) requires:
1. Create `src/api/routes/chat_plan.py` with handler function
2. Add mode detection to `_select_mode()` in `chat_routing.py`
3. Add `elif execution_mode == "plan":` branch in `chat.py`'s `_handle_chat()`
4. Write tests in `tests/unit/test_chat_plan.py`

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
