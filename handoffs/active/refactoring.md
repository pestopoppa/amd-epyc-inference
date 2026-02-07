# Handoff: Refactoring — src/api/routes

**Status**: ALL PHASES COMPLETE (0-4)
**Created**: 2026-02-07
**Updated**: 2026-02-07
**Priority**: Done
**Scope**: `src/api/routes/` (26 files after split, ~6,200 lines after dead code removal)
**Estimated effort**: 9 issues across 12 files, 5 phases (all resolved)

## Problem

The `src/api/routes/` scope has significant technical debt from the Phase 1 chat.py decomposition: a deprecated module still actively imported, duplicated logic across execution stages, a 350-line streaming endpoint that reimplements the entire orchestration pipeline independently, dead code from completed migrations, and 18/23 source files with zero test coverage.

## Test Coverage Map

| Source File | Test File | Coverage | Notes |
|-------------|-----------|----------|-------|
| `chat_pipeline/repl_executor.py` | `test_repl_executor.py` (23 tests) | Good | Covers `run_task` integration |
| `chat_vision.py` | `test_chat_vision.py` (35 tests) | Good | OCR, VL routing, ReAct vision |
| `chat_summarization.py` | `test_chat_summarization.py` (32 tests) | Good | Two-stage pipeline |
| `path_validation.py` | `test_path_validation.py` (6 tests) | Good | Path traversal checks |
| `openai_compat.py` | `test_openai_compat.py` (integration) | Partial | Integration only, no unit |
| `chat.py` | None | **None** | 23 git changes, highest churn |
| `chat_pipeline/stages.py` | None | **None** | 816 lines, largest file |
| `chat_pipeline/routing.py` | None | **None** | Routing + preprocessing |
| `chat_review.py` | None | **None** | Quality gates, plan review |
| `chat_routing.py` | None | **None** | Mode selection, confidence routing |
| `chat_delegation.py` | None | **None** | TOON parsing, multi-loop |
| `chat_react.py` | None | **None** | DEPRECATED, still imported |
| `chat_utils.py` | None | **None** | Constants, answer resolution |
| `sessions.py` | None | **None** | 440 lines |
| `documents.py` | None | **None** | 369 lines |
| `delegate.py` | None | **None** | 138 lines |
| `health.py` | None | **None** | 164 lines |
| `vision.py` | None | **None** | 302 lines |
| `config.py` | None | **None** | 42 lines |
| `stats.py` | None | **None** | 23 lines |
| `gates.py` | None | **None** | 63 lines |

**18 of 23 source files have zero test coverage.**

## Issue Inventory

| # | Issue | File:Line | Sev | Freq | Risk | Effort | Priority | Phase |
|---|-------|-----------|-----|------|------|--------|----------|-------|
| 1 | `delegation_allowed` computed twice identically | `stages.py:208` + `chat.py:217` | 2 | 5 | 2.0 | 1 | 20.0 | 2 |
| 2 | Quality-check-then-escalate duplicated | `stages.py:576` + `stages.py:700` | 3 | 3 | 2.0 | 1 | 18.0 | 2 |
| 3 | Streaming endpoint reimplements orchestration | `chat.py:272-622` | 5 | 5 | 2.0 | 4 | 12.5 | 3 |
| 4 | `_should_use_direct_mode()` never called | `chat_routing.py:20` | 2 | 3 | 2.0 | 1 | 12.0 | 1 |
| 5 | ChatResponse construction duplicated 4x | `stages.py:284,485,603,761` | 3 | 3 | 2.0 | 2 | 9.0 | 2 |
| 6 | Module-level `_get_config()` at import time | `chat_utils.py:34,69` | 3 | 3 | 2.0 | 2 | 9.0 | 2 |
| 7 | Deprecated `chat_react.py` still imported | `chat_delegation.py:16` + `stages.py:22` | 3 | 2 | 2.0 | 2 | 6.0 | 1 |
| 8 | `stages.py` is 816-line catch-all (7 stages) | `chat_pipeline/stages.py` | 3 | 3 | 2.0 | 4 | 4.5 | 4 |
| 9 | `_is_ocr_heavy_prompt()` always returns True | `chat_vision.py:47` | 1 | 2 | 1.0 | 1 | 2.0 | 1 |

Risk column: 1.0 = well-tested, 1.5 = partially tested, 2.0 = untested

## Phase 0: Safety Net

Add tests for the files that Phase 1-2 will modify. Without these, refactoring is blind.

### Tests to Add

| Source Function | What to Assert |
|----------------|----------------|
| `stages._execute_direct()` | Returns ChatResponse, quality escalation triggers on bad output, retry logic works |
| `stages._execute_react()` | Returns ChatResponse or None, quality escalation triggers |
| `stages._execute_delegated()` | Returns None on feature disabled, ChatResponse on success |
| `chat_routing._should_use_direct_mode()` | Delegates to classifier (verify import path) |
| `chat_routing._select_mode()` | Returns "repl" by default, respects hybrid_router |
| `chat_routing._parse_confidence_response()` | Parses CONF\| format correctly |
| `chat_utils._resolve_answer()` | Stub detection, tool output stripping |
| `chat_utils._truncate_looped_answer()` | Truncation on prompt echo, passthrough on clean |
| `chat_review._detect_output_quality_issue()` | Repetition detection, garbled detection, None on clean |
| `chat_review._should_review()` | Returns False for architects, False without hybrid_router |
| `chat_delegation._parse_architect_decision()` | TOON D\|, I\|, JSON, markdown-wrapped JSON, bare text |

Target: 40+ new tests across `test_chat_routing.py`, `test_stages.py`, `test_chat_review.py`, `test_chat_delegation.py`, `test_chat_utils.py`.

### Verification

```bash
cd /mnt/raid0/llm/claude
python3 -m pytest tests/unit/test_chat_routing.py tests/unit/test_stages.py tests/unit/test_chat_review.py tests/unit/test_chat_delegation.py tests/unit/test_chat_utils.py -v
python3 -m pytest tests/unit/ -x -q  # Full suite, no regressions
```

## Phase 1: Dead Code Removal

Low-risk cleanup. Remove code that is confirmed unused or deprecated.

### Files to Modify

| File | Changes |
|------|---------|
| `chat_vision.py:47-60` | Delete `_is_ocr_heavy_prompt()` — always returns True, dead function |
| `chat_react.py` | Mark for removal tracking (still imported, actual deletion in Phase 1b) |
| `chat_routing.py:20-47` | Delete `_should_use_direct_mode()` — never called by `_select_mode()` which always returns `"repl"` |
| `chat_routing.py:87-133` | Delete `_should_use_react_mode()` — React deprecated, uses hardcoded keywords (bypassed by classifiers) |
| `chat_delegation.py:16` | Change `from src.api.routes.chat_react import _react_mode_answer` to direct REPL usage |
| `stages.py:22` | Remove `from src.api.routes.chat_react import _react_mode_answer` |

### Implementation Order

1. Delete `_is_ocr_heavy_prompt()` from `chat_vision.py` and remove any callers (only tests)
2. Delete `_should_use_direct_mode()` and `_should_use_react_mode()` from `chat_routing.py`
3. In `chat_delegation.py:268-278`: Replace `_react_mode_answer` call with `REPLEnvironment(structured_mode=True)` — the module already imports `REPLEnvironment` on line 17
4. In `stages.py:542-549`: Replace `_react_mode_answer` call in `_execute_react()` with `REPLEnvironment(structured_mode=True)` — or flag `_execute_react()` itself as the next removal target since `_select_mode()` never returns "react"
5. Once no imports remain, delete `chat_react.py` entirely

### Verification

```bash
# Confirm no remaining imports of deleted functions
grep -rn "_is_ocr_heavy_prompt\|_should_use_direct_mode\|_should_use_react_mode" src/api/routes/ --include="*.py"

# Confirm chat_react imports are gone
grep -rn "from src.api.routes.chat_react" src/ --include="*.py"

# Tests pass
python3 -m pytest tests/unit/ -x -q
```

## Phase 2: Duplication Extraction

Extract repeated patterns into shared helpers.

### Files to Modify

| File | Changes |
|------|---------|
| `chat_pipeline/stages.py` | Extract `_build_chat_response()` helper, `_quality_escalate()` helper |
| `chat.py:217-224` | Remove duplicated `delegation_allowed` computation, use `stages._execute_delegated()`'s internal check |
| `chat_utils.py:34-36,69-88` | Convert module-level config to lazy properties or a function |

### Implementation Order

1. **Extract `_quality_escalate()`** from the duplicated pattern at `stages.py:576-590` and `stages.py:700-725`:

```python
def _quality_escalate(
    answer: str, prompt: str, primitives: LLMPrimitives, initial_role
) -> tuple[str, Any]:
    """Detect quality issue and escalate to coder_escalation if needed."""
    if not (answer and not answer.startswith("[ERROR") and features().generation_monitor):
        return answer, initial_role
    quality_issue = _detect_output_quality_issue(answer)
    if not quality_issue:
        return answer, initial_role
    try:
        escalated = primitives.llm_call(
            prompt, role="coder_escalation", n_tokens=2048, skip_suffix=True,
        )
        if escalated.strip():
            return escalated.strip(), Role.CODER_ESCALATION
    except Exception as exc:
        log.debug("Quality escalation failed: %s", exc)
    return answer, initial_role
```

2. **Extract `_build_stage_response()`** to construct ChatResponse from common fields:

```python
def _build_stage_response(
    answer: str, routing: RoutingResult, primitives: LLMPrimitives,
    state, start_time: float, initial_role, mode: str,
    tools_used: int = 0, tools_called: list | None = None,
    tool_timings: list | None = None,
    delegation_events: list | None = None,
    delegation_success: bool | None = None,
    role_history: list | None = None,
    turns: int = 1,
) -> ChatResponse:
    """Build ChatResponse with common fields populated."""
    elapsed = time.perf_counter() - start_time
    state.increment_request(mock_mode=False, turns=turns)
    # ... common scoring, progress logging, cache_stats ...
```

3. **Remove duplicated `delegation_allowed`** in `chat.py:217-224` — the check already exists inside `_execute_delegated()` at `stages.py:208-214`. Just pass the request through and let the stage handle it.

4. **Make config lazy in `chat_utils.py`** — replace module-level `_get_config()` calls with a cached function:

```python
@functools.cache
def _chat_config():
    return _get_config().chat

# Replace all references: THREE_STAGE_CONFIG → _chat_config_dict()
```

### Verification

```bash
# Run phase 0 tests first (new tests)
python3 -m pytest tests/unit/test_stages.py tests/unit/test_chat_review.py -v

# Full suite
python3 -m pytest tests/unit/ -x -q

# Verify no duplicate delegation_allowed pattern
grep -n "delegation_allowed" src/api/routes/chat.py src/api/routes/chat_pipeline/stages.py
```

## Phase 3: Streaming Parity (High Risk)

The streaming endpoint `chat_stream()` (lines 272-622 in `chat.py`) reimplements the entire orchestration loop: REPL creation, escalation tracking, review gates, MemRL scoring. Any pipeline change must be made in two places.

### Approach

Refactor `chat_stream()` to reuse the pipeline stages. The generator should wrap `_handle_chat()` stages and emit SSE events at stage boundaries rather than reimplementing each stage.

**This is the highest-risk change.** Recommend implementing behind a feature flag:

```python
# src/features.py
unified_streaming: bool = False  # Phase 3: streaming uses pipeline stages
```

### Implementation Order

1. Add `unified_streaming` feature flag
2. Create `chat_pipeline/stream_adapter.py` that wraps pipeline stages with SSE event emission
3. In `chat_stream()`, branch on feature flag: old path vs new adapter
4. Test with flag on, verify SSE event format matches
5. Remove old path once validated

### Verification

```bash
# Compare SSE output between old and new paths
curl -X POST http://localhost:8000/chat/stream -H 'Content-Type: application/json' \
  -d '{"prompt":"2+2","mock_mode":true}' 2>/dev/null | head -20

python3 -m pytest tests/unit/ -x -q
```

## Phase 4: Structural Cleanup

Split the 816-line catch-all `stages.py` into focused modules.

### Proposed Split

| Current location | New file | Functions |
|-----------------|----------|-----------|
| `stages.py:44-77` | Keep in stages.py | `_execute_mock()` |
| `stages.py:83-190` | `chat_pipeline/vision_stage.py` | `_execute_vision()` |
| `stages.py:196-312` | `chat_pipeline/delegation_stage.py` | `_execute_delegated()`, delegation helpers |
| `stages.py:358-506` | `chat_pipeline/proactive_stage.py` | `_execute_proactive()`, `_parse_plan_steps()` |
| `stages.py:512-624` | (delete or move to delegation) | `_execute_react()` — deprecated path |
| `stages.py:630-780` | `chat_pipeline/direct_stage.py` | `_execute_direct()` |
| `stages.py:786-817` | Keep in stages.py | `_annotate_error()` |

Update `chat_pipeline/__init__.py` imports accordingly.

**Do this last** — it's a structural reorg with many import changes. Only worth it after the duplication is resolved in Phase 2.

## Success Criteria

1. All 9 issues addressed across 4 phases
2. 40+ new tests in Phase 0 (test coverage for modified files)
3. No test regressions (2677+ tests passing)
4. `make gates` passes
5. `chat_react.py` deleted (275 lines removed)
6. `_is_ocr_heavy_prompt`, `_should_use_direct_mode`, `_should_use_react_mode` deleted
7. ChatResponse construction DRY (single helper)
8. Quality-escalation pattern DRY (single function)
9. Streaming endpoint shares pipeline stages (behind feature flag)

## Notes

- `chat_react.py` is marked DEPRECATED but `chat_delegation.py` still calls `_react_mode_answer()` for the ReAct investigation path inside architect delegation. The replacement (REPL with `structured_mode=True`) already exists in the same function for the REPL delegation path — so the migration is straightforward.
- `_execute_react()` in `stages.py` is unreachable from normal flow because `_select_mode()` never returns `"react"`. It can only be reached via `request.force_mode="react"`. Consider whether to keep that escape hatch.
- `chat.py` streaming has 350 lines of escalation logic that the non-streaming path handles via graph nodes. The graph module (`src/graph/`) already supports `iter_task()` which could yield intermediate states for streaming.
- Module-level `_get_config()` in `chat_utils.py` means tests must have config available at import time. This has caused test isolation issues in the past.
- `_should_use_direct_mode()` delegates to `src.classifiers` but `_select_mode()` ignores it entirely (always returns "repl"). This is either dead code or a regression from when direct mode was unified into REPL.
