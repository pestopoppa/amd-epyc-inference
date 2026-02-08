# Handoff: Refactoring — scripts/benchmark/

**Status**: COMPLETE
**Created**: 2026-02-08
**Updated**: 2026-02-08
**Priority**: High
**Scope**: `scripts/benchmark/` (34 Python files, ~20K lines)
**Estimated effort**: 14 issues across 8 files, 4 phases

## Problem

`scripts/benchmark/` has three dominant problems: (1) `run_benchmark()` is a 532-line function with 5-level nesting, duplicated error extraction, and magic numbers; (2) `seed_specialist_routing.py` (2252 lines, 26 commits since Dec) has two 200+ line evaluation functions with parallel structure but no shared extraction; (3) test coverage is 3/34 files (8.8%), meaning most refactoring targets have no safety net. The most-changed file (`seed_specialist_routing.py`, 26 commits) has zero test coverage.

## Test Coverage Map

| Source File | Test File | Coverage | Notes |
|-------------|-----------|----------|-------|
| `scripts/benchmark/dataset_adapters.py` | `tests/unit/test_dataset_adapters.py` | Partial | Adapter sampling, tier; no adapter-specific edge cases |
| `scripts/benchmark/eval_log_format.py` | `tests/unit/test_eval_log_format.py` | Good | Format fns covered |
| `scripts/benchmark/seeding_tui.py` | `tests/unit/test_seeding_tui.py` | Good | DequeHandler, TapTailer, SeedingTUI |
| `scripts/benchmark/seed_specialist_routing.py` | — | **None** | 2252 lines, 26 commits, zero tests |
| `scripts/benchmark/run_benchmark.py` | — | **None** | 997 lines, 17 commits, zero tests |
| `scripts/benchmark/seeding_types.py` | — | **None** | 239 lines, 11 commits; dataclasses only |
| `scripts/benchmark/seeding_rewards.py` | — | **None** | 363 lines, pure functions |
| `scripts/benchmark/seeding_infra.py` | — | **None** | 324 lines, health/recovery |
| `scripts/benchmark/debug_scorer.py` | — | **None** | 525 lines; used in seeding + benchmark |
| `scripts/benchmark/results.py` | — | **None** | 568 lines; ResultsManager |
| `scripts/benchmark/score_outputs.py` | — | **None** | 365 lines |
| `scripts/benchmark/suites.py` | — | **None** | 307 lines |
| `scripts/benchmark/context_generator.py` | — | **None** | 735 lines |
| 21 other files | — | **None** | Lower churn, lower priority |

## Issue Inventory

| # | Issue | File:Line | Severity | Freq | Risk | Effort | Score | Phase |
|---|-------|-----------|----------|------|------|--------|-------|-------|
| 1 | 532-line `run_benchmark()` function | `run_benchmark.py:322` | 5 | 5 | 2.0 | 3 | 16.7 | 2 |
| 2 | Duplicated stderr error extraction (10 lines × 2) | `run_benchmark.py:644,792` | 4 | 5 | 2.0 | 1 | 40.0 | 1 |
| 3 | Timeout formula duplication (5 occurrences) | `run_benchmark.py:438,497,535,574,606` | 3 | 5 | 2.0 | 1 | 30.0 | 1 |
| 4 | Dead code: `use_server_for_speed = False` branch | `run_benchmark.py:611` | 2 | 5 | 2.0 | 1 | 20.0 | 1 |
| 5 | Dead code: `if args.summary or True:` | `score_outputs.py:315` | 1 | 1 | 2.0 | 1 | 2.0 | 1 |
| 6 | Magic numbers (timeout bases, max_tokens, temps) | `run_benchmark.py:436-629` | 3 | 5 | 2.0 | 2 | 15.0 | 1 |
| 7 | `evaluate_question_3way()` 362 lines | `seed_specialist_routing.py:584` | 4 | 5 | 2.0 | 3 | 13.3 | 3 |
| 8 | `evaluate_question()` + `evaluate_question_3way()` parallel structure | `seed_specialist_routing.py:1199,584` | 3 | 5 | 2.0 | 3 | 10.0 | 3 |
| 9 | `_ensure_loaded()` boilerplate repeated in 10+ adapters | `dataset_adapters.py:231-245` (× 10) | 3 | 2 | 1.5 | 3 | 3.0 | 4 |
| 10 | Bare imports (fragile `sys.path` manipulation) | `seed_specialist_routing.py:63-66` | 2 | 5 | 1.0 | 2 | 5.0 | 4 |
| 11 | `seeding_rewards.py` zero test coverage (pure functions) | `seeding_rewards.py:*` | 3 | 4 | 2.0 | 2 | 12.0 | 0 |
| 12 | `debug_scorer.py` zero test coverage (used everywhere) | `debug_scorer.py:*` | 4 | 3 | 2.0 | 2 | 12.0 | 0 |
| 13 | `seeding_types.py` zero test coverage (shared dataclasses) | `seeding_types.py:*` | 2 | 5 | 2.0 | 1 | 20.0 | 0 |
| 14 | `score_outputs.py:extract_answer()` naming (public but internal-only) | `score_outputs.py:109` | 1 | 1 | 2.0 | 1 | 2.0 | 1 |

Risk column: 1.0 = well-tested, 1.5 = partially tested, 2.0 = untested.
Score = (Severity × Freq × Risk) / Effort.

## Phase 0: Safety Net

Add tests for the three most-used untested modules **before** refactoring them in later phases. Without these, Phases 1–3 are blind.

### 0A: `tests/unit/test_seeding_rewards.py`

All functions are pure (no I/O, no network). Test:

| Function | What to assert |
|----------|----------------|
| `success_reward(True)` → `1.0`, `success_reward(False)` → `0.0` | Binary return |
| `compute_comparative_rewards(role_results, baseline_key)` | Reward dict keys match inputs; baseline gets 0.0 |
| `detect_escalation_chains(role_results)` | Returns list of dicts with `from_role`, `to_role`, `reward` |
| `compute_tool_value(passed_direct, passed_repl)` | `tools_helped=True` when `not passed_direct and passed_repl` |
| `score_delegation_chain(role_results)` | Worker rewards derived from delegation events |

### 0B: `tests/unit/test_debug_scorer.py`

| Function | What to assert |
|----------|----------------|
| `score_answer("42", "42", "exact_match")` → `True` | Exact match |
| `score_answer("B", "B", "multiple_choice")` → `True` | MC extraction |
| `score_answer("wrong", "42", "exact_match")` → `False` | Mismatch |
| `score_answer("", "42", "exact_match")` → `False` | Empty answer |
| `score_answer(code, test_cases, "code_execution")` | Happy path |
| `_score_programmatic(answer, config)` | IFEval constraint checks |

### 0C: `tests/unit/test_seeding_types.py`

| Class | What to assert |
|-------|----------------|
| `RoleResult(...)` | Dataclass fields, default values |
| `ComparativeResult(...)` | Serialization round-trip via `asdict()` |
| `state` singleton | `shutdown` starts False, `close_poll_client()` no-ops safely |

### Verification

```bash
pytest tests/unit/test_seeding_rewards.py tests/unit/test_debug_scorer.py tests/unit/test_seeding_types.py -v
```

## Phase 1: Quick Wins in `run_benchmark.py` and `score_outputs.py`

Low-effort, high-impact changes. Each is independently deployable.

### Files to Modify

| File | Changes |
|------|---------|
| `scripts/benchmark/run_benchmark.py` | Extract helpers, add constants, remove dead code |
| `scripts/benchmark/score_outputs.py` | Remove dead branch, rename functions |

### Implementation Order

**1a. Extract `_extract_error_hint(stderr, max_chars=80)` helper** (`run_benchmark.py`)

Two identical blocks at lines 644–653 and 792–803 differ only in truncation length (60 vs 80). Extract to:

```python
def _extract_error_hint(stderr: str, max_chars: int = 80) -> str:
    """Extract meaningful error from stderr, filtering log noise."""
    _LOG_PREFIXES = ('build:', 'main:', 'llama_model_loader:', 'print_info:', 'load_')
    _ERROR_KEYWORDS = ('error:', 'error ', 'failed', 'fatal', 'abort', 'segfault', 'exception')
    for line in reversed(stderr.split('\n')):
        line = line.strip()
        if not line:
            continue
        if any(line.startswith(p) for p in _LOG_PREFIXES):
            continue
        if any(x in line.lower() for x in _ERROR_KEYWORDS):
            return line[:max_chars]
    return ""
```

Replace both blocks with `err_hint = _extract_error_hint(result.stderr) or f"exit={result.exit_code}"`.

**1b. Extract `_compute_timeout(size_gb, base=180)` helper** (`run_benchmark.py`)

```python
_TIMEOUT_SIZE_MULTIPLIER = 3
_TIMEOUT_SIZE_BUFFER = 120

def _compute_timeout(size_gb: float, base: int = 180) -> int:
    return max(base, int(size_gb * _TIMEOUT_SIZE_MULTIPLIER) + _TIMEOUT_SIZE_BUFFER)
```

Replace 5 occurrences:
- Line 438: `timeout=_compute_timeout(size_gb)`
- Line 497: `server_timeout = _compute_timeout(size_gb, base=600)`
- Line 535: same
- Line 574: same
- Line 606: `speed_timeout = _compute_timeout(size_gb, base=300 if is_lookup else 180)`

**1c. Remove dead `use_server_for_speed` branch** (`run_benchmark.py:611–621`)

`use_server_for_speed = False` is hardcoded; the `if use_server_for_speed:` branch is unreachable. Delete lines 611–621 and dedent the else-branch.

**1d. Constants for magic numbers** (`run_benchmark.py`)

Add at top of file:

```python
_DEFAULT_MAX_TOKENS = 256
_LOOKUP_MAX_TOKENS = 512
_DEFAULT_TEMPERATURE = 0.6
_SERVER_STARTUP_TIMEOUT_BASE = 600
```

Replace occurrences at lines 436, 437, 605, 618, 629.

**1e. Fix `score_outputs.py:315`** — remove dead `or True`:

```python
# Before
if args.summary or True:  # Always show summary
# After
# Summary always shown
```

**1f. Rename private functions** (`score_outputs.py:109,139,147`):

`extract_answer` → `_extract_answer`, `extract_speed` → `_extract_speed`, `extract_acceptance` → `_extract_acceptance`. Verify no external callers first:

```bash
grep -rn "extract_answer\|extract_speed\|extract_acceptance" /mnt/raid0/llm/claude/ --include="*.py" | grep -v score_outputs.py | grep -v __pycache__
```

### Verification

```bash
# Syntax check
python -c "import ast; ast.parse(open('scripts/benchmark/run_benchmark.py').read())"
python -c "import ast; ast.parse(open('scripts/benchmark/score_outputs.py').read())"
# Grep for removed patterns
grep -n "use_server_for_speed" scripts/benchmark/run_benchmark.py  # Should be empty
grep -n "or True" scripts/benchmark/score_outputs.py  # Should be empty
# Full gate check
cd /mnt/raid0/llm/claude && make gates
```

## Phase 2: Decompose `run_benchmark()` (532 → ~4 × 130 lines)

### Files to Modify

| File | Changes |
|------|---------|
| `scripts/benchmark/run_benchmark.py` | Split `run_benchmark()` into 4 functions |

### Implementation Order

**2a. Extract `_ensure_server(active_server, model_path, ...) → ServerManager | None`** (lines 470–504)

Server lifecycle: stop if wrong model, start if needed, wait_ready. Returns the server or None.

**2b. Extract `_run_speed_test(executor, active_server, model_path, config, ...) → dict | None`** (lines 598–695)

Prompt selection, subprocess/server dispatch, output parsing, result storage. Returns speed result dict or None on error.

**2c. Extract `_run_quality_question(executor, active_server, model_path, config, question, ...) → dict | None`** (lines 707–846)

Single question execution: timeout calculation, dispatch, error extraction, result storage. Returns result dict or None.

**2d. Slim `run_benchmark()`** to orchestration only

After extraction, the main function becomes:
1. Build work items (existing `build_work_items()`)
2. For each role: discover model, measure baseline TPS
3. For each config: ensure server, run speed test, run quality questions
4. Cleanup server

~130 lines of orchestration glue, no business logic.

### Verification

```bash
python -c "import ast; ast.parse(open('scripts/benchmark/run_benchmark.py').read())"
# If Phase 0 tests exist:
pytest tests/unit/ -v -k "benchmark"
```

## Phase 3: Decompose `evaluate_question_3way()` (362 → helpers)

### Files to Modify

| File | Changes |
|------|---------|
| `scripts/benchmark/seed_specialist_routing.py` | Extract shared helpers from 3way and legacy eval |

### Implementation Order

**3a. Extract `_eval_single_config(prompt_info, role, mode, url, timeout, client, **kw) → RoleResult`**

Both `evaluate_question_3way()` (lines 594–632, 653–704, 717–784) and `evaluate_question()` (lines 1210–1268) do the same thing: call orchestrator, score answer, build `RoleResult`, erase slots on infra error, format log lines. Extract the common core.

Reduces `evaluate_question_3way()` from 362 to ~120 lines (3 calls to `_eval_single_config` + reward computation + metadata assembly).

**3b. Extract `_compute_3way_metadata(role_results, arch_results, prompt, suite) → dict`**

Lines 828–894 in `evaluate_question_3way()` build cost metrics and architect eval metadata. Pure computation, no I/O.

### Verification

```bash
python -c "import ast; ast.parse(open('scripts/benchmark/seed_specialist_routing.py').read())"
pytest tests/unit/test_seeding_tui.py tests/unit/test_inference_tap.py -v
# If Phase 0 tests exist:
pytest tests/unit/test_seeding_rewards.py -v
```

## Phase 4: Lower-Priority Cleanup (optional)

### 4a. Adapter boilerplate in `dataset_adapters.py`

Move `_ensure_loaded()` try/except/print to `BaseAdapter`:

```python
class BaseAdapter:
    def _ensure_loaded(self):
        if self._dataset is not None:
            return
        try:
            self._load_datasets()
        except Exception as e:
            print(f"  [adapter] {self.suite_name} load failed: {e}")
            self._dataset = []

    def _load_datasets(self):
        """Override in subclass to load HF datasets."""
        raise NotImplementedError
```

Subclasses only implement `_load_datasets()`. Eliminates ~100 lines of boilerplate across 10 adapters.

### 4b. Replace bare imports with relative or explicit paths

Currently `seed_specialist_routing.py` does `sys.path.insert(0, ...)` then `from seeding_types import ...`. Fragile if run from a different CWD. Options:

- Add `__init__.py` to `scripts/benchmark/` and use `from . import seeding_types` (breaks standalone execution)
- Keep `sys.path` manipulation but centralize it in a single `_bootstrap()` function
- **Recommended**: Leave as-is (low severity, works today, breaking change for all callers)

## Success Criteria

1. `run_benchmark()` has no function longer than 150 lines
2. Zero duplicated error extraction blocks
3. Zero hardcoded timeout formulas (all via `_compute_timeout()`)
4. Phase 0 adds ≥15 tests for `seeding_rewards`, `debug_scorer`, `seeding_types`
5. No test regressions: `pytest tests/ -v` passes
6. `make gates` passes
7. `python -c "import ast; ast.parse(...)"` passes for all modified files

## Notes

- **Phase 0 is mandatory before Phases 2-3.** Without tests for `seeding_rewards` and `debug_scorer`, extracting helpers from `evaluate_question_3way()` is blind.
- **Phase 1 is safe without new tests** — it's mechanical extraction (rename, move, delete dead code). AST parse + grep verification is sufficient.
- **Do not touch `dataset_adapters.py` adapter logic** — each adapter has HF-specific quirks. Only refactor the `_ensure_loaded()` boilerplate.
- **The `deprecated/` subdirectory** contains 3 archived scripts (~2500 lines). Do not refactor — they exist for git history reference only.
- **`seed_specialist_routing.py` re-exports** (lines 82–130) exist for backward compatibility with tests that import from this file. Do not remove without updating all test imports.
- **`run_benchmark.py` uses `print()` not `logger`** — this is intentional (interactive CLI output). Don't convert to logging.
