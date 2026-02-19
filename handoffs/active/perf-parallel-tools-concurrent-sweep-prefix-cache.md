# Performance: Parallel Tools, Concurrent Sweep, Prefix Cache

**Created**: 2026-02-17
**Updated**: 2026-02-19
**Status**: 🔥 ACTIVE (implementation complete, validation remaining)
**Priority**: HIGH
**Original commit**: `882aaa0`

## Summary

Three targeted optimizations for the orchestration stack. ~99.8% of request wall time is I/O-blocked on llama-server HTTP calls; Python overhead is <20ms/request. These address real bottlenecks, not CPU overhead.

| Workstream | Goal | Status | Impact |
|-----------|------|--------|--------|
| WS1: Parallel read-only tools | ThreadPoolExecutor for independent REPL calls | ✅ IMPLEMENTED | 2-4x on multi-tool turns |
| WS2: Concurrent inference sweep | Benchmark optimal `-np`/concurrency per model tier | ⚠️ SCRIPT READY, NOT RUN | Data for tuning |
| WS3A: Wire id_slot | Fix dead PrefixRouter code — pass slot_id to llama-server | ✅ IMPLEMENTED | Enables prefix cache hits |
| WS3B: Escalation compression | LLMLingua-2 prompt compression on architect escalation | ✅ IMPLEMENTED (flag OFF) | 1.7s per escalation |
| WS3C: Pre-warm architect | Speculative prefill for COMPLEX tasks at turn 1 | ✅ IMPLEMENTED | 0.4s per escalation |

## What Was Built (commit 882aaa0, Feb 17)

### WS1: Parallel Read-Only Tool Execution

**Core**: AST-based two-pass dispatch. Parse code → extract independent read-only calls → if all safe, dispatch via `ThreadPoolExecutor`; otherwise fall through to sequential `exec()`.

**Files created/modified**:
| File | Change |
|------|--------|
| `src/repl_environment/parallel_dispatch.py` | NEW: `_ParallelCall` dataclass, `_extract_parallel_calls()`, `execute_parallel_calls()`, `_eval_ast_arg()` |
| `src/repl_environment/environment.py` | `_state_lock`, `_features`, parallel dispatch at `all_read_only` branch |
| `src/repl_environment/file_exploration.py` | `_increment_exploration()` thread-safe helper |
| `src/repl_environment/routing.py` | Lock around `_exploration_calls += 1` |
| `src/repl_environment/code_search.py` | Lock around `_exploration_calls += 1` |
| `src/features.py` | `parallel_tools: bool = True` flag |
| `tests/unit/test_repl_parallel_dispatch.py` | 22+ unit tests |

**Post-commit evolution** (by subsequent sessions):
- `parallel_dispatch.py`: Added `extract_tool_calls()`, `_ToolCallSite` dataclass for tool chaining
- `environment.py`: Gained tool chaining modes, `_READ_ONLY_REPL_TOOLS` class-level frozenset, `_get_read_only_tools()`, deferred tool results, credential redaction (+407 lines)
- `prefix_cache.py`: Added `_should_bypass_slot_routing()` for frontdoor REPL requests (+22 lines)
- `features.py`: Added `deferred_tool_results`, `script_interception`, `credential_redaction`, `cascading_tool_policy`, `depth_model_overrides` flags (+27 lines)

### WS2: Concurrent Inference Sweep

**File**: `scripts/benchmark/concurrent_inference_sweep.py` (478 lines)

Script written, **never executed**. Requires orchestration stack running.

- asyncio + httpx.AsyncClient for true concurrent HTTP
- Fixed ~300-token code prompt, `n_predict=128`, `temperature=0.0`, `cache_prompt=false`
- 2 warmup + 5 measured batches per (port, concurrency) config
- Incremental CSV output
- TTFT baseline via streaming probe
- CLI: `--roles`, `--concurrency`, `--n-measured`, `--n-predict`, `--skip-architects`, `--dry-run`, `--yes`

**Test matrix**:
| Role | Port | Current -np | Test concurrency |
|------|------|------------|-----------------|
| frontdoor (30B MoE) | 8080 | 1 | 1, 2, 3 |
| coder (32B dense) | 8081 | 1 | 1, 2 |
| worker (7B) | 8082 | 2 | 1, 2, 3, 4 |
| architect_general (235B) | 8083 | 1 | 1, 2 |
| fast_worker (1.5B) | 8102 | 4 | 1, 2, 4, 6, 8 |

### WS3A: Wire id_slot (Fix Dead Code)

| File | Change |
|------|--------|
| `src/model_server.py` | `slot_id: int | None = None` on `InferenceRequest` |
| `src/backends/llama_server.py` | `"id_slot": request.slot_id` in `_build_payload()` |
| `src/prefix_cache.py` | `CachingBackend.infer()` and `infer_stream_text()` pass computed slot via `dataclasses.replace()` |

### WS3B: Escalation Compression

| File | Change |
|------|--------|
| `src/features.py` | `escalation_compression: bool = False` (OFF by default) |
| `src/graph/helpers.py` | `_maybe_compress_for_escalation()` — triggers when `escalation_count > 0` AND prompt > 16K chars, 50% target ratio, preserves code tokens |

### WS3C: Pre-warm Architect

| File | Change |
|------|--------|
| `src/services/escalation_prewarmer.py` | NEW: `EscalationPrewarmer` with `prewarm_if_complex()`, `/slots` check, `n_predict=0` prefill |
| `src/graph/helpers.py` | `_maybe_prewarm_architect()` fired at turn 1 for COMPLEX tasks |

## Remaining Tasks

### 1. Run Concurrent Inference Sweep (WS2) — HIGH

The script is ready. Requires live servers.

```bash
# Dry run first
python scripts/benchmark/concurrent_inference_sweep.py --dry-run

# Safe subset (no architect ports)
python scripts/benchmark/concurrent_inference_sweep.py --roles frontdoor,worker,fast_worker --skip-architects

# Full run
python scripts/benchmark/concurrent_inference_sweep.py --yes
```

**After**: Update `docs/reference/benchmarks/RESULTS.md` with optimal concurrency findings.

### 2. Validate WS3B Compression Quality — MEDIUM

`escalation_compression` is flagged OFF. Before enabling:
- A/B compare architect escalation latency with/without compression on 3+ sample tasks
- Verify 50% compression doesn't drop critical context
- Check `force_tokens=["FINAL", "def ", "class ", "import "]` adequately preserves code structure

```bash
# Enable temporarily
ORCHESTRATOR_ESCALATION_COMPRESSION=1 python -c "from src.features import get_features; f = get_features(production=True); print(f.escalation_compression)"
```

### 3. Validate WS3C Pre-warmer Hit Rate — LOW

Log what % of pre-warmed slots are actually used by subsequent escalation.

```python
from src.services.escalation_prewarmer import EscalationPrewarmer
pw = EscalationPrewarmer(...)
print(pw.get_stats())  # prewarm_count, prewarm_hits, hit_rate
```

**Risk note**: Architect runs `-np 1`. Pre-warming the only slot may be counterproductive if the actual escalation uses a different prompt prefix.

### 4. Verify WS3A id_slot Routing — LOW

With running servers, check `PrefixRouter.get_stats()` shows non-zero `cache_hits` after multi-turn conversation.

## Tests

```bash
# All unit tests for this workstream
python -m pytest tests/unit/test_repl_parallel_dispatch.py -v
python -m pytest tests/unit/test_features.py -v

# Full suite (should pass — 1762+ at time of commit, 3700+ now)
make test-all
```

## Known Issue: worker_summarize SERIAL_ROLES Inconsistency

**Discovered**: 2026-02-19 during context-window-management research

`worker_summarize` has contradictory configuration across three files:

| Source | Maps to | Parallel? |
|--------|---------|-----------|
| `orchestration/model_registry.yaml:351` | 7B pool (port 8082, 8 slots) via `shared_with: [worker_math, worker_summarize, toolrunner]` | Yes (8 slots) |
| `orchestration/README.md:137` | 32B (port 8081) | Implied serial (1 slot) |
| `scripts/server/orchestrator_stack.py:157` | — (`SERIAL_ROLES` set) | **No — forced serial** |

**Runtime behavior**: `SERIAL_ROLES` in `orchestrator_stack.py` is the runtime authority. So `worker_summarize` is **serialized in practice**, even though its backing 7B server has 8 slots and could handle concurrent requests.

**Impact**: Any feature using `worker_summarize` (e.g., conversation compaction in `handoffs/active/context-window-management.md`) will be serialized. Using `worker_general` or `worker_explore` (same 7B server, NOT in `SERIAL_ROLES`) avoids this bottleneck.

**Resolution options** (pick during WS2 sweep):
1. Remove `worker_summarize` from `SERIAL_ROLES` — allows parallel summarization on 7B
2. Keep serial but document that compaction should use `worker_explore` role instead
3. Clarify whether summarize should target 32B (README says 8081) or 7B (registry says 8082) — currently contradictory

This should be resolved as part of the WS2 concurrent inference sweep, which will provide the data to decide optimal concurrency per role.

## Design Decisions

1. **Conservative fallback**: `_extract_parallel_calls()` returns `None` on any ambiguity → sequential `exec()`. Safe default.
2. **Coarse locking**: Single `_state_lock` on REPLEnvironment, held only for counter increments/list appends (nanoseconds). I/O stays outside lock.
3. **Feature flags**: `parallel_tools=True` (on by default), `escalation_compression=False` (off until validated).
4. **No `-np` restart in sweep**: Testing `concurrency > current_np` gives queue-behind data (still useful), but doesn't test actual parallel slot performance. Defer server restarts to follow-up.
