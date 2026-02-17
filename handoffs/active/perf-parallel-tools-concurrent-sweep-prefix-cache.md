# Performance: Parallel Tools, Concurrent Sweep, Prefix Cache

**Created**: 2026-02-17
**Status**: 🔥 ACTIVE
**Priority**: HIGH

## Summary

Three targeted optimizations for the orchestration stack, addressing real I/O bottlenecks (not Python overhead which is <0.1% of wall time).

| Workstream | Goal | Expected Impact |
|-----------|------|-----------------|
| WS1: Parallel read-only tools | Dispatch independent read-only REPL calls via ThreadPoolExecutor | 2-4x on multi-tool turns |
| WS2: Concurrent inference sweep | Benchmark optimal `-np`/concurrency per model tier | Data for tuning |
| WS3: Prefix cache optimization | Fix id_slot stub, compress on escalation, pre-warm architect | 0.4-1.7s per escalation |

## Workstream 1: Parallel Read-Only Tool Execution

**Files modified**:
- `src/repl_environment/parallel_dispatch.py` — NEW: AST extraction + parallel executor
- `src/repl_environment/environment.py` — Insert parallel dispatch at all_read_only branch, add `_state_lock`
- `src/repl_environment/file_exploration.py` — Lock state mutations
- `src/repl_environment/routing.py` — Lock `_tool_outputs` appends
- `src/repl_environment/code_search.py` — Lock state mutations
- `src/features.py` — Add `parallel_tools` feature flag
- `tests/unit/test_repl_parallel_dispatch.py` — NEW: unit tests

**Resume**:
```bash
python -m pytest tests/unit/test_repl_parallel_dispatch.py -v
```

## Workstream 2: Concurrent Inference Sweep

**Files created**:
- `scripts/benchmark/concurrent_inference_sweep.py` — asyncio + httpx benchmark script

**Resume**:
```bash
python scripts/benchmark/concurrent_inference_sweep.py --dry-run
python scripts/benchmark/concurrent_inference_sweep.py --roles frontdoor,worker
```

## Workstream 3: Prefix Cache Optimization

### 3A: Wire id_slot
- `src/model_server.py` — Add `slot_id` to `InferenceRequest`
- `src/backends/llama_server.py` — Add `id_slot` to payload
- `src/prefix_cache.py` — Pass computed slot_id via `dataclasses.replace`

### 3B: Escalation compression
- `src/features.py` — Add `escalation_compression` flag
- `src/graph/helpers.py` — Call `_maybe_compress_for_escalation()` when escalating

### 3C: Pre-warm architect
- `src/services/escalation_prewarmer.py` — NEW: `EscalationPrewarmer`
- `src/graph/helpers.py` — Fire prewarm task at turn 1

**Resume**:
```bash
python -m pytest tests/unit/test_prefix_cache.py -v
python -m pytest tests/ -k "escalation" -v
```

## Completion Checklist

- [x] WS1: parallel_dispatch.py with unit tests (22 tests)
- [x] WS2: concurrent_inference_sweep.py with CSV output
- [x] WS3A: id_slot wired through
- [x] WS3B: escalation compression feature-flagged
- [x] WS3C: pre-warmer implemented
- [x] Progress report updated
- [x] CHANGELOG updated
- [x] `make gates` passes (except NextPLAID reindex timeout — Docker issue)
- [ ] Git commit + push
