# Orchestration Integration with RadixAttention - YOLO Agent Handoff

**Status**: ✅ VERIFIED (2026-01-09) - 80% cache hit rate achieved
**Target**: YOLO Agent (devcontainer with --dangerously-skip-permissions)
**Created**: 2026-01-07
**Updated**: 2026-01-07 (All 9 phases implemented)
**Depends On**: RadixAttention Implementation (COMPLETE)

---

## Overview

All code for RadixAttention integration is **COMPLETE**. The YOLO agent only needs to:
1. Start llama-server instances
2. Run tests and fix any issues
3. Iterate until all tests pass
4. Benchmark cache performance

**Expected Impact**: 40-60% reduction in prefill time for RLM workloads through prefix reuse.

---

## Quick Start (YOLO Agent)

```bash
# 1. Launch devcontainer
export PATH="/mnt/raid0/llm/npm-global/bin:/mnt/raid0/llm/tools/devc/bin:$PATH"
devc /mnt/raid0/llm/claude

# 2. Start test servers (in background)
/mnt/raid0/llm/llama.cpp/build/bin/llama-server \
  -m /mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q8_0.gguf \
  --host 0.0.0.0 --port 8080 -c 4096 -np 4 -t 16 &

/mnt/raid0/llm/llama.cpp/build/bin/llama-server \
  -m /mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q8_0.gguf \
  --host 0.0.0.0 --port 8082 -c 4096 -np 4 -t 16 &

# 3. Verify servers
curl http://localhost:8080/health
curl http://localhost:8082/health

# 4. Run unit tests
python -m pytest tests/unit/test_prefix_cache.py -v

# 5. Run integration tests (mock mode)
python -m pytest tests/integration/test_cache_integration.py -v

# 6. Run integration tests (live server)
python -m pytest tests/integration/test_cache_hits.py -v --run-server

# 7. Start API server
uvicorn src.api:app --host 0.0.0.0 --port 8000 &

# 8. Run e2e validation
python scripts/test_recursive_orchestration.py -v

# 9. Run cache benchmark
python scripts/benchmark/bench_cache_performance.py --dry-run  # Mock first
python scripts/benchmark/bench_cache_performance.py  # Then live
```

---

## Implementation Status

### All Phases COMPLETE

| Phase | Description | File | Status |
|-------|-------------|------|--------|
| 1 | Server Infrastructure | (manual startup) | Ready |
| 2 | LLM Primitives Integration | `src/llm_primitives.py` | ✅ DONE |
| 3 | Model Server Factory | `src/model_server.py` | ✅ DONE |
| 4 | Registry Update | `orchestration/model_registry.yaml` | ✅ DONE |
| 5 | Integration Tests | `tests/integration/test_cache_integration.py` | ✅ DONE |
| 6 | Benchmark Script | `scripts/benchmark/bench_cache_performance.py` | ✅ DONE |
| 7 | API Integration | `src/api.py` | ✅ DONE |
| 8 | Root LM Loop | `src/api.py` | ✅ DONE |
| 9 | E2E Validation | `scripts/test_recursive_orchestration.py` | ✅ DONE |

---

## Key Code Changes Made

### `src/llm_primitives.py`
- Added `server_urls` parameter for CachingBackend initialization
- Added `_init_caching_backends()` method
- Added `get_backend()` and `get_cache_stats()` methods
- Updated `_real_call()` to use CachingBackend first
- Updated `_real_batch()` for parallel CachingBackend calls

### `src/model_server.py`
- Added `create_caching_server()` factory function
- Added `CachingModelServer` class with cache stats

### `orchestration/model_registry.yaml`
- Added `server_mode` section with role → URL mapping
- Added `dev` section for testing with smaller models

### `src/api.py`
- Added `real_mode` parameter to ChatRequest
- Added `server_urls` parameter for custom server config
- Added `cache_stats` to ChatResponse
- Implemented full Root LM Loop:
  - `_build_root_lm_prompt()` - builds prompt for frontdoor
  - `_extract_code_from_response()` - parses code from LLM output
  - Loop: get state → call frontdoor → execute in REPL → check FINAL()

---

## Test Commands

### Unit Tests (No Server Required)
```bash
# RadixAttention components
python -m pytest tests/unit/test_prefix_cache.py -v

# LLM Primitives
python -m pytest tests/unit/test_llm_primitives.py -v

# REPL environment
python -m pytest tests/unit/test_repl_environment.py -v
```

### Integration Tests (Mocked)
```bash
# Cache integration
python -m pytest tests/integration/test_cache_integration.py -v
```

### Integration Tests (Live Server Required)
```bash
# Start servers first, then:
python -m pytest tests/integration/test_cache_hits.py -v --run-server
```

### End-to-End Tests
```bash
# Start API and servers first, then:
python scripts/test_recursive_orchestration.py -v
```

### Benchmarks
```bash
# Mock mode (no server)
python scripts/benchmark/bench_cache_performance.py --dry-run

# Live mode (server required)
python scripts/benchmark/bench_cache_performance.py --server http://localhost:8082
```

---

## Troubleshooting Guide

### Server Not Responding
```bash
# Check if running
pgrep -af llama-server

# Check health
curl http://localhost:8080/health

# Restart if needed
pkill -f llama-server
# Then restart servers
```

### Import Errors
```bash
# Ensure you're in project root
cd /mnt/raid0/llm/claude

# Install dependencies if needed
pip install -e .
```

### Cache Stats All Zeros
- Verify `cache_prompt: true` is in request payload
- Check that prefix is long enough (>256 chars by default)
- Run multiple queries with same prefix

### Low Hit Rate
- Check prompt canonicalization is being applied
- Ensure prompts share common prefix
- Review `PrefixRouter.prefix_length` setting

### Root LM Loop Errors
- Check frontdoor server is running on port 8080
- Review LLM output for code extraction issues
- Check REPL execution logs for syntax errors

---

## Success Criteria

### Phase 1: RadixAttention Integration

| Metric | Target | How to Verify |
|--------|--------|---------------|
| Unit tests | All pass | `pytest tests/unit/ -v` |
| Integration tests | All pass | `pytest tests/integration/ -v --run-server` |
| Cache hit rate | >50% | `bench_cache_performance.py` |
| Prefill speedup | >40% on cached | Compare cold vs warm latency |
| Root LM loop | Works | `test_recursive_orchestration.py` |
| API real_mode | Works | `curl -X POST .../chat -d '{"real_mode": true}'` |

### Phase 2: Escalation & Throughput Tuning

| Metric | Target | How to Verify |
|--------|--------|---------------|
| Escalation latency | <500ms routing | Time from task to first token |
| Multi-turn throughput | >10 t/s aggregate | E2E test with parallel workers |
| Early abort savings | >20% token reduction | Compare with/without early abort |
| Memory pool hot-load | <2s model switch | Time warm→hot transition |

### References for Escalation Tuning

- Escalation flow: `research/ESCALATION_FLOW.md`
- Gate-based triggers: entropy, spike detection, repetition thresholds
- Context-based routing: >64K → SSM, has_images → VL worker
- Early abort: Check at token 50, 100 for entropy spikes

---

## Files Modified/Created

### Modified
| File | Changes |
|------|---------|
| `src/llm_primitives.py` | CachingBackend integration |
| `src/model_server.py` | `create_caching_server()` factory |
| `orchestration/model_registry.yaml` | `server_mode` section |
| `src/api.py` | `real_mode` + Root LM loop |

### Created
| File | Purpose |
|------|---------|
| `tests/integration/test_cache_integration.py` | LLMPrimitives cache tests |
| `scripts/benchmark/bench_cache_performance.py` | RLM cache benchmark |
| `scripts/test_recursive_orchestration.py` | E2E validation |

### Do NOT Modify (Already Tested)
- `src/radix_cache.py` - 46/46 tests passing
- `src/prefix_cache.py` - 46/46 tests passing
- `src/backends/llama_server.py` - 46/46 tests passing
- `src/repl_environment.py` - working

---

## Agent Workflow

1. **Start Servers** → Wait for health checks
2. **Run Unit Tests** → Fix any failures
3. **Run Integration Tests (mock)** → Fix any failures
4. **Run Integration Tests (live)** → Fix any failures
5. **Start API Server** → Verify health
6. **Run E2E Validation** → Fix any failures
7. **Run Benchmarks** → Verify >50% hit rate
8. **Report Results** → Update BLOCKED_TASKS.md

---

## References

- Full plan: `/home/daniele/.claude/plans/twinkly-sniffing-crescent.md`
- RadixAttention implementation: `research/radix_attention_handoff.md`
- Orchestrator overview: `research/orchestrator_handoff.md`
- Blocked tasks: `orchestration/BLOCKED_TASKS.md`
