# Backend Saturation Under Sequential Load (504/429)

**Created**: 2026-02-20
**Priority**: HIGH — blocks feature validation comparison and any sustained-load benchmark
**Status**: INVESTIGATING

## Problem Statement

When running 10 prompts sequentially through the orchestrator (`POST /chat`), the system degrades catastrophically after 5-7 prompts. Failure mode: **504 Gateway Timeout** and **429 Too Many Requests**. The degradation worsens significantly when 15 validated features are enabled (candidate mode), but also appears with all features disabled (baseline mode).

### Observed Behavior

**Run 1** (no inter-prompt cooldown, 120s client timeout):
- Baseline: 9/10 success, 1 client-side timeout at 120.1s (tool_02)
- Candidate: 5/10 success, 3×429, 1×504, 1×client timeout

**Run 2** (5s inter-prompt cooldown, 60s phase cooldown, 180s client timeout):
- Baseline: 7/10 success, 1×504 (tool_02, 105.3s), 1×504 (tool_03, 89.7s), 1×429 (tool_04, 92.5s)
- Candidate: 0/4 success before test was aborted (3×504, 1×client timeout)

Key pattern: **failures cluster in the second half of each phase** and are **worse in the candidate phase** (which runs after baseline, so backends have been under load longer).

### What This Blocks

- Clean baseline-vs-candidate comparison for the 15 validated features
- Any sustained-load benchmark (seeding runs, multi-prompt evaluations)
- Confidence that production feature enablement doesn't cause degradation

## System Architecture (Relevant Layers)

```
Client (httpx, 180s read timeout)
  → Orchestrator API (uvicorn, 6 workers, limit-concurrency 4)
    → Rate Limiter (60 RPM + 10 burst, per-IP token bucket)
      → Chat Pipeline (routing → formalization → generation → output)
        → Admission Controller (per-backend semaphore)
          → Circuit Breaker (per-backend, 3-failure threshold)
            → llama-server (per-model, N slots, spec decode)
```

### Protection Layers (all added 2026-02-11)

| Layer | Config | Effect |
|-------|--------|--------|
| Rate limiter | 60 RPM + 10 burst | 429 if bucket empty |
| Admission control | Per-backend semaphore (1-4 slots) | 429 if all slots busy + 2s wait exhausted |
| Circuit breaker | 3 consecutive failures → open | 503 for 30s (doubles on repeated probe failure, max 300s) |
| Per-role timeout | 30-600s depending on model | 504 if backend doesn't respond in time |
| Request deadline | Propagated via contextvars, post-lock re-clamp | Ensures backend timeout ≤ remaining request budget |

### Backend Capacity

| Backend | Port | Model | Slots | Admission Limit | Throughput | Role Timeout |
|---------|------|-------|-------|-----------------|------------|-------------|
| frontdoor | 8080 | Qwen3-Coder-30B-A3B | 4 | 2 | 47 t/s | 180s |
| coder_escalation | 8081 | Qwen2.5-Coder-32B | 4 | 2 | 39 t/s | 120s |
| worker | 8082 | Qwen2.5-7B | 8 | 4 | 44 t/s | 60s |
| architect_general | 8083 | Qwen3-235B-A22B | 2 | 1 (serial) | 6.1 t/s | 600s |
| architect_coding | 8084 | Qwen3-Coder-480B-A35B | 2 | 1 (serial) | 9.0 t/s | 600s |

## Hypothesis Space

### H1: Cumulative KV Cache Pressure (LIKELY)

llama-server maintains KV cache across requests within a slot. With spec decode, the KV cache grows proportionally to `accepted_tokens + rejected_drafts`. After 5-7 prompts, KV cache may be near capacity, causing:
- Slower token generation (cache thrashing)
- Increased latency → breaches role timeout → 504
- Backend becomes unresponsive → circuit breaker opens → 503/429 cascade

**Evidence**: Run 2 baseline gen_03 and gen_04 took 91.5s and 87.0s (near the old 90s frontdoor limit), suggesting the frontdoor model was already degraded by prompt 3-4.

**Test**: Compare `/health` or llama-server `/metrics` KV cache utilization before prompt 1 vs after prompt 5. Check if `POST /slots/erase` between prompts resolves degradation.

### H2: Feature Pipeline Amplifies Latency (CONTRIBUTING)

With 15 features ON, each request goes through additional pipeline stages:
- MemRL retrieval (FAISS + episodic lookup)
- Input formalization
- Specialist routing (Q-value scoring)
- Plan review
- Output formalization

These stages add serial latency. If the base generation takes 50s and pipeline overhead adds 30s, the total (80s) approaches timeout thresholds. Under KV pressure (H1), the base generation slows further, pushing totals past timeouts.

**Evidence**: Candidate failures are 504 at ~90s (old timeout) even after raising to 180s — suggesting the orchestrator's config change wasn't applied to the running process. Need to verify `TimeoutsConfig.for_role("frontdoor")` returns 180 at runtime.

**Test**: `curl -s localhost:8000/health | jq .config.timeouts.frontdoor` (if exposed), or add a `/config/timeouts` debug endpoint.

### H3: Admission Controller / Circuit Breaker Cascade (CONTRIBUTING)

If one backend times out, the admission slot is held for the full timeout duration. With frontdoor limited to 2 concurrent slots and 180s timeout, a single stuck request blocks 50% of capacity for up to 3 minutes. If the stuck request also triggers a circuit breaker open (3 failures), the backend becomes unavailable for 30s.

**Evidence**: The transition from 429 to 504 within the same run suggests circuit state changes mid-benchmark.

**Test**: Log admission slot acquire/release timestamps and circuit breaker state transitions during a benchmark run.

### H4: Uvicorn Worker Starvation (POSSIBLE)

6 uvicorn workers with `--limit-concurrency 4`. If workers are blocked waiting on backend responses (synchronous `asyncio.to_thread` calls), subsequent requests queue at the ASGI layer. With long timeouts (180s), a few stalled workers can exhaust the pool.

**Evidence**: The 2026-02-18 progress report documents stale uvicorn workers surviving reloads and holding locks.

**Test**: Monitor `ss -tlnp | grep 8000` during benchmark — check accept queue depth and worker thread count.

### H5: Config Reload Not Applied (CONFIRMED for timeout change)

The frontdoor timeout was changed from 90→180s in `model_registry.yaml`, and `orchestrator_stack.py reload orchestrator` was run. However, run 2 still shows failures at exactly 90s, suggesting the running process didn't pick up the change.

**Evidence**: `run.log` candidate failures at 90.5s, 90.6s — exactly the old 90s timeout.

**Test**: Restart orchestrator fully (`stop` + `start`) rather than hot-reload. Verify timeout via request that deliberately takes >90s.

### H6: Speculative Decode Overhead Under Concurrent Load (POSSIBLE)

Spec decode draft evaluation contends for the same KV cache slots as the target model. Under concurrent load (2 admission slots on frontdoor), the draft model's cache may evict target model's cache entries, degrading acceptance rate and causing net slowdown.

**Evidence**: The worker model (7B, 8 slots, admission=4) didn't show the same degradation pattern — it's smaller and has more headroom. The frontdoor (30B MoE, 4 slots, admission=2) degraded first.

**Test**: Run the same benchmark with `--no-speculative` on frontdoor. If degradation disappears, spec decode contention is the cause.

## Reproduction Steps

```bash
# 1. Start orchestrator stack (HOT tier only)
python3 scripts/server/orchestrator_stack.py start --hot-only

# 2. Verify health
curl -s localhost:8000/health | python3 -m json.tool

# 3. Run comparison (will likely show degradation after prompt 5-7)
python3 -u scripts/benchmark/feature_comparison.py 2>&1 | tee /tmp/comparison.log

# 4. Monitor during run (separate terminal):
watch -n 5 'curl -s localhost:8000/health | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d.get(\"backend_health\",{}), indent=2))"'
```

## Investigation Playbook

### Step 1: Confirm timeout config is applied

```bash
# Full restart (not reload)
python3 scripts/server/orchestrator_stack.py stop --all
sleep 5
python3 scripts/server/orchestrator_stack.py start --hot-only

# Verify with a deliberately slow prompt
time curl -s -X POST localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Write a 2000-word essay on quantum computing","role":"frontdoor"}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error_code','none'), d.get('elapsed_seconds','?'))"
```

If this returns 504 at ~90s, the timeout config isn't being loaded. Check `TimeoutsConfig` initialization path.

### Step 2: Isolate KV cache degradation

```bash
# Run 5 prompts, checking llama-server metrics between each
for i in 1 2 3 4 5; do
  curl -s localhost:8080/metrics | grep -E 'kv_cache|slots_state'
  curl -s -X POST localhost:8000/chat \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"Explain TCP vs UDP briefly","role":"frontdoor"}'
  echo "--- After prompt $i ---"
  curl -s localhost:8080/metrics | grep -E 'kv_cache|slots_state'
  echo ""
done
```

### Step 3: Test with inter-prompt slot erasure

Add `/slots/erase` calls between prompts in `feature_comparison.py`:

```python
# After each prompt, erase all slots on all backends
for port in [8080, 8081, 8082]:
    try:
        httpx.post(f"http://localhost:{port}/slots/erase", timeout=8)
    except Exception:
        pass
```

If this resolves degradation, KV cache pressure is the root cause.

### Step 4: Test with reduced concurrency

Edit `admission.py` to set frontdoor limit to 1 (serial). If degradation disappears, the issue is concurrent KV cache contention.

### Step 5: Test without features (pure baseline stress test)

Run 20 prompts with all features OFF. If degradation still appears after prompt 10-15, the issue is purely backend/KV cache, not feature pipeline overhead.

### Step 6: Monitor circuit breaker transitions

Add temporary logging to `health_tracker.py`:

```python
def record_failure(self, backend_url, error=None):
    # ... existing code ...
    logger.warning(f"[circuit] {backend_url}: failure #{info.failure_count}, state={info.state}, error={error}")
```

## Prior Hardening Work (Timeline)

| Date | Work | Files | Impact |
|------|------|-------|--------|
| 2026-02-11 | KV cache pressure / cascading timeouts fix | admission.py, health_tracker.py, model_registry.yaml, config.py, inference.py, stages.py | Introduced admission control, circuit breaker, per-role timeouts, explicit HTTP error codes |
| 2026-02-11 | Slot erase timeout fix | seeding_eval.py, seeding_orchestrator.py | Fixed 3s erase timeout → 8s, added proactive cancellation, inter-strategy cleanup |
| 2026-02-11 | 5 seeding script bugs | seed_specialist_routing.py, seeding_eval.py | Fixed double-timeout retry, dedup race, vision token counting |
| 2026-02-15 | Worker concurrency benchmark | model_registry.yaml | Measured optimal concurrency per backend (worker: 2 optimal) |
| 2026-02-18 | Lock-starvation closure | inference.py, primitives.py | Post-lock timeout re-clamp, request-budget diagnostics, deadline propagation |
| 2026-02-18 | Delegation loop hardening | chat_delegation.py | Break on specialist completion, timeout string classification |
| 2026-02-18 | Stale uvicorn worker fix | orchestrator_stack.py | Kill stale workers on reload, prevent lock holding by dead workers |
| 2026-02-20 | Frontdoor timeout increase | model_registry.yaml | 90→180s (NOT confirmed applied to running process) |

## Key Files

| File | Purpose |
|------|---------|
| `src/api/admission.py` | Per-backend concurrency semaphores |
| `src/api/health_tracker.py` | Circuit breaker state machine |
| `src/api/rate_limit.py` | Per-IP token bucket |
| `src/config/__init__.py` | TimeoutsConfig, HealthTrackerConfigData |
| `src/llm_primitives/inference.py` | Admission acquire/release, circuit check |
| `src/llm_primitives/primitives.py` | Request deadline, post-lock timeout clamp |
| `src/api/routes/chat_pipeline/stages.py` | Error code mapping (429/502/503/504) |
| `orchestration/model_registry.yaml` | Timeout source of truth, slot counts |
| `scripts/benchmark/feature_comparison.py` | Comparison script that triggered discovery |
| `benchmarks/results/runs/feature_validation/comparison/` | Partial results |

## Fix Applied: Slot/Admission Alignment (2026-02-20)

During investigation, discovered that every backend had **2x more llama-server slots than the admission controller allowed**. llama-server partitions KV cache evenly across all slots, so 50% of KV memory was wasted on slots that could never be occupied.

Additionally, the Feb 19 concurrency sweep (`benchmarks/results/eval/concurrent_sweep_20260219_144159.summary.json`) showed that **coder_escalation and worker should run serial** (p95 latency nearly doubles at concurrency=2).

### Changes Made

| Backend | Old Slots → New | Old Admission → New | Rationale |
|---------|-----------------|---------------------|-----------|
| frontdoor (8080) | 4 → **2** | 2 (unchanged) | Sweep: optimal at 2, p95 1.33x (ok) |
| coder_escalation (8081) | 4 → **1** | 2 → **1** | Sweep: p95 1.98x at concurrency=2 — serial only |
| worker (8082) | 8 → **1** | 4 → **1** | Sweep: all concurrent levels rejected on p95 |
| architect_general (8083) | 2 → **1** | 1 (unchanged) | Already serial |
| architect_coding (8084) | 2 → **1** | 1 (unchanged) | Already serial |
| ingest (8085) | 2 → **1** | 1 (unchanged) | Already serial |

**Expected impact**: Each active request gets the full KV cache budget instead of sharing with idle slots. For worker (was 8 slots), each request now gets 8x the KV headroom. This directly addresses H1 (KV cache pressure).

**Files changed**: `orchestration/model_registry.yaml` (slot counts), `src/api/admission.py` (admission limits).

**Not yet validated**: Requires orchestrator restart and re-run of comparison benchmark.

## Acceptance Criteria

1. 10 sequential prompts complete with ≥90% success rate in both baseline and candidate modes
2. No 504/429 errors attributable to backend saturation (as opposed to genuinely slow responses)
3. Clean comparison data: baseline vs candidate with matched success counts
4. Root cause identified and documented (which hypothesis confirmed)

## Resume Command

```bash
# Start fresh
python3 scripts/server/orchestrator_stack.py stop --all && sleep 5
python3 scripts/server/orchestrator_stack.py start --hot-only
curl -s localhost:8000/health | python3 -m json.tool

# Then follow Investigation Playbook above
```
