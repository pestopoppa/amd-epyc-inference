# Handoff: Orchestrator Architecture Audit Remediation

**Created**: 2026-02-04
**Status**: COMPLETE
**Priority**: HIGH
**Triggered by**: Senior developer audit of orchestrator architecture (7 findings)

---

## Context

An external audit identified 7 issues across the orchestrator's routing, evaluation, and telemetry subsystems. The findings fall into three severity tiers:

- **3 HIGH**: Routing policy drift, infrastructure error corruption of Q-values, brittle delegation attribution
- **3 MEDIUM**: Architect routing heuristic, tool/delegation signal conflation, eval cache bias
- **1 LOW**: JSONL checkpoint concurrency

The overnight 3-way evaluation (134 questions, ~8 hours) confirmed a key symptom: Q-values are nearly all 1.0, providing no semantic discrimination. Direct answer mode won 84% of questions on speed alone. The routing system is currently unable to learn *when* escalation is worthwhile because the reward signal lacks fidelity.

### Phase → Finding Mapping

| Phase | Findings | Severity |
|-------|----------|----------|
| Phase 1 | Finding 1 (routing policy drift) | HIGH |
| Phase 2 | Finding 2 (infra errors corrupt Q-values) | HIGH |
| Phase 3 | Findings 3 + 5 (delegation attribution + tool/delegation signal) | HIGH + MEDIUM |
| Phase 4 | Findings 4 + 6 + 7 (architect heuristic + eval cache + JSONL concurrency) | MEDIUM + MEDIUM + LOW |

---

## Phase 1: Unified Routing Facade (Finding 1 — HIGH)

### Problem

Two parallel escalation systems exist and can drift:

| System | File | Used? |
|--------|------|-------|
| `EscalationPolicy.decide()` | `src/escalation.py` | YES — called in `chat.py:467,488,562` and `repl_executor.py:267,341,358,442` |
| `FailureRouter.route_failure()` | `src/failure_router.py` | NO — initialized in `src/api/__init__.py:89` but never called |

Both duplicate `ErrorCategory` enums. `FailureRouter` has the MemRL integration (`LearnedEscalationPolicy`) that `EscalationPolicy` lacks, but this learned component is unreachable because `route_failure()` is never called from production code.

### Pre-Flight Verification (MANDATORY)

Before starting this phase, confirm FailureRouter call sites:
```bash
rg "route_failure" src/     # Should only show definition + docstring examples
rg "FailureRouter" src/     # Should show: failure_router.py, state.py, api/__init__.py
rg "FailureRouter" tests/   # Will show ~40 tests in test_failure_router.py + test_generation_monitor.py
```

**Verified 2026-02-04**: `route_failure()` is never called from production API code. It IS used in:
- `tests/unit/test_failure_router.py` (~40 tests exercising FailureRouter directly)
- `tests/unit/test_generation_monitor.py` (2 integration tests)

**Consequence**: The facade must preserve `FailureRouter`'s public API for test compatibility. Either:
- Keep `FailureRouter` as a thin wrapper delegating to the facade, OR
- Update tests to use `RoutingFacade` directly and deprecate `FailureRouter`

Recommended: Keep wrapper for now, migrate tests in a follow-up.

### Fix

Create a `RoutingFacade` that wraps both: rules are authoritative, MemRL is advisory.

### Files

| File | Action | Detail |
|------|--------|--------|
| `src/routing_facade.py` | CREATE | `RoutingFacade` class with `decide(context)` method. Tries `LearnedEscalationPolicy.query()` first; if confident and suggestion doesn't violate rule constraints (e.g., FORMAT/SCHEMA errors must not escalate), uses learned decision. Otherwise falls back to `EscalationPolicy.decide()`. Tracks strategy counts (`learned` vs `rules`) for telemetry |
| `src/escalation.py` | MODIFY | Add `ErrorCategory.INFRASTRUCTURE = "infrastructure"` to enum (used by Phase 2). No other changes — this module remains the authoritative rule source |
| `src/failure_router.py` | MODIFY | Add deprecation notice. Remove duplicated `ErrorCategory` enum — import from `src.escalation` instead. `FailureRouter` class remains for backwards compat but becomes a thin wrapper around the facade |
| `src/api/__init__.py` | MODIFY | Line ~89: Replace `FailureRouter()` init with `RoutingFacade(policy=EscalationPolicy(), learned=LearnedEscalationPolicy(retriever))`. Store on `state.routing_facade`. Keep `state.failure_router` as deprecated alias |
| `src/api/state.py` | MODIFY | Add `routing_facade: RoutingFacade | None = None` field |
| `src/api/routes/chat.py` | MODIFY | Lines 467, 488, 562: Replace `EscalationPolicy().decide(...)` with `request.app.state.routing_facade.decide(...)` |
| `src/api/routes/chat_pipeline/repl_executor.py` | MODIFY | Lines 267, 341, 358, 442: Same replacement |
| `tests/unit/test_routing_facade.py` | CREATE | ~15 tests: facade delegates to rules by default; uses learned when confident; rejects learned when it violates constraints (FORMAT/SCHEMA must not escalate); tracks strategy counts; handles `learned=None` gracefully |

### RoutingFacade Design

```python
class RoutingFacade:
    """Single entry point for all escalation/routing decisions.

    Rules (EscalationPolicy) are authoritative.
    MemRL (LearnedEscalationPolicy) is advisory — consulted first,
    but only used when confident and when the suggestion doesn't
    violate rule constraints.
    """

    def __init__(
        self,
        policy: EscalationPolicy,
        learned: LearnedEscalationPolicy | None = None,
        confidence_threshold: float = 0.7,
    ):
        self.policy = policy
        self.learned = learned
        self.confidence_threshold = confidence_threshold
        self._strategy_counts = {"learned": 0, "rules": 0}

    def decide(self, context: EscalationContext) -> EscalationDecision:
        # 1. Try learned first (advisory)
        if self.learned is not None:
            learned_result = self.learned.query(context)
            if (
                learned_result
                and learned_result.should_use_learned
                and learned_result.confidence >= self.confidence_threshold
            ):
                # 2. Validate against rule constraints
                decision = self._validate_learned(context, learned_result)
                if decision is not None:
                    self._strategy_counts["learned"] += 1
                    return decision

        # 3. Fall back to rules (authoritative)
        self._strategy_counts["rules"] += 1
        return self.policy.decide(context)

    def _validate_learned(self, context, learned_result) -> EscalationDecision | None:
        """Reject learned suggestions that violate rule constraints."""
        # FORMAT/SCHEMA: rules say never escalate
        if context.error_category in (ErrorCategory.FORMAT, ErrorCategory.SCHEMA):
            if learned_result.suggested_action == "escalate":
                return None  # Reject — rules forbid escalation for format errors
        # INFRASTRUCTURE: skip entirely (Phase 2)
        if context.error_category == ErrorCategory.INFRASTRUCTURE:
            return None  # Let rules handle
        # Convert learned suggestion to EscalationDecision
        return EscalationDecision(
            action=_map_action(learned_result.suggested_action),
            target_role=learned_result.suggested_role,
            reason=f"learned (confidence={learned_result.confidence:.2f}, "
                   f"similar_cases={learned_result.similar_cases})",
            strategy="learned",
        )
```

### Verification

```bash
pytest tests/unit/test_routing_facade.py tests/unit/test_escalation.py -v
```

---

## Phase 2: Infrastructure Error Classification (Finding 2 — HIGH)

### Problem

When `call_orchestrator_forced()` hits a timeout or connection error during seeding, the code sets `passed=False` and injects `reward=0.0`. This tells the Q-learner "this action failed" when really the infrastructure failed — the action was never tested.

Current code path (in `seed_specialist_routing.py`):
```python
# Line 547: error makes passed=False
passed_direct = score_answer_deterministic(...) if not error_direct else False
# Line 698: False → 0.0 reward
rewards[ACTION_SELF_DIRECT] = success_reward(passed_direct)  # 0.0
```

Over time, heavy models (architect_coding at 10.3 t/s, architect_general at 6.75 t/s) accumulate more timeouts than fast models, systematically depressing their Q-values regardless of actual capability.

### Fix

Classify errors as infrastructure vs task failure. Skip reward injection for infra errors entirely (no retry — the question gets retried naturally in the next seeding batch).

### Files

| File | Action | Detail |
|------|--------|--------|
| `scripts/benchmark/seeding_types.py` | MODIFY | Add `error_type: str = "none"` field to `RoleResult` dataclass |
| `scripts/benchmark/seed_specialist_routing.py` | MODIFY | Add `_classify_error()` function. In `evaluate_question_3way()`, set `error_type` on RoleResult. Only add action to rewards dict when error is absent or `task_failure`. Log infra errors as `INFRA_SKIP` |
| `scripts/benchmark/seed_specialist_routing.py` | MODIFY | In `_inject_3way_rewards_http()`, skip injection for actions not present in rewards dict |
| `scripts/benchmark/seeding_rewards.py` | MODIFY | `_has_delegation()` returns `False` for infra-errored results. `score_delegation_chain()` skips infra-errored architect results |
| `tests/unit/test_3way_routing.py` | MODIFY | Add `TestInfraErrorClassification`: timeout → infra, connection refused → infra, "model produced wrong answer" → task_failure, None → none. Test that infra errors produce no reward entries |

**Note**: `orchestration/repl_memory/q_scorer.py` does NOT need changes. Since infra errors are excluded from the rewards dict entirely, `_inject_3way_rewards_http()` never posts them to `/chat/reward`, so the Q-scorer never sees them. If we later want to *log* infra events (with `action_type="infra"`) for observability, add the q_scorer guard at that time.

### Error Classification Logic

```python
INFRA_PATTERNS = [
    "timed out", "timeout", "connection", "refused",
    "unreachable", "502", "503", "504", "ConnectError",
    "ReadTimeout", "backend down", "server error",
]

def _classify_error(error_str: str | None) -> str:
    """Classify error as infrastructure or task failure."""
    if error_str is None:
        return "none"
    error_lower = error_str.lower()
    if any(p.lower() in error_lower for p in INFRA_PATTERNS):
        return "infrastructure"
    return "task_failure"
```

### Modified Reward Computation

```python
# In evaluate_question_3way(), replace direct reward assignment:
error_type_direct = _classify_error(error_direct)
if error_type_direct == "infrastructure":
    logger.info(f"    {ACTION_SELF_DIRECT} -> INFRA_SKIP (not injecting reward)")
    # Do NOT add to rewards dict — action was never fairly tested
else:
    passed_direct = score_answer_deterministic(...) if not error_direct else False
    rewards[ACTION_SELF_DIRECT] = success_reward(passed_direct)
```

### Verification

```bash
pytest tests/unit/test_3way_routing.py::TestInfraErrorClassification -v
# Dry-run to verify infra errors are skipped:
python scripts/benchmark/seed_specialist_routing.py --3way --dry-run --suites thinking --sample-size 2
```

---

## Phase 3: Delegation Telemetry (Findings 3 + 5 — HIGH/MEDIUM)

### Problem (Finding 3)

WORKER reward attribution uses brittle string matching:

```python
# seeding_rewards.py:309-325 — current _has_delegation()
def _has_delegation(rr: RoleResult) -> bool:
    # String matching on tools_called for "delegate" substrings
    if rr.tools_called:
        delegation_tools = {"delegate", "delegate_to_worker", "spawn_worker"}
        for tool in rr.tools_called:
            if any(dt in tool.lower() for dt in delegation_tools):
                return True
    # String matching on role_history for worker role substrings
    if rr.role_history and len(rr.role_history) > 1:
        worker_roles = {"worker_explore", "worker_math", "worker_vision", "worker_summarize"}
        for role in rr.role_history:
            if any(wr in role.lower() for wr in worker_roles):
                return True
    return False
```

This will silently miss new delegation tools, renamed worker roles, or delegation via the new ProactiveDelegator system.

### Problem (Finding 5)

A single run can include both tool usage and delegation. The reward for WORKER is tied to the overall `passed` flag. If tools fix the answer but delegation doesn't (or vice versa), attribution is ambiguous.

### Fix

Add `delegation_events` list to `ChatResponse` for canonical telemetry. Add separate `tools_success` and `delegation_success` signals.

### Files

| File | Action | Detail |
|------|--------|--------|
| `src/api/models/responses.py` | MODIFY | Add `DelegationEvent` model and three new fields to `ChatResponse` |
| `src/api/routes/chat_delegation.py` | MODIFY | Emit `DelegationEvent` when architect delegates. This file exists and handles delegation routing |
| `src/proactive_delegation/delegator.py` | MODIFY | Emit `DelegationEvent` in `ProactiveDelegator.delegate()` — this is where wave-based parallel delegation actually executes |
| `src/api/routes/delegate.py` | MODIFY | Emit `DelegationEvent` in the `/delegate` endpoint handler |
| `src/api/routes/chat_pipeline/repl_executor.py` | MODIFY | Emit `DelegationEvent` when REPL delegates to worker. Set `tools_success` based on tool call outcomes |
| `src/api/routes/chat.py` | MODIFY | Thread `delegation_events` from pipeline stages into final `ChatResponse` assembly |
| `scripts/benchmark/seeding_types.py` | MODIFY | Add `delegation_events: list[dict]`, `tools_success: bool | None`, `delegation_success: bool | None` to `RoleResult` |
| `scripts/benchmark/seed_specialist_routing.py` | MODIFY | Extract new fields from API response into `RoleResult` |
| `scripts/benchmark/seeding_rewards.py` | MODIFY | Rewrite `_has_delegation()` and `score_delegation_chain()` to prefer canonical telemetry with legacy fallback |
| `tests/unit/test_delegation_telemetry.py` | CREATE | ~12 tests for DelegationEvent model, canonical vs legacy detection, separate success signals |

Delegation can be initiated via chat pipeline, proactive delegator, or direct `/delegate` endpoint; all emit `DelegationEvent` into a shared aggregation list.

### New ChatResponse Fields

```python
class DelegationEvent(BaseModel):
    """Record of a delegation that occurred during request processing."""
    from_role: str = Field(..., description="Role that delegated")
    to_role: str = Field(..., description="Role that received delegation")
    task_summary: str = Field(default="", description="What was delegated")
    success: bool | None = Field(default=None, description="Whether delegated work succeeded")
    elapsed_ms: float = Field(default=0.0, description="Time spent on delegated work")
    tokens_generated: int = Field(default=0, description="Tokens generated by delegate")


class ChatResponse(BaseModel):
    # ... existing fields ...

    # Delegation telemetry (canonical attribution)
    delegation_events: list[DelegationEvent] = Field(
        default_factory=list,
        description="Structured delegation events during this request",
    )
    # Separate success signals (Finding 5)
    tools_success: bool | None = Field(
        default=None,
        description="Whether tool invocations contributed to a successful outcome",
    )
    delegation_success: bool | None = Field(
        default=None,
        description="Whether delegation(s) contributed to a successful outcome",
    )
```

### Source of Truth for `tools_success` and `delegation_success`

**`tools_success`**: Set to `True` only when a tool call directly changed the final answer — i.e., the tool result is referenced or incorporated in the final output. If tools were called but had no effect on the answer (e.g., a failed search followed by a direct answer), leave as `None`. This avoids injecting noisy signals into WORKER attribution. The emitting code in `repl_executor.py` should compare the pre-tool and post-tool answer states, or at minimum check whether tool output appears in the final response.

**`delegation_success`**: Set based on whether the delegated subtask returned a usable result. If the delegate returned an answer that the delegator incorporated, `True`. If the delegate errored or returned garbage that was discarded, `False`. If no delegation occurred, leave as `None`.

**Key invariant**: `None` means "not applicable" (no tools/delegation occurred), not "unknown". This three-state (True/False/None) design prevents score_delegation_chain() from falling back to the overall `passed` flag when the signal is actually absent vs ambiguous.

### Event Aggregation

`DelegationEvent` is emitted from multiple pipeline layers (repl_executor, proactive delegator, `/delegate` endpoint). The final `ChatResponse.delegation_events` is a **simple concatenation** across all stages — no deduplication or ordering guarantees. Events appear in emission order (which follows pipeline execution order). Consumers should not assume uniqueness; the same delegate role could appear in multiple events if it was invoked more than once during a request.

### Updated Reward Attribution

```python
def _has_delegation(rr: RoleResult) -> bool:
    """Check if delegation occurred. Prefers canonical telemetry."""
    # Phase 3: canonical delegation_events
    if rr.delegation_events:
        return True
    # Legacy fallback for old data without delegation_events
    if rr.tools_called:
        delegation_tools = {"delegate", "delegate_to_worker", "spawn_worker"}
        for tool in rr.tools_called:
            if any(dt in tool.lower() for dt in delegation_tools):
                return True
    if rr.role_history and len(rr.role_history) > 1:
        worker_roles = {"worker_explore", "worker_math", "worker_vision", "worker_summarize"}
        for role in rr.role_history:
            if any(wr in role.lower() for wr in worker_roles):
                return True
    return False


def score_delegation_chain(results: dict[str, RoleResult]) -> dict[str, float]:
    """Score WORKER using canonical delegation_success when available."""
    rewards: dict[str, float] = {}
    for key, rr in results.items():
        if not _has_delegation(rr):
            continue
        # Prefer canonical delegation_success (Finding 5)
        if rr.delegation_success is not None:
            score = success_reward(rr.delegation_success)
        else:
            score = success_reward(rr.passed)  # Legacy fallback
        if ACTION_WORKER in rewards:
            rewards[ACTION_WORKER] = max(rewards[ACTION_WORKER], score)
        else:
            rewards[ACTION_WORKER] = score
    return rewards
```

### Verification

```bash
pytest tests/unit/test_delegation_telemetry.py tests/unit/test_3way_routing.py -v
```

---

## Phase 4: Eval Script Hardening (Findings 4, 6, 7 — MEDIUM/LOW)

### Finding 4 + Auditor Dual-Architect Recommendation

**Problem**: `_is_coding_task()` at `seed_specialist_routing.py:464-477` uses keyword matching to choose `architect_coding` vs `architect_general`. Words like "error", "test", "class" appear frequently in non-code contexts (biology, philosophy, general knowledge), systematically misrouting non-code questions to architect_coding.

**Auditor recommendation**: During seeding, evaluate BOTH architects for every question. This provides evidence for whether `architect_coding` is competitive on non-code suites without changing production routing.

**Rationale for keeping architect_coding primarily for coding in production**:
- It's huge and specialized; lower prior for non-code reasoning tasks
- BOS mismatch already restricts speculation and other optimizations
- Using it broadly risks waste and quality degradation from specialization bias

**But evaluating both in seeding is low-risk, high-signal**:
- Learn whether architect_coding is competitive on specific non-code suites
- Evidence gathering without changing routing rules
- If it performs well, update routing thresholds later with data

**Fix**: Replace single-architect config with dual-architect evaluation.

**Implementation in `evaluate_question_3way()`**:

1. **Define both architect roles** (after `arch_role` assignment block):
```python
# Dual-architect evaluation (audit recommendation)
# Run both to gather comparative data. VL uses vision_escalation only.
if is_vl:
    arch_roles_to_eval = [arch_role]  # vision_escalation only
else:
    arch_roles_to_eval = ["architect_general", "architect_coding"]
```

2. **Replace single architect call with loop** (use helper to avoid duplication):
```python
arch_results: dict[str, dict] = {}
for ar in arch_roles_to_eval:
    if len(arch_roles_to_eval) > 1:
        _wait_for_heavy_models_idle()  # Both are heavy ports
    logger.info(f"  -> {ACTION_ARCHITECT} ({ar})...")
    t0 = time.perf_counter()
    resp = call_orchestrator_forced(
        prompt=prompt, force_role=ar, force_mode="delegated",
        url=url, timeout=max(timeout, 300), client=client,
        allow_delegation=True, cache_prompt=False,
    )
    elapsed = time.perf_counter() - t0
    answer = resp.get("answer", "")
    error = resp.get("error")
    error_type = _classify_error(error)  # Phase 2 integration
    if error_type == "infrastructure":
        passed = None  # Infra error — don't score
    else:
        passed = score_answer_deterministic(answer, expected, ...) if not error else False
    arch_results[ar] = {
        "passed": passed, "elapsed_seconds": elapsed,
        "tokens_generated": resp.get("tokens_generated", 0),
        "predicted_tps": resp.get("predicted_tps", 0.0),
        "generation_ms": resp.get("generation_ms", 0.0),
        "tools_used": resp.get("tools_used", 0),
        "tools_called": resp.get("tools_called", []),
        "role_history": resp.get("role_history", []),
        "error": error, "error_type": error_type,
    }
```

3. **Compute best-of-two for ARCHITECT reward** (skip infra errors):

The injected action key is always `ACTION_ARCHITECT` regardless of which underlying architect(s) passed. Individual per-architect results are stored in metadata (step 4) for later analysis, but the Q-learner sees a single `ARCHITECT` action.

Partial infra failure handling: if one architect hits an infra error but the other returns a valid result, the valid result is used. Only when *all* architect calls are infra errors is the reward skipped entirely.

```python
# Best-of-two: ARCHITECT reward = best non-infra result
valid_results = {k: v for k, v in arch_results.items() if v["passed"] is not None}
if valid_results:
    passed_arch = any(v["passed"] for v in valid_results.values())
    rewards[ACTION_ARCHITECT] = success_reward(passed_arch)
else:
    logger.info(f"    {ACTION_ARCHITECT} -> ALL INFRA_SKIP")
    # Don't add to rewards — all architect calls were infra errors
```

4. **Store nested metadata** (right after `metadata = compute_tool_value(...)`):

Tie-break for `best`: when both architects pass, prefer lower `generation_ms` (pure model time) if available, falling back to `elapsed_seconds` (wall clock, subject to scheduling jitter). This avoids noisy tie-breaks from infra variance.

```python
# Determine which architect was best
best_arch = None
for ar, res in arch_results.items():
    if res["passed"] is True:
        if best_arch is None:
            best_arch = ar
        else:
            # Prefer generation_ms (pure model time) over elapsed_seconds (wall clock)
            cur_time = res.get("generation_ms") or (res["elapsed_seconds"] * 1000)
            best_time = arch_results[best_arch].get("generation_ms") or (arch_results[best_arch]["elapsed_seconds"] * 1000)
            if cur_time < best_time:
                best_arch = ar
if best_arch is None:
    # Neither passed — pick the one that at least ran (non-infra)
    for ar, res in arch_results.items():
        if res["passed"] is not None:
            best_arch = ar
            break

metadata["architect_eval"] = {
    "general": arch_results.get("architect_general"),  # None for VL
    "coding": arch_results.get("architect_coding"),     # None for VL
    "best": best_arch,
    "heuristic_would_pick": "architect_coding" if _is_coding_task(prompt) else "architect_general",
}
```

5. **Keep `_is_coding_task()` for metadata annotation only** — tag which architect the heuristic *would have* picked so we can measure its accuracy against actual results.

6. **Also store both in `role_results`** so `score_delegation_chain()` can attribute WORKER rewards from either architect's delegation events.

**Critical**: `error_type` MUST be set on the `RoleResult` so `score_delegation_chain()` and `_has_delegation()` can skip infra-errored results. Without this, worker rewards could be inadvertently assigned from a failed infra run where `passed=False` due to timeout rather than actual task failure.

```python
for ar, res in arch_results.items():
    role_results[f"{ar}:delegated"] = RoleResult(
        role=ar, mode="delegated",
        answer=..., passed=res["passed"] if res["passed"] is not None else False,
        elapsed_seconds=res["elapsed_seconds"], error=res.get("error"),
        error_type=res.get("error_type", "none"),  # MUST propagate for infra-skip
        tokens_generated=res["tokens_generated"],
        # ... remaining fields ...
    )
```

### Files for Finding 4

| File | Action | Detail |
|------|--------|--------|
| `scripts/benchmark/seed_specialist_routing.py` | MODIFY | Config 3 becomes dual-architect loop as described above. Add `_wait_for_heavy_models_idle()` between calls. Keep `_is_coding_task()` for annotation only |
| `scripts/benchmark/seeding_rewards.py` | MODIFY | `compute_3way_rewards()` handles multiple architect keys — picks best for ARCHITECT reward |

### Finding 6: Cache State Across 3-Way Eval Runs

**Problem**: In `evaluate_question_3way()`, the 3 configurations run sequentially. The first run primes the server's KV cache; subsequent runs on the same port benefit from cached prompt prefixes. The `cache_prompt` parameter is never passed — it uses the server default (caching enabled).

This means SELF:direct always runs cold (first), SELF:repl runs warm (second, same port 8080), and ARCHITECT runs cold on a different port. Latency metrics are systematically biased.

**Fix**: Pass `cache_prompt=False` to **every** `call_orchestrator_forced()` call in `evaluate_question_3way()`. This includes:

1. SELF:direct (frontdoor, no tools)
2. SELF:repl (frontdoor, with tools)
3. ARCHITECT calls (1 for VL, 2 for text — both architect_general and architect_coding per Finding 4)

All 4-5 calls get the same parameter:

```python
resp = call_orchestrator_forced(
    ...,
    cache_prompt=False,  # Disable KV cache for fair timing comparison
)

# Add to metadata:
metadata["cache_disabled"] = True
```

Note: The dual-architect loop in Finding 4 already includes `cache_prompt=False` in its snippet. Ensure SELF:direct and SELF:repl calls match.

Also log cache control status once at batch start (not per-question) in `run_batch_3way()`:
```python
logger.info(f"Cache control: cache_prompt=False for all 3-way eval calls (fair timing)")
```
This makes it obvious in long overnight run logs that cache was intentionally disabled.

### Files for Finding 6

| File | Action | Detail |
|------|--------|--------|
| `scripts/benchmark/seed_specialist_routing.py` | MODIFY | Add `cache_prompt=False` to all `call_orchestrator_forced()` calls in `evaluate_question_3way()` (5 calls total: direct, repl, and 2-3 architect calls). Add `metadata["cache_disabled"] = True` |

### Finding 7: JSONL Checkpoint Concurrency

**Problem**: `append_checkpoint()` and `record_seen()` open files with `open(path, "a")` without file locking. If two seeding sessions run concurrently (e.g., different suites in separate terminals), writes can interleave and produce corrupted lines.

Other benchmark scripts (`run_orchestrator_benchmark.py`) already use `fcntl.flock()` for JSONL appends. The seeding scripts do not.

**Fix**: Add `fcntl.flock()` to both functions, matching existing pattern.

```python
import fcntl

def append_checkpoint(session_id: str, result: ComparativeResult):
    """Append one result to the session's JSONL file (atomic with flock)."""
    _ensure_eval_dir()
    path = EVAL_DIR / f"{session_id}.jsonl"
    line = json.dumps(asdict(result), ensure_ascii=False)
    fd = open(path, "a")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        fd.write(line + "\n")
        fd.flush()
        os.fsync(fd.fileno())
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


def record_seen(prompt_id: str, suite: str, session_id: str):
    """Append to the global seen questions file (atomic with flock)."""
    _ensure_eval_dir()
    entry = {
        "prompt_id": prompt_id,
        "suite": suite,
        "session": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    fd = open(SEEN_FILE, "a")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        fd.write(json.dumps(entry) + "\n")
        fd.flush()
        os.fsync(fd.fileno())
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()
```

### Files for Finding 7

| File | Action | Detail |
|------|--------|--------|
| `scripts/benchmark/seed_specialist_routing.py` | MODIFY | Add `import fcntl`. Wrap `append_checkpoint()` and `record_seen()` with `flock(LOCK_EX)` / `flock(LOCK_UN)` |

### Tests for Phase 4

Add to `tests/unit/test_3way_routing.py`:
- `TestDualArchitectEval`: dual roles generated for text, single role for VL; best-of-two reward correct; metadata structure valid; tie-break for `architect_eval.best` uses `generation_ms` then `elapsed_seconds`; `_is_coding_task()` annotation present but not used for routing
- `TestCacheControl`: `cache_prompt=False` passed in all forced calls
- `TestCheckpointLocking`: mock `fcntl.flock` and verify `LOCK_EX`/`LOCK_UN` called

### Verification

```bash
pytest tests/unit/test_3way_routing.py -v
# End-to-end dry run with dual-architect:
python scripts/benchmark/seed_specialist_routing.py --3way --dry-run --suites thinking --sample-size 2
```

---

## Dependency Graph

```
Phase 1 (Routing Facade)
  |
  +---> Adds ErrorCategory.INFRASTRUCTURE to src/escalation.py
  |
Phase 2 (Infra Error Classification)
  |
  +---> Uses ErrorCategory.INFRASTRUCTURE
  |     Uses _classify_error() in seeding script
  |
Phase 3 (Delegation Telemetry)        [independent]
  |
  +---> Adds DelegationEvent to ChatResponse
  |     Finding 3 (delegation_events) enables Finding 5 (delegation_success)
  |
Phase 4 (Eval Hardening)              [independent, uses Phase 2's _classify_error]
  |
  +---> Finding 4: dual-architect evaluation
  +---> Finding 6: cache_prompt=False
  +---> Finding 7: fcntl.flock
```

Phases 2, 3, 4 can proceed in parallel after Phase 1 adds the shared `ErrorCategory.INFRASTRUCTURE` enum value.

Phase 4's dual-architect loop integrates Phase 2's `_classify_error()` for infra-aware architect scoring.

---

## Documentation Updates

The implementation changes require updates to 5 documentation files to keep the architecture description current for human readers.

### 1. Chapter 18: Escalation, Failure Routing & Proactive Delegation (MAJOR)
**File**: `docs/chapters/18-escalation-and-routing.md`

- **Introduction**: Currently says "three complementary systems" (escalation policy, failure router, proactive delegation). Rewrite: unified routing facade wrapping rules + learned advisory. FailureRouter deprecated
- **ErrorCategory enum**: Add `INFRASTRUCTURE = "infrastructure"`. Add row to decision rules table: `INFRASTRUCTURE → skip reward injection (seeding only), rules handle escalation`
- **Failure Router section**: Major rewrite. FailureRouter deprecated. LearnedEscalationPolicy accessed through RoutingFacade. "Hybrid Strategy" and "Strategy Tracking" subsections move under new "RoutingFacade" section showing rules-authoritative, MemRL-advisory pattern
- **3-Way Confidence Routing**: Add note that WORKER scored via canonical `DelegationEvent` telemetry (not string matching)
- **References**: Add `src/routing_facade.py`. Update `src/failure_router.py` to "(deprecated, thin wrapper)"

### 2. ARCHITECTURE.md — Living Technical Reference (MODERATE)
**File**: `docs/ARCHITECTURE.md`

- **Architecture diagram**: Replace `failure_router` box with `routing_facade` box wrapping both `escalation.py` (rules) and `LearnedEscalationPolicy` (MemRL)
- **Module responsibilities table**: Add `src/routing_facade.py` row. Update `src/failure_router.py` to "Deprecated — use routing_facade.py"
- **Request flow**: Change `failure_router.route_failure(context)` to `routing_facade.decide(context)`
- **ChatResponse fields**: Add `delegation_events`, `tools_success`, `delegation_success` telemetry fields
- **Version bump**: 2.3 → 2.4. Add "Recent Updates" entry for audit remediation

### 3. Chapter 25: Cost-Aware Reward Design (MINOR)
**File**: `docs/chapters/25-cost-aware-rewards.md`

- Add subsection "Infrastructure Error Handling": infra errors produce NO reward — excluded from rewards dict entirely. Reference `_classify_error()` pattern
- Add note: dual-architect evaluation uses best-of-two for ARCHITECT Q-value, individual results in metadata

### 4. Chapter 17: Memory Seeding (MINOR)
**File**: `docs/chapters/17-memory-seeding.md`

- **3-way strategy**: Add bullet: "Infrastructure errors (timeouts, connection failures) produce no reward — question skipped, retried next batch"
- **3-way strategy**: Update: ARCHITECT now runs dual-architect (both architect_general and architect_coding) with best-of-two reward
- **Action keys table**: Update ARCHITECT row Source Role from `architect_*` to `architect_general + architect_coding (best-of-two)`

### 5. Model Routing Guide (MINOR)
**File**: `docs/guides/model-routing.md`

- Update any references to `FailureRouter` or `route_failure()` → `RoutingFacade.decide()`
- Note new `INFRASTRUCTURE` error category if escalation decision tree is described

### Documentation update order

Interleave with implementation:
1. Phase 1 → update Ch18 (RoutingFacade section), ARCHITECTURE.md (diagram, module table, request flow)
2. Phase 2 → update Ch25 (infra error handling), Ch17 (infra skip note)
3. Phase 3 → update Ch18 (DelegationEvent in 3-way routing), ARCHITECTURE.md (ChatResponse fields)
4. Phase 4 → update Ch25 (dual-architect), Ch17 (dual-architect, action keys table)
5. Final pass → version bump ARCHITECTURE.md, verify cross-references

---

## Complete File Manifest

### New Files (4)

| File | Phase | Purpose |
|------|-------|---------|
| `src/routing_facade.py` | 1 | Unified routing decision facade |
| `tests/unit/test_routing_facade.py` | 1 | Facade unit tests (~15) |
| `tests/unit/test_delegation_telemetry.py` | 3 | DelegationEvent and attribution tests (~12) |
| `handoffs/active/AUDIT_REMEDIATION.md` | — | This document (included for completeness) |

### Modified Files — Code (14)

| File | Phases | Changes |
|------|--------|---------|
| `src/escalation.py` | 1 | Add `ErrorCategory.INFRASTRUCTURE` |
| `src/failure_router.py` | 1 | Deprecation notice; import ErrorCategory from escalation.py |
| `src/api/__init__.py` | 1 | Init `RoutingFacade` instead of `FailureRouter` |
| `src/api/state.py` | 1 | Add `routing_facade` field |
| `src/api/routes/chat.py` | 1, 3 | 3 call sites → use facade; thread `delegation_events` into `ChatResponse` |
| `src/api/routes/chat_pipeline/repl_executor.py` | 1, 3 | 4 call sites → use facade; emit DelegationEvent; set `tools_success` |
| `src/api/models/responses.py` | 3 | Add DelegationEvent, delegation_events, tools_success, delegation_success |
| `src/api/routes/chat_delegation.py` | 3 | Emit DelegationEvent on architect delegation |
| `src/proactive_delegation/delegator.py` | 3 | Emit DelegationEvent in `ProactiveDelegator.delegate()` (wave-based parallel delegation) |
| `src/api/routes/delegate.py` | 3 | Emit DelegationEvent in `/delegate` endpoint handler |
| `scripts/benchmark/seed_specialist_routing.py` | 2, 4 | `_classify_error()`, dual-architect loop, `cache_prompt=False`, `fcntl.flock` |
| `scripts/benchmark/seeding_rewards.py` | 2, 3, 4 | Infra-aware scoring, canonical delegation attribution, multi-architect reward |
| `scripts/benchmark/seeding_types.py` | 2, 3 | `error_type` field, delegation telemetry fields |
| `tests/unit/test_3way_routing.py` | 2, 4 | Infra classification tests, dual-architect tests, cache/locking tests |

### Modified Files — Documentation (5)

| File | Phases | Changes |
|------|--------|---------|
| `docs/chapters/18-escalation-and-routing.md` | 1, 3 | RoutingFacade section, deprecate FailureRouter section, ErrorCategory.INFRASTRUCTURE, DelegationEvent in 3-way routing |
| `docs/ARCHITECTURE.md` | 1, 3 | Architecture diagram, module table, request flow, version bump 2.3→2.4 |
| `docs/chapters/25-cost-aware-rewards.md` | 2, 4 | Infra error skip subsection, dual-architect best-of-two note |
| `docs/chapters/17-memory-seeding.md` | 2, 4 | Infra skip note, dual-architect strategy, action keys table update |
| `docs/guides/model-routing.md` | 1 | FailureRouter → RoutingFacade reference update |

---

## Verification (All Phases)

```bash
# Unit tests
pytest tests/unit/test_routing_facade.py tests/unit/test_escalation.py \
       tests/unit/test_3way_routing.py tests/unit/test_delegation_telemetry.py -v

# Full test suite
pytest tests/ -x

# Gates
cd /mnt/raid0/llm/claude && make gates

# End-to-end dry run (no reward injection)
python scripts/benchmark/seed_specialist_routing.py \
    --3way --dry-run --suites thinking coder --sample-size 2

# Verify dual-architect metadata in dry-run output
# Look for: architect_eval.general, architect_eval.coding, architect_eval.best
```

## Completion Notes (2026-02-04)

- Implemented RoutingFacade end-to-end (rules-authoritative + learned advisory).
- Added delegation telemetry (`delegation_events`, `tools_success`, `delegation_success`) to ChatResponse.
- Hardened 3-way seeding (infra skip, dual-architect best-of-two, cache control, flocked JSONL).
- Updated docs: Ch18, ARCHITECTURE.md, Ch17, Ch25.
- Tests: `pytest tests/unit/test_routing_facade.py tests/unit/test_escalation.py tests/unit/test_3way_routing.py tests/unit/test_delegation_telemetry.py -v`
  - Result: **45 passed**, warning about unknown pytest `timeout` config.

---

## Resume Commands

```bash
# Phase 1: Start with routing facade
cd /mnt/raid0/llm/claude
cat handoffs/active/AUDIT_REMEDIATION.md  # This document
# Create src/routing_facade.py, then update call sites
# Then update: docs/chapters/18-escalation-and-routing.md, docs/ARCHITECTURE.md

# Phase 2: Infra error classification
# Modify seed_specialist_routing.py, seeding_rewards.py, seeding_types.py
# Then update: docs/chapters/25-cost-aware-rewards.md, docs/chapters/17-memory-seeding.md

# Phase 3: Delegation telemetry
# Modify responses.py, chat_delegation.py, delegator.py, delegate.py, repl_executor.py, chat.py
# Then update: docs/chapters/18-escalation-and-routing.md (DelegationEvent), docs/ARCHITECTURE.md (ChatResponse)

# Phase 4: Eval hardening
# Modify seed_specialist_routing.py (dual-architect + cache + flock)
# Then update: docs/chapters/25-cost-aware-rewards.md, docs/chapters/17-memory-seeding.md

# Final: version bump docs/ARCHITECTURE.md 2.3 → 2.4, verify cross-references
```
