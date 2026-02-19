# Orchestration Debug Playbook

Purpose: capture high-value operational lessons from lock-starvation and delegation-loop incidents so future sessions converge faster.

## Scope

Use this for:
- Delegation hangs/timeouts
- Inference lock contention/starvation
- Report-handle delegation hydration issues
- Worker role wiring confusion (`worker_coder` vs legacy `worker_code`)

Do not use this as a replacement for architecture docs in `docs/chapters/`.

## Fast Start

1. Reload only the API (do not restart full stack):
   - `python3 scripts/server/orchestrator_stack.py reload orchestrator`
2. For contention debugging, prefer profile:
   - `python3 scripts/server/orchestrator_stack.py reload orchestrator --profile contention-debug`
3. Confirm API health:
   - `curl -sS http://127.0.0.1:8000/health`

## Baseline Telemetry

Enable these when diagnosing lock/delegation behavior:
- `ORCHESTRATOR_FRONTDOOR_TRACE=1`
- `ORCHESTRATOR_DELEGATION_TRACE=1`
- `ORCHESTRATOR_INFERENCE_LOCK_TRACE=1`
- `ORCHESTRATOR_DELEGATION_TOTAL_MAX_SECONDS=55`
- `ORCHESTRATOR_DELEGATION_SPECIALIST_MAX_SECONDS=25`
- `ORCHESTRATOR_INFERENCE_LOCK_TIMEOUT_EXCLUSIVE_S=45`
- `ORCHESTRATOR_INFERENCE_LOCK_TIMEOUT_SHARED_S=45`

Lock logs now include request tags (`request=<task_id>`) and owner attribution (via `/proc/locks` with fallback).

## Delegation Diagnostics Checklist

Inspect response-level delegation diagnostics first:
- `break_reason`
- `cap_reached`
- `effective_max_loops`
- `report_handles` / `report_handles_count`

If delegation never starts or lock wait aborts, verify explicit reason fields (for example pre-delegation lock timeout) are present in diagnostics.

## Report Handle Flow

Large specialist outputs should return compact answer + handle:
- Marker: `[REPORT_HANDLE id=...]`
- REPL tool: `fetch_report(report_id, offset=0, max_chars=2400)`
- API: `GET /chat/delegation-report/{report_id}?offset=0&max_chars=...`

If retrieval fails with HTTP 422, verify request bounds (`max_chars >= 64`).

## Tool-Chaining Closure Nuance

Programmatic tool chaining is complete through persistence integration. During debugging:
- Inspect `tool_chains` first to verify wave execution mode (`dep` vs `sequential`) and fallback behavior.
- Inspect `session_persistence` to confirm restore/save lifecycle (`restore_success`, `checkpoint_saved`, `checkpoint_id`).
- For checkpoint-restore compatibility issues, inspect `session_persistence.restore_protocol` (`source_version`, `compat_mode`, `missing_required_fields`, `dropped_fields`).
- For depth-override rollout tuning, inspect `budget_diagnostics` fields:
  - `depth_override_enabled`
  - `depth_override_events`
  - `depth_override_roles` (e.g. `worker_general->worker_math`)
- For long delegated outputs, expect summary + handle, not full report text in every loop turn.
- Do not infer regressions from missing full text when `[REPORT_HANDLE ...]` and `fetch_report()` are available.

## Worker Role Semantics

- Primary coding worker semantic role: `worker_coder`
- `worker_code` remains compatibility alias only
- Runtime defaults align both to fast worker endpoint (`8102`)

When debugging routing/delegation behavior, treat `worker_code` mentions as legacy naming.

## Seeding Script Guardrail

`seed_specialist_routing.py` seeds episodic memory for production routing decisions. Avoid over-constraining routing inside seeding logic; preserve behavioral diversity so MemRL can learn route quality.

## Operational Non-Goals

During orchestration lock/delegation debugging, do not casually mutate:
- MemRL reward/scoring mechanisms
- SkillRL / `--evolve` pathways

Only touch these when the task explicitly targets learning-policy behavior.

## Safe Defaults (R6)

- Production default-on:
  - `session_compaction=1` (via `get_features(production=True)` default)
  - `tool_result_clearing=1` (via `get_features(production=True)` default)
  - `depth_model_overrides=1` (via `get_features(production=True)` default)
- Fast rollback toggle:
  - `ORCHESTRATOR_SESSION_COMPACTION=0`
  - `ORCHESTRATOR_TOOL_RESULT_CLEARING=0`
  - `ORCHESTRATOR_DEPTH_MODEL_OVERRIDES=0`
- Validation-only tuning knob:
  - `ORCHESTRATOR_CHAT_SESSION_COMPACTION_MIN_TURNS=1` can be used to force earlier C1 trigger during live benchmarking (default is `5`).
- Keep default-off unless the task explicitly validates them:
  - `content_cache`
  - `model_fallback`
  - `structured_tool_output`
  - `side_effect_tracking` / `approval_gates`

## R3/Phase6 Closure Evidence (2026-02-19)

- R3 depth-override rollout closure checklist:
  - run one delegated probe with overrides OFF (`DEPTH_MODEL_OVERRIDES=0`) and one with ON (`DEPTH_MODEL_OVERRIDES=1`),
  - confirm `budget_diagnostics.depth_override_enabled` toggles accordingly,
  - confirm delegated path remains bounded and diagnostics populated (`break_reason`, `loops`, `delegation_inference_hops`).
- Phase 6 early-failure/load validation checklist:
  - run targeted monitor tests:
    - `python3 -m pytest -n 0 tests/unit/test_chat_pipeline_stages.py tests/unit/test_stages.py tests/unit/test_generation_monitor.py -k "generation_monitor or early_abort" -q`
  - run a small concurrent live probe with `GENERATION_MONITOR=1`,
  - acceptance: no silent hangs; failures must surface as explicit bounded responses (`error_code` set).

## Evidence Logging

For every closure pass, update all three:
- Active handoff (`handoffs/active/...`)
- Progress log (`progress/YYYY-MM/YYYY-MM-DD.md`)
- Audit log (`logs/agent_audit.log`)

Keep entries evidence-first: exact commands/probes, observed outcomes, and residual risk.
