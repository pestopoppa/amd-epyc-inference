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

## Evidence Logging

For every closure pass, update all three:
- Active handoff (`handoffs/active/...`)
- Progress log (`progress/YYYY-MM/YYYY-MM-DD.md`)
- Audit log (`logs/agent_audit.log`)

Keep entries evidence-first: exact commands/probes, observed outcomes, and residual risk.
