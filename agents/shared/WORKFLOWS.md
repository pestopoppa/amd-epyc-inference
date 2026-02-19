# Shared Workflows

## New Feature

1. Add or extend a feature flag.
2. Implement guarded behavior.
3. Add tests for enabled and disabled states.
4. Document architecture impact in `docs/`.

## API Change

1. Update route, models, and service boundaries.
2. Verify request/response validation at boundaries.
3. Run focused API tests.
4. Document behavior changes.

## Escalation Logic Change

1. Modify canonical escalation modules only.
2. Add tests for expected decisions.
3. Validate no regressions in existing routes.

## System Change

1. Capture current system state.
2. Log rollback command.
3. Apply change via audited commands.
4. Validate expected impact and stability.

## Benchmark Update

1. Run benchmark with explicit config capture.
2. Record results and anomalies.
3. Compare against baseline.
4. Update `docs/reference/benchmarks/RESULTS.md` when appropriate.

## Handoff Closure And Roadmap Refresh

1. Reconcile handoff checklist against real code/tests before marking complete.
2. Extract durable findings into `docs/chapters/` and agent playbooks; keep handoff as execution log, not the only source of truth.
3. Update roadmap/blocker trackers (`handoffs/README.md`, `orchestration/BLOCKED_TASKS.md`) in the same change.
4. Record evidence in `CHANGELOG.md` and progress log with exact commands/tests used.
5. Archive handoff from `handoffs/active/` only after docs + trackers + evidence are in place.

## Orchestration Stabilization Closure (RLM)

1. For orchestrator lifecycle work, prefer API-only reload: `python3 scripts/server/orchestrator_stack.py reload orchestrator`.
2. In restricted environments, socket-based health/probe commands may require escalated execution; treat sandbox `PermissionError` on local sockets as an environment constraint, not an orchestration regression.
3. Validate fixes with both unit coverage and contention probes; lock/delegation changes are not complete until seeded contention runs confirm no stale lock holders.
4. Treat response diagnostics as first-class acceptance criteria: `delegation_diagnostics.break_reason`, `budget_diagnostics.*`, and `error_code` must be explicit on bounded failures.
5. Keep roadmap status synchronized with evidence. As of this closure cycle: R1, R2, R3, R5, R6 and Phase 6 are complete; Phase 7 tuning remains open (D5/D6 deferred).
