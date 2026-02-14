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
