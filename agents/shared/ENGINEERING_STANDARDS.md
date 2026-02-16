# Engineering Standards

## Code Invariants

- Prefer typed boundaries for external data.
- Use enums and constants, not ad hoc strings.
- Gate optional features with `src/features.py`.
- Log exceptions with context; do not use silent `except: pass`.
- Use thread-safe state update paths for shared mutable state.

## Numerical Parameter Policy

- Treat numeric values as one of two classes:
  - `tunable`: runtime behavior controls likely to change during evaluation/tuning.
  - `invariant`: stable semantic limits or shared hard boundaries.
- `tunable` values must live in typed config/dataclass surfaces, with env override path when operationally relevant.
- `invariant` values must be named constants (global or subsystem-local), not magic literals.
- Do not consolidate all numbers into one global file; preserve subsystem ownership of tunables.
- PRs adding numerics should include a one-line classification note (`tunable` vs `invariant`).

## Change Style

- Keep each change scoped to one concern.
- Reuse existing modules and utilities before adding new helpers.
- Place new files according to existing project layout.

## Placement Rules

- Feature flags: `src/features.py`
- Roles/routing metadata: `src/roles.py` and model registry
- API routes/models/services/state: `src/api/`
- Tests: `tests/unit/` and `tests/integration/`
- Architecture and design rationale: `docs/`

## Verification Minimum

Before finalizing:

1. Syntax check for modified Python files.
2. Run targeted tests for touched behavior.
3. Confirm feature-flag behavior where applicable.
4. Update docs when behavior or interfaces change.
