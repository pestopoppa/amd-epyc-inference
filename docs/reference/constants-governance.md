# Constants Governance

This repository applies a two-class policy for numeric values:

1. `tunable`: runtime behavior controls expected to change during evaluation.
2. `invariant`: stable semantic boundaries or protocol limits.

## Placement

- `tunable` values:
  - live in typed config/dataclass control planes
  - should expose env overrides when operationally relevant
  - in agent/runtime modules (`src/api/routes/*`, `src/graph/*`, `src/session/*`), do not add inline tuning literals; route through `src/config` first
- `invariant` values:
  - live in constants modules (global or subsystem-local)
  - should not remain as inline magic literals

## Tooling

- Inventory report:
  - `python3 scripts/validate/inventory_numerics.py`
- Soft lint report:
  - `python3 scripts/validate/check_numeric_literals.py --mode report`
- Changed-file enforcement (local/PR workflows):
  - `python3 scripts/validate/check_numeric_literals.py --mode enforce-changed --diff-range origin/main...HEAD`

Allowlist: `scripts/validate/numeric_literal_allowlist.yaml`

## PR Expectation

When introducing a new numeric value, include one short note:
- classification: `tunable` or `invariant`
- placement rationale
- runtime use site and (if tunable) its config key/env override
