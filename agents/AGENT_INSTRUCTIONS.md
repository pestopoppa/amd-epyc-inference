# Agent Execution Contract

This file is the top-level contract for agents working in this repository.

## Scope

- This file is intentionally short.
- It points to canonical policy and workflow docs.
- It does not duplicate deep implementation details.

## Read Order

1. `agents/shared/OPERATING_CONSTRAINTS.md`
2. `agents/shared/ENGINEERING_STANDARDS.md`
3. `agents/shared/WORKFLOWS.md`
4. `docs/guides/agent-workflows/INDEX.md`
5. Role file in `agents/*.md` relevant to the task
6. Domain docs in `docs/` and current status in `CLAUDE.md`

## Non-Negotiables

- No writes outside `/mnt/raid0/` for LLM-related artifacts.
- Never run `pytest -n auto` on this host.
- Use feature flags for optional modules and expensive runtime components.
- Use enums and typed boundaries instead of magic strings.
- Classify every new numeric value as either:
  - `tunable` (belongs in typed config/dataclass + env override path), or
  - `invariant` (belongs in constants modules, not inline literals).
- Do not create monolithic "all numerics" files; keep tunables in owning subsystem configs.
- Never silently swallow exceptions.
- Keep changes small, testable, and documented.

## Output Contract

Each substantial task should end with:

1. What changed
2. Why this approach
3. Verification run (or why not run)
4. Risks and follow-up actions

## File Ownership Model

- `agents/shared/*.md`: cross-cutting policy and reusable workflows.
- `agents/*.md`: role behavior and role-specific playbooks.
- `docs/`: long-lived system-of-record knowledge.

If guidance conflicts:

1. Safety constraints win.
2. Architectural invariants win.
3. Role guidance applies next.
4. Local task prompt resolves remaining ambiguity.

## Validation Commands

- `python3 scripts/validate/validate_agents_structure.py`
- `python3 scripts/validate/validate_agents_references.py`
- `python3 scripts/validate/validate_claude_md_matrix.py`
