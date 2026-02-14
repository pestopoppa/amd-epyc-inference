---
name: agent-file-architecture
description: Design, refactor, and validate project agent files using a thin-map architecture (shared policy plus lean role overlays). Use when creating/updating `agents/*.md`, splitting monolithic agent instructions, enforcing role schema, or migrating long operational guidance into docs.
---

# Agent File Architecture

Use this skill for work in `agents/`.

Use when:

- Refactoring role prompts into schema-driven overlays.
- Moving duplicated policy into `agents/shared/`.
- Splitting long operational content into docs.

Do not use when:

- Implementing runtime code features in `src/` or `orchestration/`.
- Changing benchmark/model behavior unrelated to prompt architecture.

## Workflow

1. Apply the role schema in `references/schema.md`.
2. Use migration guidance in `references/migration.md`.
3. Run `scripts/validate_agents.py`.
4. If validation fails, fix schema or references before finalizing.

## Boundaries

- Keep role files concise.
- Keep cross-cutting policy in `agents/shared/`.
- Keep long operational details in `docs/guides/agent-workflows/`.
