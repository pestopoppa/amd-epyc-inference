# Agent System Map

This directory is organized for agent legibility and low drift.

## Start Here

1. Read `agents/AGENT_INSTRUCTIONS.md` for the global execution contract.
2. Read `agents/shared/OPERATING_CONSTRAINTS.md` for safety and environment constraints.
3. Read `agents/shared/ENGINEERING_STANDARDS.md` for coding invariants.
4. Read `agents/shared/WORKFLOWS.md` for common operating procedures.
5. Read workflow depth docs in `docs/guides/agent-workflows/`.
6. Open only the role file needed for the current task.

## Roles

| Role | File | Primary Use |
|---|---|---|
| Lead Developer | `agents/lead-developer.md` | Architecture decisions, sequencing, arbitration |
| Research Engineer | `agents/research-engineer.md` | Implementation and deep debugging |
| Benchmark Analyst | `agents/benchmark-analyst.md` | Benchmark execution and interpretation |
| Research Writer | `agents/research-writer.md` | Reports, synthesis, literature framing |
| Build Engineer | `agents/build-engineer.md` | Build configuration and compiler tuning |
| Model Engineer | `agents/model-engineer.md` | Model conversion, quantization, pairing |
| Sysadmin | `agents/sysadmin.md` | Host tuning and runtime system state |
| Safety Reviewer | `agents/safety-reviewer.md` | Risk gate before high-impact operations |

## Model Routing (Task-Based)

- `Haiku`: routine execution, data collection, simple edits.
- `Sonnet`: most engineering, synthesis, and implementation.
- `Opus`: novel architecture, hard debugging, high-stakes technical arbitration.

Rule: start with the cheapest model likely to succeed, escalate only when blocked.

## Design Principles

- Keep role files focused on role-specific behavior.
- Keep cross-cutting policy in `agents/shared/`.
- Keep durable project knowledge in `docs/`, not in role prompts.
- Prefer mechanical checks over prose-only reminders.

## Migration Note

- Legacy long-form role playbooks were intentionally split.
- Operational detail moved to `docs/guides/agent-workflows/`.
- Schema and reference consistency is enforced via `scripts/validate/` and hooks in `scripts/hooks/`.
- Full design rationale: `docs/reference/agent-config/AGENT_FILE_LOGIC.md`.
