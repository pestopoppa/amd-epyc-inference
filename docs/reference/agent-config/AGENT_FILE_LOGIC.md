# Agent File Logic

This document explains how agent files work in this repository and why they are structured this way.

## Core Architecture

The agent system is split into layers to keep prompts small and deterministic:

1. `agents/AGENT_INSTRUCTIONS.md`
   - Global execution contract.
   - Read order and conflict resolution.

2. `agents/shared/*.md`
   - Cross-cutting policy that applies to all roles.
   - Examples: operating constraints, engineering standards, reusable workflows.

3. `agents/*.md` role files
   - Role-local behavior only.
   - Required schema:
     - `## Mission`
     - `## Use This Role When`
     - `## Inputs Required`
     - `## Outputs`
     - `## Workflow`
     - `## Guardrails`

4. `docs/guides/agent-workflows/*.md`
   - Procedure-heavy operational detail moved out of prompts.
   - Prevents role prompts from becoming brittle megadocs.
5. `docs/reference/agent-config/*_PLAYBOOK.md`
   - Incident/runbook knowledge that should be discoverable but not always loaded.
   - Example: orchestration lock/delegation debugging playbook.

## Why This Design

- Small top-level context improves routing and reduces drift.
- Shared policy in one place prevents contradictory duplication.
- Role prompts stay easy to audit and update.
- Heavy operational guidance remains available, but loaded as docs when needed.

## Enforcement Path

The design is backed by multiple guardrails:

1. Hooks (`scripts/hooks/` + `.claude/settings.json`)
   - Block malformed role edits and broken references.
   - Inject CLAUDE-accounting and skill-parity reminders.

2. Validators (`scripts/validate/`)
   - `validate_agents_structure.py`
   - `validate_agents_references.py`
   - `validate_claude_md_matrix.py`

3. Aggregated check target
   - `make check-agent-config`

## CLAUDE Governance Boundaries

CLAUDE policy files are explicitly scoped via:

- `docs/reference/agent-config/CLAUDE_MD_MATRIX.md`
- `docs/reference/agent-config/claude_md_matrix.json`

Governed files for this repo:

- `CLAUDE.md`

## Skill Design Principles Folded In

This project applies the same practical skill patterns described in OpenAI's skills/shell guidance:

- Write skill descriptions like routing boundaries (use when / do not use when).
- Include negative examples to reduce misfires.
- Keep templates/examples in skills/docs, not always-loaded prompts.
- Use explicit skill invocation for deterministic production workflows.

Source: https://developers.openai.com/blog/skills-shell-tips
