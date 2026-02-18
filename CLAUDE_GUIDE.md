# Understanding the Agent Configuration

This guide helps human readers understand how Claude Code is configured in this project.

## Architecture Overview

The agent configuration is split across five layers:

```
CLAUDE.md              ← Always loaded. Core rules, system identity, routing tables.
.claude/commands/*.md  ← On-demand skills. Loaded only when invoked via /skill-name.
.claude/skills/*       ← Packaged local skills (SKILL.md + references/scripts) for reusable workflows.
agents/*               ← Agent execution contract, shared policy, and role overlays.
.claude/settings.json  ← Hooks. Run automatically on tool use to enforce safety rules.
CHANGELOG.md           ← Dated record of system changes. Referenced, not loaded.
```

This layered design keeps the always-loaded context small (~370 lines / 16KB) while making situational knowledge available on demand.

## File-by-File Breakdown

### `CLAUDE.md` — Core Context (Always Loaded)

Loaded into every Claude Code session. Contains only what the agent needs on every task:

| Section | What It Does |
|---------|--------------|
| Critical Constraints | Filesystem rules (`/mnt/raid0/` only), env vars |
| Test Memory Safety | `pytest -n auto` prohibition (192-thread machine) |
| System Identity | Host, user, key file paths |
| Hardware Specs | CPU, RAM, storage for inference planning |
| Available Skills | Table of `/skill-name` commands |
| Current Status | Best speedups achieved, deprecated approaches |
| Orchestration System | Agent tiers (A-D), model assignments, component flow |
| Directory Structure | Project layout, branch safety rules |
| Session Startup | 3-command quickstart sequence |
| Model Routing | Which model/port for which task type |
| Logging | Mandatory agent audit log pattern |
| Code Style / Git | Conventions and commit workflow |

**Design principle**: If the agent needs it on >50% of sessions, it stays in CLAUDE.md. Otherwise, it's a skill.

### `.claude/commands/*.md` — Command Skills (On-Demand)

Skills are loaded only when the agent (or user) invokes them with `/skill-name`. Each skill is a self-contained reference document for a specific workflow.

| Skill | File | Lines | When It's Needed |
|-------|------|-------|------------------|
| `/benchmark` | `benchmark.md` | 308 | Running benchmarks, scoring models, analyzing eval logs |
| `/draft-compat` | `draft-compat.md` | 49 | Validating speculative decoding draft-target pairs |
| `/research-update` | `research-update.md` | 39 | Updating results tables after benchmarking |
| `/new-model` | `new-model.md` | — | Onboarding a new model into the registry |
| `/refactor` | `refactor.md` | — | Code technical debt analysis |
| `/mcp-knowledge` | `mcp-knowledge.md` | — | Knowledge tools integration |
| `/agent-files` | `agent-files.md` | — | Agent file schema and migration workflow |
| `/agent-governance` | `agent-governance.md` | — | Prompt governance checks and CLAUDE accounting |

**Why skills?** Benchmarking, eval scoring, and draft compatibility validation are detailed workflows that are only relevant ~10-20% of sessions. Keeping them as skills saves ~500 lines of context on every other session.

### `.claude/skills/*` — Packaged Local Skills

Packaged skills are local reusable bundles with `SKILL.md`, references, and scripts.

| Skill | Path | Purpose |
|------|------|---------|
| Agent File Architecture | `.claude/skills/agent-file-architecture/` | Role schema, migration guardrails, validation runner |
| CLAUDE MD Accounting | `.claude/skills/claude-md-accounting/` | Governance scoping and matrix consistency |

### `agents/` — Agent Prompt Architecture

`agents/` now follows a split design:

- `agents/AGENT_INSTRUCTIONS.md`: top-level execution contract
- `agents/shared/*.md`: cross-cutting policy
- `agents/*.md`: lean role overlays
- `docs/guides/agent-workflows/`: operational detail moved out of prompts

Design details: `docs/reference/agent-config/AGENT_FILE_LOGIC.md`

### `.claude/settings.json` — Hooks (Automatic)

Hooks run automatically before certain tool calls. They enforce safety rules that were previously only written as prose in CLAUDE.md.

| Hook | Trigger | What It Does |
|------|---------|--------------|
| `check_pytest_safety.sh` | Any `Bash` call | Blocks `pytest -n auto` and `-n N` where N > 16 |
| `check_filesystem_path.sh` | `Write` or `Edit` | Blocks file writes outside `/mnt/raid0/` |
| `benchmark_context.sh` | `Write` or `Edit` | Reminds agent to use `/benchmark` when editing benchmark files |
| `agents_schema_guard.sh` | `Write` or `Edit` | Blocks non-conforming role schema changes in `agents/*.md` |
| `agents_reference_guard.sh` | `Write` or `Edit` | Blocks unresolved local markdown references in governance files |
| `claude_accounting_context.sh` | `Write` or `Edit` | Reminds CLAUDE matrix sync when editing CLAUDE policy files |
| `skills_context.sh` | `Write` or `Edit` | Reminds command-skill and packaged-skill parity |

Hook scripts live in `scripts/hooks/`. The configuration in `.claude/settings.json` maps tool names to hook scripts. A separate `.claude/settings.local.json` (not committed) holds permission allow-lists.

### `CHANGELOG.md` — Change Log (Referenced)

Dated entries documenting system changes (new features, bug fixes, architecture updates). CLAUDE.md links to it but doesn't include its content. The agent reads it when it needs historical context about what changed and when.

## How It All Fits Together

```
Session Start
│
├── CLAUDE.md loaded (370 lines, ~16KB, ~4K tokens)
│   ├── Safety rules in memory
│   ├── Routing tables in memory
│   └── Skill table in memory (agent knows skills exist)
│
├── Hooks registered from .claude/settings.json
│   └── Run before every Write/Edit/Bash call
│
│   User asks: "benchmark the new Qwen model"
│   │
│   ├── Agent sees /benchmark in skill table
│   ├── Agent loads benchmark.md (308 lines, one-time)
│   └── Agent follows the workflow
│
│   Agent tries: Write to /tmp/results.json
│   │
│   └── check_filesystem_path.sh → BLOCKED (exit 2)
│       "All files must be on /mnt/raid0/"
```

## For Human Reading

If you're a human trying to understand the project (not configure the agent), these are better starting points:

| Topic | Read This |
|-------|-----------|
| Research journey | [docs/chapters/INDEX.md](docs/chapters/INDEX.md) |
| Model reference | [docs/reference/models/MODELS.md](docs/reference/models/MODELS.md) |
| Orchestration incident/debug runbook | [docs/reference/agent-config/ORCHESTRATION_DEBUG_PLAYBOOK.md](docs/reference/agent-config/ORCHESTRATION_DEBUG_PLAYBOOK.md) |
| Launch commands | [docs/reference/commands/QUICK_REFERENCE.md](docs/reference/commands/QUICK_REFERENCE.md) |
| Benchmark results | [docs/reference/benchmarks/RESULTS.md](docs/reference/benchmarks/RESULTS.md) |
| Getting started | [docs/guides/getting-started.md](docs/guides/getting-started.md) |

## Updating This Configuration

### Adding a new skill
1. Create `.claude/commands/your-skill.md`
2. Add a row to the "Available Skills" table in `CLAUDE.md`
3. The skill is immediately available as `/your-skill`

### Adding a new hook
1. Create `scripts/hooks/your-hook.sh` (must read JSON from stdin, exit 0 to allow, exit 2 to block)
2. Add the hook to `.claude/settings.json` under the appropriate tool matcher
3. Make it executable: `chmod +x scripts/hooks/your-hook.sh`

### Updating CLAUDE.md
- Keep it under 500 lines — if a section grows beyond ~20 lines and is only needed sometimes, extract it to a skill
- Use tables over prose for structured data
- Include full paths in commands
- Test that Claude Code can find new content

---

*This guide is for human orientation. For actual work, use the structured documents in [docs/](docs/).*
