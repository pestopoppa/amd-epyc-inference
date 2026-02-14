# Handoff: Agent Files Refactor Completion (Harness-Aligned)

**Status**: IMPLEMENTATION COMPLETE
**Created**: 2026-02-14
**Updated**: 2026-02-14 (execution complete)
**Priority**: HIGH
**Scope**: `agents/` + repo references + CLAUDE accounting (`CLAUDE.md`, `kernel-dev/llama-cpp-dev/CLAUDE.md`)
**Track**: Agent prompt architecture modernization

## Objective

Complete the in-progress refactor of all agent files using harness-engineering best practices:

1. Thin top-level instructions.
2. Shared canonical policy docs.
3. Lean role overlays.
4. Durable operational detail moved to docs.
5. Mechanical enforcement via hooks and lightweight validators.
6. Skill-driven reuse across `.claude/commands` and local packaged skills.

## Current State Snapshot

Agent refactor is already partially implemented with these changed files:

- `agents/AGENT_INSTRUCTIONS.md`
- `agents/README.md`
- `agents/benchmark-analyst.md`
- `agents/build-engineer.md`
- `agents/lead-developer.md`
- `agents/model-engineer.md`
- `agents/research-engineer.md`
- `agents/research-writer-guide.md`
- `agents/research-writer.md`
- `agents/safety-reviewer.md`
- `agents/sysadmin.md`
- `agents/shared/` (new folder)

## In-Scope Decisions (Locked)

1. Scope is `agents + references`.
2. CLAUDE accounting includes:
   - `CLAUDE.md`
   - `kernel-dev/llama-cpp-dev/CLAUDE.md`
3. Detailed content removed from prompts moves to docs, not back into role files.
4. Skill leverage uses both surfaces:
   - `.claude/commands/*.md`
   - local packaged skills (`SKILL.md` format)
5. Hook strategy is broad hook suite, not minimal guard-only.
6. Enforcement level is lightweight checks.

## Out of Scope

1. Vendored/plugin/cache/backup `CLAUDE.md` trees under `config/plugins/`, `config/.claude/plugins/`, and `backups/`.
2. Runtime orchestration behavior changes in `src/` unless strictly required for validation utilities.
3. Archival rewrites of deprecated docs.

## Required Deliverables

1. Finalized `agents/` architecture and role schema compliance.
2. CLAUDE coverage matrix with governance classification.
3. Migrated operational guidance in docs (`docs/guides/agent-workflows/`).
4. Expanded hooks in `scripts/hooks/` and `.claude/settings.json`.
5. New command skills in `.claude/commands/`.
6. New local packaged skills in `.claude/skills/`.
7. Lightweight validation scripts + runner.
8. Final completion updates to logs, progress, and docs.

## Implementation Phases

## Phase 1: Finalize Agent Prompt Architecture

1. Normalize all role files to required schema:
   - Mission
   - Use This Role When
   - Inputs Required
   - Outputs
   - Workflow
   - Guardrails
2. Keep cross-cutting policy only in `agents/shared/*.md`.
3. Keep `agents/AGENT_INSTRUCTIONS.md` thin and pointer-based.
4. Add migration note in `agents/README.md` for old-to-new map.

## Phase 2: Migrate Legacy Operational Detail to Docs

1. Create `docs/guides/agent-workflows/INDEX.md`.
2. Create `docs/guides/agent-workflows/research-writer.md`.
3. Create `docs/guides/agent-workflows/benchmark-analyst.md`.
4. Create `docs/guides/agent-workflows/safety-reviewer.md`.
5. Replace role-level long instructions with doc links and script references.

## Phase 3: CLAUDE.md Accounting

1. Add `docs/reference/agent-config/CLAUDE_MD_MATRIX.md`.
2. Include explicit entries for:
   - `CLAUDE.md` as governed.
   - `kernel-dev/llama-cpp-dev/CLAUDE.md` as governed (external-subproject policy).
3. Add classified exclusions for vendor/cache/backup trees (`external` or `archival`).
4. Add link from matrix to `kernel-dev/llama-cpp-dev/AGENTS.md`.

## Phase 4: Update Repo References

1. Update `README.md` agent architecture section.
2. Update `CLAUDE_GUIDE.md` to include:
   - `agents/shared/` model.
   - dual skill surfaces.
   - expanded hook suite.
3. Update `docs/guides/system-prompt-guide.md` examples to current paths and conventions.
4. Ensure references to the old long writer guide point to new workflow docs.

## Phase 5: Broad Hook Suite

Add scripts under `scripts/hooks/`:

1. `agents_schema_guard.sh` (blocking): enforce required role headings.
2. `agents_reference_guard.sh` (blocking): detect broken local path refs in edited agent/doc files.
3. `claude_accounting_context.sh` (context): remind matrix update when editing CLAUDE files.
4. `skills_context.sh` (context): remind command/local-skill parity on relevant edits.

Register hooks in `.claude/settings.json` with scoped matchers and timeouts.

## Phase 6: Skill Surfaces

1. Add `.claude/commands/agent-files.md`.
2. Add `.claude/commands/agent-governance.md`.
3. Add local packaged skill `.claude/skills/agent-file-architecture/SKILL.md` with references/scripts.
4. Add local packaged skill `.claude/skills/claude-md-accounting/SKILL.md` with references/scripts.

## Phase 7: Lightweight Validation

1. Add `scripts/validate/validate_agents_structure.py`.
2. Add `scripts/validate/validate_agents_references.py`.
3. Add `scripts/validate/validate_claude_md_matrix.py`.
4. Add runner target `make check-agent-config`.
5. Keep this non-blocking in CI for initial rollout.

## Phase 8: Finalize and Document Completion

1. Verify changed file matrix and links.
2. Verify hook JSON validity and script executability.
3. Run local validation commands.
4. Update handoff status and checklist.
5. Perform required logging/progress/docs updates.

## Verification Commands

```bash
git status --short
python3 scripts/validate/validate_agents_structure.py
python3 scripts/validate/validate_agents_references.py
python3 scripts/validate/validate_claude_md_matrix.py
rg -n "agents/shared|agent-workflows|CLAUDE_MD_MATRIX" README.md CLAUDE_GUIDE.md docs -g '*.md'
```

## Completion Checklist (Mandatory)

- [x] `agents/` schema and shared-policy split complete.
- [x] `docs/guides/agent-workflows/` created and linked.
- [x] `docs/reference/agent-config/CLAUDE_MD_MATRIX.md` complete.
- [x] `.claude/commands` additions complete.
- [x] `.claude/skills` packaged skills complete.
- [x] `.claude/settings.json` updated with new hooks.
- [x] New hook scripts added and executable.
- [x] Validation scripts added and passing.
- [x] `README.md`, `CLAUDE_GUIDE.md`, and relevant docs updated.
- [x] Final handoff status updated.

## Post-Implementation Reporting Requirements

When implementation is finished, update all of the following:

1. `logs/agent_audit.log`
   - record task start, key decisions, and task end using `scripts/utils/agent_log.sh`.
2. Progress report
   - update `progress/2026-02/2026-02-14.md` (or current date file if different at execution time).
   - add concise agent-file refactor completion summary and verification status.
   - update `progress/INDEX.md` if required by convention.
3. Changelog and docs
   - add an entry to `CHANGELOG.md` describing:
     - agent architecture split
     - new hook suite
     - new skill surfaces
     - CLAUDE matrix
   - update any newly touched docs to avoid stale references.
4. Handoff registry
   - update `handoffs/README.md` table/status if needed.
   - keep this handoff in `handoffs/active/` until all checklist items pass.
   - after completion and archival decision, follow repository handoff lifecycle.

## Risks

1. Hook false positives causing edit friction.
2. Drift between `.claude/commands` skills and `.claude/skills` packaged skills.
3. Loss of critical operational guidance during prompt thinning.

## Mitigations

1. Scope hooks narrowly and make errors actionable.
2. Add parity checks in validation scripts.
3. Preserve operational depth in `docs/guides/agent-workflows/` before final prompt cleanup.

## Final Success Criteria

This track is complete when:

1. The new `agents/` architecture is stable and enforced.
2. All governed CLAUDE files are explicitly accounted for.
3. Skill and hook systems support the architecture operationally.
4. Validation scripts pass.
5. Logs, progress, changelog, and docs are fully updated.
