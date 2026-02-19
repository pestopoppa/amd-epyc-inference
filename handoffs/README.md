# Handoff Documents

Handoff documents track work-in-progress for future sessions or agents. They are **ephemeral** - once work completes, their content is extracted and they are deleted.

## Directory Structure

```
handoffs/
├── active/       # Currently active work
└── blocked/      # Work awaiting dependencies
    └── BLOCKED.md
```

## Handoff Lifecycle

```
CREATE → handoffs/active/{topic}.md
    │
    ▼
WORK → Update with progress, daily summary to progress log
    │
    ▼ (optional)
BLOCKED → Move to handoffs/blocked/, update BLOCKED.md
    │
    ▼
COMPLETE → Extract content, then DELETE
```

## On Task Completion

When a handoff task is complete, follow this checklist:

- [ ] **Technical findings** → appropriate `docs/chapters/` chapter
- [ ] **Key metrics** → `docs/reference/benchmarks/RESULTS.md`
- [ ] **Model quirks discovered** → `docs/reference/models/QUIRKS.md`
- [ ] **Process summary** → today's `progress/YYYY-MM/YYYY-MM-DD.md`
- [ ] **Handoff file deleted** from `handoffs/active/`
- [ ] **BLOCKED.md updated** (mark complete, unblock dependents)

## Current Active Handoffs

7 handoffs in `active/`. Key active items:

| Handoff | Purpose | Status |
|---------|---------|--------|
| [rlm-orchestrator-roadmap.md](active/rlm-orchestrator-roadmap.md) | Consolidated orchestrator roadmap and integration sequencing | Active |
| ~~hybrid-lookup-spec-decode.md~~ | Prompt lookup + speculative decode + corpus strategy | **Archived** (2026-02-19) |
| [routing-intelligence.md](active/routing-intelligence.md) | Routing quality and decision-policy improvements | Phase 1 COMPLETE, Phases 2-6 deferred |
| [skillbank-distillation.md](active/skillbank-distillation.md) | Skill distillation/evolution planning | Active |
| [multimodal-pipeline.md](active/multimodal-pipeline.md) | Multimodal and vision pipeline enhancements | Active |
| [claude-code-local-constellation-routing.md](active/claude-code-local-constellation-routing.md) | Claude Code local constellation routing behavior | Active |
| [open_source_orchestrator.md](active/open_source_orchestrator.md) | Open-source packaging and public orchestrator path | Active |

> Note: Several COMPLETE handoffs need extraction per lifecycle above. Run `ls handoffs/active/` for full listing.

**Last Updated**: 2026-02-19

## Blocked Tasks

See [blocked/BLOCKED.md](blocked/BLOCKED.md) for tasks awaiting dependencies.

## Navigation

- [Progress Logs](../progress/INDEX.md)
- [Research Chapters](../docs/chapters/INDEX.md)
- [Back to README](../README.md)
