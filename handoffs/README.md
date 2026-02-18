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

35 handoffs in `active/`. Key active items:

| Handoff | Purpose | Status |
|---------|---------|--------|
| [middleware-hardening-trio.md](active/middleware-hardening-trio.md) | Credential redaction + script interception + cascading tool policy (from Clawzempic/OpenClaw gap analysis) | READY FOR IMPLEMENTATION |
| [programmatic-tool-chaining.md](active/programmatic-tool-chaining.md) | Deferred tool results + multi-mutation chaining + persistent REPL sessions (from Anthropic API analysis) | RESEARCH COMPLETE |
| [inference-lock-starvation-bug.md](active/inference-lock-starvation-bug.md) | Lock contention diagnosis + bounded wait + embedder isolation | Active (mitigated, quality follow-up) |
| [perf-parallel-tools-concurrent-sweep-prefix-cache.md](active/perf-parallel-tools-concurrent-sweep-prefix-cache.md) | Parallel read-only tools, concurrent sweep, prefix cache | IMPLEMENTED |
| [orchestration-architecture-optimization-handoff.md](active/orchestration-architecture-optimization-handoff.md) | Risk-controlled posterior routing + telemetry hardening | COMPLETE (100% review coverage) |
| [hybrid-lookup-spec-decode.md](active/hybrid-lookup-spec-decode.md) | Prompt lookup + spec decode + corpus | Phase 2A SHIPPED, 2B ABANDONED |
| [seed-routing-decomposition.md](active/seed-routing-decomposition.md) | Seed specialist routing decomposition | IN PROGRESS |
| [orchestration-roadmap.md](active/orchestration-roadmap.md) | Orchestration phases 6-8 | Active |
| [simpleqa-debugger-agency.md](active/simpleqa-debugger-agency.md) | SimpleQA 0% fix, debugger action bias | IMPLEMENTED |
| [vision-pipeline.md](active/vision-pipeline.md) | Vision pipeline chat integration | NEEDS LIVE VALIDATION |
| [skillbank-distillation.md](active/skillbank-distillation.md) | Skill distillation pipeline | SPEC COMPLETE |
| [nextplaid-code-retrieval.md](active/nextplaid-code-retrieval.md) | NextPLAID code retrieval | Active |
| [nextplaid-phase5-upgrade.md](active/nextplaid-phase5-upgrade.md) | NextPLAID phase 5 upgrade | Active |
| [minicpm-o-4_5-integration.md](active/minicpm-o-4_5-integration.md) | MiniCPM-o multimodal integration | Active |
| [classifier-refactoring.md](active/classifier-refactoring.md) | Classifier refactoring | READY TO IMPLEMENT |
| [qwen3-tts-voice-synthesis.md](active/qwen3-tts-voice-synthesis.md) | TTS voice synthesis | BLOCKED (audio quality) |
| [agent-files-refactor-complete.md](active/agent-files-refactor-complete.md) | Agent prompt architecture & governance | COMPLETE |
| [ui-consolidated.md](active/ui-consolidated.md) | UI index & evaluation reference | Active |
| [infra-seeding-regression.md](active/infra-seeding-regression.md) | Seeding infrastructure regression | Active |
| [open_source_orchestrator.md](active/open_source_orchestrator.md) | Open source orchestrator | STUB (future) |

> Note: Several COMPLETE handoffs need extraction per lifecycle above. Run `ls handoffs/active/` for full listing.

**Last Updated**: 2026-02-18

## Blocked Tasks

See [blocked/BLOCKED.md](blocked/BLOCKED.md) for tasks awaiting dependencies.

## Navigation

- [Progress Logs](../progress/INDEX.md)
- [Research Chapters](../docs/chapters/INDEX.md)
- [Back to README](../README.md)
