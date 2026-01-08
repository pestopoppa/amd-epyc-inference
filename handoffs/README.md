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

| Handoff | Purpose | Status |
|---------|---------|--------|
| [amd-pace-testing.md](active/amd-pace-testing.md) | AMD PACE native PyTorch vs llama.cpp | Ready |
| [cpu-optimization.md](active/cpu-optimization.md) | T-MAC evaluation, Tree speculation | Research |
| [draft-benchmark.md](active/draft-benchmark.md) | Draft model speed tests & formalizer eval | Ready |
| [early-failure-prediction.md](active/early-failure-prediction.md) | Heuristic failure detection | In Progress |
| [formalizer-evaluation.md](active/formalizer-evaluation.md) | Formalizer model evaluation | Blocked |
| [kernel-development.md](active/kernel-development.md) | AVX-512 kernel optimization | Blocked |
| [mathsmith-reconversion.md](active/mathsmith-reconversion.md) | MathSmith GGUF re-conversion | Complete |
| [orchestration-integration.md](active/orchestration-integration.md) | RadixAttention integration | Code Complete |
| [orchestrator.md](active/orchestrator.md) | Main orchestrator implementation | Ready |
| [radix-attention.md](active/radix-attention.md) | Prefix caching implementation | Complete |

## Blocked Tasks

See [blocked/BLOCKED.md](blocked/BLOCKED.md) for tasks awaiting dependencies.

## Navigation

- [Progress Logs](../progress/INDEX.md)
- [Research Chapters](../docs/chapters/INDEX.md)
- [Back to README](../README.md)
