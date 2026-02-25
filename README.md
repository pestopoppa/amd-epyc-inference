# amd-epyc-inference (Archived)

> **This repository has been split into focused repos. See [epyc-root](https://github.com/pestopoppa/epyc-root) for setup instructions.**

## New repositories

| Repository | Purpose |
|------------|---------|
| [epyc-root](https://github.com/pestopoppa/epyc-root) | Umbrella repo — governance, coordination, setup scripts |
| [epyc-orchestrator](https://github.com/pestopoppa/epyc-orchestrator) | Production multi-model orchestration system |
| [epyc-inference-research](https://github.com/pestopoppa/epyc-inference-research) | Benchmarks, experiments, model evaluation |
| [epyc-llama](https://github.com/pestopoppa/llama.cpp) | Custom llama.cpp fork with AMD EPYC patches |

## Quick start (new setup)

```bash
git clone https://github.com/pestopoppa/epyc-root.git
cd epyc-root
./scripts/setup.sh
```

## Why the split?

This monorepo mixed four distinct concerns: production orchestration, inference research, llama.cpp patches, and cross-repo governance. Splitting improves:

- **FOSS accessibility** — orchestrator can be used without research/governance baggage
- **Focused development** — each repo has its own CLAUDE.md, tests, and conventions
- **Dependency clarity** — formal dependency map in epyc-root

## Historical reference

This repository is preserved as-is for historical reference. All commit history remains intact. For active development, use the repositories listed above.
