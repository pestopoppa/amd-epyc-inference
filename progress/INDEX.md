# Progress Log Index

Lab notebook entries for the AMD EPYC 9655 inference optimization project.

## 2026

### February 2026

| Date | Summary | Key Outcomes |
|------|---------|--------------|
| [2026-02-03](2026-02/2026-02-03.md) | UI Consolidated + Test Coverage + Hard Benchmarks | UI handoff index, 223 new tests (97-100% chat coverage), Phase 1 hard benchmark adapters (GPQA, SimpleQA, HotpotQA, LiveCodeBench) |
| [2026-02-02](2026-02/2026-02-02.md) | MemRL Mode-Advantage Tasks + HF Adapters | 90 mode-advantage tasks, GAIA/CRUXEval/BigCodeBench adapters |
| [2026-02-01](2026-02/2026-02-01.md) | Vision Pipeline + WI-9/10/11 + Eval Bugfix | Doc pipeline, staged rewards, parallel gates, prompt_builders decomp, eval pre-launch fixes |

### January 2026

| Date | Summary | Key Outcomes |
|------|---------|--------------|
| [2026-01-31](2026-01/2026-01-31.md) | Overhead Root-Cause + Pipeline Perf Optimization | KV cache overhead diagnosed, 6 mitigations, pipeline −239s via POST /config + persistent httpx |
| [2026-01-30](2026-01/2026-01-30.md) | Phase 3 MemRL Orchestration + Vision Integration | 884 tests, architect delegation, VL suite, input formalizer |
| [2026-01-21](2026-01/2026-01-21.md) | LightOnOCR Document Pipeline | Full pipeline integration: models, client, chunker, REPL, API |
| [2026-01-20](2026-01/2026-01-20.md) | LightOnOCR-2 GGUF Optimization | **19x speedup** (0.17 pg/s), 8×12t optimal, orchestrator integration |
| [2026-01-16](2026-01/2026-01-16.md) | Engram Tokenizer Compression Experiment | Negative result - BPE encodes case differently |
| [2026-01-15](2026-01/2026-01-15.md) | Unified Orchestrator Stack Launcher | `orchestrator_stack.py` for model management |
| [2026-01-14](2026-01/2026-01-14.md) | MemRL Integration, API Refactoring | Lazy loading, memory safety guards |
| [2026-01-13](2026-01/2026-01-13.md) | MemRL Implementation Phases 1-3 | Episodic store, embedder, retriever |
| [2026-01-12](2026-01/2026-01-12.md) | Benchmark Spec Decode Optimization | New benchmark results with spec decode |
| [2026-01-11](2026-01/2026-01-11.md) | Role Enum Fix | `__str__` returns value not repr |
| [2026-01-10](2026-01/2026-01-10.md) | REPL Tool Integration | TOOL() function wired into environment |
| [2026-01-09](2026-01/2026-01-09.md) | Benchmark Infrastructure | Results indexing, Claude-as-Judge reviews |
| [2026-01-08](2026-01/2026-01-08.md) | Orchestrator Tooling Architecture | 6 phases complete, 29 tests, 92% token savings |
| [2026-01-07](2026-01/2026-01-07.md) | Formalizer, RadixAttention, Claude-as-Judge | RadixAttention complete (46 tests), Optuna optimization |
| [2026-01-05](2026-01/2026-01-05.md) | CPU Optimization Research | T-MAC evaluation, NUMA findings |
| [2026-01-04](2026-01/2026-01-04.md) | Orchestration Planning | Implementation plan, model registry design |
| [2026-01-01](2026-01/2026-01-01.md) | Q1 Planning | Priorities set for 2026 |

## 2025

### December 2025

| Date | Summary | Key Outcomes |
|------|---------|--------------|
| [2025-12-24](2025-12/2025-12-24.md) | Holiday Maintenance | Bug fixes, documentation updates |
| [2025-12-22](2025-12/2025-12-22.md) | Benchmark Hardening | 8 suites updated, T3 questions added |
| [2025-12-21](2025-12/2025-12-21.md) | Parallel Tensor Repack | 2.2x loading speedup, PR submitted |
| [2025-12-16](2025-12/2025-12-16.md) | Phase 1 Complete | 184 unit tests + 9 integration, E2E working |

---

## Quick Stats

- **Total Entries**: 24
- **Date Range**: 2025-12-16 to 2026-02-03
- **Major Milestones**:
  - Phase 1 Foundation Complete (Dec 16)
  - Benchmark Hardening (Dec 22)
  - RadixAttention Complete (Jan 7)
  - Tooling Architecture Complete (Jan 8)
  - MemRL Episodic Memory (Jan 13-14)
  - Orchestrator Stack Launcher (Jan 15)
  - LightOnOCR-2 GGUF Optimization: **19x speedup** (Jan 20)
  - LightOnOCR Document Pipeline Integration (Jan 21)
  - Phase 3 MemRL Orchestration + Vision Integration (Jan 30)
  - Pipeline Perf Optimization: `POST /config` + persistent httpx (Jan 31)
  - Vision → Document Pipeline /chat Integration (Feb 1)
  - MemRL Mode-Advantage Tasks + HF Adapters (Feb 2)
  - Phase 1 Hard Benchmark Adapters: GPQA, SimpleQA, HotpotQA, LiveCodeBench (Feb 3)

---

## Navigation

- [Active Handoffs](../handoffs/README.md)
- [Research Chapters](../docs/chapters/INDEX.md)
- [Back to README](../README.md)
