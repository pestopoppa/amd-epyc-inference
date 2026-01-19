# Research Chapters Index

Completed research findings for AMD EPYC 9655 inference optimization.

## Reading Order

For newcomers, read chapters in order for the research journey narrative. For reference, jump directly to topics of interest.

## Chapter List

| # | Title | Summary | Key Result |
|---|-------|---------|------------|
| 01 | [Hardware System](01-hardware-system.md) | AMD EPYC 9655 specs, baseline performance | ~460 GB/s bandwidth |
| 02 | [Speculative Decoding](02-speculative-decoding.md) | External draft model approach | **11x speedup** on code |
| 03 | [MoE Optimization](03-moe-optimization.md) | Expert reduction technique | **+52%** on 30B MoE |
| 04 | [Prompt Lookup](04-prompt-lookup.md) | N-gram matching for grounded tasks | **12.7x** on summarization |
| 05 | [Benchmarking Framework](05-benchmarking-framework.md) | 8-suite methodology, Claude-as-Judge | 61 models evaluated |
| 06 | [Orchestration Architecture](06-orchestration-architecture.md) | Hierarchical agent design | TaskIR, escalation chains |
| 07 | [RadixAttention](07-radix-attention.md) | Prefix caching for orchestrator | >50% cache hit target |
| 08 | [Deprecated Approaches](08-deprecated-approaches.md) | EAGLE-1, CAS-Spec failures | Documented dead ends |

---

## By Topic

### Inference Optimization
- Chapter 02: Speculative Decoding (Track 1)
- Chapter 03: MoE Optimization (Track 2)
- Chapter 04: Prompt Lookup (Track 8)

### Infrastructure
- Chapter 01: Hardware System
- Chapter 05: Benchmarking Framework

### Orchestration
- Chapter 06: Orchestration Architecture
- Chapter 07: RadixAttention

### Historical
- Chapter 08: Deprecated Approaches

---

## Navigation

- **[Master Benchmark Results](../reference/benchmarks/RESULTS.md)** — All model scores and speeds
- [Progress Logs](../../progress/INDEX.md)
- [Active Handoffs](../../handoffs/README.md)
- [Back to README](../../README.md)
