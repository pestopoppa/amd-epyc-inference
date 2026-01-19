# Getting Started

Welcome to the AMD EPYC 9655 Inference Optimization project.

## Prerequisites

- Access to the system (Beelzebub)
- Basic familiarity with LLM inference
- Understanding of llama.cpp

## Quick Orientation

### What This Project Does

We optimize LLM inference on CPU using three main techniques:
1. **Speculative Decoding** (Track 1) - Small model proposes tokens, large model verifies
2. **MoE Expert Reduction** (Track 2) - Use fewer experts for faster inference
3. **Prompt Lookup** (Track 8) - Match n-grams from input prompt

### Best Results

| Technique | Speedup | Best For |
|-----------|---------|----------|
| Speculative Decoding | 11x | Code generation |
| MoE Reduction | +52% | MoE models |
| Prompt Lookup | 12.7x | Summarization |

## First Steps

1. **Read the hardware overview**: [Chapter 01](../chapters/01-hardware-system.md)
2. **Understand the techniques**: [Chapter 02](../chapters/02-speculative-decoding.md)
3. **Try a command**: See [Quick Reference](../reference/commands/QUICK_REFERENCE.md)

## Key Constraints

- **All files on /mnt/raid0/** - Never write to root filesystem
- **OMP_NUM_THREADS=1** - Prevents threading conflicts
- **numactl --interleave=all** - Maximizes memory bandwidth

## Where to Find Things

| Need | Location |
|------|----------|
| Research narrative | [docs/chapters/](../chapters/INDEX.md) |
| Benchmark results | [docs/reference/benchmarks/](../reference/benchmarks/RESULTS.md) |
| Commands | [docs/reference/commands/](../reference/commands/QUICK_REFERENCE.md) |
| Model info | [docs/reference/models/](../reference/models/MODELS.md) |

## Next Steps

- Read the [Research Chapters](../chapters/INDEX.md) for the full story
- Check the [Benchmark Results](../reference/benchmarks/RESULTS.md) for model scores
- Try running a benchmark

---

*See [CLAUDE_GUIDE.md](../../CLAUDE_GUIDE.md) for understanding the AI context file.*
