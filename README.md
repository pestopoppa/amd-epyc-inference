# AMD EPYC 9655 Inference Optimization

LLM inference optimization research on AMD EPYC 9655 "Turin" (96 cores, 1.13TB DDR5).

## Best Results

| Configuration | Speed | Speedup | Use Case |
|---------------|-------|---------|----------|
| Prompt Lookup (summarization) | 95.18 t/s | **12.7x** | Document QA |
| Qwen2.5-Coder-32B + 0.5B (K=24) | 33.0 t/s | **11x** | Code generation |
| Prompt Lookup (code editing) | 25.82 t/s | **8.6x** | Refactoring |
| Qwen2.5-72B + 0.5B (K=16) | 8.53 t/s | **5.8x** | General tasks |

## Quick Start

```bash
# Track 1: External Draft Model (11x speedup on code)
OMP_NUM_THREADS=1 numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-speculative \
  -m /mnt/raid0/llm/models/Qwen2.5-Coder-32B-Q4_K_M.gguf \
  -md /mnt/raid0/llm/models/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf \
  --draft-max 24 -t 96 -p "Your prompt"

# Track 2: MoE Expert Reduction (+52% on MoE models)
numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-cli \
  -m /mnt/raid0/llm/models/Qwen3-235B-A22B-Q4_K_M.gguf \
  --override-kv qwen3moe.expert_used_count=int:4 \
  -t 96 -p "Your prompt"
```

---

## Documentation

### For Humans (Narrative)

| Document | Description |
|----------|-------------|
| [Research Chapters](docs/chapters/INDEX.md) | The project journey as numbered chapters |
| [Getting Started](docs/guides/getting-started.md) | New contributor onboarding |
| [Model Routing Guide](docs/guides/model-routing.md) | When to use which model |
| [Benchmarking Guide](docs/guides/benchmarking-guide.md) | How to benchmark models |

### For AI / Reference (Topical)

| Document | Description |
|----------|-------------|
| [CLAUDE.md](CLAUDE.md) | AI context file (loaded at session start) |
| [CLAUDE_GUIDE.md](CLAUDE_GUIDE.md) | Human guide to understanding CLAUDE.md |
| [Model Reference](docs/reference/models/MODELS.md) | All model specs and constraints |
| [Benchmark Results](docs/reference/benchmarks/RESULTS.md) | Complete benchmark data |
| [Command Reference](docs/reference/commands/QUICK_REFERENCE.md) | All inference commands |

### Progress & Work Tracking

| Document | Description |
|----------|-------------|
| [Progress Index](progress/INDEX.md) | Lab notebook entries by date |
| [Active Handoffs](handoffs/README.md) | Work in progress |
| [Blocked Tasks](handoffs/blocked/BLOCKED.md) | Awaiting dependencies |

---

## Project Structure

```
/mnt/raid0/llm/claude/
├── CLAUDE.md                  # AI context file
├── CLAUDE_GUIDE.md            # Human guide to CLAUDE.md
├── README.md                  # This file
│
├── docs/
│   ├── chapters/              # Research papers (numbered, permanent)
│   │   ├── INDEX.md
│   │   ├── 01-hardware-system.md
│   │   ├── 02-speculative-decoding.md
│   │   └── ...
│   ├── reference/             # Quick lookup (AI-optimized)
│   │   ├── models/MODELS.md
│   │   ├── models/QUIRKS.md
│   │   ├── benchmarks/RESULTS.md
│   │   └── commands/QUICK_REFERENCE.md
│   ├── guides/                # Human tutorials
│   └── deprecated/            # Historical archive
│
├── progress/                  # Lab notebook (chronological)
│   ├── INDEX.md
│   └── YYYY-MM/YYYY-MM-DD.md
│
├── handoffs/                  # Work in progress
│   ├── active/
│   └── blocked/
│
├── orchestration/             # Schemas and configs (UNCHANGED)
│   ├── model_registry.yaml
│   ├── task_ir.schema.json
│   └── ...
│
├── agents/                    # Agent definitions
├── scripts/                   # Automation scripts
├── benchmarks/                # Test prompts and results
└── patches/                   # llama.cpp patches
```

---

## Key Insights

1. **Small drafts win on CPU**: 0.5B at 85 t/s beats 7B at 8 t/s
2. **MoE models don't need speculation**: Already at "draft speed" (~25 t/s)
3. **K-tuning matters**: K=8 for 7B, K=16-24 for 32B+
4. **Temperature helps**: temp=0.5-0.7 can double performance on some models

## Track Status

| Track | Method | Speedup | Status |
|-------|--------|---------|--------|
| Track 1 | External Draft Model | 5.9-11x | **Production** |
| Track 2 | MoE Expert Reduction | +21-52% | **Production** |
| Track 8 | Prompt Lookup | 8.6-12.7x | **Production** |
| Track 3 | EAGLE-1 | — | Deprecated (0% acceptance) |

---

## Hardware

| Component | Specification |
|-----------|---------------|
| CPU | AMD EPYC 9655 "Turin" (96 cores, 192 threads, Zen 5) |
| RAM | 1.13 TB DDR5-5600 ECC (12 channels, ~460 GB/s) |
| Storage | 2× Solidigm P44 Pro 2TB NVMe RAID0 |

## Dependencies

### llama.cpp (Modded Fork)

This project uses a modified llama.cpp with performance optimizations:

**Fork:** https://github.com/pestopoppa/llama.cpp

| Optimization | Speedup | PR |
|--------------|---------|-----|
| Parallel tensor repacking | 2.2x loading | [#18239](https://github.com/ggml-org/llama.cpp/pull/18239) |

---

## Links

- [llama.cpp (upstream)](https://github.com/ggml-org/llama.cpp)
- [llama.cpp (modded fork)](https://github.com/pestopoppa/llama.cpp)
- [Speculative Decoding Papers](https://github.com/hemingkx/SpeculativeDecodingPapers)

---

*January 2026 | AMD EPYC 9655 Inference Optimization Project*
