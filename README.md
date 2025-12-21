# AMD EPYC 9655 Inference Optimization

LLM inference optimization research on AMD EPYC 9655 "Turin" (96 cores, 1.13TB DDR5).

## Best Results

| Configuration | Speed | Speedup | Use Case |
|---------------|-------|---------|----------|
| Prompt Lookup (summarization) | 95.18 t/s | **12.7x** | Document QA |
| Qwen2.5-Coder-32B + 0.5B (K=24) | 33.0 t/s | **11x** | Code generation |
| Prompt Lookup (code editing) | 25.82 t/s | **8.6x** | Refactoring |
| Qwen2.5-72B + 0.5B (K=16) | 8.53 t/s | **5.8x** | General tasks |

## Key Insights

1. **Small drafts win on CPU**: 0.5B at 85 t/s beats 7B at 8 t/s
2. **MoE models don't need speculation**: Already at "draft speed" (~25 t/s)
3. **K-tuning matters**: K=8 for 7B, K=16-24 for 32B+
4. **Temperature helps**: temp=0.5-0.7 can double performance

## Track Status

| Track | Method | Speedup | Status |
|-------|--------|---------|--------|
| Track 1 | External Draft Model | 5.9-11x | **Production** |
| Track 2 | MoE Expert Reduction | +21-48% | **Production** |
| Track 8 | Prompt Lookup | 8.6-12.7x | **Production** |
| Track 3 | EAGLE-1 | — | Deprecated (0% acceptance) |

## Quick Start

```bash
# Track 1: External Draft Model (5.9x speedup)
numactl --interleave=all \
  llama-speculative \
  -m Qwen2.5-Coder-32B-Q4_K_M.gguf \
  -md Qwen2.5-0.5B-Instruct-Q8_0.gguf \
  --draft-max 16 -t 96 -p "prompt"

# Track 2: MoE Expert Reduction (+21-48%)
numactl --interleave=all \
  llama-cli -m Qwen3-30B-A3B-Q4_K_M.gguf \
  --override-kv qwen3moe.expert_used_count=int:4 \
  -t 96 -p "prompt"
```

> **Note:** Do NOT use `OMP_NUM_THREADS=1` - it disables parallel tensor repacking and hurts prompt processing (49 vs 119 t/s).

## Project Structure

```
/mnt/raid0/llm/claude/
├── CLAUDE.md              # Main project configuration
├── README.md              # This file
├── agents/                # Claude Code agent definitions
│   ├── lead-developer.md
│   ├── research-engineer.md
│   ├── research-writer.md
│   └── benchmark-analyst.md
├── docs/                  # Consolidated documentation
│   └── model-routing.md   # Model selection strategy
├── research/              # Research documents
│   └── speculative_decoding_research.md
├── scripts/
│   ├── benchmark/         # Benchmarking scripts
│   ├── session/           # Session management
│   └── utils/             # Utility scripts
└── logs/                  # Benchmark results (not in repo)
```

## Model Routing

Select model based on task complexity:

| Task Type | Model | Examples |
|-----------|-------|----------|
| Novel design, complex debugging | **Opus** | "Debug 0% acceptance", "Design new approach" |
| Research, synthesis, routine code | **Sonnet** | "Find papers", "Update report", "Add CLI flag" |
| Benchmark execution, log parsing | **Haiku** | "Run benchmark", "Parse CSV" |

**Default:** Sonnet. Escalate to Opus if blocked.

## Key Files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Project configuration, track status, commands |
| `docs/model-routing.md` | Detailed model selection guide |
| `research/speculative_decoding_research.md` | Research methodology |
| `agents/README.md` | Agent definitions and coordination |

## Hardware

- **CPU:** AMD EPYC 9655 "Turin" (96 cores, 192 threads, Zen 5)
- **RAM:** 1.13 TB DDR5-5600 ECC (12 channels, ~460 GB/s)
- **Storage:** 2x Solidigm P44 Pro 2TB NVMe RAID0

## Documentation

| File | Purpose |
|------|---------|
| [RESULTS_SUMMARY.md](research/RESULTS_SUMMARY.md) | Compact results for quick reference |
| [research_report_template.md](research/research_report_template.md) | Full report template for blog post |
| [speculative_decoding_research.md](research/speculative_decoding_research.md) | Methodology and track details |

## Dependencies

### llama.cpp (Modded Fork)

This project uses a modified llama.cpp with performance optimizations for many-core CPUs:

**Fork:** https://github.com/pestopoppa/llama.cpp

| Optimization | Speedup | PR |
|--------------|---------|-----|
| Parallel tensor repacking | 2.2x loading | [#18239](https://github.com/ggml-org/llama.cpp/pull/18239) |

Local patches are in `patches/` for upstream submission.

```bash
# Clone the modded fork
git clone https://github.com/pestopoppa/llama.cpp.git
cd llama.cpp
git checkout parallel-repack  # or apply patches manually
cmake -B build && cmake --build build -j
```

## Links

- [llama.cpp (upstream)](https://github.com/ggml-org/llama.cpp)
- [llama.cpp (modded fork)](https://github.com/pestopoppa/llama.cpp)
- [Speculative Decoding Papers](https://github.com/hemingkx/SpeculativeDecodingPapers)
- [SuffixDecoding](https://suffix-decoding.github.io/)

---

December 2025 | AMD EPYC Speculative Decoding Project
