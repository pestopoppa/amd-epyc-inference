# AMD EPYC 9655 Inference Optimization

LLM inference optimization research on AMD EPYC 9655 "Turin" (96 cores, 1.13TB DDR5).

## Current Results

| Optimization | Speedup | Status |
|--------------|---------|--------|
| External Draft Model (Track 1) | **5.9x** | Production |
| MoE Expert Reduction (Track 2) | **21-48%** | Production |
| Prompt Lookup (Track 8) | **12.7x** (summarization) | Testing |

## Quick Start

```bash
# Track 1: External Draft Model (5.9x speedup)
OMP_NUM_THREADS=1 numactl --interleave=all \
  llama-speculative \
  -m Qwen2.5-Coder-32B-Q4_K_M.gguf \
  -md Qwen2.5-0.5B-Instruct-Q8_0.gguf \
  --draft-max 16 -t 96 -p "prompt"

# Track 2: MoE Expert Reduction (+21-48%)
llama-cli -m Qwen3-30B-A3B-Q4_K_M.gguf \
  --override-kv qwen3moe.expert_used_count=int:4 \
  -t 96 -p "prompt"
```

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

## Research Tracks

| Track | Method | Status |
|-------|--------|--------|
| Track 1 | External Draft Model | **Production** (5.9x) |
| Track 2 | MoE Soft Mask | **Production** (21-48%) |
| Track 6 | SuffixDecoding | Planned |
| Track 8 | Prompt Lookup | Testing |
| Track 3 | EAGLE-1 | Deprecated (0% acceptance) |

## Links

- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Speculative Decoding Papers](https://github.com/hemingkx/SpeculativeDecodingPapers)
- [SuffixDecoding](https://suffix-decoding.github.io/)

---

December 2025 | AMD EPYC Speculative Decoding Project
