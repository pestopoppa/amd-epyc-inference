# AMD EPYC 9655 Inference Optimization

LLM inference optimization research and production orchestration system on AMD EPYC 9655 "Turin" (96 cores, 1.13TB DDR5). Achieves up to **12.7x speedup** on CPU-only inference through speculative decoding, MoE expert reduction, and prompt lookup techniques.

> **[Master Benchmark Results](docs/reference/benchmarks/RESULTS.md)** — 77 models evaluated across 8 suites with Claude-as-Judge scoring

## Best Results

| Configuration | Speed | Speedup | Use Case |
|---------------|-------|---------|----------|
| Prompt Lookup (summarization) | 95.18 t/s | **12.7x** | Document QA |
| Qwen2.5-Coder-32B + 0.5B (K=24) | 33.0 t/s | **11x** | Code generation |
| Prompt Lookup (code editing) | 25.82 t/s | **8.6x** | Refactoring |
| Qwen2.5-72B + 0.5B (K=16) | 8.53 t/s | **5.8x** | General tasks |
| MoE Expert Reduction (235B) | 6.75 t/s | **+87%** | Architecture design |

## Architecture Overview

```
                     User Request
                          |
                    [Orchestrator API :8000]
                          |
              +-----------+-----------+
              |                       |
         [Front Door]           [MemRL System]
         Qwen3-Coder-30B       2714 memories
         MoE6, 18 t/s          FAISS + Q-scoring
              |                       |
    +---------+---------+    Learned Routing
    |         |         |             |
 [Coder]  [Architect] [Ingest]        |
 32B+0.5B  235B-A22B  80B-A3B         |
 33 t/s    6.75 t/s   6.3 t/s         |
    |         |         |             |
    +----+----+---------+-------------+
         |
    [Workers]  [Vision]  [Document OCR]
    7B models   VL-7B    LightOnOCR-2-1B
    44 t/s     ~15 t/s   :9001
```

**4 Agent Tiers**: Front Door (routing) -> Specialists (code, architecture, ingestion) -> Workers (parallel execution) -> Draft models (speculative decoding)

**Memory Tiers**: HOT ~535GB always resident (47% of RAM) | WARM on-demand | COLD disk-only

## Feature Highlights

### Inference Optimization (Part II)
- **[Speculative Decoding](docs/chapters/05-speculative-decoding.md)** — 0.5B draft model achieves 11x on code generation via K=24 lookahead
- **[MoE Expert Reduction](docs/chapters/06-moe-optimization.md)** — Reduce active experts at runtime for +21-87% speedup, quality preserved
- **[Prompt Lookup](docs/chapters/07-prompt-lookup.md)** — N-gram matching on input context yields 12.7x on summarization tasks
- **[RadixAttention](docs/chapters/08-radix-attention.md)** — Prefix caching for KV reuse across orchestrator requests

### System Architecture (Part III)
- **[Orchestration](docs/chapters/10-orchestration-architecture.md)** — Hierarchical 4-tier agent system with TaskIR emission and escalation chains
- **[REPL Environment](docs/chapters/11-repl-environment.md)** — 106KB sandboxed Python executor with AST-based security
- **[Production Server Stack](docs/chapters/12-production-server-stack.md)** — 9 llama-servers + 2 services, managed via orchestrator_stack.py
- **[Data Pipelines](docs/chapters/13-data-processing-pipelines.md)** — LightOnOCR (19x speedup), vision pipeline with CLIP embeddings and batch processing
- **[TOON Encoding](docs/chapters/14-toon-encoding.md)** — 55% token compression for structured data, 41.8% TTFT improvement

### Intelligence & Learning (Part IV)
- **[MemRL](docs/chapters/15-memrl-system.md)** — Memory-augmented RL with 2714 episodic memories, FAISS O(log n) retrieval, Q-value scoring
- **[Graph Reasoning](docs/chapters/16-graph-reasoning.md)** — Kuzu-backed failure (13 modes, 45 symptoms) and hypothesis (15 hypotheses) graphs
- **[Memory Seeding](docs/chapters/17-memory-seeding.md)** — 56 canonical seeds, 8 diverse seeding strategies for bootstrap
- **[Escalation & Routing](docs/chapters/18-escalation-and-routing.md)** — Learned + rule-based routing with complexity-aware delegation
- **[Procedure Registry](docs/chapters/19-procedure-registry.md)** — 11 YAML procedures for self-management at ~350 tokens/op
- **[Session Persistence](docs/chapters/20-session-persistence.md)** — 7-phase checkpoint/resume with document caching

### Operations (Part V)
- **[Benchmarking](docs/chapters/21-benchmarking-framework.md)** — 8 suites (thinking, coder, VL, general, agentic, math, long-context, instruction-precision), Claude-as-Judge scoring
- **[Tool Registry](docs/chapters/22-tool-registry.md)** — 40+ callable tools with role-based permissions, 8 agent role definitions
- **[Security & Monitoring](docs/chapters/23-security-and-monitoring.md)** — AST sandboxing, entropy-based early failure detection

## Setup

See **[docs/SETUP.md](docs/SETUP.md)** for comprehensive setup instructions, including:
- System requirements and prerequisites
- Environment configuration (`.env.example` template)
- Model downloads via HuggingFace
- llama.cpp build from source

**Quick Setup:**
```bash
# 1. Clone and configure
git clone https://github.com/your-org/claude.git && cd claude
cp .env.example .env  # Edit paths for your system

# 2. Install dependencies
pip install -e ".[dev]"  # or: uv sync

# 3. Verify setup
make validate-paths && make gates
```

## Quick Start

After setup, all paths are configured via environment variables:

```bash
# Track 1: External Draft Model (11x speedup on code)
OMP_NUM_THREADS=1 numactl --interleave=all \
  ${LLAMA_SERVER} -m ${MODELS_DIR}/Qwen2.5-Coder-32B-Q4_K_M.gguf \
  -md ${MODELS_DIR}/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf \
  --draft-max 24 -t 96 -p "Your prompt"

# Track 2: MoE Expert Reduction (+87% on MoE models)
numactl --interleave=all \
  ${LLAMA_CPP_BIN}/llama-cli \
  -m ${MODELS_DIR}/Qwen3-235B-A22B-Q4_K_M.gguf \
  --override-kv qwen3moe.expert_used_count=int:4 \
  -t 96 -p "Your prompt"

# Start orchestrator stack (HOT tier)
python3 scripts/server/orchestrator_stack.py start --hot-only
```

## Container Setup

For reproducible environments, use Docker or Nix:

```bash
# Docker: Build and run API server
make docker-build && make docker-run

# Docker: Development shell
make docker-dev

# Nix: Enter development shell
make nix-develop
```

See [docs/SETUP.md#container-setup](docs/SETUP.md#container-setup) for detailed container instructions.

---

## Documentation

### Research Chapters (27 chapters in 5 parts)

See **[Chapter Index](docs/chapters/INDEX.md)** for the full list with audience-specific reading paths.

| Part | Chapters | Focus |
|------|----------|-------|
| I: Foundation | 01-04 | Hardware, runtime, toolchain, safety |
| II: Inference Optimization | 05-09 | Speculative decoding, MoE, prompt lookup |
| III: System Architecture | 10-14 | Orchestration, REPL, servers, pipelines |
| IV: Intelligence & Learning | 15-20 | MemRL, graphs, seeding, procedures |
| V: Operations & Quality | 21-27 | Benchmarking, tools, security, debugger, skills |

### Reference & Guides

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Living technical reference (updated continuously) |
| [CLAUDE.md](CLAUDE.md) | AI context file (loaded at session start) |
| [Getting Started](docs/guides/getting-started.md) | New contributor onboarding |
| [Model Reference](docs/reference/models/MODELS.md) | All model specs and constraints |
| [Model Quirks](docs/reference/models/QUIRKS.md) | Known issues and workarounds |
| [Benchmark Results](docs/reference/benchmarks/RESULTS.md) | Complete benchmark data (77 models) |
| [Command Reference](docs/reference/commands/QUICK_REFERENCE.md) | Copy-paste inference commands |

### Progress & Work Tracking

| Document | Description |
|----------|-------------|
| [Progress Index](progress/INDEX.md) | Lab notebook entries by date |
| [Active Handoffs](handoffs/README.md) | Work in progress |
| [Blocked Tasks](orchestration/BLOCKED_TASKS.md) | Awaiting dependencies |

---

## Project Structure

```
$PROJECT_ROOT/                 # Set via ORCHESTRATOR_PATHS_PROJECT_ROOT
├── CLAUDE.md                  # AI context file
├── README.md                  # This file
├── .env.example               # Environment configuration template
│
├── docs/
│   ├── chapters/              # 27 research chapters (5 parts)
│   │   ├── INDEX.md           # Chapter index with reading paths
│   │   ├── 01-hardware-system.md ... 04-storage-and-safety.md
│   │   ├── 05-speculative-decoding.md ... 09-deprecated-approaches.md
│   │   ├── 10-orchestration-architecture.md ... 14-toon-encoding.md
│   │   ├── 15-memrl-system.md ... 20-session-persistence.md
│   │   └── 21-benchmarking-framework.md ... 23-security-and-monitoring.md
│   ├── ARCHITECTURE.md        # Living technical reference
│   ├── reference/             # Quick lookup (models, benchmarks, commands)
│   ├── guides/                # Human tutorials
│   └── deprecated/            # Historical archive
│
├── src/                       # Python source (orchestrator, REPL, services)
├── orchestration/             # Model registry, TaskIR schema, procedures, MemRL
├── agents/                    # Agent map, execution contract, shared policy, role overlays
├── scripts/                   # Server management, benchmarking, utilities
├── benchmarks/                # Test prompts and results (8 suites)
├── progress/                  # Lab notebook (daily entries)
├── handoffs/                  # Active and blocked work items
└── patches/                   # llama.cpp upstream patches
```

---

## Key Insights

1. **Small drafts win on CPU**: 0.5B at 85 t/s beats 7B at 8 t/s for speculative decoding
2. **MoE models don't need speculation**: Already at "draft speed" (~25 t/s), use expert reduction instead
3. **K-tuning matters**: K=8 for 7B, K=16-24 for 32B+ models
4. **Prompt lookup stacks**: Combines with speculative decoding for 5.4x additional gain
5. **Memory accelerates routing**: MemRL Q-scores reduce escalation latency vs rule-based routing

## Track Status

| Track | Method | Speedup | Status |
|-------|--------|---------|--------|
| Track 1 | External Draft Model | 5.9-11x | **Production** |
| Track 2 | MoE Expert Reduction | +21-87% | **Production** |
| Track 8 | Prompt Lookup | 8.6-12.7x | **Production** |
| Track 3 | EAGLE-1 | — | Deprecated (0% acceptance) |
| Track 7 | CAS-Spec | — | Deprecated (0.446% acceptance) |

---

## Hardware

| Component | Specification |
|-----------|---------------|
| CPU | AMD EPYC 9655 "Turin" (96 cores, 192 threads, Zen 5) |
| RAM | 1.13 TB DDR5-5600 ECC (12 channels, ~460 GB/s) |
| Storage | 2x Solidigm P44 Pro 2TB NVMe RAID0 |
| Architecture | True 512-bit AVX-512 (not double-pumped) |

## Dependencies

### llama.cpp (Modded Fork)

This project uses a modified llama.cpp with performance optimizations:

**Fork:** https://github.com/pestopoppa/llama.cpp

| Optimization | Speedup | PR |
|--------------|---------|-----|
| Parallel tensor repacking | 2.2x loading | [#18239](https://github.com/ggml-org/llama.cpp/pull/18239) |
| SWA spec decode fix | Gemma-3 support | [#18720](https://github.com/ggml-org/llama.cpp/pull/18720) |
| Prompt lookup in llama-server | 1.48x standalone | merged locally |

## Contributing

1. **Setup:** Follow [docs/SETUP.md](docs/SETUP.md) to configure your environment
2. **Onboard:** Read the [Getting Started guide](docs/guides/getting-started.md)
3. **Learn:** Browse the [Chapter Index](docs/chapters/INDEX.md) for your area of interest
4. **Reference:** Check [ARCHITECTURE.md](docs/ARCHITECTURE.md) for system internals
5. **Validate:** Run `make gates` after any changes (schema, shellcheck, format, lint)

> **Model Selection:** See [docs/MODEL_MANIFEST.md](docs/MODEL_MANIFEST.md) for role-based model configuration. You don't need the exact models we use—the orchestrator supports any compatible models.

---

## Links

- [llama.cpp (upstream)](https://github.com/ggml-org/llama.cpp)
- [llama.cpp (modded fork)](https://github.com/pestopoppa/llama.cpp)
- [Speculative Decoding Papers](https://github.com/hemingkx/SpeculativeDecodingPapers)

---

*January 2026 | AMD EPYC 9655 Inference Optimization Project*
