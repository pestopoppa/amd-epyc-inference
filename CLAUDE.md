# AMD EPYC 9655 "Turin" Inference Optimization Project

## Table of Contents

- [Critical Constraints](#-absolute-rule-no-root-filesystem-writes-)
- [Test Memory Safety](#-test-memory-safety-)
- [System Identity](#system-identity)
- [Hardware Specifications](#hardware-specifications)
- [Plan Mode](#plan-mode)
- [Available Skills](#available-skills)
- [Current Status](#current-status-production-orchestration-december-2025)
- [Hierarchical Orchestration System](#hierarchical-orchestration-system)
- [Directory Structure](#directory-structure)
- [Session Startup](#session-startup-mandatory)
- [Orchestration Workflow](#orchestration-workflow)
- [Verification Gates](#verification-gates)
- [Model Routing Strategy](#model-routing-strategy)
- [Logging Requirements](#mandatory-append-only-agent-logging)
- [Key Resources](#key-resources)
- [Code Style](#code-style)
- [Git Workflow](#git-commit-workflow)

> **For human readers**: See [CLAUDE_GUIDE.md](CLAUDE_GUIDE.md) for a guide to navigating this document.
> **For detailed reference**: See [docs/reference/](docs/reference/) for extracted reference material.
> **Recent changes**: See [CHANGELOG.md](CHANGELOG.md). **New changelog entries go there, NEVER in this file.**

---

## ⛔⛔⛔ ABSOLUTE RULE: NO ROOT FILESYSTEM WRITES ⛔⛔⛔

**ALL LLM-related files MUST reside on `/mnt/raid0/` — NEVER on root (`/`).** The root filesystem is a 120GB SSD — writing large files causes system crashes and disk exhaustion.

### Allowed vs Forbidden Paths

| ✅ ALLOWED (RAID Array) | ❌ FORBIDDEN (Root FS) |
|-------------------------|------------------------|
| `/mnt/raid0/llm/` | `/home/` (except symlinks) |
| `/mnt/raid0/llm/claude/` | `/tmp/` (except via bind mount) |
| `/mnt/raid0/llm/claude/logs/` | `/var/` |
| `/mnt/raid0/llm/cache/` | `~/.cache/` |
| `/mnt/raid0/llm/models/` | `~/.local/` |
| `/mnt/raid0/llm/tmp/` | Any path not starting with `/mnt/raid0/` |

### Environment Variables

Set `HF_HOME`, `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE` → `/mnt/raid0/llm/cache/huggingface`; `PIP_CACHE_DIR` → `/mnt/raid0/llm/cache/pip`; `TMPDIR` → `/mnt/raid0/llm/tmp`; `XDG_CACHE_HOME` → `/mnt/raid0/llm/claude/cache`; `XDG_DATA_HOME` → `/mnt/raid0/llm/claude/share`; `XDG_STATE_HOME` → `/mnt/raid0/llm/claude/state`.

> A hook (`scripts/hooks/check_filesystem_path.sh`) enforces this automatically on Write/Edit.

---

## ⚠️ Test Memory Safety ⚠️

**NEVER use `pytest -n auto` on this 192-thread machine!** It spawns ~192 workers, each loading models, exhausting 1.13TB RAM.

```bash
pytest tests/            # Default: -n 8 via pyproject.toml (safe, 4x speedup)
pytest tests/ -n 4       # Conservative
# DANGEROUS — DO NOT USE: pytest tests/ -n auto
```

> A hook (`scripts/hooks/check_pytest_safety.sh`) blocks `pytest -n auto` and `-n N` where N > 16.

Safeguards: lazy MemRL loading (mock mode skips model init), memory guard in `tests/conftest.py` (< 100GB = fail), `make check-memory`.

---

## System Identity

- **Host**: Beelzebub | **User**: daniele | **Working Dir**: `/mnt/raid0/llm/` | **Python Env**: `pace-env`

### Key Files

| File | Purpose |
|------|---------|
| `docs/reference/benchmarks/RESULTS.md` | **MASTER BENCHMARK TABLE** — scores, speeds, optimizations |
| `orchestration/model_registry.yaml` | Model configs, paths, compatible drafts, launch commands |
| `docs/reference/models/QUIRKS.md` | Known model issues & workarounds |
| `benchmarks/results/reviews/summary.csv` | Claude-as-Judge scores (CSV) |

---

## Hardware Specifications

| Component | Specification |
|-----------|---------------|
| CPU | AMD EPYC 9655 "Turin" — 96 cores, 192 threads (Zen 5) |
| RAM | 1.13 TB DDR5-5600 ECC, 12 channels (~460 GB/s) |
| Storage | 2x Solidigm P44 Pro 2TB NVMe RAID0 (models), 120GB SSD (OS) |
| Architecture | Zen 5 with true 512-bit AVX-512 (not double-pumped) |

## Plan Mode

- Make the plan extremely concise. Sacrifice grammar for the sake of concision.
- At the end of each plan, give me a list of unresolved questions to answer, if any.
- when presenting potential solutions to problems, generate 2-3 primary and 5 alternative fixes, each within a separate <response> tag.
Each <response> must include a <text> and a numeric <probability>.  Please sample the 5 alternative fixes at random from the tails of the distribution (e.g., probabilities < 0.10). This should help us ultrathink outside the box a little more and consider an even wider range of options.

---

## Available Skills

| Skill | Purpose |
|-------|---------|
| `/new-model` | Register new model in registry |
| `/benchmark` | Benchmarking workflow, scoring rubric, eval analysis, pitfalls |
| `/draft-compat` | Validate draft-target model compatibility |
| `/research-update` | Update results across tracking docs |
| `/refactor` | Code technical debt analysis |
| `/mcp-knowledge` | Knowledge tools integration |

---

## Current Status: Production Orchestration (December 2025)

### Best Results Achieved

| Configuration | Speed | Speedup | Use Case |
|---------------|-------|---------|----------|
| Prompt Lookup (summarization) | 95.18 t/s | **12.7x** | Document QA, summarization |
| Qwen2.5-Coder-32B + 0.5B (K=24) | 33.0 t/s | **11x** | Code generation |
| Prompt Lookup (code editing) | 25.82 t/s | **8.6x** | Refactoring, code review |
| MoE Expert Reduction (4 experts) | +21-48% | — | MoE models |

### Deprecated (Do Not Use)

| Track | Method | Reason |
|-------|--------|--------|
| 3 | EAGLE-1 | 0% acceptance (quantization mismatch) |
| 7 | CAS-Spec | 0.446% acceptance |

---

## Hierarchical Orchestration System

> *One model thinks. Many models work. Tools decide who is right.*

### Agent Tiers

| Tier | Role | Model | Acceleration |
|------|------|-------|--------------|
| **A** | Front Door / Orchestrator | Qwen3-Coder-30B-A3B | MoE6 → 18 t/s |
| **B1** | Coder | Qwen2.5-Coder-32B | Spec K=24 → 33 t/s |
| **B2** | Ingestion | Qwen3-Next-80B-A3B | Expert reduction only (SSM!) |
| **B3** | Architect (General) | Qwen3-235B-A22B | MoE4 → 6.75 t/s |
| **B4** | Architect (Coding) | Qwen3-Coder-480B-A35B | MoE3 → 10.3 t/s |
| **C** | Workers (parallel) | Qwen2.5-7B/VL-7B/1.5B | Various |
| **D** | Draft / Embedder | Qwen2.5-Coder-0.5B / BGE-large | Co-loaded |

### Critical Constraints

- **SSM Models (Qwen3-Next)**: NEVER use speculative decoding or prompt lookup. SSM requires consecutive positions.
- **Qwen3-Coder-480B**: BOS token mismatch (`BOS=','`) breaks all speculation. Expert reduction only.

### Component Flow

> Last updated: 2026-02-11

```
Request:    API(:8000) → AppState → ChatPipeline → REPLExecutor → run_task() → [graph nodes]
Graph:      orchestration_graph (pydantic-graph) → 7 node classes → LLMPrimitives → [model servers]
Memory:     EpisodicStore(SQLite) → FAISSStore(4042 vectors) → ParallelEmbedder → BGE pool(:8090-8095)
Retrieval:  NextPLAID(:8088) → LateOn-Code-edge(ONNX INT8) → code+docs indices(mmap) — multi-vector search
Escalation: Graph nodes use EscalationPolicy(rules) + MemRL(advisory) via TaskDeps injection
Graphs:     QScorer reads FailureGraph(anti-memory) + HypothesisGraph(confidence)
Tools:      REPLExecutor → ToolRegistry → PluginLoader(5 plugins, 10 tools)
Prompts:    resolve_prompt(name) → orchestration/prompts/{name}.md (hot-swap) → fallback constant
REPL:       sanitize_code_unicode() → exec(code) — strips non-ASCII before execution
```

**Visual topology**: `logs/canvases/component_topology.canvas` (for Obsidian)

---

## Directory Structure

**llama.cpp Fork:** https://github.com/pestopoppa/llama.cpp
- Worktree setup: [docs/reference/LLAMA_CPP_WORKTREES.md](docs/reference/LLAMA_CPP_WORKTREES.md)

### Branch Safety (CRITICAL)

**Production must use `production-consolidated` branch.** Verify: `scripts/session/verify_llama_cpp.sh`
**Never run benchmarks on a feature branch.** Use `llama.cpp-experimental/` for feature work.

```
/mnt/raid0/llm/
├── llama.cpp/              # PRODUCTION - production-consolidated branch
├── llama.cpp-experimental/ # EXPERIMENTAL - feature branches
├── hf/                     # HuggingFace format models
├── models/                 # GGUF converted models
├── cache/                  # HF/pip caches
├── tmp/                    # Temporary files (TMPDIR)
└── claude/                 # Project root
    ├── CLAUDE.md           # This file
    ├── CHANGELOG.md        # Dated knowledgebase updates
    ├── orchestration/      # Orchestration layer (registry, prompts, IR schema)
    ├── docs/               # chapters/, reference/, guides/
    ├── benchmarks/         # prompts/, results/ (runs/, reviews/)
    ├── handoffs/           # active/, blocked/
    ├── research/           # Research docs & findings
    ├── scripts/            # benchmark/, session/, hooks/, utils/, nextplaid/
    ├── cache/next-plaid/   # NextPLAID indices (mmap'd, Docker volume)
    └── logs/               # Runtime logs, agent_audit.log
```

### Where to Save What

| Content | Location |
|---------|----------|
| Benchmark results (raw) | `benchmarks/results/runs/{timestamp}/` |
| Benchmark summary | `docs/reference/benchmarks/RESULTS.md` |
| Model registry | `orchestration/model_registry.yaml` |
| Model quirks | `docs/reference/models/QUIRKS.md` |
| Handoffs | `handoffs/active/` → `blocked/` → complete → delete |
| Progress logs | `progress/YYYY-MM/YYYY-MM-DD.md` |

---

## SESSION STARTUP (MANDATORY)

```bash
# 1. Set environment & init logging
export HF_HOME=/mnt/raid0/llm/cache/huggingface TMPDIR=/mnt/raid0/llm/tmp XDG_CACHE_HOME=/mnt/raid0/llm/claude/cache
source /mnt/raid0/llm/claude/scripts/utils/agent_log.sh
agent_session_start "Session purpose"

# 2. Discover models & verify llama.cpp branch
bash /mnt/raid0/llm/claude/scripts/session/session_init.sh

# 3. Load research context
cat /mnt/raid0/llm/claude/docs/reference/benchmarks/RESULTS.md
```

### Orchestrator Stack (For Live Inference)

```bash
python3 scripts/server/orchestrator_stack.py start --dev          # Dev mode (0.5B only)
python3 scripts/server/orchestrator_stack.py start --hot-only     # HOT tier (~40GB)
python3 scripts/server/orchestrator_stack.py status               # Check status
python3 scripts/server/orchestrator_stack.py stop --all           # Stop all
```

Port topology: run `orchestrator_stack.py status` for current server map. See [Model Routing](#model-routing-strategy) for role→port mapping.

### After Any Work
```bash
cd /mnt/raid0/llm/claude && make gates
```

---

## Quick Reference Commands

See [docs/reference/commands/QUICK_REFERENCE.md](docs/reference/commands/QUICK_REFERENCE.md) for full launch commands (spec decode, MoE reduction, prompt lookup, SSM).

---

## Orchestration Workflow

1. **TaskIR emission**: Front door emits `TaskIR` JSON → `orchestration/last_task_ir.json`
2. **Validate**: `python3 orchestration/validate_ir.py task orchestration/last_task_ir.json`
3. **Route**: Read `orchestration/model_registry.yaml` — `code` → coder, `ingest` → Qwen3-Next, workers parallel
4. **Gates**: `make gates` (schema → shellcheck → format → lint)
5. **Failures**: 1st → return report to agent; 2nd → escalate one tier; 3rd → B3 Architect

---

## Verification Gates

Every artifact must pass, in order:

1. **Schema validation** (`validate_ir.py`)
2. **Shell lint** (`shellcheck`)
3. **Format check** (`shfmt`, `mdformat`)
4. **Markdown lint** (`markdownlint`)
5. **Unit tests** (when applicable)
6. **Integration tests** (when applicable)
7. **Index freshness** (`make nextplaid-reindex`) — when NextPLAID is running

Run via `make gates`.

---

## Model Routing Strategy

### Claude Code Tier Selection

| Tier | Model | Use When |
|------|-------|----------|
| **Opus 4.5** | Deep reasoning | Novel design, complex debugging, architecture |
| **Sonnet 4.5** | Default | Research, synthesis, parallel questions |
| **Haiku 4.5** | Fast execution | Repetitive benchmarks, log parsing, known commands |

### Local Model Routing (Orchestrator)

| Task Type | Model | Port | Speed |
|-----------|-------|------|-------|
| Interactive chat | Qwen3-Coder-30B-A3B | 8080 | 18 t/s (MoE6) |
| Code gen / escalation | Qwen2.5-Coder-32B | 8081 | 39 t/s (spec+lookup) |
| Explore / summarize | Qwen2.5-7B-f16 | 8082 | 44 t/s (spec+lookup) |
| Long-context ingest | Qwen3-Next-80B-A3B | 8085 | 6.3 t/s (**NO SPEC**) |
| Architecture (general) | Qwen3-235B-A22B | 8083 | 6.75 t/s (MoE4) |
| Architecture (coding) | Qwen3-Coder-480B-A35B | 8084 | 10.3 t/s (MoE3) |
| Vision | Qwen2.5-VL-7B / Qwen3-VL-30B | 8086/8087 | ~15 / ~10 t/s |

---

## MANDATORY: Append-Only Agent Logging

```bash
source /mnt/raid0/llm/claude/scripts/utils/agent_log.sh
agent_task_start "Description" "Reasoning"
# ... do work ...
agent_task_end "Description" "success|failure"
```

Analysis: `scripts/utils/agent_log_analyze.sh --summary | --loops | --errors`

Loop prevention: max 3 retries → log blocker → STOP.

---

## Benchmarking & Eval

For model testing, scoring rubric, eval analysis, pitfalls: use `/benchmark`
For draft-target compatibility checks: use `/draft-compat`
For updating results docs after benchmarks: use `/research-update`

---

## Key Resources

### Documentation

| Document | Location |
|----------|----------|
| Results Summary | `logs/research_report.md` |
| Model Registry | `orchestration/model_registry.yaml` |
| TaskIR Schema | `orchestration/task_ir.schema.json` |
| Benchmark Prompts | `benchmarks/prompts/v1/` |
| Benchmark Results | `benchmarks/results/` |
| Blocked Tasks | `orchestration/BLOCKED_TASKS.md` |

### Commands

| Action | Command |
|--------|---------|
| Run gates | `make gates` |
| Validate TaskIR | `python3 orchestration/validate_ir.py task FILE.json` |
| Analyze logs | `scripts/utils/agent_log_analyze.sh --summary` |
| Discover models | `scripts/session/session_init.sh` |

---

## Code Style

- Use `#!/bin/bash` with `set -euo pipefail`
- **ALWAYS log all actions**
- All files on `/mnt/raid0/`
- Prefix inference: `OMP_NUM_THREADS=1 numactl --interleave=all`
- Run `make gates` after producing artifacts

---

## Git Commit Workflow

1. **Run tests first:** `make test-all`
2. **Update progress report:** `orchestration/progress/PROGRESS_YYYY-MM-DD.md`
3. **Update research summary** if benchmarks changed: `research/RESULTS_SUMMARY.md`
4. **Commit with descriptive message**: what changed, test status, perf metrics if applicable

When creating/updating handoff documents, **always update `orchestration/BLOCKED_TASKS.md`** (status, resume commands, completion checklist).
