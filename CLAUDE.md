# AMD EPYC 9655 "Turin" Inference Optimization Project

## ⛔⛔⛔ ABSOLUTE RULE: NO ROOT FILESYSTEM WRITES ⛔⛔⛔

**ALL LLM-related files MUST reside on `/mnt/raid0/` — NEVER on root (`/`).**

**THIS IS NON-NEGOTIABLE.** The root filesystem is a 120GB SSD. Writing large files there causes:
- System instability and crashes
- Paging storms that freeze the machine
- Disk exhaustion that corrupts the OS

### Path Verification (MANDATORY before any file operation)

```bash
# The path MUST start with /mnt/raid0/
[[ "$TARGET_PATH" == /mnt/raid0/* ]] || { echo "ERROR: Path not on RAID!"; exit 1; }
```

### Allowed vs Forbidden Paths

| ✅ ALLOWED (RAID Array) | ❌ FORBIDDEN (Root FS) |
|-------------------------|------------------------|
| `/mnt/raid0/llm/` | `/home/` (except symlinks) |
| `/mnt/raid0/llm/claude/` | `/tmp/` (except via bind mount) |
| `/mnt/raid0/llm/claude/logs/` | `/var/` |
| `/mnt/raid0/llm/cache/` | `~/.cache/` |
| `/mnt/raid0/llm/models/` | `~/.local/` |
| `/mnt/raid0/llm/tmp/` | Any path not starting with `/mnt/raid0/` |

### Environment Variables (MUST be set in every session)

```bash
export HF_HOME=/mnt/raid0/llm/cache/huggingface
export TRANSFORMERS_CACHE=/mnt/raid0/llm/cache/huggingface
export HF_DATASETS_CACHE=/mnt/raid0/llm/cache/huggingface/datasets
export PIP_CACHE_DIR=/mnt/raid0/llm/cache/pip
export TMPDIR=/mnt/raid0/llm/tmp
export XDG_CACHE_HOME=/mnt/raid0/llm/claude/cache
export XDG_DATA_HOME=/mnt/raid0/llm/claude/share
export XDG_STATE_HOME=/mnt/raid0/llm/claude/state
```

---

## System Identity

- **Host**: Beelzebub
- **User**: daniele
- **Working Directory**: `/mnt/raid0/llm/`
- **Python Environment**: `pace-env`

---

## Hardware Specifications

| Component | Specification |
|-----------|---------------|
| CPU | AMD EPYC 9655 "Turin" — 96 cores, 192 threads (Zen 5) |
| RAM | 1.13 TB DDR5-5600 ECC, 12 channels (~460 GB/s) |
| Storage | 2× Solidigm P44 Pro 2TB NVMe RAID0 (models), 120GB SSD (OS) |
| Architecture | Zen 5 with true 512-bit AVX-512 (not double-pumped) |

---

## Current Status: Production Orchestration (December 2025)

### Best Results Achieved

| Configuration | Speed | Speedup | Use Case |
|---------------|-------|---------|----------|
| Prompt Lookup (summarization) | 95.18 t/s | **12.7x** | Document QA, summarization |
| Qwen2.5-Coder-32B + 0.5B (K=24) | 33.0 t/s | **11x** | Code generation |
| Prompt Lookup (code editing) | 25.82 t/s | **8.6x** | Refactoring, code review |
| Qwen2.5-72B + 0.5B (K=16) | 8.53 t/s | **5.8x** | General tasks |
| MoE Expert Reduction (4 experts) | +21-48% | — | MoE models |

### Production Tracks

| Track | Method | Status | Result |
|-------|--------|--------|--------|
| 1 | External Draft | **Production** | 5.9-11x |
| 2 | MoE Expert Reduction | **Production** | +21-48% |
| 8 | Prompt Lookup | **Production** | 8.6-12.7x |

### Deprecated (Do Not Use)

| Track | Method | Reason |
|-------|--------|--------|
| 3 | EAGLE-1 | 0% acceptance (quantization mismatch) |
| 7 | CAS-Spec | 0.446% acceptance |

---

## Hierarchical Orchestration System

This project uses a **hierarchical local-agent workflow** for production inference. The system is designed around measured CPU performance with explicit attention to throughput, latency, and correctness.

### Core Philosophy

> *One model thinks. Many models work. Tools decide who is right.*

### Agent Tiers

#### Tier A — Front Door / Orchestrator
- Interactive chat, intent classification, task routing
- Emits `TaskIR` JSON
- **Model**: Qwen3-Coder-30B-A3B with expert reduction (41.55 t/s)
- Always resident, low-latency

#### Tier B — Specialists

| Role | Purpose | Model | Acceleration |
|------|---------|-------|--------------|
| **B1: Coder** | Code generation, refactors | Qwen2.5-Coder-32B | Speculative (K=24) → 33 t/s |
| **B2: Ingestion** | Long-context synthesis | Qwen3-Next-80B-A3B | Expert reduction only (SSM!) |
| **B3: Architect** | System design, invariants | Qwen3-235B-A22B | Expert reduction |

#### Tier C — Workers (Parallel)
- File-level implementation, tests, docs
- **Models**: Meta-Llama-3-8B, Qwen2.5-Math-7B, Qwen2.5-VL-7B
- Stateless, cheap, many run concurrently

#### Tier D — Draft
- Speculative decoding draft models
- **Model**: Qwen2.5-Coder-0.5B-Instruct Q8_0 (85 t/s)

### Critical Constraints

**SSM Models (Qwen3-Next)**: NEVER use speculative decoding or prompt lookup. SSM architecture requires consecutive positions — incompatible with ALL speculation methods.

**Qwen3-Coder-480B**: BOS token mismatch (`BOS=','`) breaks all speculation. Use expert reduction only.

---

## Directory Structure

```
/mnt/raid0/llm/
├── llama.cpp/                    # Main inference engine
│   └── build/                    # CMake build directory
├── hf/                           # HuggingFace format models
├── models/                       # GGUF converted models
├── lmstudio/                     # LM Studio models
├── cache/                        # HF/pip caches
├── tmp/                          # Temporary files (TMPDIR)
└── claude/                       # Project documentation & scripts
    ├── CLAUDE.md                 # This file
    ├── OPENING_PROMPT.md         # Opening prompt template
    ├── logs/                     # Benchmark and runtime logs
    │   ├── research_report.md    # Main results document
    │   ├── agent_audit.log       # Agent action log
    │   └── benchmarks/           # Benchmark CSV results
    ├── orchestration/            # NEW: Orchestration layer
    │   ├── task_ir.schema.json   # TaskIR JSON Schema
    │   ├── architecture_ir.schema.json
    │   ├── model_registry.yaml   # Deterministic model mapping
    │   ├── validate_ir.py        # IR validator
    │   └── last_task_ir.json     # Current task (gitignored)
    ├── agents/                   # Specialized agent definitions
    ├── research/                 # Research documents
    └── scripts/                  # All scripts organized here
        ├── benchmark/            # bench_zen5.sh, run_inference.sh
        ├── session/              # session_init.sh, health_check.sh
        ├── system/               # system_audit.sh
        └── utils/                # agent_log.sh, agent_log_analyze.sh
```

---

## SESSION STARTUP (MANDATORY)

### 1. Set Environment & Initialize Logging
```bash
export HF_HOME=/mnt/raid0/llm/cache/huggingface
export TMPDIR=/mnt/raid0/llm/tmp
export XDG_CACHE_HOME=/mnt/raid0/llm/claude/cache

source /mnt/raid0/llm/claude/scripts/utils/agent_log.sh
agent_session_start "Session purpose description"
```

### 2. Discover Models
```bash
bash /mnt/raid0/llm/claude/scripts/session/session_init.sh
```

### 3. Load Research Context
```bash
head -100 /mnt/raid0/llm/claude/logs/research_report.md
```

### 4. Run Gates (After Any Work)
```bash
cd /mnt/raid0/llm/claude && make gates
```

---

## Quick Reference Commands

### Track 1: External Draft (11x on code)
```bash
OMP_NUM_THREADS=1 numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-speculative \
  -m /mnt/raid0/llm/models/Qwen2.5-Coder-32B-Q4_K_M.gguf \
  -md /mnt/raid0/llm/models/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf \
  --draft-max 24 -t 96 -p "prompt"
```

### Track 2: MoE Expert Reduction (+87% on 235B)
```bash
numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-cli \
  -m /mnt/raid0/llm/models/Qwen3-235B-A22B-Q4_K_M.gguf \
  --override-kv qwen3moe.expert_used_count=int:4 \
  -t 96 -p "prompt"
```

### Track 8: Prompt Lookup (12.7x on summarization)
```bash
numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-cli \
  -m /mnt/raid0/llm/models/Qwen2.5-Coder-32B-Q4_K_M.gguf \
  --lookup-ngram-min 3 \
  -t 96 -f prompt_with_source_material.txt
```

### SSM Model (Expert Reduction ONLY)
```bash
# ⛔ DO NOT use --draft or --lookup with Qwen3-Next!
numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-cli \
  -m /mnt/raid0/llm/models/Qwen3-Next-80B-A3B-Q4_K_M.gguf \
  --override-kv qwen3next.expert_used_count=int:2 \
  -t 96 -p "prompt"
```

---

## Orchestration Workflow

### 1. TaskIR Emission (Front Door)

The Front Door emits a `TaskIR` JSON for every non-trivial request:

```json
{
  "task_id": "uuid",
  "task_type": "code",
  "priority": "interactive",
  "objective": "Implement error handling for registry loader",
  "agents": [{"tier": "B", "role": "coder"}],
  "plan": {"steps": [...]},
  "gates": ["schema", "format", "lint", "typecheck", "unit"],
  "definition_of_done": ["Tests pass", "Type hints complete"],
  "escalation": {"max_level": "B3", "on_second_failure": true}
}
```

Save to: `orchestration/last_task_ir.json`

### 2. Validate IR

```bash
python3 orchestration/validate_ir.py task orchestration/last_task_ir.json
```

### 3. Route to Specialist/Workers

Read `orchestration/model_registry.yaml` for deterministic model selection:
- `task_type == 'code'` → coder_primary (Qwen3-Coder-30B-A3B + MoE)
- `task_type == 'ingest'` → ingest_long_context (Qwen3-Next-80B)
- Workers run in parallel for file-level tasks

### 4. Run Gates

```bash
make gates  # schema → shellcheck → format → lint
```

### 5. Handle Failures

- First failure → return gate report to producing agent
- Second failure → escalate one tier
- Third failure → escalate to B3 Architect

---

## Verification Gates

Every artifact must pass, in order:

1. **Schema validation** (`validate_ir.py`)
2. **Shell lint** (`shellcheck`)
3. **Format check** (`shfmt`, `mdformat`)
4. **Markdown lint** (`markdownlint`)
5. **Unit tests** (when applicable)
6. **Integration tests** (when applicable)

Gates are run via `make gates` or `just gates`.

---

## Model Routing Strategy

### Tier Selection (Claude Code Context)

This section applies to Claude Code (Opus/Sonnet/Haiku) routing, not local inference.

| Tier | Model | Use When |
|------|-------|----------|
| **Opus 4.5** | Deep reasoning | Novel design, complex debugging, architecture |
| **Sonnet 4.5** | Default | Research, synthesis, parallel questions |
| **Haiku 4.5** | Fast execution | Repetitive benchmarks, log parsing, known commands |

### Local Model Routing (Orchestrator)

| Task Type | Model | Acceleration |
|-----------|-------|--------------|
| Interactive chat | Qwen3-Coder-30B-A3B | Expert reduction (4) |
| Code generation | Qwen2.5-Coder-32B | Speculative (K=24) |
| Long-context ingestion | Qwen3-Next-80B-A3B | Expert reduction (2), **NO SPEC** |
| Architecture/escalation | Qwen3-235B-A22B | Expert reduction (4) |
| Boilerplate/docs | Meta-Llama-3-8B | Prompt lookup |
| Math/invariants | Qwen2.5-Math-7B | Speculative (K=8) |
| Vision/UI | Qwen2.5-VL-7B | Speculative (K=8, temp=0.7) |

---

## MANDATORY: Append-Only Agent Logging

### Using the Logging Library
```bash
source /mnt/raid0/llm/claude/scripts/utils/agent_log.sh
```

### Required Pattern
```bash
agent_task_start "Description" "Reasoning"
# ... do work ...
agent_task_end "Description" "success|failure"
```

### Log Analysis
```bash
/mnt/raid0/llm/claude/scripts/utils/agent_log_analyze.sh --summary
/mnt/raid0/llm/claude/scripts/utils/agent_log_analyze.sh --loops
/mnt/raid0/llm/claude/scripts/utils/agent_log_analyze.sh --errors
```

### Loop Prevention
1. Max 3 retries on failures
2. If stuck: log blocker, document attempts, STOP
3. Always verify success via exit codes

---

## ⚠️ Benchmarking Pitfalls

### Interactive Mode Hangs
**CRITICAL**: `llama-cli` can hang waiting for user input if not configured correctly.

**ALWAYS use these flags when benchmarking:**
```bash
llama-cli -m MODEL.gguf -f prompt.txt -n 128 \
    --no-display-prompt \
    --simple-io \
    --no-warmup \
    --temp 0
```

**Never use:**
- `-i` or `--interactive` in automated scripts
- Pipes without proper EOF handling

**If a benchmark hangs:**
1. Check for interactive mode prompts
2. Verify timeout is set: `timeout 300 llama-cli ...`
3. Kill stuck processes: `pkill -f llama-cli`

### MANDATORY: Document Model Quirks

**After every new model benchmark**, update `orchestration/model_registry.yaml`:

1. **Add performance data** under the appropriate role entry:
   ```yaml
   performance:
     baseline_tps: <measured>
     optimized_tps: <measured>
     speedup: <calculated>
   benchmark_date: YYYY-MM-DD
   ```

2. **Document any runtime quirks** in the `runtime_quirks` section:
   ```yaml
   runtime_quirks:
     model_name:
       description: "Model full name"
       quirks:
         - issue: "What breaks or behaves unexpectedly"
           workaround: "How to fix or avoid it"
           discovered: YYYY-MM-DD
   ```

3. **Required quirk documentation includes:**
   - Speculative decoding acceptance rates (if unusually low)
   - MoE override key names (`qwen3moe.*` vs `qwen3next.*` etc.)
   - BOS/EOS token mismatches that break draft compatibility
   - Timeout/wrapper issues specific to model or binary
   - Architecture-specific constraints (SSM incompatibility, etc.)

4. **Reference the model registry** before running benchmarks to avoid rediscovering known quirks.

---

## Key Resources

### Documentation
| Document | Location |
|----------|----------|
| Results Summary | `logs/research_report.md` |
| Model Registry | `orchestration/model_registry.yaml` |
| TaskIR Schema | `orchestration/task_ir.schema.json` |
| Agent Definitions | `agents/` |

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

When committing orchestration-related changes:

1. **Run tests first:** `make test-all`
2. **Update progress report:** `orchestration/progress/PROGRESS_YYYY-MM-DD.md`
   - Update implementation status table
   - Add completed items to "Completed This Period"
   - Update test counts
   - Add new files to "Files Created/Modified"
3. **Update research summary** if benchmarks changed: `research/RESULTS_SUMMARY.md`
4. **Commit with descriptive message** including:
   - What was added/changed
   - Test status (X tests passing)
   - Performance metrics if applicable
