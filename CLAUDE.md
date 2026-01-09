# AMD EPYC 9655 "Turin" Inference Optimization Project

## Table of Contents

- [Critical Constraints](#-absolute-rule-no-root-filesystem-writes-)
- [System Identity](#system-identity)
- [Hardware Specifications](#hardware-specifications)
- [Current Status](#current-status-production-orchestration-december-2025)
- [Hierarchical Orchestration System](#hierarchical-orchestration-system)
- [Directory Structure](#directory-structure)
- [Session Startup](#session-startup-mandatory)
- [Quick Reference Commands](#quick-reference-commands)
- [Orchestration Workflow](#orchestration-workflow)
- [Verification Gates](#verification-gates)
- [Model Routing Strategy](#model-routing-strategy)
- [Logging Requirements](#mandatory-append-only-agent-logging)
- [Model Testing Workflow](#-new-model-testing-workflow)
- [Benchmarking Pitfalls](#%EF%B8%8F-benchmarking-pitfalls)
- [Claude-as-Judge Review](#claude-as-judge-quality-review)
- [Benchmark Hardening](#benchmark-hardening-2025-12-18)
- [Research Summary Maintenance](#research-summary-maintenance)
- [Key Resources](#key-resources)
- [Code Style](#code-style)
- [Git Workflow](#git-commit-workflow)

> **For human readers**: See [CLAUDE_GUIDE.md](CLAUDE_GUIDE.md) for a guide to navigating this document.
> **For detailed reference**: See [docs/reference/](docs/reference/) for extracted reference material.

---

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

### Key Files (Most Commonly Needed)

| File | Purpose |
|------|---------|
| `/mnt/raid0/llm/claude/orchestration/model_registry.yaml` | **Model configurations, paths, compatible drafts** |
| `/mnt/raid0/llm/claude/docs/reference/benchmarks/RESULTS.md` | Benchmark results summary |
| `/mnt/raid0/llm/claude/docs/reference/models/QUIRKS.md` | Known model issues & workarounds |
| `/mnt/raid0/llm/claude/benchmarks/results/reviews/summary.csv` | Claude-as-Judge scores |

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
| **B3: Architect (General)** | System design, invariants | Qwen3-235B-A22B | Expert reduction → 6.75 t/s |
| **B4: Architect (Coding)** | Ultimate code escalation | Qwen3-Coder-480B-A35B | MoE3 only → 10.3 t/s |

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

**llama.cpp Fork:** https://github.com/pestopoppa/llama.cpp
- Contains local optimizations (parallel tensor repack, etc.)
- Patches in `claude/patches/` for upstream submission
- PR: https://github.com/ggml-org/llama.cpp/pull/18239

```
/mnt/raid0/llm/
├── llama.cpp/                    # Main inference engine (modded)
│   └── build/                    # CMake build directory
├── hf/                           # HuggingFace format models
├── models/                       # GGUF converted models
├── lmstudio/                     # LM Studio models
├── cache/                        # HF/pip caches
├── tmp/                          # Temporary files (TMPDIR)
└── claude/                       # Project documentation & scripts
    ├── CLAUDE.md                 # This file
    ├── OPENING_PROMPT.md         # Opening prompt template
    ├── handoffs/                 # Work-in-progress handoffs
    │   ├── active/               # Currently active work
    │   ├── blocked/              # Work awaiting dependencies
    │   └── README.md             # Handoff lifecycle docs
    ├── docs/                     # Curated documentation
    │   ├── chapters/             # Permanent research chapters
    │   ├── reference/            # Quick-lookup reference
    │   │   ├── benchmarks/RESULTS.md  # Canonical benchmark results
    │   │   └── models/           # MODELS.md, QUIRKS.md
    │   ├── guides/               # Human tutorials
    │   └── deprecated/           # Superseded documents
    ├── benchmarks/               # Benchmark infrastructure
    │   ├── prompts/              # Test prompts by category
    │   └── results/              # Raw benchmark data
    │       ├── index.jsonl       # Master index
    │       ├── runs/             # Timestamped run directories
    │       └── reviews/          # Claude-as-Judge scores
    ├── orchestration/            # Orchestration layer
    │   ├── task_ir.schema.json   # TaskIR JSON Schema
    │   ├── model_registry.yaml   # Deterministic model mapping
    │   ├── validate_ir.py        # IR validator
    │   └── progress/             # Weekly progress snapshots
    ├── research/                 # Research docs & findings (NOT handoffs)
    ├── progress/                 # Lab notebook (daily entries)
    ├── agents/                   # Specialized agent definitions
    ├── patches/                  # llama.cpp patches for upstream
    ├── logs/                     # Runtime logs
    │   └── agent_audit.log       # Agent action log
    └── scripts/                  # All scripts organized here
        ├── benchmark/            # bench_zen5.sh, run_inference.sh
        ├── session/              # session_init.sh, health_check.sh
        ├── system/               # system_audit.sh
        └── utils/                # agent_log.sh, agent_log_analyze.sh
```

### Where to Save What

| Content Type | Location |
|--------------|----------|
| **Handoffs (new work)** | `handoffs/active/{topic}.md` |
| **Blocked handoffs** | `handoffs/blocked/` (update `BLOCKED.md`) |
| **Benchmark results (raw)** | `benchmarks/results/runs/{timestamp}/` |
| **Benchmark results (summary)** | `docs/reference/benchmarks/RESULTS.md` |
| **Model registry updates** | `orchestration/model_registry.yaml` |
| **Model quirks discovered** | `docs/reference/models/QUIRKS.md` |
| **Research findings** | `research/` (NOT handoffs) |
| **Progress logs** | `progress/YYYY-MM/YYYY-MM-DD.md` |
| **Permanent documentation** | `docs/chapters/` |

**Handoff Lifecycle:** Create in `active/` → Work → (optional) Move to `blocked/` → Complete → Extract findings to `docs/`, then DELETE handoff.

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
cat /mnt/raid0/llm/claude/docs/reference/benchmarks/RESULTS.md
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

## 🔧 New Model Testing Workflow

**CRITICAL: When testing a NEW model, follow this order:**

### Step 1: Establish Reliable Launch (FIRST)
Before ANY testing:
1. Run a minimal test: `llama-completion -m MODEL.gguf -p "Hello"`
2. Identify and document launch quirks:
   - Does it need specific flags?
   - Does it auto-enable interactive/conversation mode?
   - Are there output format quirks (e.g., `<think>` tags)?
   - Does piping output cause errors?
3. Add quirks to `orchestration/model_registry.yaml` immediately

### Step 2: Run Quality Rubric (Captures Speed Automatically)
Once launch is reliable:
1. Run quality rubric script (e.g., `run_thinking_rubric.sh`)
2. Script captures BOTH quality scores AND speed per question
3. Apply known optimizations during testing:
   - MoE models: `--override-kv ARCH.expert_used_count=int:4`
   - Dense models: spec decode if compatible
4. Assign role based on tier scores

**DO NOT do separate speed benchmarks** - the rubric captures speed data.

### Step 3: Run Full Benchmark Suites (MANDATORY for New Models)
After registry entry and quirks are documented, run the complete benchmark suite:

```bash
# Run all 8 benchmark suites for comprehensive evaluation
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite all

# Or run specific suites based on model role:
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite thinking      # Reasoning models
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite coder         # Code models
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite instruction_precision  # Orchestration candidates
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite long_context  # Context window testing
```

**8 Benchmark Suites:**
1. **Thinking** - Chain-of-thought, multi-step reasoning
2. **Coder** - Code generation, debugging, refactoring
3. **VL** - Vision-language (OCR, image understanding)
4. **General** - Instruction following, summarization
5. **Agentic** - Tool calling, function extraction
6. **Math** - Mathematical reasoning, step verification
7. **Long Context** - Information retrieval across 4K-50K token contexts
8. **Instruction Precision** - Exact format compliance (critical for orchestration)

**Why this matters:**
- Results are stored permanently in `/mnt/raid0/llm/claude/benchmarks/results/`
- JSONL index enables faithful comparison with future models
- Models can be deleted after benchmarking - results persist for comparison
- Instruction Precision suite identifies models that will break orchestration parsing

**Why this order matters:**
- Debugging launch issues DURING quality tests wastes time
- Quality rubric captures speed - no separate benchmark needed
- Registry should always have working launch commands
- Full benchmark suite provides permanent record for future comparison

---

## Draft-Target Compatibility Validation (MANDATORY)

**Before adding ANY draft-target pairing to `model_registry.yaml`, run the compatibility check:**

```bash
python3 scripts/utils/check_draft_compatibility.py DRAFT.gguf TARGET.gguf
```

### What It Checks

1. **Vocab size match** - If draft has fewer tokens than target, spec decode may SIGSEGV when target generates a token ID the draft can't handle
2. **BOS/EOS token match** - Mismatch causes generation failures or garbage output
3. **Tokenizer model/pre match** - Different tokenizer families are usually incompatible

### Known Cases

| Draft | Target | Vocab Diff | Result |
|-------|--------|------------|--------|
| Gemma-3-1B-IT | Gemma-3-27B-QAT | 64 fewer | **std::bad_alloc crash** (SWA incompatibility, NOT vocab) |
| Qwen2.5-Coder-0.5B | Qwen2.5-Coder-32B | 128 fewer | **Works (11x speedup)** |

**Important**: The Gemma-3 crash is due to **Sliding Window Attention (SWA)** being incompatible with speculative decoding in llama.cpp, NOT vocab mismatch. The crash happens in `llama_kv_cache::slot_info` during KV cache initialization.

The script warns about vocab mismatch but can't detect SWA incompatibility. Always test before adding to registry.

### Workflow

1. **Run the check** before adding to registry
2. **If warnings**, run a test generation: `llama-speculative -m TARGET -md DRAFT -p "test" -n 50`
3. **If SIGSEGV or garbage**, do NOT add the pairing - document in `runtime_quirks`
4. **If works**, add to registry with `benchmark_date` and test results

### Example Output

```
Draft Model: gemma-3-1b-it-Q8_0.gguf
  vocab_size: 262,144
  bos_token_id: 2
  tokenizer_model: 108

Target Model: gemma-3-27B-it-QAT-Q4_0.gguf
  vocab_size: 262,208
  bos_token_id: 2
  tokenizer_model: 108

============================================================
RESULT: COMPATIBLE with warnings:
  VOCAB MISMATCH: draft=262,144, target=262,208 (64 fewer tokens in draft) - TESTING REQUIRED!
```

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

## Claude-as-Judge Quality Review

### Overview

Claude-as-Judge is our framework for independent quality evaluation of model benchmark answers. The algorithmic rubric was found to severely underscore models (38% vs 89% for the same model) due to pattern matching failures.

**Use this framework to:**
- Score new model benchmark results
- Compare quality across models
- Identify models with empty output issues
- Make role assignment decisions

### Scoring Rubric

| Score | Meaning | Examples |
|-------|---------|----------|
| 3 | Correct answer with good reasoning | Complete solution, accurate math, valid logic |
| 2 | Partially correct or correct but truncated | Right approach but incomplete, minor errors |
| 1 | Wrong answer but reasonable attempt | Plausible but incorrect, misunderstood question |
| 0 | Completely wrong, empty, or no answer | Garbage output, empty response, unrelated text |

### File Locations

```
benchmarks/results/reviews/
├── {model_name}_baseline.csv      # Per-model review
├── {model_name}_{config}.csv      # Per-config review (if applicable)
└── summary.csv                    # Comparative summary
```

### Per-Model Review CSV Format

```csv
suite,question_id,tokens_per_second,claude_score,score_reason
thinking,t1_q1_logic,21.0,3,Correctly identified syllogism fallacy
thinking,t1_q2_sequence,20.8,3,Answer 42 is correct
general,t1_q1_reformat,18.5,2,Reformatted but truncated
agentic,t1_q1_single_tool,19.2,3,Tool call structure present
```

### Summary CSV Format

```csv
model,thinking,general,math,agentic,coder,instruction_precision,total,pct_str,avg_tps
thinking_deepseek_r1_distill_llama_8b,28/30,24/30,30/30,30/30,-,-,112/120,93%,7.2
```

### How to Review a New Model

1. **Locate benchmark results:**
   ```bash
   ls benchmarks/results/runs/*/  # Find the run directory
   # Look for {model_name}_baseline.json or similar
   ```

2. **Read the benchmark output:**
   - Each JSON file contains questions and model answers
   - Note the `tokens_per_second` from each answer

3. **Score each answer (0-3):**
   - Read the question and expected answer format
   - Evaluate the model's response
   - Assign score based on rubric above
   - Note brief reason

4. **Create review CSV:**
   ```bash
   # Create file at: benchmarks/results/reviews/{model_name}_baseline.csv
   ```

5. **Update summary.csv:**
   - Calculate totals per suite (e.g., "28/30")
   - Calculate overall percentage
   - Calculate average tokens/second
   - Add row to summary.csv (sorted by percentage descending)

### Batch Scoring Heuristics

For efficiency, use these heuristics for common patterns:

| Pattern | Score | Reason |
|---------|-------|--------|
| Empty or `<think>` only | 0 | Empty or minimal output |
| Tool call JSON present | 3 | Tool call structure present |
| JSON structure valid | 3 | JSON structure present |
| Reformatting response | 2 | Reformatting response |
| General text response | 2 | General response generated |

### Current Coverage (as of 2026-01-07)

- **61 baseline models reviewed** (381 total configs including MoE/spec decode variants)
- **Top performers:** See RESULTS_SUMMARY.md for current rankings
- **Score inheritance:** Speculative decoding configs inherit quality scores from their baseline (same model, different speed)

### When to Run Claude-as-Judge

- After any new model completes benchmark suite
- When algorithmic scores seem suspiciously low
- Before making role assignment decisions
- When comparing models for a specific role

---

## Benchmark Hardening (2025-12-18)

### Overview

Benchmark questions were hardened to address ceiling effects. Top models were scoring 89-93%, indicating questions were too easy for expert-level differentiation.

**Changes made:**
- Removed 3 trivial T1 questions from each of 8 suites
- Shifted T2 → T1, T3 → T2 (relabeling)
- Added 3 post-doctoral level T3 questions to each suite

### Reference Model for Score Conversion

Models benchmarked before 2025-12-18 were tested on easier questions. To compare old vs new scores:

| Reference Model | Old Score | New Score | Conversion Factor |
|-----------------|-----------|-----------|-------------------|
| DeepSeek-R1-Distill-Llama-8B | 112/120 (93%) | TBD | TBD |

**After testing reference model on new questions:**
```
conversion_factor = new_score / old_score
converted_score = old_claude_score × conversion_factor
```

### New T3 Question Difficulty

New T3 questions require expert-level reasoning:

| Suite | Example Question | Why It's Hard |
|-------|------------------|---------------|
| thinking | Causal inference DAG (collider bias) | Requires formal causal reasoning |
| thinking | Gödel/Penrose philosophy | Cross-domain philosophy of mind |
| math | f(x) = Σ(x^n/n!)sin(n) analysis | Closed-form via complex exponentials |
| math | E[N] where S_n > 1 (uniform sum) | Answer is e, requires two proof methods |
| coder | Lock-free stack ABA problem | Concurrent programming edge case |
| coder | Distributed consistency strategies | CAP theorem trade-offs |
| agentic | Multi-agent coordination | Time-budgeted agent orchestration |
| agentic | Adversarial input handling | Security-aware tool use |
| vl | Scientific figure analysis | Statistical critique of graphs |
| long_context | Multi-hop temporal reasoning | 4+ document chain reasoning |
| instruction_precision | Self-referential constraints | Meta-accurate self-description |

### Expected Score Distribution (Post-Hardening)

| Model Class | Expected Score |
|-------------|----------------|
| 0.5B-1.5B draft models | 30-50% |
| 4B-8B general models | 50-70% |
| 8B+ specialized thinking models | 60-80% |
| 14B+ large models | 70-85% |

Top models should no longer hit 90%+ ceiling.

---

## Research Summary Maintenance

### When Adding New Benchmark Results

1. **Add to summary.csv** with Claude-as-Judge scores
2. **Add to research_report.md** in the Complete Claude Score Table (line ~440)
3. **If optimization tested:** Add notes about baseline vs optimized speeds
4. **Update other relevant tables** (MoE, spec decode, etc.) if applicable

### Speed Reporting Rules

| Column | Meaning |
|--------|---------|
| `Baseline t/s` | Raw speed during benchmark (no optimization) |
| `Optimized t/s` | Speed with best optimization for that model |

### Optimization Configuration Tracking

When MoE reduction or other quality-affecting optimizations are tested:
- Document baseline AND optimized configurations separately
- Note quality impact (e.g., "2 experts = garbage" vs "4 experts = quality preserved")
- If optimization degrades quality, mark it explicitly

Example entry in research_report.md optimization tables:
```
| Model | Baseline | 4 Experts | 2 Experts | Quality Notes |
| Qwen3-235B | 3.6 t/s | 6.75 t/s | 3.80 t/s | 4 experts OK, 2 experts garbage |
```

### Keeping Tables in Sync

The research_report.md has multiple tables that may need updates:
- **Complete Claude Score Table** - ALL models with scores
- **Top Performers** - Summary of role recommendations
- **MoE Optimization Results** - Expert reduction benchmarks
- **Speculative Decoding Results** - Draft model combinations
- **Per-model Performance** - Baseline speeds

When benchmarking a new model, check if it belongs in any of these tables.

---

## Key Resources

### Documentation
| Document | Location |
|----------|----------|
| Results Summary | `logs/research_report.md` |
| Model Registry | `orchestration/model_registry.yaml` |
| TaskIR Schema | `orchestration/task_ir.schema.json` |
| Agent Definitions | `agents/` |
| Benchmark Prompts | `benchmarks/prompts/v1/` |
| Benchmark Results | `benchmarks/results/` |
| Benchmark Index | `benchmarks/results/index.jsonl` |
| Claude-as-Judge Reviews | `benchmarks/results/reviews/` |
| Blocked Tasks Checklist | `orchestration/BLOCKED_TASKS.md` |
| Research Handoffs | `research/*_handoff.md` |

### Commands
| Action | Command |
|--------|---------|
| Run gates | `make gates` |
| Validate TaskIR | `python3 orchestration/validate_ir.py task FILE.json` |
| Analyze logs | `scripts/utils/agent_log_analyze.sh --summary` |
| Discover models | `scripts/session/session_init.sh` |
| Run all benchmarks | `scripts/benchmark/run_overnight_benchmark_suite.sh --suite all` |
| Compare benchmark runs | `scripts/benchmark/compare_results.sh --baseline ID --current ID` |
| List benchmark runs | `scripts/benchmark/compare_results.sh --list-runs` |

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

### When Creating/Updating Handoff Documents

Handoff documents (`research/*_handoff.md`) track work for future sessions or agents.

**ALWAYS update `orchestration/BLOCKED_TASKS.md`** when:
- Creating a new handoff document
- Completing a task from the blocked list
- Changing the blocking dependency

The blocked tasks file should reflect:
- Current status (blocked/ready/complete)
- Resume commands (copy-paste ready)
- Completion checklist items
