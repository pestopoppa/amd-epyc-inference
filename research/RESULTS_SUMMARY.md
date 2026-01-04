# Research Results Summary

**Last Updated:** 2026-01-04 (Research consolidation: RESEARCH_PROGRESS.md created, all literature refs preserved)
**System:** AMD EPYC 9655 (96 cores, 1.13TB DDR5), llama.cpp

---

## Best Results

| Configuration | Speed | Speedup | Use Case |
|---------------|-------|---------|----------|
| Prompt Lookup (summarization) | 95.18 t/s | **12.7x** | Document QA, summarization |
| **Qwen3-Coder-30B-A3B + MoE 4** | **45.3 t/s** | **+31%** | **Code generation (coder_primary)** |
| Qwen3-Coder-53B-A3B + MoE 4 | 30.4 t/s | +49% | Code escalation (coder_escalation) |
| Qwen2.5-Coder-32B + 0.5B (K=24) | 33.0 t/s | **11x** | Code generation (DEPRECATED) |
| Prompt Lookup (code editing) | 25.82 t/s | **8.6x** | Refactoring, code review |
| Qwen2.5-72B + 0.5B (K=16) | 8.53 t/s | **5.8x** | General tasks |
| MoE Expert Reduction (4 experts) | +21-48% | — | MoE models |

---

## 🆕 Parallel Tensor Repacking (2025-12-21)

**Problem:** Model loading on 96-core EPYC was bottlenecked by single-threaded tensor repacking for AVX-512 optimization.

**Solution:** OpenMP parallelization of repack functions in llama.cpp.

| Model Size | Before | After | Speedup |
|------------|--------|-------|---------|
| 6.8GB Q4_K | 5.0s | 3.3s | **1.5x** |
| 19GB Q4_K | 11.9s | 5.3s | **2.2x** |
| 271GB Q4_K | ~150s | ~60s | **~2.5x** |

**Additional finding:** Removing `OMP_NUM_THREADS=1` also improved prompt processing 2.4x (49 → 119 t/s).

**Status:**
- PR submitted: https://github.com/ggml-org/llama.cpp/pull/18239
- Patch applied locally: `patches/llama-cpp-parallel-repack.patch`
- Benchmark scripts updated to use parallel repack

**Affected quant types:** Q4_0, Q4_K, Q2_K, IQ4_NL (Q6_K and Q8_0 don't use repacking)

---

## 🆕 Coder Model Selection (2025-12-16)

**Quality Evaluation Task:** Binary search with docstring and empty array handling.

| Model | Baseline | Optimized | Method | Quality |
|-------|----------|-----------|--------|---------|
| **Qwen3-Coder-30B-A3B-Instruct** | 34.6 t/s | **45.3 t/s** | MoE 4 experts | ⭐⭐⭐⭐⭐ |
| Qwen3-Coder-53B-A3B-TOTAL-RECALL | 20.4 t/s | 30.4 t/s | MoE 4 experts | ⭐⭐⭐⭐⭐ |
| Qwen2.5-Coder-32B-Instruct | 7.0 t/s | 9.7 t/s | Spec decode (16% accept) | ⭐⭐⭐⭐⭐ |

**Finding:** All three models produce equivalent quality code (docstrings, edge cases, correct algorithm). Speed is the only differentiator.

**Decision:**
- `coder_primary` = Qwen3-Coder-30B-A3B-Instruct (45.3 t/s) - 4.7x faster than dense
- `coder_escalation` = Qwen3-Coder-53B-A3B-TOTAL-RECALL (30.4 t/s) - generalist support
- Qwen2.5-Coder-32B-Instruct = DEPRECATED (16% spec accept too low)

**Coding Escalation Hierarchy:**
```
coder_primary (45 t/s) → coder_escalation (30 t/s) → architect_coding (5 t/s)
```

---

## Very Large Models (100B+)

### Baseline Performance
| Model | Size | Quant | Active Params | Baseline | Notes |
|-------|------|-------|---------------|----------|-------|
| Qwen3-235B-A22B | 133GB | Q4_K_M | ~22B | **3.6 t/s** | MoE, fits in RAM |
| Qwen3-VL-235B-A22B-Thinking | 124GB | Q4_K_S | ~22B | **3.23 t/s** | VL+MoE, thinking variant |
| Qwen3-Coder-480B-A35B | 271GB | Q4_K_M | ~35B | **6.53 t/s** | MoE, largest tested (updated 2025-12-21) |
| GLM-4.6-355B-A32B | 189GB | Q4_K_S | ~32B | **2.24 t/s** | MoE (glm4moe) |
| Qwen3-Next-80B-A3B | 45GB | Q4_K_M | ~3B | **8.43-10.12 t/s** | SSM+MoE hybrid |

### Optimization Results
| Model | Baseline | +Expert Reduction | +Lookup | Best |
|-------|----------|-------------------|---------|------|
| **Qwen3-Coder-480B** | 6.53 t/s | 10.30 t/s (+58% MOE3) | Garbage | **+58%** |
| **Qwen3-VL-235B-Thinking** | 3.23 t/s | 7.12 t/s (+120%) | 3.82 t/s | **+120%** |
| **Qwen3-235B** | 3.6 t/s | 6.75 t/s (+87%) | 6.35 t/s | **+87%** |
| **GLM-4.6-355B** | 2.24 t/s | 3.97 t/s (+77%) | 3.37-3.65 t/s | **+77%** |

**Note (2025-12-21):** Qwen3-Coder-480B results updated after parallel repack fix:
- Architecture: 160 total experts, **8 experts used by default** (35B active params)
- Old baseline (with OMP_NUM_THREADS=1): 2.25 t/s
- New baseline (with parallel repack): 6.53 t/s (+190%)
- **MOE3 (3 experts): 10.30 t/s (+58%) - OPTIMAL** ✓
- MOE4 (4 experts): 9.25 t/s (+42%) - good fallback
- MOE5 (5 experts): 8.50 t/s (+30%)
- MOE2 (2 experts): 11.51 t/s - **GARBAGE output, unusable** ✗

### MoE + Lookup Combination (Detailed)

**Key Finding: SSM models (like Qwen3-Next) are incompatible with speculation-based methods.**

| Model | Hard Mask Alone | Lookup + Hard Mask | Combination Benefit |
|-------|-----------------|--------------------|--------------------|
| **Qwen3-Next-80B-A3B** | 11.55 t/s | ❌ FAILS | SSM incompatible |
| Qwen3-Coder-30B-A3B | 41.55 t/s | 29.92 t/s | 0.72x ❌ |
| Qwen3-VL-30B-A3B | 36.84 t/s | 29.88 t/s | 0.81x ❌ |
| Qwen3-235B-A22B | 6.75 t/s | 6.35 t/s | 0.94x ❌ |

**When to combine vs use standalone:**

| Model Type | Best Approach | Reasoning |
|------------|---------------|-----------|
| **SSM/Hybrid (Qwen3-Next)** | Expert reduction only | Speculation incompatible |
| **30B MoE** | Hard Mask only | Already fast; lookup adds overhead |
| **235B+ MoE** | Hard Mask only | Large active params limit lookup benefit |

**Commands:**
```bash
# SSM models (Qwen3-Next): Expert reduction only
llama-cli -m Qwen3-Next-80B-A3B.gguf --override-kv qwen3next.expert_used_count=int:4 -t 96

# 30B MoE: Expert reduction only (fastest)
llama-cli -m Qwen3-Coder-30B-A3B.gguf --moe-n-expert 4 -t 96
```

### Qwen3-Next-80B (SSM+MoE Hybrid)

**Architecture:** SSM + MoE hybrid with 512 experts, 10 active by default (~3B active params)

| Configuration | Speed | vs Baseline | Quality |
|---------------|-------|-------------|---------|
| Baseline (10 experts) | 10.12 t/s | — | ✅ |
| 4 experts | 11.49 t/s | +13.5% | ✅ Good |
| **2 experts** | **11.55 t/s** | **+14%** | ✅ Good |
| Speculative decoding | ❌ FAILS | — | SSM incompatible |
| Prompt lookup | ❌ FAILS | — | SSM incompatible |

**Absolute performance limit: ~11.6 t/s** (2 experts)

**Key insight:** Unlike Qwen3-235B (which produces garbage at 2 experts), Qwen3-Next-80B maintains quality even at 2 experts. This is likely because:
- 512 experts with 2 active still provides reasonable routing options
- SSM component provides additional sequence modeling capacity

### Key Finding: Largest Models Benefit Most
- 480B model: **+48-80% speedup** from expert reduction + lookup
- Expert reduction more effective than speculative decoding on MoE
- All 100B+ models run entirely in RAM (no GPU needed)
- **SSM models:** Expert reduction only - speculation/lookup incompatible

---

## Dense Models (32B-72B)

### Baselines
| Model | Size | Quant | Baseline | Notes |
|-------|------|-------|----------|-------|
| DeepSeek-R1-32B | 18.5GB | Q4_K_M | **6.01 t/s** | Fastest 32B |
| Qwen2.5-Coder-32B | 18.5GB | Q4_K_M | **5.79 t/s** | Code specialist |
| Gemma-3-27B-QAT | 14.5GB | Q4_0 | **4.72 t/s** | QAT quantized |
| Qwen3-32B | 18.4GB | Q4_K_M | **3.67 t/s** | Slower than R1 |
| Meta-Llama-3.1-70B | 40GB | Q4_K_M | **1.96 t/s** | Dense 70B |
| Hermes-4-70B | 40GB | Q4_K_M | **1.73 t/s** | Llama-based |
| DeepSeek-R1-Llama-70B | 40GB | Q4_K_M | **1.73 t/s** | R1 distilled |
| Meta-Llama-3-70B | 40GB | Q4_K_M | **1.72 t/s** | Original Llama 3 |
| Qwen2.5-72B-Instruct | 41GB | Q4_K_M | **1.70 t/s** | Qwen 72B |
| Qwen2.5-Math-72B | 41GB | Q4_K_M | **1.41 t/s** | Math specialist |
| Qwen2.5-72B | 41GB | Q4_K_M | **0.85 t/s** | Base (slow) |

### Speculative Decoding Results (Dense)
| Model + Draft | Speed | Speedup | Accept | K |
|---------------|-------|---------|--------|---|
| **Qwen2.5-Coder-32B + 0.5B** | **33.0 t/s** | **11x** | 70.8% | K=24 |
| Qwen2.5-Coder-32B + 0.5B | 27.9 t/s | 9.3x | 75% | K=16 |
| Qwen2.5-Coder-32B + 0.5B | 25.3 t/s | 8.5x | 100% | K=8 |
| **Qwen2.5-72B-Instruct + 0.5B** | **8.53 t/s** | **5.8x** | 44.3% | K=16 |
| Qwen2.5-Math-72B + 0.5B (t=0.5) | **7.55 t/s** | **7.3x** | 60.3% | K=12 |
| Qwen2.5-Math-72B + 0.5B | 6.83 t/s | 5.9x | 42% | K=16 |
| Meta-Llama-70B + PARD-1B | 6.42 t/s | 3.7x | 79.2% | K=8 |
| Qwen3-32B + Qwen3-0.6B | 5.87 t/s | 3.1x | 39.1% | K=8 |

### Prompt Lookup Results (Dense)
| Model | Summarize | Code | Edit |
|-------|-----------|------|------|
| Qwen2.5-Coder-32B | 6.50 t/s | 4.78 t/s | 4.94 t/s |
| Qwen3-32B | 5.09 t/s | 4.51 t/s | 3.99 t/s |
| DeepSeek-R1-32B | 4.78 t/s | 4.74 t/s | 4.17 t/s |
| Gemma-3-27B | 8.03 t/s | 6.52 t/s | 6.42 t/s |
| Meta-Llama-3.1-70B | 3.15 t/s | 1.67 t/s | 1.76 t/s |
| Hermes-4-70B | 3.72 t/s | 2.54 t/s | 2.76 t/s |
| DeepSeek-R1-Llama-70B | 3.02 t/s | 2.38 t/s | 2.23 t/s |
| Qwen2.5-72B-Instruct | 3.46 t/s | 1.97 t/s | 2.02 t/s |
| Qwen2.5-Math-72B | 2.04 t/s | 0.88 t/s | 0.85 t/s |

---

## MoE Models (30B-A3B Class)

### Baselines (Fastest MoE)
| Model | Quant | Active Params | Baseline | Notes |
|-------|-------|---------------|----------|-------|
| Qwen3-Coder-30B-A3B | Q4_K_M | ~3B | **27.14 t/s** | Code specialist |
| Qwen3-VL-30B-A3B | Q4_K_M | ~3B | **26.88 t/s** | Vision-Language |
| **Qwen3-Coder-53B-A3B** | Q4_K_M | ~3B | **18.54 t/s** | TOTAL-RECALL-v2 finetune (30GB) |
| Qwen3-1.7B (draft) | Q4_K_M | 1.7B | **51.31 t/s** | Draft model |
| Qwen3-VL-2B (draft) | Q4_K_M | 2B | **42.19 t/s** | VL draft |

### Expert Reduction (Hard Mask)
| Model | Baseline | 4 experts | 3 experts | 6 experts |
|-------|----------|-----------|-----------|-----------|
| Qwen3-Coder-30B-A3B | 27.14 t/s | **41.55 t/s** | — | 30.05 t/s |
| Qwen3-VL-30B-A3B | 26.88 t/s | **36.84 t/s** | 37.66 t/s | 28.41 t/s |
| **Qwen3-Coder-53B-A3B** | 18.54 t/s | **27.9 t/s (+50%)** | — | — |

### Prompt Lookup (MoE)
| Model | Summarize | Code |
|-------|-----------|------|
| Qwen3-Coder-30B-A3B | 43.21 t/s | 40.85 t/s |
| Qwen3-VL-30B-A3B | 46.34 t/s | 43.29 t/s |

---

## Small Models (7B-14B)

### Baselines
| Model | Size | Quant | Baseline | Notes |
|-------|------|-------|----------|-------|
| Meta-Llama-3-8B | 4.7GB | Q4_K_M | **17.52 t/s** | Fastest 8B |
| Qwen2.5-VL-7B | 4.4GB | Q4_K_M | **15.28 t/s** | VL model |
| DeepSeek-R1-Llama-8B | 4.6GB | Q4_K_M | **13.42 t/s** | R1 distilled |
| DeepSeek-R1-Qwen-7B | 4.4GB | Q4_K_M | **13.15 t/s** | R1 distilled |
| Qwen2.5-Math-7B | 4.4GB | Q4_K_M | **12.44 t/s** | Math specialist |
| Gemma-3-12B | 6.8GB | Q4_K_M | **10.42 t/s** | Medium |
| DeepSeek-R1-Qwen-14B | 8.4GB | Q4_K_M | **6.44 t/s** | Larger R1 |

### Speculative Decoding (7B)
| Model + Draft | Speed | Speedup | Accept | Notes |
|---------------|-------|---------|--------|-------|
| **Qwen2.5-VL-7B + 0.5B (t=0.7)** | **57.1 t/s** | **3.7x** | 74.2% | Temp tuned! |
| **Qwen2.5-Math-7B + 0.5B** | **48.5 t/s** | **3.9x** | 65.6% | K=8 optimal |
| Qwen2.5-VL-7B + 0.5B (t=0) | 28.3 t/s | 1.9x | — | Baseline temp |

### Prompt Lookup (Small)
| Model | Summarize | Code |
|-------|-----------|------|
| Meta-Llama-3-8B | 37.07 t/s | 36.64 t/s |
| Qwen2.5-Math-7B | 38.74 t/s | 27.44 t/s |
| DeepSeek-R1-Qwen-7B | 20.71 t/s | 19.39 t/s |
| DeepSeek-R1-Llama-8B | 13.50 t/s | 19.10 t/s |
| DeepSeek-R1-Qwen-14B | 20.19 t/s | 7.65 t/s |
| Gemma-3-12B | 9.31 t/s | 8.59 t/s |

---

## Key Insights

### 1. Small Drafts Win on CPU
- 0.5B draft at 85 t/s vs 7B draft at 8 t/s
- More speculation rounds beat higher acceptance rates
- **Rule:** Use smallest compatible draft model

### 2. MoE Models Don't Need Speculative Decoding
- Qwen3-VL-30B-A3B baseline: 24.82 t/s
- With speculation: 20.99 t/s (0.84x slower)
- **Why:** 3B active params already "draft speed"

### 3. K-Value Tuning
| Model Size | Optimal K | Reason |
|------------|-----------|--------|
| 7B | K=8 | High baseline, diminishing returns |
| 32B | K=16-24 | Verification cost amortized |
| 72B | K=16 | Balance acceptance vs overhead |

### 4. Temperature Tuning
- Non-zero temperature can improve speculative decoding
- Qwen2.5-VL-7B: temp=0.7 → 57.1 t/s vs temp=0 → 28.3 t/s
- **Rule:** Try temp=0.5-0.7 if acceptance rate is low

---

## Track Status

| Track | Method | Status | Result |
|-------|--------|--------|--------|
| 1 | External Draft | **Production** | 5.9-11x |
| 2 | MoE Expert Reduction | **Production** | +21-48% |
| 8 | Prompt Lookup | **Production** | 8.6-12.7x |
| A | System (Hugepages/NUMA) | **Tested - Already Optimal** | interleave=all best |
| 6 | SuffixDecoding | **= Track 8** | Same as Prompt Lookup |
| C | Draft Quantization | **Tested - No Benefit** | Q8_0 optimal |
| 3 | EAGLE-1 | Deprecated | 0% acceptance |
| 7 | CAS-Spec | Blocked | 0.446% acceptance |

## New Draft Models Available

| Model | Quantization | Size | Path |
|-------|--------------|------|------|
| Qwen2-0.5B | Q2_K | 323MB | `QuantFactory/Qwen2-0.5B-GGUF/Qwen2-0.5B.Q2_K.gguf` |
| Qwen2.5-Coder-1.5B | Q2_K | 645MB | `QuantFactory/Qwen2.5-Coder-1.5B-GGUF/Qwen2.5-Coder-1.5B.Q2_K.gguf` |
| Qwen3-0.6B | Q2_K | 283MB | `unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q2_K.gguf` |
| Qwen3-Embedding-0.6B | Q8_0 | — | `Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf` |

**Benchmark Results (2025-12-15):**

| Model | Q2_K Speed | vs Q8_0 |
|-------|------------|---------|
| Qwen3-0.6B | **221 t/s** | 3.4x faster |
| Qwen2-0.5B | **208 t/s** | 2.4x faster |
| Qwen2.5-Coder-1.5B | **98 t/s** | (no Q8_0 baseline) |

**Speculative Decoding Results:**

| Draft Model | Accept | Spec Speed | Verdict |
|-------------|--------|------------|---------|
| Qwen2.5-Coder-0.5B Q8_0 | 58% | **22.5 t/s** | Best (smaller = faster) |
| Qwen2.5-Coder-1.5B Q4_K_M | 58% | 12.5 t/s | Works but slower than 0.5B |
| Qwen2.5-Coder-1.5B Q2_K | 57% | 13.1 t/s | Slower despite faster raw speed |
| Qwen2-0.5B Q2_K | FAIL | — | Wrong vocab family |
| Qwen3-0.6B Q2_K | N/A | — | Wrong model family |

**Conclusion:** Q2_K raw speed gains don't translate to speculative decoding — smaller models still win on CPU.

---

## Quick Commands

```bash
# Track 1: External Draft (5.9-11x)
OMP_NUM_THREADS=1 numactl --interleave=all \
  llama-speculative -m TARGET.gguf -md DRAFT.gguf \
  --draft-max 16 -t 96

# Track 2: MoE Expert Reduction (+21-48%)
llama-cli -m MOE.gguf \
  --override-kv ARCH.expert_used_count=int:4 -t 96

# Track 8: Prompt Lookup (8.6-12.7x)
# Use --lookup-ngram-min 3 with prompt containing repeated patterns
```

---

## Failed Approaches (Lessons)

### EAGLE-1 (0% Acceptance)
- Problem: Architecture/checkpoint incompatibility
- Lesson: "Zero-shot" EAGLE requires exact model-checkpoint matching

### CAS-Spec Layer Skip (0.446% Acceptance)
- Problem: Knowledge not evenly distributed across layers
- Lesson: Self-drafting without training produces garbage

### MoE + Speculative Decoding
- Problem: Slower than baseline (0.26-0.84x)
- Lesson: Don't add speculation overhead to already-fast MoE

### SSM/Hybrid Models + Speculation
- Problem: "inconsistent sequence positions" error
- Models affected: Qwen3-Next (SSM+MoE hybrid)
- Lesson: SSM requires consecutive positions - incompatible with ALL speculation methods (speculative decoding, prompt lookup, EAGLE, etc.)

### Qwen3-Coder-480B + Speculative Decoding
- Problem: "draft model special tokens must match target model" error
- BOS token mismatch: Qwen3-Coder-480B has BOS=',' (token 11) vs standard BOS='<|endoftext|>' (token 151643)
- Tested drafts: Qwen3-0.6B, Qwen2.5-Coder-0.5B - both fail
- Lesson: Verify tokenizer compatibility before attempting speculation; unusual BOS tokens block all compatible draft models
- **Workaround:** Use 4-expert reduction (9.25 t/s, +42% vs baseline)
- **Warning:** 2-expert reduction produces GARBAGE output despite higher speed (11.51 t/s)

### Qwen3-Coder-53B-A3B + Speculative Decoding
- Problem: Token mismatch with Qwen2.5 drafts, low acceptance (8.96%) with Qwen3-0.6B
- Tested drafts: Qwen2.5-Coder-0.5B (fails - token mismatch), Qwen3-0.6B (works but 8.96% accept)
- Lesson: MoE models with different distributions don't benefit from small dense drafts
- **Workaround:** Use expert reduction instead (+50% with 4 experts)

---

## Benchmark Framework (2025-12-16)

### 8 Quality Benchmark Suites

| Suite | Purpose | Auto-Scoring |
|-------|---------|--------------|
| **Thinking** | Chain-of-thought, multi-step reasoning | Manual |
| **Coder** | Code generation, debugging, refactoring | Manual |
| **VL** | Vision-language (OCR, image understanding) | Manual |
| **General** | Instruction following, summarization | Manual |
| **Agentic** | Tool calling, function extraction | Partial |
| **Math** | Mathematical reasoning, step verification | Partial |
| **Long Context** | Information retrieval (4K-50K tokens) | Auto |
| **Instruction Precision** | Exact format compliance | **Full Auto** |

### Permanent Storage

```
benchmarks/
├── prompts/v1/          # Versioned YAML prompt files
│   ├── thinking.yaml
│   ├── coder.yaml
│   ├── vl.yaml
│   ├── general.yaml
│   ├── agentic.yaml
│   ├── math.yaml
│   ├── long_context.yaml
│   └── instruction_precision.yaml
├── results/
│   ├── runs/            # Raw outputs per run with metadata
│   └── index.jsonl      # Structured index for querying
└── baselines/           # Reference checkpoints
```

### Commands

```bash
# Run all 8 suites
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite all

# Run specific suite
./scripts/benchmark/run_overnight_benchmark_suite.sh --suite instruction_precision

# Compare runs
./scripts/benchmark/compare_results.sh --baseline ID --current ID

# List all runs
./scripts/benchmark/compare_results.sh --list-runs
```

### Why Instruction Precision Matters for Orchestration

Models that fail instruction precision tests will break orchestration:
- "Output only JSON" → model adds "Here's the JSON:" → **parsing failure**
- "Exactly 3 items" → model gives 4 → **schema validation failure**
- "Do not mention X" → model mentions X → **context pollution**

**Orchestration readiness thresholds:**
- Workers: T1 100%, T2 75%+
- Orchestrators: T1 100%, T2 100%, T3 75%+

---

## Claude-as-Judge Quality Review (2025-12-18)

### Overview

Independent quality evaluation of 33 models using Claude as judge. Algorithmic rubric found severely underscoring due to pattern matching failures.

**Key Finding:** Algorithmic rubric scored ~38% average; Claude-as-Judge scored ~70% for same models.

### Role Recommendations (Top Performers)

| Role | Model | Score | Baseline t/s |
|------|-------|-------|--------------|
| **General Best** | Gemma-3-12B-IT | **91.7%** | 9.2 |
| **Thinking Primary** | DeepSeek-R1-Distill-Llama-8B | 93%* | 7.2 |
| **Thinking Fast** | Qwen3-4B-Thinking-2507-Q8_0 | **69%** | 18.0 |
| **General Worker** | Qwen2.5-7B-Q4_K_S | **69.4%** | 16.0 |
| **Math Worker** | Qwen2.5-Math-7B-Instruct | **60.0%** | 11.3 |
| **Vision Primary** | Qwen2.5-VL-7B-Instruct | **69.4%** | 11.8 |
| **Draft (Dense)** | Qwen2.5-Coder-0.5B | **63%** | 182.8 |
| **Draft (Thinking)** | DeepSeek-R1-Distill-Qwen-1.5B | **65%** | 29.5 |

⚠️ **Qwen3-VL Warning:** All Qwen3-VL models (2B/4B/8B) score 0% on agentic tasks - all tool-call prompts return empty. Use Qwen2.5-VL for vision tasks requiring tool coordination.

⚠️ **DeepSeek-R1-0528-Qwen3-8B Warning:** 100% thinking accuracy but only 24% instruction precision. Model echoes prompts and outputs verbose reasoning instead of clean structured output. Unsuitable for orchestration tasks requiring exact format compliance.

⚠️ **MathSmith-Hard-Problem-Synthesizer-Qwen3-8B Warning:** Runs at 3.5 t/s instead of expected 15+ t/s for an 8B model (5x slower). Uses only 6% of memory bandwidth - severely compute-bound. Likely mradermacher GGUF conversion issue or hidden architecture differences. Use Qwen2.5-Math-7B-Instruct (11.3 t/s) instead.

`*` = scored on old (easier) questions before hardening
`†` = incomplete benchmark (partial test run)
Scores without markers = tested on complete 60-120 question suites (2025-12-19)

### Complete Model Database with Claude Scores

All models from registry (~70 unique models). Empty cells = not yet benchmarked. `Baseline t/s` = raw benchmark speed.

**Note:** Scores marked with `*` were evaluated on the OLD (easier) benchmark questions before hardening. Scores without `*` are on the new post-doctoral level T3 questions.

#### Tier A-B: Production & Specialist Models (MoE/Hybrid Architecture)

| Model | Role | Thinking | General | Math | Agentic | Coder | Inst.Prec | VL | Pct | Baseline t/s | Optimized t/s (config) |
|-------|------|----------|---------|------|---------|-------|-----------|-----|-----|--------------|------------------------|
| **Qwen3-Coder-480B-A35B** | architect_coding | 28/30 | - | - | 28/30 | 27/27 | - | - | **95%**† | 6.53 | (8 experts default) |
| Qwen3-Coder-480B-A35B (MoE 2) | architect_coding | 3/30 | - | - | 1/30 | 10/30 | - | - | **14%** | 7.6 | GARBAGE output ❌ |
| **Qwen3-Coder-480B-A35B (MoE 3)** | architect_coding | - | - | - | - | - | - | - | - | 10.30 | **+58% OPTIMAL ✓** |
| Qwen3-Coder-480B-A35B (MoE 4) | architect_coding | 27/30 | - | - | 30/30 | 30/30 | - | - | **88%** | 7.1 | +9% |
| Qwen3-Coder-480B-A35B (MoE 5) | architect_coding | - | - | - | - | - | - | - | - | 8.50 | +30% |
| Qwen3-Coder-480B-A35B (MoE 6) | architect_coding | - | 29/30 | - | 30/30 | - | - | - | **95%**‡ | 6.2 | -5% |
| **GLM-4.6-355B-A32B** | general | 24/30 | - | - | 12/30 | 5/6 | 18/33 | - | **59%** | 3.4 | Lower quality than Qwen |
| GLM-4.6-355B-A32B (MoE 2) | general | - | - | - | - | - | - | - | - | - | - |
| GLM-4.6-355B-A32B (MoE 4) | general | - | - | - | - | - | - | - | - | 3.97 | 3.5 (+lookup) ❌ |
| GLM-4.6-355B-A32B | ingest | 29/30 | - | - | 13/30 | - | 19/33 | - | **60%** | 3.7 | ~60% across roles |
| **Qwen3-VL-235B-A22B-Thinking** | vision | - | 21/30 | - | 9/30 | - | - | 27/30 | **63%** | 5.8 | - |
| **Qwen3-VL-235B-A22B-Thinking (MoE 2)** | vision | - | 27/30 | - | 17/30 | - | - | - | **73%**† | 7.6 | **+31% ✓** |
| Qwen3-VL-235B-A22B-Thinking (MoE 4) | vision | - | 30/30 | - | 15/30 | - | - | - | **75%**† | 6.8 | +17% |
| Qwen3-VL-235B-A22B-Thinking (MoE 6) | vision | - | 23/30 | - | 12/30 | - | - | - | **58%**† | 6.4 | +10% |
| **Qwen3-235B-A22B** | architect_general | 28/30 | 23/30 | 29/30 | 28/30 | 30/30 | 25/33 | - | **88%** | 5.9 | - |
| **Qwen3-235B-A22B (MoE 2)** | architect_general | 28/30 | 28/30 | 26/30 | 25/30 | 27/30 | 28/33 | - | **88%** | 8.2 | **+39% OPTIMAL ✓** |
| Qwen3-235B-A22B (MoE 4) | architect_general | 28/30 | 25/30 | 28/30 | 27/30 | 25/30 | 27/33 | - | **87%** | 7.3 | +24% |
| Qwen3-235B-A22B (MoE 6) | architect_general | 23/30 | 24/30 | 27/30 | 26/30 | 28/30 | 28/33 | - | **86%** | 6.7 | +14% |
| **Qwen3-Next-80B-A3B** | ingest | 29/30 | - | - | 25/30 | - | 12/33 | - | **74%** | 9.7 | SSM (no spec) |
| Qwen3-Next-80B-A3B (MoE 2) | ingest | 28/30 | 21/30 | 30/30 | 22/30 | 23/30 | 24/33 | - | **80%** | 9.8 | +1% (SSM) |
| Qwen3-Next-80B-A3B (MoE 4) | ingest | 30/30 | 29/30 | 30/30 | 30/30 | 27/30 | 15/33 | - | **89%** | 9.9 | +2% (SSM) |
| Qwen3-Next-80B-A3B (MoE 6) | ingest | 30/30 | - | - | 30/30 | - | 16/33 | - | **84%**‡ | 9.8 | +1% (SSM) |
| **Qwen3-Next-80B-A3B-Thinking** | thinking | 30/30 | 30/30 | 30/30 | 27/30 | 21/30 | 28/33 | - | **92%** | 9.2 | SSM (no spec) |
| **Qwen3-Next-80B-A3B-Thinking (MoE 2)** | thinking | 30/30 | - | 30/30 | - | 27/30 | 33/33 | - | **98%**† | 10.3 | **+12% ✓** |
| Qwen3-Next-80B-A3B-Thinking (MoE 4) | thinking | 30/30 | 29/30 | 30/30 | 29/30 | 30/30 | 31/33 | - | **98%** | 9.8 | +7% (SSM) |
| Qwen3-Next-80B-A3B-Thinking (MoE 6) | thinking | 30/30 | 30/30 | 30/30 | 29/30 | 24/30 | 30/33 | - | **95%** | 9.7 | +5% (SSM) |
| **Qwen3-Coder-53B-A3B** | coder_escalation | - | - | - | - | - | - | - | - | 18.54 | - |
| Qwen3-Coder-53B-A3B (MoE 2) | coder_escalation | - | - | - | - | - | - | - | - | - | - |
| Qwen3-Coder-53B-A3B (MoE 4) | coder_escalation | - | - | - | - | - | - | - | - | 27.9 | - |
| **Qwen3-Coder-30B-A3B** | frontdoor/coder | - | - | - | - | - | - | - | - | 27.14 | - |
| Qwen3-Coder-30B-A3B (MoE 2) | frontdoor/coder | - | - | - | - | - | - | - | - | - | - |
| Qwen3-Coder-30B-A3B (MoE 4) | frontdoor/coder | - | - | - | - | - | - | - | - | 41.55 | 29.92 (+lookup) ❌ |
| Qwen3-Coder-30B-A3B (MoE 6) | frontdoor/coder | - | - | - | - | - | - | - | - | 30.05 | - |
| **Qwen3-30B-A3B-Thinking-2507 (Q8_0)** | thinking | 24/30 | 22/30 | 20/30 | 20/30 | 21/30 | 11/33 | - | **64%** | 17.4 | - |
| Qwen3-30B-A3B-Thinking-2507 (Q8_0, MoE 2) | thinking | 0/30 | 0/27 | 0/18 | 0/30 | 1/24 | 0/33 | - | **1%** ⚠️ | 24.2 | GARBAGE |
| Qwen3-30B-A3B-Thinking-2507 (Q8_0, MoE 4) | thinking | 21/30 | 16/27 | - | 11/24 | 12/30 | - | - | **54%** | 19.6 | - |
| **Qwen3-30B-A3B-Thinking-2507 (Q4_K_S)** | thinking | - | - | - | - | - | - | - | - | - | - |
| Qwen3-30B-A3B-Thinking-2507 (Q4_K_S, MoE 2) | thinking | - | - | - | - | - | - | - | - | - | - |
| Qwen3-30B-A3B-Thinking-2507 (Q4_K_S, MoE 4) | thinking | - | - | - | - | - | - | - | - | - | - |
| **Qwen3-VL-30B-A3B** | vision_escalation | - | - | - | - | - | - | - | - | 26.88 | - |
| Qwen3-VL-30B-A3B (MoE 3) | vision_escalation | - | - | - | - | - | - | - | - | 37.66 | - |
| Qwen3-VL-30B-A3B (MoE 4) | vision_escalation | - | - | - | - | - | - | - | - | 36.84 | 29.88 (+lookup) ❌ |
| Qwen3-VL-30B-A3B (MoE 6) | vision_escalation | - | - | - | - | - | - | - | - | 28.41 | - |

**Notes:**
- † Qwen3-Coder-480B score excludes long_context suite (4/18) due to timeout issues at 40K+ token contexts. Score of 83/87 = 95% on thinking+agentic+coder only.
- ‡ MOE6 partial run (22 questions: agentic+general+2 long_context). Full benchmark pending.
- ⚠️ Qwen3-Coder-480B has a tokenizer quirk: occasionally outputs Chinese characters (e.g., "6日消息1" instead of "60") in numerical contexts. Does not affect reasoning quality - correct answer usually follows.
- **MOE Quality Summary (2025-12-24):** MOE2=14% (garbage), MOE4=88% (good), MOE6=95% (partial). MOE3 untested but expected similar to MOE4/6.
- **MOE8 is redundant (2025-12-29):** Qwen3-Coder-480B uses 8 experts by default. MOE8 test confirmed: 96% at 4.4 t/s = baseline.
- **SSM Model Finding (2025-12-30):** SSM+MoE hybrids hit a **ceiling effect** where moe2/moe4 produce identical speeds (~10.2 t/s). The ~12% speedup from baseline→moeX is real, but further expert reduction doesn't help. Instruct variant: +1-2% (SSM bottleneck from start). Thinking variant: +12% (baseline slower at 9.2 t/s, moeX hits same 10.2 t/s ceiling).

#### Tier A-B: Production & Specialist Models (Dense 70B+)

| Model | Role | Thinking | General | Math | Agentic | Coder | Inst.Prec | Pct | Baseline t/s | Opt t/s | Draft | K | Temp |
|-------|------|----------|---------|------|---------|-------|-----------|-----|--------------|---------|-------|---|------|
| **Meta-Llama-3.1-70B** | architect | 29/30 | 29/30 | 25/30 | 30/30 | 27/30 | 25/33 | **90%** | 2.1 | 84.31 | PARD-Llama-3.2-1B Q4_0 | 24 | - |
| **Hermes-4-70B** | architect | 29/30 | 29/30 | 22/30 | 28/30 | 30/30 | 15/33 | **84%** | 2.7 | - | - | - | - |
| Hermes-4-70B | ingest | 30/30 | 29/30 | 30/30 | 24/30 | 25/30 | 15/33 | **84%** | 2.9 | - | - | - | - |
| Meta-Llama-3.1-70B | ingest | 28/30 | 28/30 | 23/30 | 26/30 | 23/30 | 21/33 | **81%** | 2.0 | 85.75 | PARD-Llama-3.2-1B Q4_0 | 24 | - |
| Qwen2.5-Math-72B (2nd run) | math | 22/30 | 30/30 | 26/30 | 20/30 | 21/30 | 22/33 | **77%** | 2.0 | 158.85 | Qwen2.5-0.5B | 24 | - |
| Qwen2.5-72B | ingest | 28/30 | 26/30 | 27/30 | 20/30 | 19/30 | 18/33 | **75%** | 2.2 | - | - | - | - |
| DeepSeek-R1-Distill-Llama-70B | thinking | 20/30 | 20/30 | 20/30 | 21/30 | 20/30 | 13/33 | **62%** | 1.0 | - | - | - | - |
| Qwen2.5-Math-72B (Q4_K_M) | math | 21/30 | 18/30 | 21/30 | 19/30 | 21/30 | 11/33 | **61%** | ~2.0 | 7.55 | Qwen2.5-0.5B | 12 | 0.5 |
| Meta-Llama-3-70B | architect | 0/30 | 13/30 | 5/30 | 12/30 | 15/30 | 16/33 | **33%** ⚠️ | 14.9 | 6.42 | PARD-Llama-3.2-1B | 8 | - |
| Qwen2.5-72B (base) | architect | - | - | - | - | - | - | - | 0.85 | - | - | - | - |
| Qwen2.5-72B-Instruct | architect | - | - | - | - | - | - | - | 1.70 | 8.53 | Qwen2.5-0.5B | 16 | - |
| Qwen2.5-Math-72B (Q6_K) | math | - | - | - | - | - | - | - | - | - | - | - | - |
| GLM-4.6 (dense) | general | - | - | - | - | - | - | - | - | - | - | - | - |

#### Tier A-B: Production & Specialist Models (Dense 27B-32B)

| Model | Role | Thinking | General | Math | Agentic | Coder | Inst.Prec | Pct | Baseline t/s | Opt t/s | Draft | K | Temp |
|-------|------|----------|---------|------|---------|-------|-----------|-----|--------------|---------|-------|---|------|
| DeepSeek-R1-Distill-Qwen-32B (Q6_K) | thinking | - | - | - | - | - | - | - | 3.73 | - | - | - | - |
| Qwen3-32B | general | - | - | - | - | - | - | - | 3.67 | 5.87 | Qwen3-0.6B | 8 | - |
| Gemma-3-27B-QAT | general | - | - | - | - | - | - | - | 4.72 | - | - | - | - |
| Qwen2.5-Coder-32B | coder/summarize | - | - | - | - | - | - | - | 5.79 | 33.0 | Qwen2.5-Coder-0.5B | 24 | - |

#### Tier B: Thinking & Reasoning Models (7B-14B)

| Model | Role | Thinking | General | Math | Agentic | Coder | Inst.Prec | Pct | Baseline t/s | Opt t/s | Draft | K | Temp |
|-------|------|----------|---------|------|---------|-------|-----------|-----|--------------|---------|-------|---|------|
| DeepSeek-R1-Distill-Qwen-14B (Q4_K_M) | thinking | 30/30 | 29/30 | 21/21 | 24/30 | 30/30 | 17/33 | **86.8%** | 6.44 | - | - | - | - |
| DeepSeek-R1-Distill-Qwen-14B (Q6_K_L) | thinking | 6/6† | - | - | - | - | - | **100%†** | 4.2 | - | - | - | - |
| DeepSeek-R1-Distill-Llama-8B | thinking | 28/30 | 24/30 | 30/30 | 30/30 | - | - | **93%*** | 7.2 | - | - | - | - |
| DeepSeek-R1-Distill-Qwen-7B | thinking | 29/30 | 19/30 | 30/30 | 24/30 | - | - | 85%* | 7.7 | - | - | - | - |
| DeepSeek-R1-0528-Qwen3-8B | thinking | 24/24 | - | - | 21/30 | - | 8/33 | **60.9%** ⚠️ | 8.7 | - | - | - | - |
| Qwen3-4B-Thinking-2507-Q8_0 | thinking | 22/30 | 19/30 | 8/9 | 5/9 | - | - | **69%** | 18.0 | - | - | - | - |

#### Tier C: Workers & General Models

| Model | Role | Thinking | General | Math | Agentic | Coder | Inst.Prec | Pct | Baseline t/s | Opt t/s | Draft | K | Temp |
|-------|------|----------|---------|------|---------|-------|-----------|-----|--------------|---------|-------|---|------|
| **Qwen2.5-7B-Q4_K_S** | general | 40/60 | 42/60 | 42/60 | 40/60 | 46/60 | 44/66 | **69.4%** | 16.0 | - | - | - | - |
| Meta-Llama-3-8B-Instruct | worker_general | 37/60 | 36/60 | 19/30 | 40/60 | 22/30 | 22/33 | **64.5%** | 12.6 | - | - | - | - |
| Meta-Llama-3-8B-Instruct | toolrunner | 19/30 | 37/60 | 20/30 | 20/30 | 22/30 | 22/33 | **65.7%** | 13.8 | - | - | - | - |
| Meta-Llama-3.1-8B (Q4_K_S) | general | 4/30 | 6/30 | 2/30 | 7/30 | 4/30 | 4/33 | **15%** | 63.5 | - | - | - | - |
| **Qwen2.5-Math-7B-Instruct** | worker_math | 43/60 | 30/60 | 27/30 | 26/60 | - | - | **60.0%** | 11.3 | - | - | - | - |
| **Qwen2.5-VL-7B-Instruct** | worker_vision | 22/30 | 43/60 | - | 40/60 | - | - | **69.4%** | 11.8 | 57.1 | Qwen2.5-0.5B | 8 | 0.7 |
| Qwen2.5-Coder-32B | worker_summarize | - | - | - | - | - | - | - | 5.79 | 95.18 | (lookup) | - | - |
| **Gemma-3-12B-IT** | general | 29/30 | 27/27 | 27/30 | 28/30 | 26/30 | 28/33 | **91.7%** | 9.2 | - | - | - | - |
| Gemma-3-27B-IT-QAT | general | 30/30 | 30/30 | - | 30/30 | 30/30 | 22/33 | **92.8%** | 2.1 | - | - | - | - |
| MathSmith-Qwen3-8B ⚠️ | math | 30/30 | 28/30 | 28/30 | - | - | - | **95.6%** | **3.4** ⚠️ | - | - | - | - |

#### Vision Models

| Model | Role | Thinking | General | Math | Agentic | Coder | Inst.Prec | VL | Pct | Baseline t/s | Opt t/s | Draft | K | Temp |
|-------|------|----------|---------|------|---------|-------|-----------|-----|-----|--------------|---------|-------|---|------|
| **Qwen3-VL-235B-A22B-Thinking** | vision | - | - | - | - | - | - | - | - | 3.23 | - | - | - | - |
| Qwen3-VL-235B-A22B-Thinking (MoE 4) | vision | - | - | - | - | - | - | - | - | 7.12 | 3.82 ❌ | (lookup) | - | - |
| **Qwen3-VL-30B-A3B** | vision_escalation | - | - | - | - | - | - | - | - | 26.88 | - | - | - | - |
| Qwen3-VL-30B-A3B (MoE 3) | vision_escalation | - | - | - | - | - | - | - | - | 37.66 | - | - | - | - |
| Qwen3-VL-30B-A3B (MoE 4) | vision_escalation | - | - | - | - | - | - | - | - | 36.84 | 29.88 ❌ | (lookup) | - | - |
| Qwen3-VL-30B-A3B (MoE 6) | vision_escalation | - | - | - | - | - | - | - | - | 28.41 | - | - | - | - |
| Qwen2.5-VL-7B | worker_vision | - | - | - | - | - | - | - | 68%* | 15.28 | 57.1 | Qwen2.5-0.5B | 8 | 0.7 |
| **Qwen3-VL-2B-Q4_K_M** | vision | - | 12/60 | - | 25/60 | - | - | 1/60 | **21.1%** | 48.0 | - | - | - | - |
| Qwen3-VL-4B (Q4_K_M) | vision | - | 21/60 | - | 0/60 | - | - | 14/60 | **19.4%** | 78.9 | - | - | - | - |
| Qwen3-VL-4B (Q8_0) | vision | - | 15/60 | - | 0/60 | - | - | 23/60 | **21.1%** | 32.5 | - | - | - | - |
| Qwen3-VL-8B (Q4_K_M) | vision | - | 13/30 | - | 0/30 | - | - | 18/33 | **33.3%** | 38.8 | - | - | - | - |
| Qwen3-VL-8B (Q8_0) | vision | - | 25/30 | - | 8/30 | - | - | 14/30 | **52.2%** | 18.7 | - | - | - | - |

#### Tier D: Draft Models (Qwen2.5 Family)

| Model | Role | Thinking | General | Math | Agentic | Coder | Inst.Prec | Pct | Baseline t/s | Compatible Targets |
|-------|------|----------|---------|------|---------|-------|-----------|-----|--------------|-------------------|
| **Qwen2.5-Coder-0.5B** (Q8_0) | draft_primary | 19/30 | 19/30 | - | - | - | - | **63%** | 182.8 | Qwen2.5-Coder-* |
| Qwen2.5-Coder-0.5B (Q4_K_M) | draft | - | - | - | - | - | - | - | - | Qwen2.5-Coder-* |
| **Qwen2.5-0.5B** (Q8_0) | draft | 20/30 | 19/30 | - | - | - | - | **65%** | 198.0 | Qwen2.5-* |
| Qwen2-0.5B (Q2_K) | draft | 19/30 | 19/30 | - | - | - | - | **63%** | 185.4 | Qwen2-* |
| Qwen2.5-Coder-1.5B (Q4_K_M) | draft | 20/30 | 19/30 | - | - | - | - | **65%** | 87.9 | Qwen2.5-* |
| Qwen2.5-Coder-1.5B (Q2_K) | draft | 20/30 | 19/30 | - | - | - | - | **65%** | 78.7 | Qwen2.5-* |
| Qwen2.5-Math-1.5B (Q4_K_M) | draft | 22/30 | 18/30 | - | - | - | - | **67%** | 52.5 | Qwen2.5-Math-* |
| Qwen2.5-Math-1.5B (Q6_K) | draft | 22/30 | 18/30 | - | - | - | - | **67%** | 65.6 | Qwen2.5-Math-* |

#### Tier D: Draft Models (Qwen3 Family)

| Model | Role | Thinking | General | Math | Agentic | Coder | Inst.Prec | Pct | Baseline t/s | Compatible Targets |
|-------|------|----------|---------|------|---------|-------|-----------|-----|--------------|-------------------|
| Qwen3-0.6B (Q8_0) | draft | 17/30 | 18/30 | - | - | - | - | **58%** ⚠️ | 89.2 | Qwen3-* (UNRELIABLE: repetition loops) |
| PARD-Qwen3-0.6B (Q4_0) | draft | 17/30 | 18/30 | - | - | - | - | **58%** ⚠️ | 78.4 | Qwen3-* (UNRELIABLE: repetition loops) |
| Qwen3-1.7B (Q4_K_M) | draft | 20/30 | 19/30 | - | - | - | - | **65%** | 39.8 | Qwen3-* |
| Co-Rewarding-II-Qwen3-1.7B-Math | draft (⚠️) | 21/30 | 21/30 | - | - | - | - | 70%* | 22.0 | incompatible |
| Qwen3-Embedding-0.6B | draft (⚠️) | - | - | - | - | - | - | - | - | embedding only |
| Qwen3-VL-1B-Merged (Q8_0) | draft (⚠️) | 4/6 | - | - | - | - | - | 67%* | 67.7 | corrupted output |

#### Tier D: Draft Models (DeepSeek-R1-Distill Family)

| Model | Role | Thinking | General | Math | Agentic | Coder | Inst.Prec | Pct | Baseline t/s | Compatible Targets |
|-------|------|----------|---------|------|---------|-------|-----------|-----|--------------|-------------------|
| **DeepSeek-R1-Distill-Qwen-1.5B** (Q8_0) | draft | 20/30 | 19/30 | - | - | - | - | **65%** | 29.5 | DeepSeek-R1-Distill-* |
| DeepSeek-R1-Distill-Qwen-1.5B (lmstudio Q8_0) | draft | 20/30 | 19/30 | - | - | - | - | **65%** | 48.0 | DeepSeek-R1-Distill-* |
| PARD-DeepSeek-R1-1.5B (Q5_K_S) | draft | 20/30 | 19/30 | - | - | - | - | **65%** | 58.0 | DeepSeek-R1-Distill-* |
| PARD-DeepSeek-R1-1.5B (Q8_0) | draft | 20/30 | 19/30 | - | - | - | - | **65%** | 40.9 | DeepSeek-R1-Distill-* |

#### Tier D: Draft Models (Llama Family)

| Model | Role | Thinking | General | Math | Agentic | Coder | Inst.Prec | Pct | Baseline t/s | Compatible Targets |
|-------|------|----------|---------|------|---------|-------|-----------|-----|--------------|-------------------|
| PARD-Llama-3.2-1B (Q4_0) | draft | 19/30 | 19/30 | - | - | - | - | **63%** | 82.3 | Llama-3.2/3.3 |
| PARD-Llama-3.2-1B (Q8_0) | draft | 20/30 | 19/30 | - | - | - | - | **65%** | 71.4 | Llama-3.2/3.3 |

### Review Files

- Location: `benchmarks/results/reviews/`
- Per-model: `{model}_baseline.csv`
- Summary: `summary.csv`

---

## Full Data

- Detailed results: `logs/research_report.md`
- Methodology: `research/speculative_decoding_research.md`
- Blog template: `research/research_report_template.md`
- Benchmark prompts: `benchmarks/prompts/v1/`
- Benchmark results: `benchmarks/results/`
- Claude-as-Judge reviews: `benchmarks/results/reviews/`
