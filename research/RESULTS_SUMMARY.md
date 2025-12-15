# Research Results Summary

**Last Updated:** December 2025
**System:** AMD EPYC 9655 (96 cores, 1.13TB DDR5), llama.cpp

---

## Best Results

| Configuration | Speed | Speedup | Use Case |
|---------------|-------|---------|----------|
| Prompt Lookup (summarization) | 95.18 t/s | **12.7x** | Document QA, summarization |
| Qwen2.5-Coder-32B + 0.5B (K=24) | 33.0 t/s | **11x** | Code generation |
| Prompt Lookup (code editing) | 25.82 t/s | **8.6x** | Refactoring, code review |
| Qwen2.5-72B + 0.5B (K=16) | 8.53 t/s | **5.8x** | General tasks |
| MoE Expert Reduction (4 experts) | +21-48% | — | MoE models |

---

## Very Large Models (100B+)

### Baseline Performance
| Model | Size | Active Params | Baseline | Notes |
|-------|------|---------------|----------|-------|
| Qwen3-235B-A22B | 133GB | ~22B | **3.6 t/s** | MoE, fits in RAM |
| Qwen3-Coder-480B-A35B | 271GB | ~35B | **2.05 t/s** | MoE, largest tested |
| GLM-4.6-355B-A32B | 189GB | ~32B | **1.82 t/s** | MoE |
| Qwen3-VL-235B-A22B-Thinking | 124GB | ~22B | TBD | VL model |

### Optimization Results
| Model | Baseline | +Expert Reduction | +Lookup | Best |
|-------|----------|-------------------|---------|------|
| **Qwen3-Coder-480B** | 2.05 t/s | 3.0 t/s (+48%) | 3.69 t/s | **+80%** |
| **Qwen3-235B** | 3.6 t/s | 6.75 t/s (+87%) | 6.35 t/s | **+87%** |
| **GLM-4.6-355B** | 1.82 t/s | N/A | 3.37 t/s | **+85%** |

### Key Finding: Largest Models Benefit Most
- 480B model: **+48-80% speedup** from expert reduction + lookup
- Expert reduction more effective than speculative decoding on MoE
- All 100B+ models run entirely in RAM (no GPU needed)

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

---

## Full Data

- Detailed results: `logs/research_report.md`
- Methodology: `research/speculative_decoding_research.md`
- Blog template: `research/research_report_template.md`
