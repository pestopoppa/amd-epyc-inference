# Speculative Decoding Research: Multi-Track Optimization Plan

## Executive Summary

This document tracks research into maximizing LLM inference throughput on AMD EPYC 9655 "Turin" (96-core Zen 5) with 1.13TB RAM.

**Current Best Results (Tested 2025-12-15):**
- Track 8 (Prompt Lookup): **12.7x speedup** on Qwen3-Next-80B (summarization)
- Track 1 (External Draft): **11x speedup** on Qwen2.5-Coder-32B (K=24 tuned)
- Track 2 (MoE Soft Mask): **52% speedup** on Qwen3-VL-30B (3 experts)
- Combined (MoE + Lookup): **47.5 t/s** on Qwen3-Coder-30B

**Failed Approaches:**
- CAS-Spec (Track 7): 0.446% acceptance - requires trained exit classifiers
- EAGLE-1 (Track 3): 0% acceptance - checkpoint incompatibility

**Remaining to Test:** SuffixDecoding (Track 6) for agentic workloads.

---

## Track Status Overview (Updated 2025-12-15)

### PRODUCTION (Tested & Working)
| Track | Approach | Status | Proven Speedup | Notes |
|-------|----------|--------|----------------|-------|
| **Track 1** | External Draft Model | ✅ Production | **5.9x-11x** | K=24 for 32B, K=8 for 7B |
| **Track 2** | MoE Soft Mask | ✅ Production | **21-52%** | 3 experts safe on 30B |
| **Track 8** | Prompt Lookup | ✅ Production | **1.9x-12.7x** | Best on MoE + summarization |

### NOT YET TESTED (Potential)
| Track | Approach | Status | Expected Gain | Effort |
|-------|----------|--------|---------------|--------|
| **Track 6** | SuffixDecoding | 🆕 Untested | +100-200% (agentic) | 1 day |

### TESTED AND FAILED
| Track | Approach | Status | Result | Analysis |
|-------|----------|--------|--------|----------|
| **Track 7** | CAS-Spec Layer Skip | ⛔ Failed | 0.446% accept | Needs trained exit classifiers |
| **Track 3** | EAGLE-1 | ⛔ Failed | 0% acceptance | Checkpoint/architecture issues |
| **Track 5** | SSM Speculation | ⛔ Blocked | N/A | Recurrent state incompatible |

### REQUIRES TRAINING (Not Pursued)
| Track | Approach | Status | Reason |
|-------|----------|--------|--------|
| **Track 4** | Medusa | ⚠️ Skipped | Requires head training per model |
| **Track 9** | CLaSp/SWIFT | ⚠️ Skipped | Similar to CAS-Spec, likely same issues |
| **Track 10** | Kangaroo | ⚠️ Skipped | Requires adapter training |

---

## Compounding vs. Mutual Exclusivity

### Key Insight: Draft Token Sources are Mutually Exclusive

Every speculative decoding method answers: "Where do draft tokens come from?"

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DRAFT TOKEN SOURCE (Pick ONE)                     │
├─────────────────────────────────────────────────────────────────────┤
│  Option A: External Draft Model (Track 1) ← RECOMMENDED             │
│            └── Qwen2.5-0.5B → 5.9x proven                           │
│                                                                      │
│  Option B: Self-Draft via Layer Skip (Track 7/9)                     │
│            └── CAS-Spec / CLaSp → 2.3x (when no draft available)    │
│                                                                      │
│  Option C: Self-Draft via Trained Head (Track 3/10)                  │
│            └── EAGLE / Kangaroo → Blocked/Training required          │
└─────────────────────────────────────────────────────────────────────┘
                              +
┌─────────────────────────────────────────────────────────────────────┐
│              RETRIEVAL AUGMENTATION (Stacks with ANY above)          │
├─────────────────────────────────────────────────────────────────────┤
│  Track 8: Prompt Lookup (zero cost, try FIRST)                       │
│           └── +2-4x on grounded tasks                               │
│                                                                      │
│  Track 6: SuffixDecoding (pattern matching)                          │
│           └── +5-10x on agentic/repetitive                          │
└─────────────────────────────────────────────────────────────────────┘
                              +
┌─────────────────────────────────────────────────────────────────────┐
│              TARGET MODEL OPTIMIZATION (Orthogonal)                  │
├─────────────────────────────────────────────────────────────────────┤
│  Track 2: MoE Soft Mask                                              │
│           └── 21-48% speedup on MoE models, INDEPENDENT              │
└─────────────────────────────────────────────────────────────────────┘
```

### Optimal Combined Stack

```python
def get_draft_tokens(context, prompt, draft_model, suffix_tree):
    # Layer 1: Prompt Lookup (FREE - zero compute)
    candidates = prompt_lookup(context, prompt, ngram_size=4)
    if candidates and len(candidates) >= 3:
        return candidates, "prompt_lookup"
    
    # Layer 2: SuffixDecoding (near-free for agentic patterns)
    candidates = suffix_tree.find_match(context[-32:])
    if candidates and len(candidates) >= 3:
        return candidates, "suffix"
    
    # Layer 3: Draft model (costs compute but always works)
    candidates = draft_model.generate(context, k=8)
    return candidates, "draft_model"
```

---

## Track 1: External Draft Model (PRODUCTION ✅)

### Status: ✅ **5.9x-11x speedup proven**

### Validated Configurations

| Target Model | Draft Model | K | Acceptance | Speedup |
|--------------|-------------|---|------------|---------|
| **Qwen2.5-Coder-32B** | Qwen2.5-Coder-0.5B | **24** | 70.8% | **11x** |
| Qwen2.5-Coder-32B | Qwen2.5-Coder-0.5B | 16 | 75% | 5.9x |
| Qwen2.5-VL-7B | Qwen2.5-Coder-0.5B | 8 (t=0.7) | 74.2% | **3.7x** |
| Qwen2.5-Math-7B | Qwen2.5-Coder-0.5B | 8 | 65.6% | **3.9x** |
| Qwen2.5-Math-72B | Qwen2.5-Coder-0.5B | 16 (t=0.5) | 60.3% | **7.3x** |
| Meta-Llama-70B | PARD-Llama-3.2-1B | 8 | 79% | **5.0x** |
| Qwen3-32B | Qwen3-0.6B | 8 | 39% | **3.1x** |
| Qwen2.5-72B-Instruct | Qwen2.5-0.5B | 8 | 47% | **3.0x** |

### K-Tuning Methodology (Discovered 2025-12-15)

**Finding:** Optimal K depends on model size and acceptance rate curve.

| Model Size | Optimal K | Reasoning |
|------------|-----------|-----------|
| 7B targets | K=8 | High baseline speed, K>8 reduces acceptance too much |
| 32B targets | K=16-24 | Lower baseline speed, more tokens per verification worthwhile |
| 72B targets | K=16 | Balance of acceptance vs verification cost |

**Process:**
1. Start with K=8, measure speed and acceptance
2. If acceptance >60%, try K=12, K=16, K=24
3. Plot speed vs K - optimal is usually where curve flattens
4. Higher K = more draft tokens but lower acceptance per token

### Temperature Tuning Discovery (2025-12-15)

**Unexpected finding:** Non-zero temperature can IMPROVE speculative decoding.

| Model | temp=0 | temp=0.5 | temp=0.7 | Best |
|-------|--------|----------|----------|------|
| Qwen2.5-VL-7B | 28.3 t/s | 37.4 t/s | **57.1 t/s** | t=0.7 |
| Qwen2.5-Math-72B | 6.0 t/s | **7.5 t/s** | N/A | t=0.5 |
| Qwen2.5-Coder-32B | **26.6 t/s** | 19.0 t/s | 19.4 t/s | t=0 |

**Hypothesis:** temp=0 produces overly deterministic drafts that diverge from target model's probability distribution. Slight temperature increases draft diversity, improving acceptance.

**Recommendation:** If acceptance is <50% at temp=0, try temp=0.3-0.7.

### Draft Model Compatibility Matrix

| Target Family | Compatible Drafts | Incompatible |
|---------------|-------------------|--------------|
| Qwen2.5-* | Qwen2.5-Coder-0.5B, Qwen2.5-0.5B | Qwen3-*, DeepSeek-* |
| Qwen3-* | Qwen3-0.6B | Qwen2.5-* |
| Meta-Llama-3.* | PARD-Llama-3.2-1B | Other families |
| DeepSeek-R1-Distill-* | **NONE FOUND** | All tested |

**Key insight:** Same model family is NOT enough. Tokenizer vocab and special tokens must match exactly.

### Quick Start Command

```bash
OMP_NUM_THREADS=1 numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-speculative \
  -m /mnt/raid0/llm/models/Qwen2.5-Coder-32B-Q4_K_M.gguf \
  -md /mnt/raid0/llm/models/Qwen2.5-0.5B-Instruct-Q8_0.gguf \
  --draft-max 24 -t 96 -p "Your prompt"  # Note: K=24 optimal for 32B
```

---

## Track 2: MoE Soft Mask (PRODUCTION ✅)

### Status: ✅ **21-50% speedup proven**

### Validated Results

| Model | Baseline | Top-4 Experts | Speedup |
|-------|----------|---------------|---------|
| Qwen3-Coder-480B-A35B | 2.5 t/s | 3.7 t/s | **+48%** |
| GLM-4.6-355B-A32B | 2.2 t/s | 3.0 t/s | **+36%** |
| Qwen3-Coder-30B-A3B | 26.6 t/s | 33.6 t/s | **+26%** |
| Qwen3-VL-30B-A3B | 24.8 t/s | 37.7 t/s (3 exp) | **+52%** |
| Qwen3-Next-80B-A3B | 7.5 t/s | 9.1 t/s (3 exp) | **+21%** |

### Expert Count Tuning (2025-12-15)

| Model | 6 Experts | 4 Experts | 3 Experts | Quality |
|-------|-----------|-----------|-----------|---------|
| Qwen3-VL-30B-A3B | 28.4 t/s | ~35 t/s | **37.7 t/s** | ✅ Good |
| Qwen3-Coder-30B-A3B | 30.1 t/s | ~35 t/s | N/A | ✅ Good |
| Qwen3-Next-80B-A3B | 9.0 t/s | ~9 t/s | 9.1 t/s | ✅ Good |

**Finding:** Reducing to 3 experts (from default 8) is safe and faster on 30B models.

### MoE + Lookup Combination Results

**Key Discovery:** Combination benefits are model-size dependent.

| Model | Lookup Only | MoE Only | Combo (3 exp + Lookup) | Verdict |
|-------|-------------|----------|------------------------|---------|
| Qwen3-Coder-30B-A3B | 43.2 t/s | 30.1 t/s | **47.5 t/s** | ✅ Combo wins |
| Qwen3-VL-30B-A3B | 46.3 t/s | 37.7 t/s | 15.3 t/s | ❌ Lookup alone |
| Qwen3-Next-80B-A3B | 29.1 t/s | 9.1 t/s | **22.0 t/s** | ✅ Combo wins |

**Recommendation:**
- **30B MoE models:** Use lookup alone (46 t/s) OR MoE alone (37 t/s), NOT combo
- **80B+ MoE models:** Use combo for best speed (22 t/s vs 9 t/s baseline = 2.4x)

### Architecture-Specific Keys

| Model Family | Override Key |
|--------------|--------------|
| Qwen3-Coder-* | `qwen3moe.expert_used_count` |
| Qwen3-VL-* | `qwen3vlmoe.expert_used_count` |
| GLM-4.6-* | `glm4moe.expert_used_count` |

### Quick Start Command

```bash
# No code changes needed - use --override-kv
OMP_NUM_THREADS=1 numactl --interleave=all \
  ./llama-cli \
  -m Qwen3-VL-30B-A3B-Q4_K_M.gguf \
  --override-kv qwen3vlmoe.expert_used_count=int:3 \
  -t 96 -p "Your prompt"
```

---

## Track 6: SuffixDecoding (NEW - NeurIPS 2025 Spotlight) 🔥

### Status: 🆕 **Implement this week**

### Why This Matters

**10.4x speedup demonstrated** on agentic workloads — model-free, no training required.

For repetitive inference patterns (multi-agent pipelines, code generation, SQL), the model often repeats token sequences. SuffixDecoding retrieves drafts from suffix trees built on prior outputs.

### Expected Performance

| Workload | Reported Speedup | Combined with Track 1 |
|----------|------------------|----------------------|
| AgenticSQL | 10.4x | Primary (no draft needed) |
| Multi-agent pipelines | 5.3x | Primary |
| Code generation | 3-5x | Supplement Track 1 |
| General chat | 1.5-2x | Fallback to Track 1 |

### Implementation

```python
# Build suffix tree from session outputs
from suffix_trees import STree

class SuffixDraftProvider:
    def __init__(self):
        self.global_tree = None
        self.outputs = []
    
    def add_output(self, text):
        self.outputs.append(text)
        if len(self.outputs) % 10 == 0:
            self.global_tree = STree("".join(self.outputs))
    
    def get_drafts(self, context, k=8):
        if not self.global_tree:
            return None
        match = self.global_tree.find_longest_match(context[-32:])
        if match and len(match.continuation) >= k:
            return match.continuation[:k]
        return None
```

### Resources

- **Paper:** https://suffix-decoding.github.io/ (NeurIPS 2025 Spotlight)
- **vLLM integration:** Available

---

## Track 8: Prompt Lookup Decoding (PRODUCTION ✅)

### Status: ✅ **12.7x speedup proven** (2025-12-15)

### Why This Matters

**Zero overhead, no model needed** — draft tokens come from the prompt itself.

In input-grounded tasks (summarization, document QA, code editing), there's high n-gram overlap between prompt and output.

### Actual Benchmark Results (2025-12-15)

| Model | Task | Speed | Accept % | vs Baseline |
|-------|------|-------|----------|-------------|
| **Qwen3-Next-80B-A3B** | Summarize | **95.2 t/s** | 8.8% | **12.7x** |
| Qwen3-VL-30B-A3B | Summarize | **46.3 t/s** | 100% | **1.9x** |
| Qwen3-VL-30B-A3B | Code | **43.3 t/s** | 100% | **1.7x** |
| Qwen3-Coder-30B-A3B | Summarize | **43.2 t/s** | 100% | **1.9x** |
| Qwen3-Coder-30B-A3B | Code | **40.9 t/s** | 100% | **1.8x** |
| Meta-Llama-3-8B | Summarize | **37.1 t/s** | 100% | **2.1x** |
| Qwen2.5-Math-7B | Summarize | **38.7 t/s** | 100% | **3.1x** |
| Meta-Llama-3.1-70B | Summarize | 3.5 t/s | 75% | **2.7x** |
| Hermes-4-70B | Summarize | 1.3 t/s | 21% | **1.0x** |

### Key Findings

1. **MoE models excel with lookup:** 40-46 t/s with near-100% acceptance
2. **Dense 70B models limited:** Even with lookup, 1-4 t/s (memory bound)
3. **Task dependency:** Summarization > Code > General chat
4. **Acceptance varies widely:** 8.8% to 100% depending on prompt/model

### Expected vs Actual Performance

| Task Type | Paper Claims | Our Results |
|-----------|--------------|-------------|
| Summarization | 2.8x | **1.9-12.7x** (model dependent) |
| Code editing | 3-4x | **1.7-1.9x** |
| General | 1.0-1.2x | ~1.0x |

### Implementation (50 lines)

```python
def prompt_lookup(input_ids, max_ngram=4, num_candidates=10):
    """Find matching n-grams in prompt, return continuation as draft."""
    for ngram_size in range(max_ngram, 0, -1):
        # Get last N tokens as search pattern
        pattern = input_ids[-ngram_size:]
        
        # Search earlier in input for match
        for i in range(len(input_ids) - ngram_size - num_candidates):
            if input_ids[i:i+ngram_size] == pattern:
                # Return tokens following the match
                return input_ids[i+ngram_size:i+ngram_size+num_candidates]
    return None
```

### llama.cpp / vLLM Support

```bash
# vLLM
python -c "
from vllm import LLM
llm = LLM(
    model='facebook/opt-6.7b',
    speculative_model='[ngram]',
    num_speculative_tokens=5,
    ngram_prompt_lookup_max=4
)
"

# Check llama.cpp support
./llama-speculative --help | grep -i ngram
```

### Resources

- **Original repo:** https://github.com/apoorvumang/prompt-lookup-decoding
- **HuggingFace:** `prompt_lookup_num_tokens` parameter in `generate()`
- **vLLM:** `speculative_model="[ngram]"`

---

## Track 7: CAS-Spec (Cascade Adaptive Self-Speculative)

### Status: ⛔ **TESTED AND FAILED** (2025-12-15)

### Why This Matters (In Theory)

**No external model needed** — creates hierarchy of drafts from target model itself via:
1. Layer sparsity (skip 30% → 50% → 70% of layers)
2. Activation quantization (INT8)
3. Dynamic Tree Cascade routing

### Expected vs Actual Performance

| Configuration | Paper Claims | Our Results |
|---------------|--------------|-------------|
| Static layer skip | 1.1-1.5x | ❌ Garbage output |
| CAS-Spec + DyTC | **2.3x** | ❌ **0.446% acceptance** |

### Experimental Methodology (2025-12-15)

**Test Setup:**
```bash
# Using llama.cpp --n-layer-exit flag
OMP_NUM_THREADS=1 numactl --interleave=all \
  ./llama-speculative \
  -m Qwen2.5-Coder-32B-Q4_K_M.gguf \
  --n-layer-exit 32 \  # 50% of 64 layers
  -t 96 -n 100
```

**Results by Exit Layer:**

| Layers Used | % of Full | Speed | Acceptance | Output Quality |
|-------------|-----------|-------|------------|----------------|
| 32/64 | 50% | ~60 t/s | 0.2% | ❌ Multilingual garbage |
| 48/64 | 75% | ~40 t/s | 0.4% | ❌ Gibberish |
| 58/64 | 90% | ~35 t/s | 0.5% | ❌ Repetitive tokens |
| 62/64 | 97% | ~32 t/s | 0.4% | ❌ Wrong language |
| 64/64 | 100% | 3 t/s | N/A | ✅ Correct |

### Failure Analysis

**Root Cause:** Early transformer layers capture syntax/structure, late layers capture semantics/knowledge. On Qwen2.5 architecture:
- Layers 1-32: Encode basic patterns, produce plausible but meaningless tokens
- Layers 33-50: Build semantic relationships
- Layers 51-64: Final knowledge retrieval and coherence

**Why Paper Results Don't Transfer:**
1. Paper tested on specific models with different layer distributions
2. Paper used trained exit classifiers to dynamically choose exit points
3. Our static layer skip has no way to detect "confident" vs "uncertain" tokens
4. Qwen2.5's knowledge may be more concentrated in final layers

### Verdict

CAS-Spec requires **per-model calibration** and potentially trained exit classifiers. The static `--n-layer-exit` flag in llama.cpp is insufficient for production use.

**Recommendation:** Do NOT use layer-skip self-speculation. Use external draft models (Track 1) or lookup decoding (Track 8) instead.

### Resources

- **Paper:** https://arxiv.org/abs/2510.26843 (NeurIPS 2025)
- **Code:** Implementation requires C++ modifications + training

---

## Track 9: CLaSp/SWIFT (Dynamic Layer Skip)

### Status: 🆕 **Quick fallback option**

### What It Does

Plug-and-play layer skipping that adapts per token based on difficulty.

### Expected Performance

| Model Size | Speedup |
|------------|---------|
| 8B | 1.3-1.5x |
| 70B | 1.5-1.8x |

### HuggingFace Integration

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-70B")
outputs = model.generate(
    **inputs,
    assistant_early_exit=24,  # Exit after layer 24 for drafts
    do_sample=False
)
```

### Resources

- **CLaSp:** https://arxiv.org/abs/2505.24196 (ACL 2025)
- **SWIFT:** https://openreview.net/forum?id=EKJhH5D5wA (ICLR 2025)
- **LayerSkip:** https://arxiv.org/abs/2404.16710

---

## Track 3: EAGLE (DEPRECATED ⛔)

### Status: ⛔ **Blocked — Pivot to Tracks 6/8 instead**

### Why Deprecated

After 20+ hours of debugging:
- 0% acceptance rate persists
- Falsified: GQA expansion, RoPE scaling, quantization, V-expansion layout
- Remaining suspects: Penultimate layer extraction, checkpoint incompatibility
- Would require PyTorch reference validation or retraining

### Resurrection Criteria

Only revisit if:
1. llama.cpp adds official EAGLE support
2. Pre-trained EAGLE checkpoints for Qwen2.5/Qwen3 families become available
3. All other tracks exhausted AND need >10x speedup

### Resources (For Reference)

- **EAGLE-1:** https://arxiv.org/abs/2401.15077 (ICML'24)
- **EAGLE-2:** EMNLP'24
- **EAGLE-3:** NeurIPS'25
- **GitHub:** https://github.com/SafeAILab/EAGLE
- **Checkpoints:** yuhuili/EAGLE-* on HuggingFace

---

## Track 5: SSM Speculation (BLOCKED ⛔)

### Status: ⛔ **Architecturally incompatible**

SSM/Mamba models (Qwen3-Next) use recurrent state that cannot be rolled back like KV cache.

### Future Research

- **STree:** Tree-based speculation for SSMs
- **SpeculativeMamba:** Mamba-specific algorithms
- **MambaInLlama:** https://github.com/jxiw/MambaInLlama

---

## Execution Priority (Updated 2025-12-15)

### COMPLETED ✅
1. ✅ Track 8 (Prompt Lookup) — **12.7x proven**, production ready
2. ✅ Track 1 K-tuning — K=24 for 32B, K=8 for 7B
3. ✅ Track 1 temperature tuning — temp=0.7 can help
4. ✅ Track 2 expert count optimization — 3 experts safe
5. ✅ Track 2 + Track 8 combination testing — model-dependent
6. ✅ Track 7 (CAS-Spec) — **FAILED**, 0.446% acceptance

### NEXT TO TRY
1. **Track 6 (SuffixDecoding)** — 1 day effort, high potential for agentic/SQL workloads
2. Better draft models — Find/train Qwen3 compatible drafts
3. DeepSeek-R1 draft model search — Currently no working drafts

### DO NOT PURSUE
- Track 3 (EAGLE) — 0% acceptance, 20+ hours wasted
- Track 7 (CAS-Spec) — 0.446% acceptance, needs training
- Track 4 (Medusa) — training required
- Track 9 (CLaSp/SWIFT) — likely same issues as CAS-Spec
- Track 10 (Kangaroo) — training required

---

## Actual Combined Performance (2025-12-15)

| Workload Type | Baseline | Best Achieved | Method | Model |
|---------------|----------|---------------|--------|-------|
| **Summarization (MoE)** | 7.5 t/s | **95.2 t/s (12.7x)** | Lookup | Qwen3-Next-80B |
| **Code (MoE)** | 22.8 t/s | **47.5 t/s (2.1x)** | MoE + Lookup | Qwen3-Coder-30B |
| **Code (Dense)** | 3.0 t/s | **33.0 t/s (11x)** | Draft K=24 | Qwen2.5-Coder-32B |
| **General (7B)** | 15.3 t/s | **57.1 t/s (3.7x)** | Draft temp=0.7 | Qwen2.5-VL-7B |
| **Math (72B)** | 1.0 t/s | **7.5 t/s (7.5x)** | Draft temp=0.5 | Qwen2.5-Math-72B |
| Agentic/SQL | TBD | **TBD** | SuffixDecoding | Not yet tested |

### Best Configuration Per Model Type

| Model Type | Recommended Stack | Expected Speed |
|------------|-------------------|----------------|
| MoE 30B | Lookup alone | 40-46 t/s |
| MoE 80B+ | MoE reduction + Lookup | 20-95 t/s |
| Dense 7B | Draft K=8, try temp tuning | 45-57 t/s |
| Dense 32B | Draft K=24 | 33 t/s |
| Dense 72B | Draft K=16, temp=0.5 | 7-8 t/s |

---

## Literature References

### Foundational Speculative Decoding
- Leviathan et al. (2023) "Fast Inference from Transformers via Speculative Decoding"
- Chen et al. (2023) "Accelerating Large Language Model Decoding with Speculative Sampling"

### NeurIPS 2025 (New)
- **SuffixDecoding:** https://suffix-decoding.github.io/ (Spotlight)
- **CAS-Spec:** https://arxiv.org/abs/2510.26843
- **AdaSPEC:** https://neurips.cc/media/neurips-2025/Slides/115055.pdf
- **SpecFormer:** https://arxiv.org/abs/2511.20340

### Retrieval-Based Methods
- **REST:** https://arxiv.org/abs/2311.08252
- **Prompt Lookup:** https://github.com/apoorvumang/prompt-lookup-decoding
- **RASD:** https://arxiv.org/abs/2503.03434

### Self-Speculative / Layer Skip
- **SWIFT:** https://openreview.net/forum?id=EKJhH5D5wA (ICLR 2025)
- **CLaSp:** https://arxiv.org/abs/2505.24196 (ACL 2025)
- **LayerSkip:** https://arxiv.org/abs/2404.16710
- **Kangaroo:** https://github.com/Equationliu/Kangaroo (NeurIPS 2024)

### EAGLE Series (Deprecated for this project)
- EAGLE-1: https://arxiv.org/abs/2401.15077 (ICML'24)
- EAGLE-2: EMNLP'24
- EAGLE-3: NeurIPS'25
- GitHub: https://github.com/SafeAILab/EAGLE

### Medusa
- Paper: ICML'24
- GitHub: https://github.com/FasterDecoding/Medusa

### Cascade Methods
- **CS-Drafting:** https://arxiv.org/pdf/2312.11462 (NeurIPS 2024)
- **Faster Cascades:** https://arxiv.org/abs/2405.19261

### Curated Paper List
- **SpeculativeDecodingPapers:** https://github.com/hemingkx/SpeculativeDecodingPapers

### Implementation Resources
- llama.cpp speculative: https://github.com/ggerganov/llama.cpp/tree/master/examples/speculative
- vLLM spec decode: https://blog.vllm.ai/2024/10/17/spec-decode.html
- HuggingFace generation strategies: https://huggingface.co/docs/transformers/generation_strategies

---

## Quick Reference Commands

### Track 1: External Draft
```bash
OMP_NUM_THREADS=1 numactl --interleave=all \
  ./llama-speculative \
  -m TARGET.gguf -md DRAFT.gguf \
  --draft-max 16 -t 96 -p "prompt"
```

### Track 2: MoE Soft Mask
```bash
./llama-cli -m MOE_MODEL.gguf \
  --override-kv ARCH.expert_used_count=int:4 \
  -t 96 -p "prompt"
```

### Track 8: Prompt Lookup (vLLM)
```python
llm = LLM(model="...", speculative_model="[ngram]", 
          num_speculative_tokens=5, ngram_prompt_lookup_max=4)
```

---

*Last updated: December 2025 — Reorganized with NeurIPS 2025 findings*
