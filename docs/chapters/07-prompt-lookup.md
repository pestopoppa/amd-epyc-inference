# Chapter 07: Prompt Lookup (Track 8)

## Introduction

Prompt Lookup is our highest-performing technique for grounded tasks, achieving **12.7x speedup** on summarization. Unlike speculative decoding which requires a draft model, prompt lookup extracts candidate tokens directly from the input prompt using n-gram matching.

## How It Works

When the model generates a token, the system searches for matching n-grams in the input prompt:

```
Input: "The quick brown fox jumps over the lazy dog. Summarize:"
Generated: "The quick brown"
→ N-gram match found! Draft: "fox jumps over"
→ Verify against model → Accept all 3 tokens
```

**Key Insight**: Summarization, code editing, and QA tasks frequently generate text that appears verbatim in the input. Prompt lookup exploits this for free speedup (no draft model needed).

## Best Results

| Task Type | Model | Baseline | With Lookup | Speedup |
|-----------|-------|----------|-------------|---------|
| Summarization | Qwen3-Next-80B | 7.5 t/s | 95.18 t/s | **12.7x** |
| Code editing | Qwen2.5-Coder-32B | 3.0 t/s | 25.82 t/s | **8.6x** |
| Document QA | Qwen2.5-72B | ~4 t/s | ~8 t/s | **2x** |
| Code generation | Any | - | - | 1.0-1.2x |
| Code generation (w/ corpus) | Coder-family | - | - | 1.06-1.17x |

**Key Finding**: Prompt lookup only helps when output overlaps with input. Pure generation tasks see minimal benefit without corpus augmentation. With corpus-augmented prompt stuffing (see below), Coder-family models gain 6-17% on novel generation.

## When to Use

| Task Type | Expected Speedup | Reasoning |
|-----------|------------------|-----------|
| Summarization | 8-13x | Output is subset of input |
| Code refactoring | 5-9x | Most code preserved |
| Document QA | 2-4x | Answers often quote source |
| Translation | 1.5-3x | Some terms preserved |
| Code generation | ~1x | Novel output, no overlap |
| Creative writing | ~1x | Novel output, no overlap |

## Configuration

### Minimum N-gram Size

The `--lookup-ngram-min` flag controls how many consecutive tokens must match:

| Setting | Behavior | Best For |
|---------|----------|----------|
| `3` | Aggressive matching | Summarization, high overlap |
| `4` | Balanced | General use |
| `5+` | Conservative | Reduce false matches |

**Recommendation**: Start with `--lookup-ngram-min 3` for grounded tasks.

## Quick Start Command

```bash
# For summarization/QA tasks with source material
numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-cli \
  -m /mnt/raid0/llm/models/Qwen2.5-Coder-32B-Q4_K_M.gguf \
  --lookup-ngram-min 3 \
  -t 96 -f prompt_with_source_material.txt
```

## Combining with Other Techniques

Prompt lookup stacks with both MoE reduction and speculative decoding via llama-server's `--lookup` flag:

| Combination | Compatible | Result |
|-------------|------------|--------|
| Lookup + MoE Reduction | ✅ Yes | **47.11 t/s** on Qwen3-Coder-30B (MoE6+spec+lookup) |
| Lookup + Speculative | ✅ Yes | **39.44 t/s** on Qwen2.5-Coder-32B (spec-first, lookup fallback) |
| Lookup + SSM | ❌ No | SSM state corruption (consecutive position requirement) |

llama-server uses **spec-first priority**: draft model proposes tokens first, prompt lookup fills gaps when the draft model has low confidence. The `--lookup` CLI flag enables this per-slot ngram cache.

**Optimal Stack**:
```python
def get_draft_tokens(context, prompt):
    # Layer 1: Draft model (higher acceptance on novel tokens)
    drafts = draft_model.generate(context, k=8)
    if drafts.confidence > threshold:
        return drafts

    # Layer 2: Prompt Lookup fallback (FREE - zero compute)
    candidates = prompt_lookup(context, prompt, ngram_size=3)
    if candidates and len(candidates) >= 3:
        return candidates

    return drafts  # Fall through to draft regardless
```

## SSM Warning

**CRITICAL**: Do not use prompt lookup with Qwen3-Next (SSM) models.

The SSM architecture requires consecutive token positions. Draft token rejection corrupts the recurrent state. See [Chapter 06: MoE Optimization](06-moe-optimization.md) for SSM-safe alternatives.

## Implementation Notes

Prompt lookup is implemented in llama.cpp and requires no additional models or training. It works by:

1. Building an n-gram index of the input prompt at inference start
2. After each generated token, searching for matching n-grams
3. If match found, proposing those tokens as drafts
4. Verifying with the main model (same as speculative decoding)

The overhead is minimal - index building is O(n) in prompt length.

## Measuring Effectiveness

To determine if prompt lookup helps your task:

```bash
# Run with and without lookup, compare speeds
# With lookup:
llama-cli -m MODEL.gguf --lookup-ngram-min 3 -f task.txt -n 500

# Without lookup (baseline):
llama-cli -m MODEL.gguf -f task.txt -n 500
```

If speedup is <1.3x, prompt lookup isn't worth enabling for that task type.

## Corpus-Augmented Prompt Lookup (Phase 2A)

Standard prompt lookup only matches against the user's input prompt. For novel code generation, there's nothing to match against — acceptance rate is ~0%. **Corpus-augmented prompt stuffing** solves this by injecting retrieved code snippets into the prompt before inference, expanding the n-gram search space.

### Architecture

```
User: "implement async retry with exponential backoff"
  │
  ▼
CorpusRetriever (sub-ms query)
  │  SQLite n-gram index → top-3 matching snippets
  ▼
Prompt Assembly
  │  <reference_code> [retrieved snippets, ~750 tokens] </reference_code>
  │  <user> [original request] </user>
  ▼
llama-server (--lookup)
  │  n-gram matches now hit retrieved snippets
  │  + original prompt + spec decode drafts
  ▼
Output (higher acceptance rate on novel generation)
```

### Implementation

- **Index**: SQLite with word-level 4-gram index. `scripts/corpus/build_index_v2.py` builds from The Stack v1 (HuggingFace streaming). Optional pruning via `scripts/corpus/prune_index.py`.
- **Retriever**: `src/services/corpus_retrieval.py` — singleton `CorpusRetriever`, auto-detects JSON (v1) vs SQLite (v2) index. Uses mmap (~200KB RAM per query regardless of DB size).
- **Prompt injection**: `build_corpus_context()` in `src/prompt_builders/builder.py`. Runs on turn 0 for lookup-enabled roles. Injects as `## Reference Code` section.
- **Telemetry**: `src/backends/llama_server.py` extracts `draft_n` / `draft_n_accepted` from llama-server timings.

### A/B Results (MVP Corpus: 73K snippets, 338MB)

| Model | Task | Acceptance Δ | Speed Δ | Verdict |
|-------|------|-------------|---------|---------|
| Qwen3-Coder-480B | BST | +15.6pp (74.9→90.5%) | +17% (8.3→9.7 t/s) | **Best** |
| Qwen2.5-Coder-32B | BST | +8.7pp (84.6→93.3%) | +6% (30.8→32.7 t/s) | **Good** |
| Qwen3-Coder-480B | HTTP | +3.4pp | +9% | Positive |
| Qwen3-235B-A22B | HTTP | +6.6pp | +2% | Marginal |
| Qwen2.5-7B | HTTP | +5.3pp | +1% | Saturated |
| Qwen3-Coder-30B | BST | +2.1pp | -12% | Negative |
| Qwen3-235B-A22B | BST | -12.1pp | -17% | Negative |

**Finding**: Coder-family models benefit most. Enabled for 32B and 480B only. The overhead from extra prompt tokens can outweigh gains on models where acceptance is already high or verification is expensive.

### Configuration

In `orchestration/model_registry.yaml`:

```yaml
runtime_defaults:
  corpus_retrieval:
    enabled: true            # Per-role: only Coder-family
    index_path: /mnt/raid0/llm/cache/corpus/mvp_index  # JSON v1, 73K snippets. Switch to full_index when ready.
    max_snippets: 3
    max_chars: 3000          # ~750 tokens budget
```

### Token Normalization

Both index builder and retriever strip non-alphanumeric characters (except underscore) from tokens before n-gram extraction. This ensures `class Foo(Bar):` and `class foo bar` produce the same n-grams for matching.

### Keyword Fallback

When 4-gram matching returns 0 results (common for natural-language queries), `CorpusRetriever` falls back to keyword-level overlap scoring. A word→snippet_ids reverse index (665K words, builds in ~3.8s) enables individual word matching. This ensures retrieval works for queries like "binary search tree iterator" that never produce matching code 4-grams.

### Phase 2B-Quality RAG: ABANDONED (2026-02-15)

Attempted to improve code quality (not just speed) by instructing the model to "study and adapt" retrieved patterns. Tested on 7B (delta -0.96) and 32B (delta -1.38) — prompt-level RAG actively hurts quality. Models either ignore the instruction or get confused by reference code. Only works with models fine-tuned for RAG (e.g., SWE-Dev-7B/32B, which use reinforcement fine-tuning on agentic trajectories). Phase 2A (speed-only, silent injection) remains the production approach.

## References

### Prompt Lookup and N-gram Methods

1. Saxena, A. (2023). *Prompt Lookup Decoding*. GitHub Repository. https://github.com/apoorvumang/prompt-lookup-decoding

2. Yang, N., Ge, T., Wang, L., Jiao, B., Jiang, D., Yang, L., ... & Wei, F. (2023). *Inference with Reference: Lossless Acceleration of Large Language Models*. arXiv preprint. https://arxiv.org/abs/2304.04487

### Retrieval-Based Speculative Decoding

3. He, Z., Zhong, Z., Cai, T., Lee, J., & He, D. (2023). *REST: Retrieval-Based Speculative Decoding*. NAACL 2024. https://arxiv.org/abs/2311.08252

4. Zhang, A., Deng, C., Oguz, B., Ott, M., & Çelikyilmaz, A. (2024). *RASD: Retrieval-Augmented Speculative Decoding*. arXiv preprint. https://arxiv.org/abs/2503.03434

### Suffix Tree Methods

5. Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J. D., Chen, D., & Dao, T. (2024). *SuffixDecoding: A Model-Free Approach to Speeding Up Large Language Model Inference*. NeurIPS 2025 Spotlight. https://suffix-decoding.github.io/

6. Cai, T. (2024). *Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads*. GitHub Repository. https://github.com/FasterDecoding/Medusa

### Implementation Resources

7. HuggingFace. (2024). *Generation Strategies: Speculative Decoding*. HuggingFace Transformers Documentation. https://huggingface.co/docs/transformers/generation_strategies

8. vLLM Team. (2024). *N-gram Prompt Lookup in vLLM*. vLLM Documentation. https://docs.vllm.ai/en/latest/features/spec_decode.html

9. Gerganov, G., et al. (2024). *llama-lookup: Prompt Lookup Decoding in llama.cpp*. GitHub. https://github.com/ggml-org/llama.cpp/tree/master/examples/lookup

---

*Previous: [Chapter 06: MoE Optimization](06-moe-optimization.md)* | *Next: [Chapter 08: RadixAttention](08-radix-attention.md)*
