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

**Key Finding**: Prompt lookup only helps when output overlaps with input. Pure generation tasks see minimal benefit.

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

Prompt lookup can stack with MoE reduction but **NOT** with speculative decoding:

| Combination | Compatible | Result |
|-------------|------------|--------|
| Lookup + MoE Reduction | ✅ Yes | **47.5 t/s** on Qwen3-Coder-30B |
| Lookup + Speculative | ❌ No | Both provide draft tokens |
| Lookup + SSM | ❌ No | SSM state corruption |

**Optimal Stack**:
```python
def get_draft_tokens(context, prompt):
    # Layer 1: Prompt Lookup (FREE - zero compute)
    candidates = prompt_lookup(context, prompt, ngram_size=3)
    if candidates and len(candidates) >= 3:
        return candidates

    # Layer 2: Fall back to draft model
    return draft_model.generate(context, k=8)
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
