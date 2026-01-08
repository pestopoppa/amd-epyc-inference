# Chapter 04: Prompt Lookup (Track 8)

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

The SSM architecture requires consecutive token positions. Draft token rejection corrupts the recurrent state. See [Chapter 03: MoE Optimization](03-moe-optimization.md) for SSM-safe alternatives.

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

- speculative_decoding_research.md (Track 8 section)
- [Prompt Lookup Decoding Paper](https://arxiv.org/abs/2309.08333)

---

*Previous: [Chapter 03: MoE Optimization](03-moe-optimization.md)*
*Next: [Chapter 05: Benchmarking Framework](05-benchmarking-framework.md)*
