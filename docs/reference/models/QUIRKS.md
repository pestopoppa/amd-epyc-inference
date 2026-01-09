# Model Runtime Quirks

Known issues and workarounds discovered during benchmarking. Check this before testing a new model.

## Critical Quirks

### Qwen3-Next (SSM Architecture)

**Issue**: State corruption when speculation is used

**Symptoms**:
- Garbage output after a few tokens
- Repetitive loops
- Model hangs

**Workaround**: Use expert reduction ONLY, never speculation or prompt lookup

```bash
# ✅ Safe
llama-cli -m Qwen3-Next-80B.gguf --override-kv qwen3next.expert_used_count=int:3

# ❌ Breaks model
llama-speculative -m Qwen3-Next-80B.gguf -md draft.gguf
```

### Qwen3-Coder-480B

**Issue**: BOS token is `,` (comma) instead of standard

**Symptoms**: 0% acceptance rate with any draft model

**Workaround**: Expert reduction only, no speculation

### DeepSeek-R1-Distill-* Models

**Issue**: Vocab size mismatch between sizes (152,064 vs 151,936)

**Symptoms**: Token mismatch errors during speculation

**Workaround**: No speculation available. Use baseline or MoE reduction if applicable.

### Gemma-3 Family (SWA Architecture)

**Issue**: ~~Sliding Window Attention (SWA) incompatible with speculative decoding in llama.cpp~~

**Status**: ✅ FIXED with PR #18720 (forward-looking SWA masking)

**Original Problem**: Gemma-3 uses Interleaved Sliding Window Attention (ISWA) with `sliding_window=1024`. The spec decode KV cache allocation failed because draft and target had incompatible cache structures.

**Solution**: PR #18720 adds forward-looking SWA masking in `find_slot()`, allowing cells that will be outside the attention window *after* batch insertion to be reused. This reduces SWA cache from 10240 MiB to 624 MiB (94% reduction).

```bash
# ✅ Now works (PR #18720 or upstream after merge)
llama-speculative -m gemma-3-27B.gguf -md gemma-3-1b.gguf --draft 4 -t 96

# Results: 42-81% acceptance rate, 12.26 t/s
```

**Note**: Vocab mismatch (1B=262144, 27B=262208) is safe - 64 token diff doesn't affect generation.

**Discovered**: 2026-01-09
**Fixed**: 2026-01-09 (PR #18720)

## Benchmarking Quirks

### Interactive Mode Hangs

**Issue**: `llama-cli` waits for user input if not configured correctly

**Symptoms**: Benchmark script hangs indefinitely

**Workaround**: Always use these flags:
```bash
llama-cli -m MODEL.gguf -f prompt.txt -n 128 \
    --no-display-prompt \
    --simple-io \
    --no-warmup \
    --temp 0
```

**Never use**: `-i` or `--interactive` in automated scripts

### Output Capture Issues

**Issue**: Some models output to stderr, breaking parsing

**Workaround**: Capture both streams
```bash
llama-cli ... 2>&1 | tee output.log
```

### `<think>` Tag Models

**Issue**: Thinking models emit `<think>...</think>` tags that inflate token counts

**Workaround**: Parse output to separate thinking from final response

## Performance Quirks

### Temperature and Speculation

**Issue**: Some models perform better with non-zero temperature during speculation

| Model | Best temp | Speed Impact |
|-------|-----------|--------------|
| Qwen2.5-VL-7B | 0.7 | 28.3 → 57.1 t/s |
| Qwen2.5-Math-72B | 0.5 | 6.0 → 7.5 t/s |
| Qwen2.5-Coder-32B | 0 | Best at temp=0 |

**Workaround**: If acceptance <50% at temp=0, try temp=0.3-0.7

### MoE Expert Count Sweet Spots

**Issue**: Below 3 experts, quality degrades significantly

| Model | Min Safe Experts | Quality Impact |
|-------|------------------|----------------|
| Qwen3-VL-30B | 3 | ✅ Good |
| Qwen3-Next-80B | 3 | ✅ Good |
| Qwen3-235B | 4 | ✅ Good |
| Any | 2 | ⚠️ Often garbage |

## Memory Quirks

### Context Length Limits

| Model Family | Max Context | Notes |
|--------------|-------------|-------|
| Llama2 | 4K | Hard limit |
| Llama3 | 8K | Default, some support 128K |
| Qwen | 131K | But slower beyond 32K |
| DeepSeek-R1 | 65K | Official limit |

### VRAM/RAM Estimates

| Quantization | Size Formula |
|--------------|--------------|
| Q4_K_M | params × 0.5 GB |
| Q8_0 | params × 1.0 GB |
| F16 | params × 2.0 GB |

Example: 70B model at Q4_K_M ≈ 35GB

## Adding New Quirks

When discovering a new quirk:

1. Add to this file with:
   - Issue description
   - Symptoms
   - Workaround
   - Discovery date

2. Update `orchestration/model_registry.yaml`:
   ```yaml
   runtime_quirks:
     model_name:
       quirks:
         - issue: "Description"
           workaround: "Fix"
           discovered: YYYY-MM-DD
   ```

---

*See [MODELS.md](MODELS.md) for model configurations.*
