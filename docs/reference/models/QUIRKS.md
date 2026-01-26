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

### Qwen3-Thinking-2507 Models (unsloth)

**Issue**: Missing BOS token metadata in GGUF

**Symptoms**: `draft model special tokens must match target model to use speculation`

**Root Cause**: The Qwen3-*-Thinking-2507 models from unsloth have `tokenizer.ggml.eos_token_id` but no `tokenizer.ggml.bos_token_id` in their GGUF metadata. All draft models have BOS tokens defined, causing a mismatch.

**Affected Models**:
- Qwen3-30B-A3B-Thinking-2507 (Q8_0 and Q4_K_S)
- Qwen3-4B-Thinking-2507

**Workaround**: Use MoE expert reduction only. Speculative decoding is incompatible.

```bash
# ✅ Safe - MoE reduction
llama-cli -m Qwen3-30B-A3B-Thinking-2507.gguf \
    --override-kv qwen3moe.expert_used_count=int:4

# ❌ Fails - spec decode (BOS mismatch)
llama-speculative -m Qwen3-30B-A3B-Thinking-2507.gguf \
    -md Qwen3-0.6B.gguf --draft-max 8
```

**Discovered**: 2026-01-12

### Gemma-3 Family (SWA Architecture)

**Speculative Decoding Issue**: ~~Sliding Window Attention (SWA) incompatible with speculative decoding in llama.cpp~~

**Status**: ✅ FIXED with PR #18720 (forward-looking SWA masking)

**Original Problem**: Gemma-3 uses Interleaved Sliding Window Attention (ISWA) with `sliding_window=1024`. The spec decode KV cache allocation failed because draft and target had incompatible cache structures.

**Solution**: PR #18720 adds forward-looking SWA masking in `find_slot()`, allowing cells that will be outside the attention window *after* batch insertion to be reused. This reduces SWA cache from 10240 MiB to 624 MiB (94% reduction).

```bash
# ✅ Now works (PR #18720 or upstream after merge)
llama-speculative -m gemma-3-27B.gguf -md gemma-3-1b.gguf --draft 4 -t 96

# Results: 42-81% acceptance rate, 12.26 t/s
```

**Prompt Lookup Issue**: ✅ FIXED (PRs #18729, #18730)

**Symptoms**: `GGML_ASSERT(batch.seq_id[...])` crash without `-c` flag (affects ALL models, not just SWA)

**Root Cause**: Two pre-existing bugs activated by upstream default changes:
- `lookup.cpp:109` - batch init with `params.n_ctx` (now defaults to 0)
- `lookahead.cpp:121` - same issue + n_seq_max validation

```bash
# ❌ Crashed without -c (before fix)
llama-lookup -m any-model.gguf -f prompt.txt --draft-max 4

# ✅ Now works (with PRs #18729 + #18730 or local cherry-pick)
llama-lookup -m gemma-3-27B.gguf -f prompt.txt --draft-max 4
```

**Test Result**: Prompt lookup works with SWA models (32.8% acceptance on Gemma-3-1b).

**Status**: PRs submitted to llama.cpp upstream, fixes cherry-picked to local fork.

**Note**: Vocab mismatch (1B=262144, 27B=262208) is safe - 64 token diff doesn't affect generation.

**Discovered**: 2026-01-09
**Spec Decode Fixed**: 2026-01-09 (PR #18720)
**Prompt Lookup**: Still broken as of 2026-01-10

### llama-lookup Binary (Large Context)

**Issue**: `llama-lookup` crashes with assertion failure on large context prompts

**Symptoms**:
```
GGML_ASSERT(src/llama-context.cpp:1008: n_tokens <= n_batch) failed
```

**Conditions**: Occurs with prompts >10K characters (e.g., document summarization)

**Workaround**: Use `llama-cli --lookup-ngram-min` instead of the dedicated binary:
```bash
# ✅ Works - llama-cli with lookup flag
numactl --interleave=all llama-cli \
    -m MODEL.gguf \
    --lookup-ngram-min 3 \
    -f large_prompt.txt \
    -n 500 --temp 0

# ❌ Crashes - llama-lookup binary
llama-lookup -m MODEL.gguf -f large_prompt.txt --draft-max 4
```

**Expected Speedup**: 12.7x on summarization tasks (per RESULTS.md)

**Discovered**: 2026-01-23

### Vision-Language (VL) Models

**Issue**: `llama-speculative` doesn't support VL models with mmproj files

**Symptoms**: Timeout or crash when running spec decode on Qwen2.5-VL, Qwen3-VL, or similar models

**Root Cause**: VL models require the mmproj (multimodal projector) file for vision processing. The `llama-speculative` binary doesn't support loading mmproj files, only the main model weights.

**Workaround**: Use baseline mode or MoE expert reduction (for MoE-VL models):

```bash
# ✅ Safe - baseline mode
llama-cli -m Qwen2.5-VL-7B.gguf --mmproj mmproj-model-f16.gguf -p "prompt"

# ✅ Safe - MoE reduction for VL-MoE models
llama-cli -m Qwen3-VL-30B-A3B.gguf --mmproj mmproj.gguf \
    --override-kv qwen3vlmoe.expert_used_count=int:4

# ❌ Broken - spec decode not supported
llama-speculative -m Qwen2.5-VL-7B.gguf -md draft.gguf  # Times out
```

**Note**: Manual testing achieved 57.1 t/s with `--temp 0.7` using a special llama-cli workaround, but this is not automated.

**Discovered**: 2026-01-09

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

**Issue**: Below 4 experts causes instability (SIGSEGV, garbage output, UTF-8 decode errors)

| Model | Min Safe Experts | Issue at 2 Experts |
|-------|------------------|-------------------|
| Qwen3-VL-30B | 4 | ⚠️ Garbage output |
| Qwen3-Next-80B | 4 | ⚠️ Garbage output |
| Qwen3-235B | 4 | ⚠️ Garbage output |
| Any MoE | 4 | ⚠️ Unstable |

**Workaround**: Benchmark system starts MoE testing at 4 experts minimum

**Note**: Earlier reports of Qwen3-30B-A3B-Thinking crashes were caused by a stale build issue, not the model itself. Model works fine with moe2 and moe4 on clean builds.

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

## REPL Tool Compliance

**Issue**: Models may use Python imports instead of REPL tools

**Symptoms**:
- `SecurityError: Dangerous operation not allowed: import os`
- Code uses `os.listdir()`, `pathlib.Path()`, `open()` instead of REPL tools
- Multiple failed turns before model adapts

**Affected Models**:
- Qwen3-Coder-30B-A3B (frontdoor) - Initially tried `pathlib` and `os.listdir`
- Other models may vary in instruction-following capability

**Workaround**: Add explicit NO IMPORTS warnings to system prompts:
```
## CRITICAL
1. **NO IMPORTS** - import/from are BLOCKED. Use ONLY the tools above.
2. **USE list_dir()** for files - NOT os.listdir or pathlib
3. **ALWAYS call FINAL(answer)** to complete the task

## Examples
List files: `result = list_dir('/path'); FINAL(result)`
Read file: `text = peek(1000, file_path='/path'); FINAL(text)`
```

**Tool → Python Equivalent Mapping**:

| REPL Tool | Forbidden Python Equivalent |
|-----------|---------------------------|
| `list_dir(path)` | `os.listdir()`, `pathlib.Path().iterdir()` |
| `peek(n, file_path)` | `open().read()`, `pathlib.Path().read_text()` |
| `grep(pattern)` | `re.findall()`, `grep` subprocess |
| `file_info(path)` | `os.stat()`, `pathlib.Path().stat()` |
| `run_shell(cmd)` | `subprocess.run()`, `os.system()` |
| `web_fetch(url)` | `requests.get()`, `urllib` |

**Testing**: Run `pytest tests/integration/test_model_tool_compliance.py -v`

**Discovered**: 2026-01-24

---

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
