# Handoff: Fix Gemma-3 SWA + Speculative Decoding Crash in llama.cpp

## Summary

Gemma-3 models crash when used with speculative decoding in llama.cpp due to incompatibility between Sliding Window Attention (SWA/ISWA) and the speculative decoding KV cache management. This is a llama.cpp bug, not a model issue.

## Priority

**Medium-High** - Blocks speculative decoding acceleration for all Gemma-3 models (1B, 12B, 27B variants).

## Environment

- **llama.cpp build**: 7508 (commit d10a5a4a5)
- **Models tested**:
  - Target: `gemma-3-27B-it-QAT-Q4_0.gguf` (262,208 vocab)
  - Draft: `gemma-3-1b-it-Q8_0.gguf` (262,144 vocab)
- **System**: AMD EPYC 9655, 1.13TB RAM, CPU-only inference

## Problem Description

### Symptoms

1. **Without `--swa-full`**: Crashes with `std::bad_alloc` in `llama_kv_cache::slot_info::operator=`
2. **With `--swa-full`**: Crashes with `realloc(): invalid pointer`

Both crashes occur during warmup/decode, after models load successfully.

### Root Cause

Gemma-3 uses **Interleaved Sliding Window Attention (ISWA)** with:
- `sliding_window = 1024` (27B model)
- `sliding_window = 512` (1B model)
- `n_swa = 512` / `n_swa = 1024` (varies by model)

The speculative decoding code path in llama.cpp doesn't correctly handle the case where:
1. Target and draft models both use SWA but with different window sizes
2. The KV cache allocation/management differs between SWA and non-SWA layers
3. The "ISWA" (interleaved) pattern means some layers use SWA and some don't

### Crash Traces

**Crash 1 (without --swa-full):**
```
#7  std::vector<llama_kv_cache::slot_info>::operator=()
#8  llama_context::initialize_decode_context(llama_batch const&, bool)
#9  llama_context::decode(llama_batch const&)
#10 llama_decode()
terminate called after throwing std::bad_alloc
```

**Crash 2 (with --swa-full):**
```
llama_kv_cache_iswa: using full-size SWA cache (ref: https://github.com/ggml-org/llama.cpp/pull/13194)
...
common_init_from_params: warming up the model with an empty run
realloc(): invalid pointer
```

## Technical Context

### Gemma-3 Architecture

From the GGUF metadata:
```
gemma3.attention.sliding_window = 1024
print_info: n_swa = 512
print_info: is_swa_any = 1
```

Gemma-3 uses **interleaved** SWA where:
- Some layers use full attention (non-SWA)
- Some layers use sliding window attention
- This creates a hybrid KV cache structure

### Relevant llama.cpp Code Paths

1. **KV Cache Creation**: `llama_kv_cache_iswa` creates separate caches for SWA and non-SWA layers
2. **Speculative Decoding**: `llama-speculative` manages two models (draft + target) with shared decode context
3. **The Bug**: When both models use ISWA, the cache slot management in `initialize_decode_context` fails

### Key Files to Investigate

- `src/llama-kv-cache.cpp` - KV cache allocation for ISWA models
- `src/llama-context.cpp` - `initialize_decode_context()` function
- `examples/speculative/speculative.cpp` - Main speculative decoding logic
- `src/llama.cpp` - `llama_decode()` and batch processing

### Related Issues/PRs

- PR #13194: Added `--swa-full` flag for SWA cache compatibility
- The `--swa-full` flag comment references: https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055
- Users report Gemma-3 spec decode works in some versions but not others

## Reproduction Steps

```bash
# This crashes with std::bad_alloc
/mnt/raid0/llm/llama.cpp/build/bin/llama-speculative \
  -m /mnt/raid0/llm/lmstudio/models/lmstudio-community/gemma-3-27B-it-qat-GGUF/gemma-3-27B-it-QAT-Q4_0.gguf \
  -md /mnt/raid0/llm/models/gemma-3-1b-it-Q8_0.gguf \
  --draft-max 4 -t 96 -n 10 -c 2048 \
  --prompt "Hello"

# This crashes with realloc(): invalid pointer
/mnt/raid0/llm/llama.cpp/build/bin/llama-speculative \
  -m /mnt/raid0/llm/lmstudio/models/lmstudio-community/gemma-3-27B-it-qat-GGUF/gemma-3-27B-it-QAT-Q4_0.gguf \
  -md /mnt/raid0/llm/models/gemma-3-1b-it-Q8_0.gguf \
  --swa-full --draft-max 4 -t 96 -n 10 -c 2048 \
  --prompt "Hello"

# Baseline (no spec decode) works fine
/mnt/raid0/llm/llama.cpp/build/bin/llama-cli \
  -m /mnt/raid0/llm/lmstudio/models/lmstudio-community/gemma-3-27B-it-qat-GGUF/gemma-3-27B-it-QAT-Q4_0.gguf \
  -t 96 -n 50 -c 2048 \
  --prompt "Hello"
```

## Possible Solutions

### Option 1: Fix KV Cache Slot Management for ISWA + Spec Decode

The `slot_info` vector assignment in `initialize_decode_context` fails when target and draft have different ISWA configurations. Need to handle the case where:
- Draft model has N SWA layers, M non-SWA layers
- Target model has P SWA layers, Q non-SWA layers
- The cache slots need to map correctly between them

### Option 2: Require Matching SWA Configuration

Add validation that draft and target models have compatible SWA configurations:
- Same `sliding_window` size
- Same `n_swa` layer count
- Same interleaving pattern

If incompatible, fail early with a clear error message.

### Option 3: Disable SWA for Speculative Decoding

When `--swa-full` is used with speculative decoding, ensure BOTH models use the full-size cache mode consistently. Currently `--swa-full` seems to apply to one model but not coordinate between draft and target.

## Additional Context

### Vocab Mismatch (Secondary Issue)

There's also a vocab mismatch (1B=262,144 vs 27B=262,208 tokens), but this is NOT the crash cause - the SWA crash happens before any tokens are generated. The vocab issue would only matter if spec decode actually ran.

### Why This Matters

Gemma-3 is a popular model family. Speculative decoding could provide 5-10x speedup for CPU inference. Currently users must use baseline (slow) mode only.

### External References

- Gemma-3 tech report describes the ISWA architecture
- Reddit reports of successful Gemma-3 spec decode suggest it's possible with the right llama.cpp version/flags
- HuggingFace GGUF maintainers note "b5554+" needed for SWA support

## Acceptance Criteria

1. `llama-speculative` runs without crashing on Gemma-3 models
2. Speculative decoding produces correct output (matches baseline)
3. Performance improvement is measurable (target: >2x speedup)
4. If incompatible configurations exist, fail with clear error message

## Files Modified (Documentation)

These files were updated to document the issue:
- `orchestration/model_registry.yaml` - `draft_gemma3` marked deprecated
- `docs/reference/models/QUIRKS.md` - Gemma-3 SWA section added
- `CLAUDE.md` - Draft-target validation workflow added
- `scripts/utils/check_draft_compatibility.py` - New validation script

## Contact

This handoff created: 2026-01-09
llama.cpp fork: https://github.com/pestopoppa/llama.cpp
