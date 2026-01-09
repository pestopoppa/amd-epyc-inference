# Handoff: Fix Gemma-3 SWA + Speculative Decoding Crash in llama.cpp

## Status: RESOLVED - MTP Branch Specific

**Resolution Date:** 2026-01-09

## Summary

~~Gemma-3 models crash when used with speculative decoding in llama.cpp~~

**CORRECTION:** The crash is **MTP-branch-specific**, not an upstream issue. Upstream llama.cpp master (build 7684+) handles Gemma-3 + speculative decoding correctly.

### Key Findings

| Branch | Gemma-3 + Spec Decode | Notes |
|--------|----------------------|-------|
| **Upstream master (b7684)** | ✅ Works | 20-66% acceptance rate |
| **MTP branch (b7508)** | ❌ Crashed | Fixed with commits below |
| **MTP branch (fixed)** | ✅ Works | 66.7% acceptance rate |

### Root Cause

The MTP (Multi-Token Prediction) branch added code in `llama-context.cpp` that assumes `llama_kv_cache_context*` type for memory contexts. For ISWA models (Gemma-3), the actual type is `llama_kv_cache_iswa_context*`, causing undefined behavior.

```cpp
// MTP branch code (problematic):
kvd->last_main_model_sinfos =
    static_cast<llama_kv_cache_context *>(mctx.get())->get_sinfos();
// ^^^^^ WRONG TYPE for ISWA models!
```

This code **does not exist in upstream master**, so upstream is unaffected.

## Fix Applied (MTP Branch Only)

Three commits on `mtp-branch`:

```
9eeaaa306 llama : fix speculative decoding crash with ISWA models
8037ef743 kv-cache : add unified slot mode for ISWA with --swa-full
c7f834f4c speculative : warn when ISWA models used without --swa-full
```

### Primary Fix (llama-context.cpp)

```cpp
// Check for ISWA before the cast
if (!model.hparams.is_swa_any()) {
    kvd->last_main_model_sinfos =
        static_cast<llama_kv_cache_context *>(mctx.get())->get_sinfos();
}
```

### When MTP Merges Upstream

When MTP code is merged to upstream llama.cpp, these fixes should be included. Otherwise, the ISWA crash will be introduced to upstream.

## Test Results (After Fix)

```bash
# MTP branch with fix
./build/bin/llama-speculative \
  -m gemma-3-27B-it-QAT-Q4_0.gguf \
  -md gemma-3-1b-it-Q8_0.gguf \
  --draft-max 4 -t 96 -n 10

# Output:
n_drafted = 12
n_accept  = 8
accept    = 66.667%
# No crash!
```

## Original Investigation (For Reference)

### Environment When Issue Was Discovered

- **llama.cpp build**: 7508 (commit d10a5a4a5) - **MTP BRANCH**
- **Models tested**:
  - Target: `gemma-3-27B-it-QAT-Q4_0.gguf` (262,208 vocab)
  - Draft: `gemma-3-1b-it-Q8_0.gguf` (262,144 vocab)
- **System**: AMD EPYC 9655, 1.13TB RAM, CPU-only inference

### Original Crash Traces

**Crash 1 (without --swa-full):**
```
#7  std::vector<llama_kv_cache::slot_info>::operator=()
#8  llama_context::initialize_decode_context(llama_batch const&, bool)
terminate called after throwing std::bad_alloc
```

**Crash 2 (with --swa-full):**
```
realloc(): invalid pointer
```

## Lessons Learned

1. **Always verify bugs on upstream master first** before developing fixes
2. **Check pending PRs** that might already address the issue
3. **Document which branch** the bug was found on
4. **The commit ID in the handoff (d10a5a4a5)** was from MTP branch, not upstream

## Files Modified

### On MTP Branch (fix applied):
- `src/llama-context.cpp` - ISWA type check before cast
- `src/llama-kv-cache-iswa.cpp` - Unified slot mode (defense-in-depth)
- `examples/speculative/speculative.cpp` - Warning message

### Documentation updated:
- `orchestration/model_registry.yaml` - `draft_gemma3` now documented as working
- This handoff - Updated with resolution

## No Upstream PR Needed

Since the bug doesn't exist in upstream, no PR is required. The fix should be included when MTP code eventually merges.

---

**Created:** 2026-01-09
**Resolved:** 2026-01-09
**Category:** MTP Branch Bug Fix
