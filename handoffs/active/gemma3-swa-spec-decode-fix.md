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

## Phase 2: Forward-Looking SWA Masking (Memory Optimization)

**Status:** ✅ COMPLETE (2026-01-09)

### Problem

With `--swa-full`, the SWA cache mirrors the base cache size (131072 cells = 10240 MiB for 27B model). This wastes memory when the sliding window only needs a fraction of that.

### Solution

Modified `find_slot()` in `src/llama-kv-cache.cpp` to use **forward-looking SWA masking**:

Instead of checking if old cells are masked relative to the **cached** max position, check against the **new batch's** max position. This allows reusing cells that will be outside the attention window *after* the batch is inserted.

### Implementation

**File:** `src/llama-kv-cache.cpp` (in `find_slot()` function)

**Change 1:** Compute batch max position before the while loop:
```cpp
// For SWA caches: compute max position in this batch for sequence s
// This is used to determine which old cells are outside the future attention window
// (forward-looking: cells masked AFTER batch insertion can be reused)
llama_pos pos_batch_max = 0;
if (n_swa > 0) {
    pos_batch_max = ubatch.pos[s * n_tokens + n_tokens - 1];
}
```

**Change 2:** Use forward-looking masking in the SWA check:
```cpp
// SWA mask - check if cell position is outside attention window
if (n_swa > 0) {
    // For SWA caches: use batch's max position (forward-looking)
    // Allows reusing cells that will be masked after batch insertion
    if (is_masked_swa(pos_cell, pos_batch_max + 1)) {
        can_use = true;
    }
} else {
    // Non-SWA: use cached sequence max (original logic)
    if (is_masked_swa(pos_cell, cells.seq_pos_max(seq_id_cell) + 1)) {
        can_use = true;
    }
}
```

### Results

| Metric | Without Fix | With Fix |
|--------|-------------|----------|
| `--swa-full` required | Yes | No |
| SWA cache size (27B) | 10240 MiB | 624 MiB |
| Memory reduction | — | **94%** |
| Crash | Yes (without flag) | No |
| Acceptance rate | N/A | 42-81% |

### Test Commands

```bash
# Without --swa-full (now works!)
/mnt/raid0/llm/llama.cpp/build/bin/llama-speculative \
  -m /mnt/raid0/llm/lmstudio/models/lmstudio-community/gemma-3-27B-it-qat-GGUF/gemma-3-27B-it-QAT-Q4_0.gguf \
  -md /mnt/raid0/llm/models/gemma-3-1b-it-Q8_0.gguf \
  --draft 4 -t 96 -n 100 \
  -p "Hello"

# Output:
# decoded  103 tokens in    8.399 seconds, speed:   12.264 t/s
# n_drafted = 152
# n_accept  = 64
# accept    = 42.105%
# SWA cache: 624 MiB (vs 10240 MiB with --swa-full)
```

### Upstream PR Status

**PR #18720:** https://github.com/ggml-org/llama.cpp/pull/18720 - **REOPENED**

**History:**
1. Initially submitted with incorrect framing about "enabling" spec decode without `--swa-full`
2. ggerganov pointed out that `--swa-full` was never required - SWA spec decode worked since PR #14131
3. PR was closed due to the misleading description
4. **Reopened** with corrected description focusing on forward-looking SWA masking as a memory/efficiency optimization

The optimization is still valid - it improves SWA cache cell reuse efficiency by allowing earlier reuse of cells that will be masked after batch completion.

---

## Phase 3: User-Specified SWA Context Size (RESEARCH ONLY)

**Status:** ✅ RESEARCH COMPLETE - NOT FOR PRODUCTION

### Summary

Explored adding `--ctx-swa N` flag to allow users to specify maximum SWA context size. Implementation was successful but deemed **not useful for production** due to:

1. **Marginal benefit**: Phase 2 already achieves 94% memory reduction
2. **Trade-offs**: Smaller SWA cache hurts acceptance rate (75% → 37.5%)
3. **Niche use case**: Most users benefit from default sizing

### Research Findings

| Flag | SWA Cache | Memory | Acceptance | Speed |
|------|-----------|--------|------------|-------|
| (none) | 1536 cells | 624 MiB | 75% | 14.5 t/s |
| `--ctx-swa 512` | 512 cells | 208 MiB | 37.5% | 9.7 t/s |

**Conclusion**: The 67% additional memory savings (624→208 MiB) comes at significant cost to acceptance rate. Phase 2's forward-looking masking provides the optimal default behavior.

### Why No PR

- Phase 2 PR (#18720) already provides major optimization
- Phase 3 adds complexity for niche use case
- Trade-offs not worth API surface increase

### Implementation Notes (for future reference)

If ever needed, the implementation requires:
- Add `n_ctx_swa` to `common_params`, `llama_context_params`, `llama_cparams`
- Add `--ctx-swa` argument in `arg.cpp`
- Add `kv_size_swa` parameter to `llama_kv_cache_iswa` constructor
- Modify sizing formula: `min(kv_size_swa, n_swa + n_ubatch)`

---

**Created:** 2026-01-09
**Phase 1 Resolved:** 2026-01-09
**Phase 2 Completed:** 2026-01-09
**Phase 2 PR Submitted:** 2026-01-09
**Phase 3 Research:** 2026-01-09
**Category:** MTP Branch Bug Fix + Memory Optimization
