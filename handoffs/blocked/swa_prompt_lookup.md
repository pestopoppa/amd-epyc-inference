# SWA + Prompt Lookup Fix

## Status: ✅ RESOLVED (PRs Submitted)
**Blocked by**: N/A - Resolved
**Created**: 2026-01-10
**Updated**: 2026-01-10
**Priority**: Low (workaround available: use `-c` flag)

---

## Resolution Summary

**This was NOT an SWA bug.** Two separate pre-existing bugs were discovered and fixed:

| PR | Fix | Status |
|----|-----|--------|
| [#18729](https://github.com/ggml-org/llama.cpp/pull/18729) | Batch init: use `llama_n_ctx(ctx)` instead of `params.n_ctx` | Submitted |
| [#18730](https://github.com/ggml-org/llama.cpp/pull/18730) | Lookahead: set `n_parallel` and `kv_unified` | Submitted (depends on #18729) |

**Local deployment works** - both fixes cherry-picked and verified.

---

## Bug 1: Batch Initialization (PR #18729)

### Problem

`llama-lookup` and `llama-lookahead` crash without explicit `-c` flag:

```
GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded") at common.cpp:1442
```

### Root Cause

```cpp
// lookup.cpp line 109 - BUG: params.n_ctx defaults to 0
llama_batch batch_tgt = llama_batch_init(params.n_ctx, 0, 1);

// speculative.cpp line 206-207 - CORRECT: uses actual runtime value
llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, n_seq_dft);
```

When user doesn't specify `-c`, `params.n_ctx = 0` (sentinel for GPU auto-fitting since PR #16653, Dec 2025). The internal context gets expanded to `n_ctx_train`, but `params.n_ctx` remains 0. This creates a batch of size 0, which crashes on the first `common_batch_add()`.

### Why This Only Appeared Now

**PR #16653 (Dec 2025)** changed the default `n_ctx` from model's training context to 0. This was a deliberate change for GPU VRAM auto-fitting, but it broke examples that assumed `params.n_ctx` would have a valid value.

### Fix

Replace `params.n_ctx` with `llama_n_ctx(ctx)`:

```cpp
// lookup.cpp line 109
- llama_batch batch_tgt = llama_batch_init(params.n_ctx, 0, 1);
+ llama_batch batch_tgt = llama_batch_init(llama_n_ctx(ctx), 0, 1);

// lookahead.cpp line 121
- llama_batch batch = llama_batch_init(params.n_ctx, 0, W + G + 1);
+ llama_batch batch = llama_batch_init(llama_n_ctx(ctx), 0, W + G + 1);
```

---

## Bug 2: Lookahead n_parallel / kv_unified (PR #18730)

### Problem

After fixing batch init, `llama-lookahead` still crashes:

```
llama_batch_allocr: error: invalid seq_id[0][1] = 1 >= 1
```

### Root Cause

```cpp
// common.cpp line 1399 - n_parallel controls n_seq_max
cparams.n_seq_max = params.n_parallel;  // defaults to 1

// Lookahead needs W + G + 1 = 31 sequences for Jacobi iteration
// But validation in llama-batch.cpp:62 rejects seq_id >= n_seq_max
```

**PR #14482 (July 2025)** changed sequence ID validation from `LLAMA_MAX_SEQ` (4096) to `n_seq_max` (derived from `n_parallel`). Lookahead was using coupled sequences but not setting `n_parallel`.

### Additional Issue

With coupled sequences, batch splitting requires unified KV cache mode:

```
split_equal: sequential split is not supported when there are coupled sequences
```

### Fix

Set required parameters before context initialization:

```cpp
// lookahead.cpp - add after W, N, G definitions (~line 52)
// lookahead requires W + G + 1 sequences for parallel Jacobi decoding
params.n_parallel = W + G + 1;

// unified KV cache is required for coupled sequences in batch splitting
params.kv_unified = true;
```

---

## Test Results (Local Deployment)

### llama-lookup (after PR #18729)

```bash
numactl --interleave=all ./build/bin/llama-lookup \
  -m /mnt/raid0/llm/lmstudio/models/.../gemma-3-1b-it-Q8_0.gguf \
  -p "Write a story about a cat:" -n 50 -t 96

# Result:
n_drafted    = 51
n_accept     = 22
accept       = 43.137%  # ✅ Working without -c flag
```

### llama-lookahead (after both PRs)

```bash
numactl --interleave=all ./build/bin/llama-lookahead \
  -m /mnt/raid0/llm/lmstudio/models/.../gemma-3-1b-it-Q8_0.gguf \
  -p "Write a story about a cat:" -n 50 -t 96

# Result:
n_accept     = 17
speed: 33.67 t/s  # ✅ Working without -c flag
```

---

## Relationship to PR #18727 (SWA Cache Optimization)

**None.** The original handoff incorrectly associated this bug with SWA models. The crash happens with ANY model when `-c` is not specified. SWA models were coincidentally being tested when the bug was discovered.

Our SWA cache optimization in PR #18727 only modifies `llama-kv-cache.cpp:find_slot()` and is unrelated to batch initialization.

---

## Git Archaeology (Claude-assisted)

| Event | PR | Date | Impact |
|-------|-----|------|--------|
| `n_ctx` default → 0 | [#16653](https://github.com/ggml-org/llama.cpp/pull/16653) | Dec 2025 | **Activated batch init bug** |
| seq_id validation tightened | [#14482](https://github.com/ggml-org/llama.cpp/pull/14482) | July 2025 | **Activated lookahead bug** |
| lookup.cpp created | Initial | 2023? | Bug dormant until Dec 2025 |
| lookahead.cpp created | Initial | 2023? | Bug dormant until July 2025 |

Both bugs were **dormant for 2+ years** until unrelated PRs changed defaults.

---

## Lessons Learned

1. **Test without `-c` flag**: Default values matter; test all binaries without explicit context size
2. **Check sentinel values**: `params.n_ctx = 0` is a valid sentinel, not an error
3. **Read git history**: Dormant bugs get activated by unrelated changes
4. **Compare to speculative.cpp**: It correctly uses `llama_n_batch(ctx)` instead of raw params

---

## Files for Future Reference

```
examples/lookup/lookup.cpp:109     - Fixed in PR #18729
examples/lookahead/lookahead.cpp   - Fixed in PRs #18729 + #18730
examples/speculative/speculative.cpp:206-207 - Reference implementation (correct)
common/common.cpp:1399             - Where n_parallel → n_seq_max happens
common/common.cpp:1442             - Crash location (batch assertion)
src/llama-batch.cpp:62             - seq_id validation that broke lookahead
```
