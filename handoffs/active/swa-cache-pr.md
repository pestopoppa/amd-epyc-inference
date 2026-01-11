# SWA Cache Optimization PR - Handoff

## Status: CLOSED - No Functional Improvement

**PR #18727:** https://github.com/ggml-org/llama.cpp/pull/18727 (closed)
**PR #18720:** https://github.com/ggml-org/llama.cpp/pull/18720 (closed)

## Outcome

**Both PRs closed.** ggerganov correctly pointed out that our optimization was functionally equivalent to existing code. No actual improvement was provided.

## Post-Mortem: What Went Wrong

### What We Thought We Were Fixing

We proposed using `pos_batch_min` (the first position in the incoming batch) instead of `cells.seq_pos_max(seq_id_cell) + 1` for SWA cache reusability checks, claiming this was "forward-looking" optimization.

### Why The Existing Code Was Already Correct

Positions are assigned contiguously in llama.cpp. From `llama-batch.cpp:100`:

```cpp
p0[s] = memory->seq_pos_max(s) + 1;
```

This means `seq_pos_max + 1` is **always** equal to `pos_batch_min` by construction. Our change computed the same value in a different way without any functional improvement.

### The Invariant We Missed

```
seq_pos_max(seq_id) + 1 == pos_batch_min
```

This is an invariant because:
1. Current sequence max = highest position in cache
2. New batch starts at max + 1
3. First position in batch = pos_batch_min

### Timeline

1. **PR #18720**: Used `pos_batch_max` - correctly rejected (could evict cells needed by earlier tokens)
2. **PR #18727**: Used `pos_batch_min` - correct, but equivalent to existing logic

### Lessons Learned

| Mistake | Fix for Future |
|---------|----------------|
| Didn't verify existing behavior before "fixing" | Trace existing code path fully before proposing changes |
| Assumed code was wrong without evidence | Test existing code to confirm bug exists |
| Didn't ask "why would these values differ?" | Question invariant assumptions explicitly |

## Original Summary (For Reference)

~~Optimizes SWA cache slot reuse by checking cell reusability against the incoming batch's minimum position.~~

**Reality**: The existing code using `seq_pos_max + 1` already achieves this. Our change was a no-op.

## Test Results

| Test | Result |
|------|--------|
| Gemma-3-12B + 1B draft, 1504 tokens | ✓ |
| SWA cache bounded at 1536 cells | ✓ |
| ~50% acceptance rate | ✓ |
| Output quality | ✓ Coherent |

## Files Changed

- `src/llama-kv-cache.cpp`: Modified `find_slot()` to use `pos_batch_min`

## Local Branches

- `swa-cache-fix-v2`: Clean branch from upstream master with the fix (pushed to fork)
- `mtp-branch`: Original branch with full history (has workflow file issues)

## Action Required

Post this response on PR #18727 and close it:

```
You're right - `seq_pos_max + 1` equals `pos_batch_min` by construction since positions are assigned contiguously. My change computes the same value differently without functional improvement.

Thanks for the explanation. Closing.
```

## Resolution

- [x] Analyzed ggerganov's feedback
- [x] Documented post-mortem
- [x] Post response and close PR #18727 (2026-01-11)
