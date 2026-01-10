# SWA Cache Optimization PR - Handoff

## Status: PR Submitted, Response Posted

**New PR:** https://github.com/ggml-org/llama.cpp/pull/18727
**Previous PR:** https://github.com/ggml-org/llama.cpp/pull/18720 (closed)

## Summary

Optimizes SWA (Sliding Window Attention) cache slot reuse by checking cell reusability against the incoming batch's **minimum** position. This enables aggressive cache reclamation while maintaining mathematical exactness.

## Key Fix

Original PR #18720 used batch **max** position, which @ggerganov correctly noted could evict cells still needed by earlier tokens. The fix uses **min** position instead:

- Min-position token has the most demanding attention window (looks furthest back)
- If we preserve everything min needs, all other tokens have their context
- Mathematically exact - no approximation

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

## Next Steps

1. Monitor PR #18727 for reviewer feedback
2. ~~User to post informal response on #18720 redirecting to #18727~~ DONE
