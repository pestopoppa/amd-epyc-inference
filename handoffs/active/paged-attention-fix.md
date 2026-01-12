# KV Cache Sizing Implementation - PR SUBMITTED

## Status: AWAITING REVIEWER RESPONSE

**Created**: 2026-01-11
**Updated**: 2026-01-12
**PR**: https://github.com/ggml-org/llama.cpp/pull/18747
**Branch**: `feature/paged-attention` on pestopoppa/llama.cpp

---

## Session Summary (2026-01-12)

### Completed
1. **Refactored codebase**: Removed ~1,150 lines of unnecessary/duplicate code
2. **Cleaned PR branch**: Reverted unrelated changes (repack.cpp OpenMP, lookahead/lookup fixes, BRANCH_RULES.md)
3. **Updated PR honestly**:
   - New title: `ggml, llama : add KV cache size limiting and block tracking infrastructure`
   - Honest description acknowledging original claims were misleading
   - Clear "What This Does NOT Do" section
   - Foundation for future PagedAttention framing
4. **CI tests**: 52/53 passed (98%) - only `test-tokenizers-ggml-vocabs` failed (Git LFS environment issue)
5. **Pushed cleaned code**: Commit `d98013d15`

### Awaiting
- User to post manual response to reviewers (ngxson, JohannesGaessler, qnixsynapse)
- Reviewer feedback on honest resubmission

---

## What The PR Actually Provides

1. **KV cache size limiting** (`--kv-cache-tokens N`): Allocate less KV cache upfront (trades memory for context length)
2. **Block tracking infrastructure**: Foundation for future PagedAttention (currently informational only)
3. **Demand-paged mmap** (`--kv-cache-demand-paged`): OS-level lazy physical page allocation on Linux/macOS

### What It Does NOT Provide
- No per-token memory reduction
- No memory sharing between sequences
- No dynamic allocation/deallocation
- The "95% savings" was comparing different context sizes (misleading)

---

## Key Files (Final State)

| File | Lines | Purpose |
|------|-------|---------|
| `src/llama-kv-cache.cpp` | +359 | KV size limiting logic |
| `src/llama-kv-block.h` | +263 | Block tracking structures |
| `tests/test-kv-block.cpp` | +444 | Unit tests (17 passing) |
| `ggml/src/ggml-backend.cpp` | +88 | mmap buffer for demand-paging |
| `src/llama-graph.cpp` | +58 | Graph integration |
| `common/arg.cpp` | +32 | CLI flags |

**Total: ~1,365 lines** (15 files)

---

## Reviewer Criticism Acknowledged

**ngxson** (correct):
> if 6400 tokens takes 2000MB, then 131072 tokens should take: 131072 / 6400 * 2000 = 40960MB
> where is improvement in memory?

**Response**: There is no per-token improvement. The "savings" came from allocating less context. PR description now honestly acknowledges this.

**JohannesGaessler**:
> The code quality of machine generated code is not high enough where the saved effort for the initial implementation outweighs the increase in the maintenance burden.

**Response**: Code refactored from ~2,100 to ~1,365 lines. Trust may still be broken.

---

## Next Steps

1. **Wait for reviewer response** to honest resubmission
2. **If accepted**: Merge PR
3. **If rejected**: Close PR, extract demand-paged mmap as smaller PR
4. **Future work**: True PagedAttention requires ggml sparse tensor allocation

---

## Resume Commands

```bash
# Check PR status
gh pr view 18747 --json state,reviews,comments

# If changes requested
cd /mnt/raid0/llm/llama.cpp-experimental
git checkout feature/paged-attention
# Make changes, commit, push to fork
git push fork feature/paged-attention
```

---

## References

- [PR #18747](https://github.com/ggml-org/llama.cpp/pull/18747)
- [vLLM PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- Plan file: `/home/daniele/.claude/plans/playful-plotting-creek.md`

---

## Related: Parallel Tensor Repack PR (#18239)

**PR:** https://github.com/ggml-org/llama.cpp/pull/18239
**Branch:** `parallel-repack` on pestopoppa/llama.cpp
**Status:** Test infrastructure added (2026-01-12)

Added `test-repack-parallel.cpp` to address maintainer concern about testing infrastructure. Test validates byte-identical output between sequential and parallel repacking across all quantization types (Q4_0, Q4_K, Q2_K, IQ4_NL).

See progress log `2026-01-12.md` for details.
