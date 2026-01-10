# Hybrid Prompt Lookup + Speculative Decoding

## Status: PROPOSAL
**Created**: 2026-01-10
**Priority**: Medium (optimization opportunity)
**Type**: Feature proposal for llama.cpp upstream

## Summary

Combine prompt lookup (ngram matching) with draft model speculative decoding in a hybrid approach that tries lookup first (free), then falls back to draft model when lookup fails to find matches.

## Motivation

Currently llama.cpp has two separate speculation methods:
- **`llama-speculative`** - Uses a draft model to predict tokens
- **`llama-lookup`** - Uses ngram matching from the prompt

These are mutually exclusive. A hybrid approach could get the best of both:
- Lookup is essentially free (string matching, no model inference)
- Draft model handles novel content where lookup fails
- Combined approach should outperform either alone for mixed content

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Hybrid Speculation Loop                   │
├─────────────────────────────────────────────────────────────┤
│  1. Try ngram lookup (last N tokens → prompt match)         │
│     ├─ If match found: use prompt continuation as draft     │
│     └─ If no match: fall through to step 2                  │
│                                                             │
│  2. Generate draft tokens with draft model                  │
│     └─ Standard speculative decoding                        │
│                                                             │
│  3. Target model verifies all draft tokens in one pass      │
│     └─ Accept/reject as normal                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Lookup-first strategy**: Lookup is O(1) with a hash table, draft model inference is O(n). Always try lookup first.

2. **Configurable minimum match length**: Only use lookup if match is >= M tokens (e.g., M=3). Short matches may not be worth the overhead.

3. **Hybrid draft buffer**: When lookup finds partial match (e.g., 3 tokens) but we want K=8 drafts, fill remaining slots with draft model predictions.

4. **Corpus extension**: Beyond prompt, allow adding external corpus (code snippets, documentation) to the ngram index.

## Implementation Plan

### Phase 1: Core Hybrid Logic

**File**: `common/speculative.cpp` (or new `common/speculative-hybrid.cpp`)

```cpp
// Pseudocode for hybrid speculation
std::vector<llama_token> hybrid_draft(
    llama_context* ctx_target,
    llama_context* ctx_draft,  // May be nullptr if lookup-only
    ngram_cache* cache,
    int n_draft,
    int min_lookup_match
) {
    std::vector<llama_token> draft_tokens;

    // Step 1: Try ngram lookup
    auto lookup_result = cache->lookup(last_n_tokens, n_draft);
    if (lookup_result.size() >= min_lookup_match) {
        draft_tokens = lookup_result;

        // If lookup found fewer than n_draft, optionally fill with draft model
        if (ctx_draft && draft_tokens.size() < n_draft) {
            auto draft_fill = draft_model_generate(
                ctx_draft,
                n_draft - draft_tokens.size()
            );
            draft_tokens.insert(draft_tokens.end(),
                               draft_fill.begin(), draft_fill.end());
        }
    } else if (ctx_draft) {
        // Step 2: Fall back to draft model
        draft_tokens = draft_model_generate(ctx_draft, n_draft);
    }

    return draft_tokens;
}
```

### Phase 2: New Binary or Flag

**Option A**: New binary `llama-speculative-hybrid`
- Pro: Clean separation, no breaking changes
- Con: Code duplication

**Option B**: Add `--lookup-fallback` flag to `llama-speculative`
- Pro: Single binary, composable
- Con: More complex argument handling

**Recommended**: Option B with flags:
```bash
llama-speculative \
  -m target.gguf \
  -md draft.gguf \
  --draft-max 8 \
  --lookup-ngram 3 \          # Enable ngram lookup with n=3
  --lookup-fallback           # Fall back to draft when lookup fails
  --lookup-min-match 2        # Minimum tokens from lookup to use it
```

### Phase 3: Server Integration

Add to `llama-server`:
```
--speculative-lookup     Enable hybrid lookup+draft speculation
--lookup-ngram N         Ngram size for lookup (default: 3)
```

## Expected Performance

### Use Cases Where Hybrid Wins

| Content Type | Lookup Hit Rate | Hybrid Benefit |
|--------------|-----------------|----------------|
| Code editing (refactoring) | High (60-80%) | Significant |
| Document Q&A with quotes | Medium (30-50%) | Moderate |
| Summarization | Medium (40-60%) | Moderate |
| Creative writing | Low (<10%) | Minimal |
| Novel code generation | Low (<20%) | Minimal |

### Benchmark Prediction

Based on our existing benchmarks:
- Prompt lookup alone: 8.6-12.7x speedup (when content is repetitive)
- Spec decode alone: 5.9-11x speedup (consistent)
- Hybrid (predicted): Should match or exceed the better of the two for any given prompt

**Key insight**: Lookup wins big when it hits, spec decode provides consistent baseline. Hybrid should never be slower than spec decode alone.

## Files to Modify (llama.cpp)

| File | Changes |
|------|---------|
| `common/speculative.cpp` | Add hybrid drafting logic |
| `common/speculative.h` | Add hybrid config struct |
| `common/ngram-cache.cpp` | Ensure compatible with speculation loop |
| `tools/speculative/speculative.cpp` | Add --lookup-* flags |
| `tools/server/server.cpp` | Add server support |
| `tools/server/README.md` | Document new flags |

## Testing Plan

1. **Correctness**: Verify output matches target-only generation
2. **Performance regression**: Hybrid should never be slower than spec-only
3. **Edge cases**:
   - Empty prompt (no lookup possible)
   - Prompt shorter than ngram size
   - Draft model unavailable (lookup-only mode)
   - Lookup-only vs hybrid vs spec-only comparison

## Prior Art

- [Discussion #4235](https://github.com/ggerganov/llama.cpp/discussions/4235) - N-gram cache API proposal
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - Has both methods but separate
- [HuggingFace prompt lookup](https://huggingface.co/blog/assisted-generation) - Original concept

## Open Questions

1. **Should lookup results be weighted by recency?** Tokens from recent prompt sections may be more relevant.

2. **Can we combine draft model output with lookup?** E.g., use lookup for first 3 tokens, draft model for next 5.

3. **How to handle KV cache with hybrid drafts?** Lookup drafts may have different attention patterns.

4. **Should lookup index be persistent across requests?** Could build up useful corpus over time.

## Success Criteria

- [ ] Hybrid mode matches or beats spec-only on all benchmarks
- [ ] Hybrid mode matches or beats lookup-only on repetitive content
- [ ] No correctness regressions (output identical to target-only)
- [ ] Clean integration with existing llama.cpp architecture
- [ ] Accepted upstream or maintained as local patch

## Local Testing Commands

Once implemented, test with:
```bash
# Baseline spec decode
./llama-speculative -m Qwen2.5-Coder-32B.gguf -md Qwen2.5-0.5B.gguf \
  --draft-max 8 -f code_edit_prompt.txt

# Hybrid mode
./llama-speculative -m Qwen2.5-Coder-32B.gguf -md Qwen2.5-0.5B.gguf \
  --draft-max 8 --lookup-ngram 3 --lookup-fallback -f code_edit_prompt.txt

# Compare t/s and acceptance rates
```

## Related

- `handoffs/blocked/swa_prompt_lookup.md` - Prompt lookup fixes for SWA models
- `docs/reference/models/QUIRKS.md` - Model-specific lookup/spec issues
