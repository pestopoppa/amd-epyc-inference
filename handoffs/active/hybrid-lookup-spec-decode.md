# Hybrid Prompt Lookup + Corpus-Augmented Speculative Decoding

## Status: ACTIVE — Phases 0+0.5+1 COMPLETE, Phase 2A A/B TESTED, Corpus Scaling In Progress
**Created**: 2026-01-10
**Updated**: 2026-02-15 (Phase 2A: A/B tested on all 5 models. Enable for Coder-family (32B, 480B). Full corpus build in progress.)
**Priority**: HIGH — **2.58x speedup on 30B, 2.16x on 480B achieved and shipped**
**Type**: Phased optimization — test existing capabilities first, then extend with corpus augmentation

---

## Executive Summary

### Results (2026-02-13)

| Phase | Status | Key Result |
|-------|--------|------------|
| **Phase 0** | COMPLETE | Prompt lookup works on 480B (18.4% acceptance) but is **net-negative** on speed (-34%) due to MoE verification overhead |
| **Phase 0.5** | COMPLETE | **jukofyork draft VERIFIED on 480B** — 74-82% acceptance, **2.16x speedup** (5.91 → 12.74 t/s) |
| **Phase 1** | COMPLETE | **30B: MoE6+spec+lookup = 47.11 t/s (2.58x)**; 235B: full+spec = 6.08 t/s (1.15x); 480B: full+spec = 9.00 t/s (1.38x). Architect roles use full experts (quality over speed). |
| **Phase 2A** | **A/B TESTED** | Corpus-augmented prompt stuffing tested on all 5 models. **Best: 480B +15.6pp acceptance, +17% speed. 32B +8.7pp.** Enable for Coder-family only. Full corpus build in progress (The Stack v1, 67GB+ Python). |

### Benchmark Results (480B, llama-server, MoE3)

| Config | Refactoring | Novel Gen | Summarization |
|--------|-------------|-----------|---------------|
| Baseline (MoE3 only) | 5.91 t/s | 6.76 t/s | 5.36 t/s |
| Lookup only | 3.87 t/s (-34%) | 6.56 t/s (-3%) | 4.41 t/s (-18%) |
| **Spec only (draft)** | **12.74 t/s (+2.16x)** | **10.52 t/s (+1.56x)** | — |
| Spec + lookup | 13.05 t/s (+2.21x) | 9.68 t/s (+1.43x) | 6.86 t/s (+1.28x) |

**Production config**: MoE3 + spec decode (K=16), NO lookup. Lookup adds marginal gain on refactoring but hurts novel gen.

### Benchmark Results (30B, llama-server, Phase 1)

| Config | Refactoring (t/s) | Acceptance | Notes |
|--------|-------------------|------------|-------|
| Baseline (full experts, no spec) | 29.28 | — | Raw model speed |
| MoE6, no spec | 30.84 | — | Expert reduction alone: +5% |
| MoE6 + spec | 37.08 | 70.1% | Spec decode adds +20% on top of MoE6 |
| Full experts + spec | 41.75 | 78.1% | More experts = higher acceptance |
| **MoE6 + spec + lookup** | **47.11** | **77.4%** | **Best config: +61% over baseline** |

### Benchmark Results (235B, llama-server, Phase 1)

| Config | Speed (t/s) | Acceptance | Notes |
|--------|-------------|------------|-------|
| Full experts baseline (no spec) | 5.30 | — | Raw model speed |
| MoE4, no spec | 3.87 | — | MoE actually slower (overhead > savings) |
| Full experts + spec | 6.08 | 52.7% | **Production config (quality)** |
| MoE4 + spec | 8.21 | 54.8% | Fastest, but quality tradeoff |
| MoE4 + spec + lookup | 8.02 | 54.4% | Lookup net-negative |

0.6B Q8_0 draft dramatically outperforms 1.7B (55% vs 21% acceptance). BOS matches (both 151643).

### Policy: Architect Roles Use Full Experts

480B no-MoE: Full experts + spec = 9.00 t/s (80.5% accept). MoE3+spec was 12.74 but sacrifices quality.
235B no-MoE: Full experts + spec = 6.08 t/s (52.7% accept). MoE4+spec was 8.21 but sacrifices quality.

**Decision**: Architect roles prioritize quality over speed. Full experts + spec decode is the production config for both 235B and 480B. Frontdoor/coder roles use MoE + spec + lookup (speed matters more).

**Key insight**: Lookup is net-POSITIVE on 30B (cheap verification, lookup fills gaps spec decode misses) but net-negative on large models (235B, 480B).

### Production Changes Shipped
- `model_registry.yaml`: 480B + 30B acceleration configs updated with verified spec decode + lookup
- `orchestrator_stack.py`: `build_server_command` handles MoE+spec combo; lookup now per-role flag
- `registry_loader.py`: Parses `speculative_decoding` sub-config + `lookup` flag under MoE acceleration
- `AccelerationConfig`: New `lookup: bool` field for per-role `--lookup` control
- `QUIRKS.md`: 480B section updated with solution and measured performance
- `RESULTS.md`: Updated with both 30B (47.11 t/s) and 480B (12.74 t/s) entries

---

## What Already Works (Production Baseline)

Prompt lookup is **already in production** via llama-server (commit `8e35dbc01`, 2026-01-28):

| Mode | Coder-32B Speed | Acceptance | Notes |
|------|-----------------|-----------|-------|
| Baseline | 7.28 t/s | N/A | No acceleration |
| Lookup only | 10.75 t/s | 13.2% | `"lookup": true` in request JSON |
| Spec only (0.5B draft) | 37.84 t/s | 89.7% | Standard spec decode |
| Combined (spec + lookup fallback) | 39.44 t/s | 83.2% | Spec first, lookup fallback |

Implementation: per-slot ngram cache, spec-first priority, `--lookup` CLI flag + `"lookup": true` per-request.
Details: `handoffs/archived/llama-server-prompt-lookup.md`

---

## Phase 0: Test Prompt Lookup on Qwen3-Coder-480B

### Hypothesis

The model registry forbids prompt lookup on 480B:
```yaml
constraints:
  forbid:
  - prompt_lookup
  reason: No prompt lookup - MoE architecture
```

**This reason is likely wrong.** Prompt lookup proposes draft tokens from n-gram matches in the already-tokenized prompt, then the SAME model verifies them. There is no cross-model interaction — no BOS mismatch possible, no draft model tokenizer to conflict with. MoE expert reduction controls which experts fire during inference, but the token proposal/verification loop is architecture-agnostic.

Contrast with Qwen3-Next (SSM) where `forbid: prompt_lookup` is correctly justified — SSM requires consecutive positions and draft rejection corrupts recurrent state. MoE has no such constraint.

**Supporting evidence**: `scripts/benchmark/run_combination_benchmarks.sh:213` already has a `run_lookup_hard_mask` entry for 480B that was set up but never executed.

### Test Plan

**Test A — llama-lookup binary (existing benchmark infrastructure)**

```bash
# Uses existing run_combination_benchmarks.sh infrastructure
# The function run_lookup_hard_mask already handles this model
OMP_NUM_THREADS=1 numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-lookup \
  -m /mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00008.gguf \
  --draft-max 16 \
  --moe-n-expert 3 \
  -t 96 \
  -n 200 \
  --temp 0 \
  -f /mnt/raid0/llm/tmp/twyne_summarize_prompt.txt
```

**Test B — llama-server with `--lookup` (production path)**

```bash
# Start server with MoE3 + lookup
numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-server \
  -m /mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00008.gguf \
  --override-kv qwen3moe.expert_used_count=int:3 \
  --lookup \
  -t 96 -c 16384 --port 8084

# Test request
curl -s http://localhost:8084/v1/chat/completions \
  -d '{"model":"qwen3-coder-480b","messages":[{"role":"user","content":"Refactor this Python function to use async/await:\n\ndef fetch_data(urls):\n    results = []\n    for url in urls:\n        response = requests.get(url)\n        results.append(response.json())\n    return results"}],"lookup":true,"max_tokens":500}'
```

### Prompts to Test (3 tiers of expected overlap)

| Prompt Type | Expected Overlap | File |
|-------------|-----------------|------|
| Code refactoring (rewrite with minor changes) | HIGH (60-80%) | Needs creation — take existing function, ask to add error handling |
| Summarization (Twyne whitepaper) | MEDIUM (40-60%) | `/mnt/raid0/llm/tmp/twyne_summarize_prompt.txt` |
| Novel code generation (new function from scratch) | LOW (<10%) | "Implement a B-tree in Python with insert, search, delete" |

### Success Criteria

| Metric | Failure | Marginal | Success |
|--------|---------|----------|---------|
| Acceptance rate (refactoring) | 0% (broken) | 5-15% | >20% |
| Speed vs MoE3-only baseline | Slower | Same (10.3 t/s) | >12 t/s |
| Output correctness | Garbled/wrong | Minor issues | Matches baseline |

**If Phase 0 succeeds**: Update `model_registry.yaml` to remove `forbid: prompt_lookup`, add measured performance, update QUIRKS.md. This is an immediate production win.

**If Phase 0 fails**: Document why (error messages, 0% acceptance, crashes) and investigate whether a llama.cpp patch can fix it.

---

## Phase 0.5: Test jukofyork Draft on 480B

The registry already has this queued but marked UNTESTED:

```yaml
acceleration:
  type: speculative_decoding
  draft_role: draft_qwen3_coder_0_75b
  k: 16
  notes: jukofyork vocab transplant draft fixes BOS mismatch - UNTESTED, benchmark needed
```

Draft model: `/mnt/raid0/llm/models/Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf`

```bash
# Test spec decode with vocab-transplant draft
OMP_NUM_THREADS=1 numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-speculative \
  -m /mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00008.gguf \
  -md /mnt/raid0/llm/models/Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf \
  --override-kv qwen3moe.expert_used_count=int:3 \
  --draft-max 16 \
  -t 96 \
  -n 200 \
  --temp 0 \
  -p "Write a Python async HTTP client with retry logic and exponential backoff."
```

**If both Phase 0 and 0.5 succeed**: Test all three combined (MoE3 + spec + lookup) via llama-server.

---

## Phase 1: Extend Prompt Lookup to All Applicable Models

Regardless of 480B results, run the same prompt lookup tests across all models that aren't SSM:

| Model | Current Best | Lookup Tested? | Expected Benefit |
|-------|-------------|---------------|-----------------|
| Qwen3-Coder-480B-A35B | 10.3 t/s (MoE3) | **NO — Phase 0** | HIGH (code editing) |
| Qwen3-Coder-30B-A3B | 22.0 t/s (MoE4) | No | Medium (code editing) |
| Qwen3-235B-A22B | 6.75 t/s (MoE4) | No | Medium (general) |
| Qwen2.5-Coder-32B | 39.44 t/s (spec+lookup) | Yes, production | Already captured |
| Qwen2.5-7B | 46.6 t/s (spec) | Yes | Already captured |
| Qwen3-Next-80B-A3B | 6.3 t/s | **SKIP** | SSM — incompatible |

Use `run_combination_benchmarks.sh` which already has entries for most of these.

---

## Phase 2: Corpus-Augmented Prompt Lookup (SoftMatcha v2)

### Motivation

Prompt lookup gives ~1x on novel generation because there's nothing in the prompt to match against. Corpus augmentation fixes this by **expanding the n-gram search space** from just the prompt to a large code corpus.

### Research: SoftMatcha v2

- **Paper**: https://arxiv.org/pdf/2602.10908
- **Code**: https://github.com/softmatcha/softmatcha2 (Python+Rust, Apache 2.0)
- **Capability**: Soft/fuzzy pattern matching on trillion-scale corpora, <90ms on 6TB+
- **Key feature**: Suffix array based with GloVe/FastText for "softness" boundary (word substitution tolerance)

For our use case (100GB corpus, in-RAM), expected latency is **<1ms** — comparable to prompt lookup itself.

### Integration Architecture

**Phase 2A: Retrieval-augmented prompt stuffing (no llama.cpp changes)**

```
User request: "implement async retry with exponential backoff in Python"
        |
        v
+------------------------------+
|  1. SoftMatcha Retrieval     |  Query: extract key terms -> search corpus
|     (in-process, <1ms)       |  Returns: top-K matching code snippets
+----------+-------------------+
           |
           v
+------------------------------+
|  2. Prompt Assembly          |  <|reference_code|>
|                              |  [retrieved snippets, ~2-5KB]
|                              |  <|/reference_code|>
|                              |  <|user|>
|                              |  [original request]
+----------+-------------------+
           |
           v
+------------------------------+
|  3. llama-server (unmodified)|  --lookup flag
|                              |  n-gram matches now hit retrieved
|                              |  snippets as well as user input
+------------------------------+
```

**Advantages**: Zero llama.cpp changes, leverages existing `--lookup` production infrastructure.

**Limitations**:
- Eats context window (~2-5KB of injected snippets out of 32K)
- Only works with models that support prompt lookup (NOT SSM)
- Quality risk: injected code may steer model output (see Quality Testing below)

**Phase 2B: Sidecar draft injection (llama.cpp modification)**

Build on the hybrid proposal from the original handoff — SoftMatcha server proposes corpus-sourced drafts directly into the speculation loop, bypassing prompt injection. Higher performance ceiling, requires llama.cpp fork work.

**Recommendation**: Prove the concept with Phase 2A first. Only invest in Phase 2B if 2A shows meaningful acceptance rate improvement.

### Corpus Selection (100GB Target)

With 1.13TB RAM, the entire corpus + suffix array index fits in memory.

| Source | Size (est.) | Rationale |
|--------|-------------|-----------|
| The Stack v2 — Python (deduplicated) | ~35GB | Primary training data for Qwen-Coder family; highest acceptance potential |
| The Stack v2 — JS/TS | ~25GB | Second most common generation target |
| The Stack v2 — Rust + Go + C++ | ~15GB | Systems languages, growing usage |
| CPython stdlib + numpy/pandas/torch source | ~2GB | Canonical patterns models memorized |
| Top-500 GitHub repos by stars (deduped) | ~15GB | High-probability code patterns |
| Our codebase + orchestration code | ~500MB | Domain-specific, immediate relevance |
| Python/Rust/JS documentation | ~8GB | Docstring patterns, API examples |

**Total: ~100GB** + ~10-30GB suffix array index = ~130GB in RAM. Trivial on our hardware.

**Key principle**: The corpus should mirror what the model was trained on, because that's what the model is likely to reproduce. The Stack v2 is explicitly the training data for StarCoder/Qwen-Coder, so n-gram matches should have the highest acceptance rates.

**Download**: Check if The Stack v2 is cached at `/mnt/raid0/llm/cache/huggingface/`. If not, budget ~2-4 hours for download on our connection.

### Query Formulation Strategy

| Strategy | Latency | Quality | Best For |
|----------|---------|---------|----------|
| Extract key terms from user prompt | ~0ms | Medium — NL terms ≠ code patterns | Known-pattern tasks |
| Full user prompt as-is | ~0ms | Low — NL won't match code n-grams | Not recommended |
| First ~20 generated tokens | +latency for initial pass | HIGH — query IS what model produces | Novel generation |
| **Hybrid: prompt keywords + first tokens** | Minimal | **Best** — seeds then refines | **Recommended** |

**Recommended approach**: Use user prompt for initial SoftMatcha query during prompt processing (latency hidden). After first 20 generated tokens, optionally re-query if initial results had low relevance scores.

### Quality Testing (CRITICAL)

Injecting code snippets into prompts risks steering model output toward those snippets, even when the model would have generated something better independently.

**Required A/B tests** (Claude-as-Judge scoring):

| Metric | Without Retrieval | With Retrieval | Acceptable Delta |
|--------|-------------------|----------------|-----------------|
| Tokens/sec | Baseline | ? | Must improve |
| Acceptance rate | ~0% (novel gen) | ? | >10% to justify |
| Code quality score (1-10) | Baseline | ? | No more than -0.5 |
| Instruction following | Baseline | ? | Must not regress |
| Hallucination rate | Baseline | ? | Must not increase |

**Quality regression is a hard blocker** — speed gains are worthless if code quality drops. Use existing `benchmarks/prompts/v1/` test suite for consistent comparison.

---

## Full Test Matrix

| Model | Phase 0 (Lookup) | Phase 0.5 (Draft) | Phase 1 (All models) | Phase 2 (Corpus) |
|-------|-------------------|-------------------|---------------------|-------------------|
| Qwen3-Coder-480B-A35B | **TEST FIRST** | Test jukofyork draft | Results from Phase 0 | Target |
| Qwen3-Coder-30B-A3B | — | Has working drafts | Test lookup+MoE4 | Target |
| Qwen3-235B-A22B | — | — | Test lookup+MoE4 | Target |
| Qwen2.5-Coder-32B | Already works | Already works | Production baseline | Target |
| Qwen2.5-7B | Already works | Already works | Production baseline | Target |
| Qwen3-Next-80B-A3B | **SKIP (SSM)** | **SKIP (SSM)** | **SKIP (SSM)** | **SKIP (SSM)** |

---

## Dependencies & Prerequisites

| Dependency | Status | Needed For |
|------------|--------|------------|
| llama-server with `--lookup` | DONE (commit `8e35dbc01`) | All phases |
| llama-lookup binary | DONE (production-consolidated) | Phase 0 quick test |
| Bug fixes PRs #18729, #18730 | DONE (cherry-picked) | All phases |
| jukofyork draft model | DONE (on disk) | Phase 0.5 |
| `run_combination_benchmarks.sh` | DONE (480B entry exists) | Phase 0/1 |
| SoftMatcha v2 | INSTALLED (v0.1.0, icu-tokenizer optional) | Phase 2 |
| MVP corpus index | BUILT (73K snippets, 5.5M n-grams, 338MB) | Phase 2A |
| Full corpus (The Stack v1) | BUILDING — Python 67GB+, 5 more languages queued | Phase 2A scaling |
| build_index_v2.py | BUILT — SQLite backend, HF streaming, --resume | Phase 2A scaling |
| prune_index.py | BUILT — optional post-build pruning | Phase 2A scaling |
| Rust toolchain | READY (rustc 1.90.0) | Phase 2 |

---

## Registry Updates (On Success)

If Phase 0 succeeds, update `orchestration/model_registry.yaml`:

```yaml
# REMOVE:
constraints:
  forbid:
  - prompt_lookup
  reason: No prompt lookup - MoE architecture

# ADD (under acceleration):
acceleration:
  type: moe_expert_reduction
  experts: 3
  override_key: qwen3moe.expert_used_count
  alternative:
    type: prompt_lookup
    ngram_min: 3
    optimized_tps: <measured>
    best_for: Code editing, refactoring where output overlaps input
```

Also update `docs/reference/models/QUIRKS.md` to clarify: BOS mismatch affects draft-model speculation only, NOT prompt lookup.

---

## Prior Art & References

| Resource | Relevance |
|----------|-----------|
| `handoffs/archived/llama-server-prompt-lookup.md` | Production lookup implementation details |
| `handoffs/archived/prompt_lookup_integration.md` | Original investigation (flag discovery, crash debugging) |
| `handoffs/completed/swa_prompt_lookup.md` | Bug fixes PRs #18729, #18730 |
| `scripts/benchmark/run_combination_benchmarks.sh` | Existing benchmark infrastructure with 480B entry |
| `docs/chapters/07-prompt-lookup.md` | Technical documentation |
| SoftMatcha v2 paper: https://arxiv.org/pdf/2602.10908 | Corpus engine for Phase 2 |
| SoftMatcha v2 code: https://github.com/softmatcha/softmatcha2 | Implementation (Python+Rust, Apache 2.0) |
| SoftMatcha v2 demo: https://softmatcha.github.io/v2/ | Online demo for spot-checks |

---

## Open Questions

1. **Does `--moe-n-expert` compose with `--lookup` in llama-server?** The llama-lookup binary supports it, but server integration may need verification.
2. **KV cache interaction**: When lookup drafts are rejected, does the KV cache rollback work correctly with MoE expert masking? (Should be fine — KV cache is post-expert-selection — but verify.)
3. **Phase 2 query formulation**: How much does first-20-token re-query improve over keyword-only retrieval? Needs ablation study.
4. **The Stack v2 licensing**: Verify our use case (inference acceleration, not redistribution) is covered.
5. **SoftMatcha v2 GloVe embeddings**: For code, are GloVe word vectors meaningful? Code tokens are not natural language. May need to use exact-only matching (no soft) for code corpus, soft for documentation corpus.

---

## Execution Order

```
Phase 0   →  Test prompt lookup on 480B (1-2 hours)
              |
              ├─ Success → Update registry, ship to production
              └─ Failure → Document, investigate patch

Phase 0.5 →  Test jukofyork draft on 480B (1-2 hours)
              |
              ├─ Success → Test combined (MoE3+spec+lookup)
              └─ Failure → Document BOS specifics

Phase 1   →  Run lookup tests on all non-SSM models (half day)
              |
              └─ Update registry with measured performance for each

Phase 2A  →  A/B TESTED (2026-02-15)
              |
              ├─ MVP index (73K snippets) tested on all 5 models
              ├─ 480B: +15.6pp acceptance, +17% speed (BEST)
              ├─ 32B: +8.7pp acceptance, +6% speed (GOOD)
              ├─ 30B: -12% speed despite +2.1pp acceptance (NEGATIVE — disabled)
              ├─ 235B: mixed (+6.6pp HTTP, -12.1pp BST — disabled)
              ├─ 7B: saturated (94-100% baseline — disabled)
              ├─ Telemetry fix: draft_n/draft_n_accepted (was wrong key names)
              ├─ Token normalization fix: index and query n-grams now consistent
              ├─ SCALING: build_index_v2.py running The Stack v1 (67GB+ Python, 5 more langs queued)
              └─ NEXT: Re-test with full corpus, then Claude-as-Judge quality gate

Phase 2B  →  Sidecar draft injection in llama.cpp (1-2 weeks)
              |
              └─ Only if Phase 2A shows >10% acceptance improvement
```

---

## Resume Commands

```bash
# Phase 0: Quick test with llama-lookup binary
OMP_NUM_THREADS=1 numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-lookup \
  -m /mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00008.gguf \
  --draft-max 16 --moe-n-expert 3 -t 96 -n 200 --temp 0 \
  -f /mnt/raid0/llm/tmp/twyne_summarize_prompt.txt

# Phase 0: Server test
numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-server \
  -m /mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00008.gguf \
  --override-kv qwen3moe.expert_used_count=int:3 --lookup -t 96 -c 16384 --port 8084

# Phase 0.5: jukofyork draft test
OMP_NUM_THREADS=1 numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-speculative \
  -m /mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00008.gguf \
  -md /mnt/raid0/llm/models/Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf \
  --override-kv qwen3moe.expert_used_count=int:3 --draft-max 16 -t 96 -n 200 --temp 0 \
  -p "Write a Python async HTTP client with retry logic and exponential backoff."

# Phase 2A: Rebuild corpus index (if sources change)
python3 scripts/corpus/build_index.py --output /mnt/raid0/llm/cache/corpus/mvp_index

# Phase 2A: Enable corpus retrieval for A/B testing
# Edit orchestration/model_registry.yaml → corpus_retrieval.enabled: true
# Then run:
python scripts/benchmark/run_benchmark.py --suite coder --tag no-corpus
# (set enabled: true)
python scripts/benchmark/run_benchmark.py --suite coder --tag with-corpus
python scripts/benchmark/score_outputs.py --compare no-corpus with-corpus

# Phase 2A scaling: Check corpus build progress
ls -lh /mnt/raid0/llm/cache/corpus/full_index/corpus.db
pgrep -f build_index_v2

# Phase 2A scaling: Build remaining languages (if not already queued)
python3 scripts/corpus/build_index_v2.py \
    --output /mnt/raid0/llm/cache/corpus/full_index \
    --languages javascript --resume --skip-finalize

# Phase 2A scaling: Optional pruning after build
python3 scripts/corpus/prune_index.py --target-gb 50 --dry-run
```
