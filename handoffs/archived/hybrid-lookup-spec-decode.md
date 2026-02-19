# Hybrid Prompt Lookup + Corpus-Augmented Speculative Decoding

## Status: ACTIVE — Phases 0+0.5+1 COMPLETE, Phase 2A SHIPPED, Phase 2B-Quality ABANDONED
**Created**: 2026-01-10
**Updated**: 2026-02-19 (V3 corpus quality gate PASSED all models. 30B: +16% speed, enabled. 32B: +72% speed, enabled. 480B: +1.4% speed, quality +1.25 — marginal, NOT enabled. Phase 2A complete.)
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
| **Phase 2A** | **SHIPPED** | V3 corpus (76.6M snippets, 5.4B n-grams) quality gate PASSED all 3 models. 30B: +16% speed (enabled). 32B: +72% speed (enabled). 480B: +1.4% speed, +1.25 quality (marginal, NOT enabled). 235B/7B remain disabled. |
| **Phase 2B-Quality** | **ABANDONED (7B & 32B)** | Prompt-level RAG hurts quality on both 7B (delta -0.96) and 32B (delta -1.38). Even relevant The Stack snippets confuse the model. Only graph_shortest benefits (+1.0/+1.8). Requires fine-tuning or GPT-4-class models. |

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
| MVP corpus index | SUPERSEDED by v3 sharded index | Phase 2A |
| V3 sharded index | BUILT (76.6M snippets, 5.4B n-grams, 651GB, 16 shards) — WAL checkpointed, quality gate PASSED | Phase 2A |
| build_index_v2.py | BUILT — SQLite backend, HF streaming, --resume | Phase 2A scaling |
| populate_shards.py | BUILT — dual-cursor merge, parallel langs, 50% sampling | Phase 2A scaling |
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

## Closed Questions (All Resolved)

1. ~~**Does `--moe-n-expert` compose with `--lookup` in llama-server?**~~ ANSWERED: Yes, 30B MoE6+spec+lookup = 47.11 t/s (Phase 1).
2. ~~**KV cache interaction**~~ ANSWERED: Works correctly — verified across all tested models in Phase 1.
3. **Phase 2 query formulation** — Q3 CLOSED (2026-02-19)
   - **Ablation**: `scripts/benchmark/q3_requery_ablation.py` + `/mnt/raid0/llm/tmp/q3_ablation_v2.py`
   - **Method**: Compared V3 4-gram hits for three strategies across 6 quality gate prompts (32B outputs):
     - A) Keyword extraction from NL prompt (production): **21 gram hits**
     - B) First-20-token re-query (model output): **53 gram hits** (+152%)
     - C) Full output n-grams (theoretical ceiling): **981 gram hits** (+4571%)
   - **Key insight**: Prompt lookup already matches against the model's own output during generation. After ~20 tokens, the model's generated code IS the best n-gram source — far exceeding any corpus re-query (981 vs 53 hits). Re-querying adds 185ms latency (~2.3 tokens at 12.6 t/s) for diminishing returns.
   - **Decision**: Keyword-only retrieval is sufficient. The model's own output provides 47x more n-gram material than first-20-token re-query, and it's free via prompt lookup's self-matching.
   - **Results**: `benchmarks/results/runs/q3_requery/results.json`
4. ~~**The Stack v1 licensing**~~ N/A: Inference acceleration (n-gram matching), not redistribution. No concern.
5. **SoftMatcha v2 GloVe embeddings for code** — Q5 CLOSED (2026-02-19)
   - **Hypothesis**: GloVe (Wikipedia/Gigaword) has poor coverage of code tokens. Most identifiers, operators, and language-specific keywords will be OOV. SoftMatcha assigns 0.0 similarity to OOV tokens — they can never soft-match. FastText (subword-aware, character n-grams) may fare better.
   - **SoftMatcha v2** installed at `/mnt/raid0/llm/tmp/softmatcha2/` (v0.1.0, ICLR 2025, Apache 2.0)
     - OOV handling: assigns 0.0 similarity — hard constraint, not soft fallback
     - Supported backends: GloVe (gensim), FastText (subword n-grams), HuggingFace transformers
     - Index format: HDF5 (inverted file + embeddings). ~30x text size. Separate from V3 SQLite corpus.
     - Tokenizer: Moses word-level (default, via sacremoses). No built-in code tokenizer.
     - AVX-512 optimized matrix multiplication (Turin CPU advantage)
   - **Key architectural difference from V3 corpus**: V3 does exact 4-gram text lookup in SQLite; SoftMatcha does fuzzy token-level matching via cosine similarity against pre-computed embeddings
   - **Evaluation script**: `scripts/benchmark/glove_code_coverage.py` — samples ~1000 snippets from V3 corpus, measures GloVe + FastText vocabulary coverage with per-language and per-category breakdown
   - **Decision tree**: Both <10% → CLOSED. GloVe <10%, FastText 10-30% → evaluate if useful identifiers covered. Either >30% → build test index, evaluate retrieval quality.
   - **Embedding models**: `glove-wiki-gigaword-300` (400K vocab, `/mnt/raid0/llm/cache/gensim-data/`), `cc.en.300` (2M known vocab + subword, `/mnt/raid0/llm/cache/fasttext/cc.en.300.bin`)
   - **Coverage results (2026-02-19)**: Hypothesis WRONG — coverage much higher than expected:
     - GloVe: **79.2%** overall (C++: 79.3%, Python: 78.0%, Rust: 84.4%)
     - FastText known: **74.0%** (C++: 76.7%, Python: 69.8%, Rust: 80.6%)
     - FastText subword: **86.1%** (C++: 87.1%, Python: 84.6%, Rust: 88.4%)
   - **Per-category analysis** (the nuanced picture):
     - operators/punctuation: ~100% all models (trivial — `(`, `)`, `,`, `:` are in any vocab)
     - keywords: GloVe 97.9%, FastText 100% (`def`, `class`, `return` are English words)
     - common identifiers: ~99-100% (`data`, `model`, `result` — again, English words)
     - compound identifiers: GloVe 66.8% (inflated by `_`), FastText known **2.9%**, FastText subword 13.2%
     - "other" (code-specific misc): GloVe 57.5%, FastText known 65.7%, FastText subword 93.9%
   - **Key insight**: High overall coverage is dominated by tokens that would match exactly anyway (operators, English keywords). The actual code-specific compound identifiers (`self.assertEqual`, `camelCase`, `getData`) have <3% coverage in FastText known vocab. GloVe's 66.8% compound coverage is inflated by `_` alone (19,960 of 30,109 compound tokens).
   - **Moses tokenizer artifact**: Top OOV tokens are XML entities (`&quot;`, `&apos;`, `&gt;`, `&#91;`) from sacremoses XML-escaping — not actual code vocabulary gaps.
   - **Step 2 results (2026-02-19)**: Built SoftMatcha HDF5 index from 10K V3 snippets (2.5M tokens, 19K vocab, 55.9MB index, 23.6s build).
     - **NL queries return 0 matches at ALL thresholds** (1.0 down to 0.5): "calculate loss predictions", "async retry exponential backoff", etc. SoftMatcha is a sequential pattern matcher — all query tokens must appear consecutively. NL phrases never appear as consecutive sequences in code.
     - **Code-pattern queries work but soft matching adds noise**: `return` finds 9,955 exact but 57,691 soft matches (because `for` ≈ `return` at 0.53 — meaningless for code). `for i in` → 354 exact, 3,210 soft (sample soft match: `to it .` — garbage).
     - **Moses tokenizer wrong for code**: `BinarySearchTree` → `binarysearchtree`, `self.search(query)` stays joined, `_` splits from identifiers.
     - **Root cause**: SoftMatcha designed for NL text matching (paraphrased sentences). Code has fundamentally different token structure — no NL phrases. GloVe similarity between code tokens is meaningless.
   - **Q5 CLOSED**: SoftMatcha v2 soft matching architecturally unsuitable for code retrieval. Exact n-gram matching via V3 SQLite corpus remains correct approach (+16%/+72% on 30B/32B).
   - **Full results**: `benchmarks/results/runs/q5_coverage/results.json`, `benchmarks/results/runs/q5_softmatcha/results.json`

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

Phase 2A  →  A/B TESTED (2026-02-15), CORPUS SCALED (2026-02-17)
              |
              ├─ MVP index (73K snippets) tested on all 5 models
              ├─ 480B: +15.6pp acceptance, +17% speed (BEST)
              ├─ 32B: +8.7pp acceptance, +6% speed (GOOD)
              ├─ 30B: -12% speed despite +2.1pp acceptance (NEGATIVE — disabled)
              ├─ 235B: mixed (+6.6pp HTTP, -12.1pp BST — disabled)
              ├─ 7B: saturated (94-100% baseline — disabled)
              ├─ Telemetry fix: draft_n/draft_n_accepted (was wrong key names)
              ├─ Token normalization fix: index and query n-grams now consistent
              ├─ SCALING (2026-02-17): V3 sharded index built from The Stack v1
              │   - Dropped JS/TS/Go (not needed). Kept Python, C++, Rust only.
              │   - 50% deterministic sampling during population to control size.
              │   - Rewrote populate_shards.py: dual-cursor merge (fixed O(N²) bug),
              │     parallel language processing, threaded shard writes.
              │   - Rust: 7.1M snippets (100%), C++: ~34.3M (96%), Python: 34.5M (75%)
              │   - Total: 76.6M snippets, 5.4B n-grams, 651GB on disk (16 shards)
              │   - WAL checkpoint completed 2026-02-19: 285G reclaimed, all shards clean.
              │
              ├─ V3 QUALITY GATE (2026-02-19): PASSED — Claude-as-Judge, 6 code gen prompts
              │   30B: avg +16.3% speed, +5.6pp acceptance, quality delta -0.04 (PASS)
              │   32B: avg +72.3% speed, +7.7pp acceptance, quality delta -0.17 (PASS)
              │   30B corpus_retrieval flipped to true in registry (was false with MVP)
              │   graph_shortest: 100% acceptance on 32B (50.3 t/s, perfect n-gram match)
              │
              └─ 480B COMPLETE (2026-02-19): +1.4% speed, quality +1.25 (PASS but marginal)
                  Already enabled in registry (line 400) from prior A/B (+15.6pp). V3 confirms no regression.

Phase 2B-Quality → RAG-augmented generation for worker models
              |
              ├─ 7B ABANDONED (3 tests, all FAIL):
              │   v1: delta -0.42 (0 snippets retrieved, n-gram mismatch)
              │   v2: delta -0.17 (keyword fallback, MVP index corpus pollution)
              │   v3: delta -0.96 (The Stack corpus, relevant code ACTIVELY HURTS)
              │
              │   Root cause: 7B cannot integrate reference code via prompt.
              │   Even perfect snippets (real BST/LRU/graph implementations from
              │   The Stack) confuse the model — it produces worse code with RAG.
              │   Only graph_shortest consistently benefits (+1.8 across tests).
              │
              │   Research papers achieve quality gains via fine-tuning on RAG data
              │   and token-level injection, not prompt stuffing.
              │
              │   Available fine-tuned RAG models (all Qwen2.5-Coder-7B fine-tunes):
              │   - SWE-Dev-7B (THUDM): 23.4% SWE-bench (→GPT-4o range), MIT, GGUF avail
              │   - SWE-agent-LM-7B (Princeton): Claude 3.5 distillation, GGUF-convertible
              │   - SWE-Dev-32B: 36.6% SWE-bench (exceeds GPT-4o)
              │   These prove RAG works WITH fine-tuning, not prompt-level injection.
              │
              ├─ 32B RESULT: delta -1.38 (WORSE than 7B's -0.96)
              │   async=-2.5 bst=-3.5 lru=-1.5 json=-0.8 rate=-1.0 graph=+1.0
              │   32B does NOT reason better about reference code.
              │   Prompt-level RAG is fundamentally wrong for local models.
              │
              ├─ Infrastructure improvements (retained regardless of quality outcome):
              │   - Keyword fallback in CorpusRetriever (word→snippet_ids index)
              │   - Auto-init from registry in build_corpus_context()
              │   - Incremental result writing in quality gate benchmark
              │   - Concurrency benchmarks (7B: 2-concurrent optimal, 1.76x throughput)
              │
              ├─ Config: rag_enabled: false PERMANENTLY. Phase 2B closed.
              │
              └─ SWE-Dev research (2026-02-15): Fine-tuned models (SWE-Dev-7B/32B,
                  THUDM) achieve 23.4%/36.6% SWE-bench via RFT on agentic trajectories.
                  NOT drop-in replacements (require OpenHands tool-use format).
                  Potential future: worker_swe_agent role if tool-use issues persist.

Phase 2B-Sidecar → Sidecar draft injection in llama.cpp
              |
              ├─ APPROVED (2026-02-19): Develop on llama.cpp-experimental branch
              │   Latency budget: corpus query <1ms vs ~50-80ms draft cycle = no bottleneck
              │   If sidecar works, Q#3 (first-20-token re-query) becomes moot —
              │   sidecar inherently re-queries using latest generated tokens each cycle.
              │
              ├─ IMPLEMENTED (2026-02-19): Branch feature/corpus-sidecar
              │   Architecture: new COMMON_SPECULATIVE_TYPE_CORPUS_SIDECAR plugin
              │   (NOT server-context.cpp hack — uses pluggable speculative framework)
              │   Files created:
              │     common/corpus-sidecar.h + .cpp — SQLite sidecar query layer
              │     common/md5.h — minimal MD5 for shard routing (matches Python indexer)
              │     tests/test-corpus-sidecar.cpp — MD5, DB init, shard routing tests (all pass)
              │   Files modified:
              │     common/common.h — enum + corpus params in common_params_speculative
              │     common/speculative.cpp — new state impl + init wiring (blocking/pre-query/async modes)
              │     common/arg.cpp — --corpus-path, --corpus-refresh, --corpus-snippets, --spec-type corpus-sidecar
              │     common/CMakeLists.txt — LLAMA_CORPUS_SIDECAR option + SQLite3 dep
              │     tests/CMakeLists.txt — conditional test target
              │   Build: cmake -DLLAMA_CORPUS_SIDECAR=ON (off by default, zero impact on normal builds)
              │   Tests: all pass (MD5 cross-validated against Python hashlib)
              │
              ├─ BENCHMARKED (2026-02-19): 30B on v3_sharded corpus — ALL MODES NEGATIVE
              │   | Mode                        | Avg t/s | vs Baseline | vs Phase 2A |
              │   |-----------------------------|---------|-------------|-------------|
              │   | Phase 2A (prompt injection)  | 29.9    | +16%        | —           |
              │   | Pre-query (--corpus-refresh 0)| 26.0   | +1%         | -13%        |
              │   | Async (--corpus-refresh -1)  | 25.8    | 0%          | -14%        |
              │   | Blocking (--corpus-refresh 64)| 24.9   | -3%         | -17%        |
              │   | Blocking + ngram_cache       | 23.6    | -8%         | -21%        |
              │
              │   Root cause: corpus n-grams injected into nc_static via
              │   common_ngram_cache_update don't match draft model proposals.
              │   Phase 2A works because injected prompt tokens ARE the context
              │   the model naturally matches against — sidecar tokenizes externally.
              │   Blocking mode also adds 24s of SQLite I/O per request.
              │
              └─ CLOSED (2026-02-19): Sidecar inferior to Phase 2A prompt injection.
                  Phase 2A remains the production approach (+16%/+72% on 30B/32B).
                  Branch feature/corpus-sidecar preserved for future reference.
                  No cherry-pick to production.
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

# Phase 2A: WAL checkpoint (recover partial writes, reclaim ~287G)
for i in $(seq -w 0 15); do
  echo "Checkpointing shard_${i}.db..."
  sqlite3 /mnt/raid0/llm/cache/corpus/v3_sharded/shard_${i}.db "PRAGMA wal_checkpoint(TRUNCATE);"
done
sqlite3 /mnt/raid0/llm/cache/corpus/v3_sharded/snippets.db "PRAGMA wal_checkpoint(TRUNCATE);"

# Phase 2A: Check corpus status
du -sh /mnt/raid0/llm/cache/corpus/v3_sharded/
sqlite3 /mnt/raid0/llm/cache/corpus/v3_sharded/shard_00.db "SELECT COUNT(*) FROM ngrams;"

# Phase 2A: Re-test with full corpus (after WAL checkpoint)
python scripts/benchmark/run_benchmark.py --suite coder --tag full-corpus
python scripts/benchmark/score_outputs.py --compare no-corpus full-corpus
```
