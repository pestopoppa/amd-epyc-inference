# Robust Orchestrator Quality (3-Phase Roadmap)

**Created**: 2026-01-29
**Status**: Phase 1-5 COMPLETE, VL Suite Rebuilt, Provenance Audit Done, Health Check Hardened, Stratified Sampling Ready, Pipeline Perf Optimized, 2015 tests / 67.48% coverage (live validation pending)
**Session transcript**: `/home/daniele/.claude/projects/-mnt-raid0-llm-claude/1c01759b-1cc7-479f-b886-7249fe6b90ca.jsonl`

---

## Overview

Fix orchestrator quality bugs, build deterministic debug scoring, then
iterate with MemRL toward intelligent routing. Three phases:

1. **Bug fixes + debug suite** (DONE)
2. **Holistic REPL/MemRL integration** (DONE)
3. **MemRL-driven intelligent orchestration** (IMPLEMENTED — live validation pending)

---

## PHASE 1: Bug Fixes + Debug Suite — COMPLETE

### Changes Made

| Step | File | What |
|------|------|------|
| 1 | `src/prefix_cache.py` | Removed lines 383-387 — `canonicalize_prompt()` was mutating the actual prompt (replacing ISO dates with `[DATE]`) before sending to model. Canonicalization now only used for cache key computation. |
| 2 | `src/backends/llama_server.py` | `_build_payload()` now forwards `stop_sequences` → `payload["stop"]`. Uses `getattr` for safety across both `InferenceRequest` types. |
| 3 | `src/llm_primitives.py` | Added `stop_sequences: list[str] | None = None` to `llm_call()` → `_llm_call_impl()` → `_real_call()` → `_call_caching_backend()`. Full chain plumbed. |
| 3b | `src/model_server.py` | Added `stop_sequences: list[str] | None = None` to legacy `InferenceRequest`. |
| 4 | `src/api/routes/chat.py` | Direct-answer mode passes `stop_sequences=["\n\n\n"]` (triple-newline = anti-loop). Both primary and retry paths. |
| 5 | `src/api/routes/chat.py` | Added `_truncate_looped_answer(answer, prompt)` — detects prompt echo in answer, truncates before it. Called after `answer.strip()`. |
| 6 | `scripts/benchmark/compare_orchestrator_direct.py` | Removed `assess_quality()` heuristic scorer. Removed `quality_match` from `ComparisonResult`, `quality_pass_rate` from summary. Added `debug_score` field. |
| 7a | `scripts/benchmark/debug_scorer.py` (NEW) | Deterministic scorer: `exact_match`, `multiple_choice`, `code_execution`, `programmatic`, `substring`. All tested. |
| 7b | `benchmarks/prompts/debug/*.yaml` (NEW, 8 files) | 111 questions across 8 suites with ground truth + scoring method per question. |
| 7c | `scripts/benchmark/compare_orchestrator_direct.py` | Added `--debug`, `--debug-sample`, `--debug-seed` flags. `load_debug_prompts()` randomly samples N questions per suite. |
| 7d | `tests/unit/test_prefix_cache.py` | Updated `test_canonicalizes_prompt` to assert correct behavior (prompt NOT mutated). |

### Test Results

- 117 unit tests pass (0 failures)
- Debug scorer self-tests pass (all 5 scoring methods)
- 111 questions load correctly, random sampling verified

### Verification Commands

```bash
# Verify no [DATE] contamination
grep -rn "canonicalize_prompt(request" src/prefix_cache.py  # Should return nothing

# Run unit tests
python3 -m pytest tests/unit/test_llm_primitives.py tests/unit/test_prefix_cache.py tests/unit/test_model_server.py tests/unit/test_api.py -q

# Test debug scorer
python3 -c "from scripts.benchmark.debug_scorer import score_answer; print(score_answer('#### 42', '42', 'exact_match', {'extract_pattern': r'####\s*(\d+)'}))"

# Run debug suite (requires live orchestrator)
python scripts/benchmark/compare_orchestrator_direct.py --debug --suite all
```

---

## PHASE 2: Holistic REPL/MemRL Integration — COMPLETE

**Problem**: Direct-answer mode = clean output, zero tool access. REPL mode = tool access, destroys format compliance. Need a middle ground.

**Solution**: Three-way mode selection (direct → react → repl), all feature-flagged. 690 unit tests pass (32 new, 0 regressions). All 6 sub-tasks implemented.

### 2.1 ReAct-style tool loop for direct mode
Give models structured tool-calling without full Python REPL. Evaluate existing tool database (tool_registry.yaml, src/tools/knowledge.py). Consider TOON encoding for tool-call format.

### 2.2 MemRL-learned mode selection
MemRL predicts: does this prompt benefit from tools vs. direct answer? Seed additional direct-vs-tool exemplars to kick-start the learning boundary.

### 2.3 Fix REPL argument-filling
REPL currently forces full Python code generation. Should be filling arguments into pre-generated tool call patterns, not writing boilerplate.

### 2.4 Output formalizer for format-sensitive tasks
Dedicated formalizer model: take output + format constraints → rewrite correct format. Lightweight 1.5B-7B model. Separation of concerns: one model thinks, another formats.

### 2.5 Tool output isolation
Improve `_strip_tool_outputs()` so REPL tool results never contaminate final answers. Currently fragile regex-based.

### 2.6 Iterative MemRL learning loop via debug suite
```
loop:
  1. Sample 10 random questions per suite from debug pool
  2. Run orchestrator
  3. Score deterministically (exact_match, multiple_choice, etc.)
  4. Feed pass/fail → MemRL Q-scorer as rewards (+1.0 pass, -0.5 fail)
  5. MemRL updates Q-values (routing, mode selection, tool usage)
  6. Repeat
```

### Key Constraint
Must not regress eval suite scores below Phase 1 baseline.

---

## PHASE 3: MemRL-Driven Intelligent Orchestration — IMPLEMENTED (2026-01-30)

**Problem**: All text prompts route to frontdoor (30B). Specialists unused.
**Solution**: Feature-flagged specialist routing + GraphEnhancedRetriever + failure veto + comparative seeding.

Code complete, 857 unit tests pass (recount after test cleanup), zero regressions. **Live validation pending.**

### What was built (8 steps + architect delegation)

| Step | What | Files |
|------|------|-------|
| 3.0 | Feature flag `ORCHESTRATOR_SPECIALIST_ROUTING` | `src/features.py`, `src/api/routes/chat.py` |
| 3.1 | `routed_to` in learning loop action space | `scripts/benchmark/memrl_learning_loop.py` |
| 3.2 | `force_role` on ChatRequest + comparative seeding script | `src/api/models/requests.py`, `scripts/benchmark/seed_specialist_routing.py` (NEW) |
| 3.3 | GraphEnhancedRetriever + FailureGraph + HypothesisGraph in init | `src/api/services/memrl.py`, `src/api/state.py` |
| 3.4 | Failure graph veto (risk > 0.5 → frontdoor) + failure recording on escalation | `src/api/routes/chat.py` |
| 3.5 | `get_action_q_summary()` + active Q-scorer per iteration | `orchestration/repl_memory/episodic_store.py`, `scripts/benchmark/memrl_learning_loop.py` |
| 3.6 | Architect plan review gate (`ORCHESTRATOR_PLAN_REVIEW`) — pre-execution architect review of MODERATE plans, 3-phase rollout (A→B→C), MemRL expert demonstrations | `src/features.py`, `src/prompt_builders.py`, `src/proactive_delegation.py`, `src/api/state.py`, `src/api/routes/chat.py`, `orchestration/repl_memory/progress_logger.py`, `orchestration/repl_memory/q_scorer.py`, `tests/unit/test_plan_review.py` (36 tests) |
| 3.6 | Per-suite routing analysis | `scripts/benchmark/analyze_routing_policy.py` (NEW) |
| 3.7 | Regression gates (`--regression-check`, `--regression-gate`) | `scripts/benchmark/memrl_learning_loop.py`, `scripts/benchmark/compare_orchestrator_direct.py` |
| 3.8 | **Architect delegation** (`ORCHESTRATOR_ARCHITECT_DELEGATION`) — architect emits TOON investigation briefs, fast specialist (32B @ 39 t/s) runs ReAct/REPL, architect synthesizes. Multi-loop (max=3). `force_mode="delegated"`. | `src/prompt_builders.py`, `src/api/routes/chat.py`, `src/features.py`, `src/api/models/requests.py`, `scripts/benchmark/seed_specialist_routing.py`, `tests/unit/test_architect_delegation.py` (27 tests) |
| 3.9 | **Validation script fixes** — env var `ORCHESTRATOR_` prefix bug (bare names were silently ignored), `ARCHITECT_DELEGATION=1` wired into steps 2-5b | `scripts/benchmark/run_phase3_validation.sh` |

### Keyword routing heuristics (when flag ON)

| Keywords | Routes to |
|----------|-----------|
| implement, write code, function, class, debug, refactor, algorithm... | `coder_primary` (32B, 39 t/s) |
| concurrent, lock-free, distributed, race condition, deadlock... | `coder_escalation` (32B, 39 t/s) |
| architecture, system design, scalab, microservice, trade-off... | `architect_general` (235B, 6.75 t/s) |
| everything else | `frontdoor` (30B, 18 t/s) |

### Success Criteria
Orchestrator >= best individual model per suite on eval suite. During iteration, only use debug suite.

---

## Phase 3 VALIDATION — Run Order

**Prerequisites**: Orchestrator stack must be running. At minimum HOT tier.

**Automated**: `bash scripts/benchmark/run_phase3_validation.sh` (runs all steps, supports `--step N` resume and `--dry-run`).

```bash
# 0. Start orchestrator stack (HOT tier)
python3 scripts/server/orchestrator_stack.py start --hot-only
# Wait for health checks to pass
python3 scripts/server/orchestrator_stack.py status

# 1. Reproducible baseline (specialist routing OFF)
ORCHESTRATOR_SPECIALIST_ROUTING=0 ORCHESTRATOR_MEMRL=0 \
  python scripts/benchmark/compare_orchestrator_direct.py --debug --suite all --debug-seed 42

# 2. Comparative seeding — populate Q-values with ground truth
#    Runs each question through frontdoor + coder_primary + coder_escalation + architect_general
#    Architect roles get direct + delegated modes; non-architects get direct + react + repl
#    Injects comparative rewards: specialist wins +1.0, both correct +0.3, etc.
ORCHESTRATOR_SPECIALIST_ROUTING=1 ORCHESTRATOR_ARCHITECT_DELEGATION=1 ORCHESTRATOR_MEMRL=1 \
  python scripts/benchmark/seed_specialist_routing.py --suites all --sample-size 10

# 3. Learning loop — verify Q-values shift and no accuracy regression
#    --regression-check: halts on 3 consecutive accuracy drops
ORCHESTRATOR_SPECIALIST_ROUTING=1 ORCHESTRATOR_ARCHITECT_DELEGATION=1 ORCHESTRATOR_MEMRL=1 \
  python scripts/benchmark/memrl_learning_loop.py --iterations 5 --sample-size 10 --regression-check

# 4. Analyze learned routing policies
python scripts/benchmark/analyze_routing_policy.py

# 5. Regression gate — per-suite frontdoor-parity check (exits non-zero on failure)
ORCHESTRATOR_SPECIALIST_ROUTING=1 ORCHESTRATOR_ARCHITECT_DELEGATION=1 ORCHESTRATOR_MEMRL=1 \
  python scripts/benchmark/compare_orchestrator_direct.py --debug --suite all --regression-gate

# 5b. Plan review gate — architect-in-the-loop pre-execution review
#     Same seed=42 benchmark suite, with plan review enabled alongside routing + delegation.
#     Compare: convergence speed, correction rate, accuracy delta vs step 5.
ORCHESTRATOR_SPECIALIST_ROUTING=1 ORCHESTRATOR_ARCHITECT_DELEGATION=1 ORCHESTRATOR_PLAN_REVIEW=1 ORCHESTRATOR_MEMRL=1 \
  python scripts/benchmark/compare_orchestrator_direct.py --debug --suite all --regression-gate

# 6. Kill switch test: disable routing + delegation, verify frontdoor-only behavior
ORCHESTRATOR_SPECIALIST_ROUTING=0 ORCHESTRATOR_ARCHITECT_DELEGATION=0 ORCHESTRATOR_MEMRL=1 \
  python scripts/benchmark/compare_orchestrator_direct.py --debug --suite all
```

### Decision after validation

- If specialists show quality gain on routing analysis (step 4) AND regression gate passes (step 5):
  → Flip `specialist_routing` default to `True` in `src/features.py` production defaults
- If `architect_general:delegated` outperforms `architect_general:direct` in seeding (step 2):
  → Flip `architect_delegation` default to `True` in production defaults
- If plan review (step 5b) shows faster Q-value convergence or fewer corrections over time:
  → Flip `plan_review` default to `True` in production defaults
- If specialists are equal or worse:
  → Keep flags OFF, Q-values still useful for future model upgrades

### Troubleshooting

- **API won't start**: Check `logs/orchestrator.log`. Ensure `pace-env` venv activated.
- **All answers empty**: llama-server backends not running. Check `orchestrator_stack.py status`.
- **seeding script errors on import**: Run from project root (`/mnt/raid0/llm/claude/`). Ensure `scripts/benchmark/` on PYTHONPATH.
- **regression gate exits 1**: Some suite dropped below frontdoor baseline. Check per-suite breakdown. Disable specialist routing if persistent.

---

## Unresolved Questions

1. ~~**`InferenceRequest` field naming**: `request.n_tokens` vs `max_tokens`.~~ **RESOLVED**: Added `max_tokens` alias with bidirectional `__post_init__` sync. Legacy callers use `n_tokens`, new callers use `max_tokens`. Both work.
2. ~~**Chat template EOS**: Test adding `<|im_end|>` as stop sequence for Qwen models.~~ **RESOLVED**: Added `QWEN_STOP = "<|im_end|>"` constant. Appended to all 3 `stop_sequences=` lists and all 4 vision httpx JSON payloads.
3. ~~**VL image datasets**: Need actual images for MMMU, ScienceQA, DocVQA, ChartQA.~~ **RESOLVED**: VL suite rebuilt from OCRBench (1,000) + ChartQA (2,500) via `extract_vl_debug_suite.py`. On-the-fly sampling from 3,500 pool. DocVQA test split has no ground truth — unusable.
4. **lm-evaluation-harness**: Use directly (60+ benchmarks free) or extract scoring logic?
5. **Formalizer model**: xLAM-2-1B, Qwen2.5-1.5B, or fine-tuned?
6. **TOON for ReAct**: Evaluate whether TOON encoding helps tool-calling format.
7. ~~**Debug question volume**: Currently 111. Hundreds ideal for random sampling.~~ **RESOLVED**: Static suites expanded to 325 questions. VL suite uses on-the-fly sampling from 3,500-question pool.
8. **Latency budget**: 235B architect at 6.75 t/s = 2.7x slower than frontdoor. Acceptable for hard tasks?
9. **480B warm-up cost**: ~120s load time. Skip in seeding if not already warm?
10. ~~**Q-value decay**: Old Q-values go stale when models updated. Time-based decay (0.99/day)?~~ **RESOLVED**: Added `temporal_decay_rate: float = 0.99` to `ScoringConfig`. `update_q_value()` now reads `updated_at`, decays toward 0.5 by `decay_rate^days_elapsed` before TD update.
11. ~~**Non-VL suite provenance**~~: **RESOLVED** — All 6 suites now sample on-the-fly from real benchmark datasets (31,820 total questions). Static YAML retained as fallback only.

---

## Resume Commands

```bash
# Unit tests (all phases, should pass)
python3 -m pytest tests/unit/ -x -q

# Phase 2 feature testing: enable react + formalizer
ORCHESTRATOR_REACT_MODE=1 ORCHESTRATOR_OUTPUT_FORMALIZER=1 \
  python scripts/benchmark/compare_orchestrator_direct.py --debug --suite all --restart-api

# Phase 3 validation: see "Phase 3 VALIDATION — Run Order" section above

# Regenerate VL suite from real benchmark data
python3 scripts/benchmark/extract_vl_debug_suite.py --total 42
```

---

## VL Suite Rebuild + Provenance Audit — COMPLETE (2026-01-30)

### Problem

All 8 debug suites claimed to source questions from public benchmarks (MMLU, GSM8K, HumanEval, etc.) but investigation revealed most were **hand-written approximations**. The VL suite was worst: 35 text-only proxy questions, zero images, despite actual VL benchmark datasets (OCRBench, ChartQA) being cached locally.

### VL Suite: Rebuilt from Real Data

Wrote `scripts/benchmark/extract_vl_debug_suite.py`:
- Reads OCRBench (1,000 q) + ChartQA (2,500 q) from HuggingFace Arrow cache
- Extracts images to disk, samples with diversity across question types
- Generates `vl.yaml` v3.0 (42 static questions) + `VLDatasetAdapter` for on-the-fly sampling
- **On-the-fly mode**: `compare_orchestrator_direct.py` now samples fresh VL questions from the full 3,500-question pool on each learning loop iteration (different seeds → zero overlap)

### Tool Usage Tracking

Added `tools_used` count to benchmark output:
- `_react_mode_answer()` returns `tuple[str, int]` (answer + tool count)
- `REPLEnvironment._tool_invocations` counter
- `ChatResponse.tools_used` field
- `ComparisonResult.tools_used` in benchmark script

### Provenance Audit Results

| Suite | Claimed Source | Actual Provenance |
|-------|---------------|-------------------|
| **vl** | OCRBench, ChartQA | **REBUILT** — real data from HF cache |
| general | MMLU | Hand-written trivia (NOT from MMLU) |
| math | GSM8K, MATH | Mixed: first ~15 GSM8K real, rest hand-written |
| coder | HumanEval, MBPP | Mixed: first 4 HumanEval real, rest hand-written |
| thinking | ARC-Challenge, HellaSwag | Mostly fabricated, zero HellaSwag |
| instruction_precision | IFEval | Hand-written in IFEval style |
| agentic | BFCL-inspired | Already honestly labeled |
| long_context | Synthetic | Already honestly labeled |

All headers updated to honestly document provenance.

### Files Changed

| File | Nature |
|------|--------|
| `scripts/benchmark/extract_vl_debug_suite.py` | **NEW** — VL extraction + adapter |
| `benchmarks/prompts/debug/vl.yaml` | Rebuilt — 42 real questions with images |
| `benchmarks/images/vl/{ocrbench,chartqa}/` | 42 images extracted from datasets |
| `scripts/benchmark/compare_orchestrator_direct.py` | On-the-fly VL loading + tool tracking + image_path fix |
| `src/api/models/responses.py` | `tools_used` field |
| `src/repl_environment.py` | `_tool_invocations` counter |
| `src/api/routes/chat.py` | `_react_mode_answer()` tuple return + tool tracking |
| `tests/unit/test_react_mode.py` | Updated for tuple returns |
| `tests/unit/test_architect_delegation.py` | Updated mocks for tuple returns |
| `scripts/benchmark/seed_specialist_routing.py` | Added `architect_coding` to `DEFAULT_ROLES` |
| `scripts/benchmark/run_phase3_validation.sh` | Added `architect_coding` to `--roles` |
| `benchmarks/prompts/debug/{general,math,coder,thinking,instruction_precision}.yaml` | Provenance headers corrected |

### All Suites: On-the-Fly Dataset Sampling — COMPLETE

Built `scripts/benchmark/dataset_adapters.py` — unified adapter for ALL suites. Downloaded 7 HuggingFace datasets:

| Suite | Dataset(s) | Pool Size |
|-------|-----------|-----------|
| general | MMLU (cais/mmlu) | 14,042 |
| math | GSM8K + MATH-500 | 1,819 |
| coder | HumanEval + MBPP | 664 |
| thinking | ARC-Challenge + HellaSwag | 11,214 |
| instruction_precision | IFEval (google/IFEval) | 541 |
| vl | OCRBench + ChartQA | 3,500 |
| **Total** | | **31,820** |

`compare_orchestrator_direct.py` now tries dataset adapter first for every suite, falls back to YAML only for `agentic` and `long_context` (no public datasets).

Each adapter handles the source dataset's specific schema: MMLU 4-choice format, GSM8K `####` answer extraction, HumanEval function signatures, ARC choices dict, HellaSwag sentence completion, IFEval constraint types.

### Files Changed (Dataset Adapters)

| File | Nature |
|------|--------|
| `scripts/benchmark/dataset_adapters.py` | **NEW** — 6 adapters (MMLU, Math, Coder, Thinking, IFEval, VL) |
| `scripts/benchmark/compare_orchestrator_direct.py` | Modified — unified `_load_from_dataset_adapter()` for all suites |

---

## Vision Specialist Integration into Phase 3 — 2026-01-30

### Design Decisions

- **Vision models = Specialists**. LLMs making judgments, routed via MemRL Q-learning.
- **OCR pipeline = Tool/Service**. LightOnOCR on port 9001, deterministic text extraction.
- **worker_vision** (Qwen2.5-VL-7B, port 8086): Supports `direct` + `react` modes (agentic).
- **vision_escalation** (Qwen3-VL-30B-A3B, port 8087): `direct` only (0% agentic, no tool calls).
- **VL baseline**: `frontdoor:direct` (text-only model → trivial +1.0 bootstraps vision Q-values fast).
- **Feature flag**: Folded into `ORCHESTRATOR_SPECIALIST_ROUTING` (no separate flag).
- **Escalation triggers**: MemRL-learned only (no keyword heuristics for vision routing).

### Architecture

```
User prompt + image
       │
       ▼
  ┌─────────────┐     force_role / MemRL Q-values
  │  Routing     │────────────────────────────────┐
  │  Decision    │                                │
  └─────────────┘                                │
       │                                          │
       ▼                                          ▼
  ┌──────────┐   force_mode="direct"    ┌──────────────────┐
  │ frontdoor │   ───────────────────►  │ _handle_vision_  │
  │ (text)    │                         │ request()        │
  └──────────┘                          │ OCR pre-chain +  │
                                        │ VL direct call   │
                  force_mode="react"    └──────────────────┘
                  ───────────────────►  ┌──────────────────┐
                                        │ _vision_react_   │
                                        │ mode_answer()    │
                                        │ VL decides OCR   │
                                        └──────────────────┘
```

### Smart Combo Filtering

VL questions (with `image_path`) only test vision roles + frontdoor baseline. Text questions skip vision roles entirely. This avoids wasting inference on impossible pairings.

### Files Changed

| File | Changes |
|------|---------|
| `scripts/benchmark/seed_specialist_routing.py` | Vision roles, `image_path` forwarding, mode constraints, smart combo filtering |
| `scripts/benchmark/run_phase3_validation.sh` | `vl` suite, vision roles, ports 8086/8087/9001 health check |
| `scripts/benchmark/memrl_learning_loop.py` | `vl` suite, `image_path` forwarding in API calls |
| `orchestration/tool_registry.yaml` | `ocr_extract` tool definition (vision category) |
| `src/prompt_builders.py` | `VISION_REACT_TOOL_WHITELIST` constant |
| `src/api/routes/chat.py` | `force_server` param, `_vision_react_mode_answer()`, `_execute_vision_tool()`, vision routing block |

### New Functions in chat.py

- **`_vision_react_mode_answer()`**: Vision ReAct loop using direct httpx to VL backend. Image in first message only. Dispatches tools via `_execute_vision_tool()`. Max 5 turns.
- **`_execute_vision_tool()`**: Tool dispatch for vision ReAct. Routes `ocr_extract` to port 9001, `calculate`/date tools inline.
- **`_handle_vision_request(force_server=)`**: Added server constraint param for forced routing to specific VL port.

### Next Steps

1. Run `run_phase3_validation.sh` with vision servers live to seed VL Q-values
2. Verify vision ReAct loop produces OCR tool calls on text-heavy images
3. Compare `worker_vision:direct` vs `worker_vision:react` accuracy on VL debug suite
4. Monitor `vision_escalation:direct` quality vs `worker_vision` to validate MemRL escalation learning

---

## Dev Tasks — COMPLETE (2026-01-30, Session 5)

6 code-quality tasks completing the remaining items from the plan. 48 new tests, all passing.

| Task | Files | What |
|------|-------|------|
| 1. `max_tokens` alias | `src/model_server.py`, `tests/unit/test_model_server.py` | Added `max_tokens` field + `__post_init__` bidirectional sync. 3 new tests. |
| 2. Q-value temporal decay | `orchestration/repl_memory/q_scorer.py`, `orchestration/repl_memory/episodic_store.py` | `temporal_decay_rate=0.99` in ScoringConfig, decay toward 0.5 by `rate^days` before TD update. All 3 call sites updated. |
| 3. Vision tool whitelist | `src/prompt_builders.py`, `src/api/routes/chat.py` | `VISION_REACT_EXECUTABLE_TOOLS` frozenset + `VISION_TOOL_DESCRIPTIONS` dict as single source of truth. Replaced hardcoded descriptions, improved error messages. |
| 4. Qwen stop sequence | `src/api/routes/chat.py` | `QWEN_STOP = "<\|im_end\|>"` constant. Appended to 3 `stop_sequences=` lists + 4 httpx JSON payloads. |
| 5. Dataset adapter tests | `tests/unit/test_dataset_adapters.py` (NEW) | 20 tests: get_adapter factory, MMLU/Math/IFEval/Base adapters. All mocked, no HF downloads. |
| 6. Vision routing tests | `tests/unit/test_vision_routing.py` (NEW) | 11 tests: constant relationships, `_execute_vision_tool` dispatch (calculate, date, time, OCR, errors). |

### Verification

```bash
# All 48 new tests pass
.venv/bin/pytest tests/unit/test_model_server.py tests/unit/test_dataset_adapters.py tests/unit/test_vision_routing.py -v -q

# Full related suite (85 tests pass)
.venv/bin/pytest tests/unit/test_model_server.py tests/unit/test_dataset_adapters.py tests/unit/test_vision_routing.py tests/unit/test_react_mode.py tests/unit/test_plan_review.py tests/unit/test_architect_delegation.py -v -q
```

### Unresolved Questions Resolved

- **#1** (field naming): `max_tokens` alias added
- **#2** (chat template EOS): `QWEN_STOP` constant added to all generation paths
- **#10** (Q-value decay): Temporal decay implemented with configurable rate

---

## Phase 3 Validation: Overhead Root-Cause + Mitigations — 2026-01-31

### Problem

Seeding script showed ~68s overhead on `direct`/`repl` modes but only ~6s on `react`. Investigation confirmed root cause is llama-server KV cache management (machine-wide memory pressure), NOT Python code.

### Evidence

After 2-token generation → ~6s overhead. After 72+ tokens → ~68s. Cross-server effect (port 8081 slow after 8080 request) proves machine-wide bottleneck.

### Mitigations Implemented

| Mitigation | File(s) | What |
|------------|---------|------|
| HTTP round-trip timing | `src/backends/llama_server.py`, `src/model_server.py` | Measure total HTTP time vs reported inference, expose `http_overhead_ms` |
| http_overhead_ms in response | `src/llm_primitives.py`, `src/api/routes/chat.py` (6 sites) | Accumulate overhead, expose in ChatResponse |
| cache_prompt parameter flow | `src/api/models/requests.py`, `src/llm_primitives.py`, chat.py | Per-request `cache_prompt` override flows through entire chain |
| Health checks in seeding | `scripts/benchmark/seed_specialist_routing.py` | `_check_server_health()` before each combo, abort question on failure |
| Cooldown between requests | `scripts/benchmark/seed_specialist_routing.py` | `--cooldown N` seconds between combos to reduce memory pressure |
| Skip-cache option | `scripts/benchmark/seed_specialist_routing.py` | `--skip-cache` disables KV cache reuse (marginal benefit) |

### Debug Suite Quality Issues Found

**Thinking suite**: 6 corrupted prompts (arc_002-005, arc_008-009) with `__(` generation artifacts. All cleaned — content and answers preserved.

**Coder suite** (not yet cleaned): 3 duplicate question pairs, extreme tier imbalance (50% T1), "T3" questions not genuinely hard.

### Test Results

483 unit tests pass, zero regressions.

### Recommended Seeding Command

```bash
# With cooldown to mitigate server pressure (skip --skip-cache, marginal benefit)
# Dedup is ON by default — coder_primary skipped (=frontdoor), saving ~6.4h per full run
ORCHESTRATOR_SPECIALIST_ROUTING=1 ORCHESTRATOR_ARCHITECT_DELEGATION=1 ORCHESTRATOR_MEMRL=1 \
  python scripts/benchmark/seed_specialist_routing.py --suites thinking coder --sample-size 10 --cooldown 2

# To force testing both frontdoor and coder_primary independently:
  python scripts/benchmark/seed_specialist_routing.py --no-dedup ...
```

---

## Role Deduplication + Random Seeds — 2026-01-31

### Problem

`frontdoor` and `coder_primary` map to the same `localhost:8080` backend. With `skip_suffix=True` in all seeding paths, the HTTP payloads are identical. Testing both wastes 210 inference calls (~6.4 hours) per full seeding run.

### Solution: URL-Based Dedup

`_deduplicate_roles()` detects URL collisions in `LLMPrimitives.DEFAULT_SERVER_URLS`, tests each unique model once, and clones rewards + results to aliased roles.

- **Enabled by default** — `--no-dedup` to opt out
- **Currently deduplicates**: `coder_primary` → `frontdoor` (same `localhost:8080`)
- **Future-safe**: If roles later get different system prompts (via `get_system_prompt()` being called), dedup would need to also check prompt identity. Currently URL-only is correct since `skip_suffix=True` everywhere.

### Random Pipeline Seeds

`run_phase3_validation.sh` now generates a random seed each run (via `$((RANDOM * RANDOM % 1000000))`), logged at pipeline start. Override with `--seed N` for reproducibility. All steps and output filenames use `PIPELINE_SEED`.

### Files Changed

| File | Changes |
|------|---------|
| `scripts/benchmark/seed_specialist_routing.py` | `_deduplicate_roles()`, `_modes_for_role()`, `--no-dedup`, alias annotations in summary, reward cloning |
| `scripts/benchmark/run_phase3_validation.sh` | Random `PIPELINE_SEED`, `--seed N` flag, dedup comment, seed in all filenames |

---

## Health Check Hardening + Stratified Sampling — 2026-01-31

### Health Check: Per-Role Skip with Retry

**Problem**: Seeding script used `break` when health check failed, aborting ALL remaining combos. After frontdoor:react generated 953 tokens, server was briefly busy → health check failed → coder_escalation, architect_general, architect_coding (all on separate servers) were skipped.

**Fix**: Replaced `break` with per-role `failed_roles` tracking:
- `failed_roles: set[str]` — tracks roles whose health check failed
- Before each combo: if `role in failed_roles`, skip (continue), don't abort everything
- On health failure: wait 5s, retry with 10s timeout, then add role to `failed_roles`
- Other roles on different servers proceed normally

### Header Log: Tested vs Cloned Combos

**Problem**: Header showed "Testing 13 combos" including coder_primary even though dedup was active.

**Fix**: Compute `tested_combos` from `unique_roles`, log separately:
```
Tested combos: 10 (frontdoor:direct, frontdoor:react, frontdoor:repl, ...)
Cloned combos: 3 (coder_primary→frontdoor)
```

### Stratified Tier-Balanced Sampling

Built `_stratified_sample()` in `dataset_adapters.py`. When `--stratify-tiers` is passed, suites with real difficulty metadata draw equal questions per tier instead of uniform random.

| Adapter | `has_real_tiers` | Tier Logic |
|---------|-----------------|------------|
| MMLUAdapter | True | Subject difficulty (physics/math → T3, humanities → T1) |
| MathAdapter | True | GSM8K → T1, MATH-500 level ≤3 → T2, level >3 → T3 |
| IFEvalAdapter | True | Constraint count (≤1 → T1, ≤3 → T2, 4+ → T3) |
| Others | False | Silent fallback to uniform random |

Wired `--stratify-tiers` through `compare_orchestrator_direct.py` → `load_debug_prompts()` → `adapter.sample()`.

### MCP Module

Installed `mcp` package. All 12 `test_mcp_server.py` tests now pass. Total test count: 857 (up from 483).

### Files Changed

| File | Changes |
|------|---------|
| `scripts/benchmark/seed_specialist_routing.py` | Per-role health skip, header dedup clarity |
| `scripts/benchmark/dataset_adapters.py` | `_stratified_sample()`, `has_real_tiers`, tier logic |
| `scripts/benchmark/compare_orchestrator_direct.py` | `--stratify-tiers` CLI flag wired through |

---

## Pipeline Performance Optimization — 2026-01-31

### Problem

Full pipeline takes ~2h24m. 4 redundant API restarts (~60s), step 6 over-sampled (10/suite for binary check), per-call httpx.Client overhead, sequential health checks.

### Changes

| # | Optimization | Saves | Key Change |
|---|-------------|-------|------------|
| 1 | Remove 2 redundant `restart_api` (steps 4, 5 = identical env to step 2) | ~30s | Comment-only replacement |
| 2 | `POST /config` endpoint for hot-toggle (steps 5b, 6) | ~30s | New `src/api/routes/config.py` |
| 3 | Step 6 sample 10→5 (kill-switch = binary property) | ~175s | `--debug-sample 5` |
| 4 | Persistent `httpx.Client` in 3 benchmark scripts | ~2s | Optional `client=` parameter |
| 5 | Parallel health checks (step 0, 8 ports) | ~2s | Background jobs + wait |

**Total: ~239s (−2.8%), pipeline ~2h24m → ~2h20m.**

### `POST /config` API

```bash
# Hot-toggle feature flags without restart
curl -s -X POST http://localhost:8000/config \
  -H 'Content-Type: application/json' \
  -d '{"plan_review": true}'
# Returns: {"status": "ok", "features": {...}}
```

### Files Changed

| File | Changes |
|------|---------|
| `scripts/benchmark/run_phase3_validation.sh` | 2 restarts removed, 2 curl toggles, parallel health, sample 10→5 |
| `src/api/routes/config.py` | **NEW** — runtime feature toggle endpoint |
| `src/api/routes/__init__.py` | Registered config router |
| `scripts/benchmark/compare_orchestrator_direct.py` | Persistent httpx.Client in `run_comparison()` |
| `scripts/benchmark/seed_specialist_routing.py` | Persistent httpx.Client in `run_seeding()` |
| `scripts/benchmark/memrl_learning_loop.py` | Persistent httpx.Client in `run_iteration()` |

### Future Opportunity

MoE 30B-A3B models (3B active) are candidates for parallel inference testing. If validated, direct-backend parallel seeding could save ~2,000s (−23%), bringing pipeline to ~1h48m.
