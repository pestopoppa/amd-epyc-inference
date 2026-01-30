# Robust Orchestrator Quality (3-Phase Roadmap)

**Created**: 2026-01-29
**Status**: Phase 1-5 COMPLETE, VL Suite Rebuilt, Provenance Audit Done (live validation pending)
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

Code complete, 884 unit tests pass, zero regressions. **Live validation pending.**

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

1. **`InferenceRequest` field naming**: `request.n_tokens` vs `max_tokens`. Legacy uses `n_tokens`, protocol uses `max_tokens`. Both coexist.
2. **Chat template EOS**: Test adding `<|im_end|>` as stop sequence for Qwen models.
3. ~~**VL image datasets**: Need actual images for MMMU, ScienceQA, DocVQA, ChartQA.~~ **RESOLVED**: VL suite rebuilt from OCRBench (1,000) + ChartQA (2,500) via `extract_vl_debug_suite.py`. On-the-fly sampling from 3,500 pool. DocVQA test split has no ground truth — unusable.
4. **lm-evaluation-harness**: Use directly (60+ benchmarks free) or extract scoring logic?
5. **Formalizer model**: xLAM-2-1B, Qwen2.5-1.5B, or fine-tuned?
6. **TOON for ReAct**: Evaluate whether TOON encoding helps tool-calling format.
7. ~~**Debug question volume**: Currently 111. Hundreds ideal for random sampling.~~ **RESOLVED**: Static suites expanded to 325 questions. VL suite uses on-the-fly sampling from 3,500-question pool.
8. **Latency budget**: 235B architect at 6.75 t/s = 2.7x slower than frontdoor. Acceptable for hard tasks?
9. **480B warm-up cost**: ~120s load time. Skip in seeding if not already warm?
10. **Q-value decay**: Old Q-values go stale when models updated. Time-based decay (0.99/day)?
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
