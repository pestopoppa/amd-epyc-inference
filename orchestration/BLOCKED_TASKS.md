# Blocked Tasks

**Last Updated**: 2026-01-15
**Blocking Resource**: PR #15225 (MTP loader)

---

## Quick Status

| Task | Blocked On | Priority | Handoff | Status |
|------|------------|----------|---------|--------|
| **Model Registry: Paged Attention Flag** | Benchmark script running | **LOW** | See below | 🔄 BLOCKED |
| **Paged Attention CoW** | Paged attention PR review | **MEDIUM** | `handoffs/active/paged-attention.md` (Section 9) | 🔄 BLOCKED |
| **MTP Refactoring** | PR #15225 merge | **HIGH** | `research/mtp_investigation.md` | ✅ PLAN READY |
| **MTP ISWA Fix** | — | **HIGH** | `handoffs/active/gemma3-swa-spec-decode-fix.md` | ✅ FIXED (3 commits on mtp-branch) |
| **Gemma-3 SWA Spec Decode** | — | **HIGH** | `handoffs/active/gemma3-swa-spec-decode-fix.md` | ✅ PR #18720 SUBMITTED (94% mem reduction) |
| **Prompt Lookup/Lookahead Bugs** | — | **LOW** | `handoffs/blocked/swa_prompt_lookup.md` | ✅ PRs #18729 + #18730 SUBMITTED (cherry-picked locally) |
| **Qwen3-A3B MoE Instability** | — | — | — | ✅ RESOLVED (was stale build issue) |
| **Hybrid Lookup+Spec Decode** | Implementation needed | **MEDIUM** | `handoffs/active/hybrid-lookup-spec-decode.md` | 📋 PROPOSAL (no upstream work exists) |
| AVX-512 VNNI Q8_0 | — | — | — | ❌ NOT SUBMITTING (8% speedup) |
| Draft model benchmarks | — | HIGH | `handoffs/active/draft-benchmark.md` | ✅ READY (registry verified) |
| Formalizer eval | — | HIGH | `handoffs/active/formalizer-evaluation.md` | ✅ READY (`nohup ./scripts/benchmark/run_all_formalizers.sh &`) |
| Tree speculation | — | MEDIUM | `handoffs/active/cpu-optimization.md` | ✅ COMPLETE (K=24 optimal) |
| RadixAttention | — | — | `handoffs/active/radix-attention.md` | ✅ VERIFIED (80% hit rate) |
| Orchestrator integration | — | HIGH | `handoffs/active/orchestration-integration.md` | ✅ VERIFIED (12/12 tests) |
| MathSmith re-conversion | — | LOW | `handoffs/active/mathsmith-reconversion.md` | ✅ COMPLETE |
| Orchestrator real mode | — | LOW | `handoffs/active/orchestrator.md` | ✅ READY (see startup commands below) |
| Kernel development | — | — | `handoffs/active/kernel-development.md` | ✅ COMPLETE (no PR - gains too small) |
| Frontend Architecture | — | — | `handoffs/active/orchestrator.md` | ✅ COMPLETE |
| CLI Parity Features | — | — | `handoffs/active/orchestrator.md` | ✅ COMPLETE |
| AMD PACE Testing | — | — | `handoffs/active/amd-pace-testing.md` | ✅ COMPLETE (not adopting) |
| **RLM Orchestrator Roadmap** | — | **HIGH** | `handoffs/active/rlm-orchestrator-roadmap.md` | 📋 NEW (8 phases documented) |
| **MemRL Episodic Memory** | — | **HIGH** | `handoffs/active/memrl-episodic-memory.md` | ✅ PHASES 1-3 COMPLETE |
| **Tool/Script Registry Wiring** | — | **MEDIUM** | `progress/2026-01/2026-01-15.md` | ✅ COMPLETE (27 tools wired) |
| **Native Computational Tools** | — | **HIGH** | `handoffs/active/native-computational-tools.md` | ✅ PHASES 1-4 COMPLETE (integration pending) |
| **Role Mapping Bug** | — | — | `progress/2026-01/2026-01-15.md` | ✅ FIXED (str(Role.X) now returns value) |
| **Orchestrator Multi-Model Live Test** | — | — | `progress/2026-01/2026-01-15.md` | ✅ VERIFIED (5 models, 459GB) |

---

## MemRL Episodic Memory Integration

**Master Handoff**: `handoffs/active/memrl-episodic-memory.md`
**Benchmark**: `benchmarks/prompts/v1/orchestrator_planning.yaml`
**Paper**: arXiv:2601.03192 (MemRL)

| Phase | Description | Status | Dependencies |
|-------|-------------|--------|--------------|
| 1 | Core Implementation | ✅ COMPLETE | None |
| 2 | Wire Logging | ✅ COMPLETE | Phase 1 |
| 3 | Enable Hybrid Routing | ✅ COMPLETE | Phase 2 |
| 4 | Escalation Learning | ✅ COMPLETE | Phase 3 |
| 4b | Memory Seeding (~5K) | ✅ COMPLETE | Phase 4 |
| 5 | REPL Exploration Learning | READY | Phase 3 |
| 6 | Claude-as-Judge | OPTIONAL | Phase 3 |

### Memory Seeding Complete (2026-01-14)

~5,000 memories seeded with 67%/33% success/failure ratio:
- Hierarchical decomposition patterns (70)
- Coding/diverse/template failures (~1,340)
- Probabilistic strategies (~450)
- Tool registry created (608 tools mined)

Seeding scripts: `scripts/seed_*.py`

### Phase 1: Core Implementation (COMPLETE)
- [x] `episodic_store.py` - SQLite + numpy memory storage
- [x] `embedder.py` - Task embedding via 0.5B model
- [x] `retriever.py` - Two-phase retrieval + hybrid router
- [x] `progress_logger.py` - Structured JSONL logging
- [x] `q_scorer.py` - Async Q-value update agent
- [x] `model_registry.yaml` - repl_memory configuration
- [x] `orchestrator_planning.yaml` - Claude-as-Judge benchmark

### Phase 2: Wire Logging (COMPLETE - 2026-01-13)
- [x] Add `ProgressLogger` to dispatcher (`src/dispatcher.py`)
- [x] Log routing decisions in Front Door (`src/api.py`)
- [x] Log gate results in GateRunner (`src/gate_runner.py`)
- [ ] Log escalations in FailureRouter (`src/failure_router.py`) - Deferred to Phase 4

### Phase 3: Enable Hybrid Routing (COMPLETE - 2026-01-13)
- [x] Replace hard-coded routing with `HybridRouter` (`src/dispatcher.py`)
- [x] Add confidence logging for monitoring
- [x] Q-scorer integrated (real-time + idle cleanup in API)

### Phase 4: Escalation Learning (COMPLETE - 2026-01-14)
- [x] Store failure contexts with escalation decisions
- [x] Implement `LearnedEscalationPolicy` in FailureRouter
- [x] Connect to episodic memory

**Implementation:**
- Added `LearnedEscalationPolicy` class that queries episodic memory
- Added `LearnedEscalationResult` dataclass for query results
- Updated `FailureRouter` with optional `retriever` and `progress_logger` parameters
- Hybrid routing: queries learned policy first, falls back to rules
- Escalation decisions logged via `progress_logger.log_escalation()`
- Strategy counts tracked for monitoring ("learned" vs "rules")

### Phase 5: REPL Exploration Learning
- [ ] Log exploration strategies in REPLEnvironment
- [ ] Implement `EpisodicREPL.suggest_exploration()`
- [ ] Track token efficiency metrics

### Phase 6: Claude-as-Judge (Optional)
- [ ] Run orchestrator_planning.yaml benchmark
- [ ] Evaluate baseline scores
- [ ] Enable graded rewards if beneficial

### MemRL Resume Commands

```bash
# Verify module imports
python3 -c "from orchestration.repl_memory import EpisodicStore, TaskEmbedder; print('OK')"

# Check memory stats
python3 -c "from orchestration.repl_memory import EpisodicStore; print(EpisodicStore().get_stats())"

# Run Q-scorer manually
python3 -c "
from orchestration.repl_memory import EpisodicStore, TaskEmbedder, ProgressLogger, ProgressReader, QScorer
scorer = QScorer(EpisodicStore(), TaskEmbedder(), ProgressLogger(), ProgressReader())
print(scorer.score_pending_tasks())
"
```

---

## RLM-Enhanced Orchestrator Development Phases

**Master Handoff**: `handoffs/active/rlm-orchestrator-roadmap.md`
**Research**: `research/rlm_analysis.md`

| Phase | Description | Status | Dependencies |
|-------|-------------|--------|--------------|
| 1 | Backend Completion | ✅ COMPLETE | None |
| 2 | RLM Enhancements | ✅ COMPLETE | Phase 1 |
| 3 | Escalation Integration | ✅ COMPLETE | Phase 1 |
| 4 | Formalizer Integration | READY | Phase 3 |
| 5 | Tool/Script Completion | READY | None |
| 6 | Early Failure Detection | READY | Phase 3 |
| 7 | Hyperparameter Tuning | BLOCKED | Benchmarks |
| 8 | Trajectory Visualization | LOW | Phase 2 |

### Phase 1: Backend Completion (COMPLETE - 2026-01-14)
- [x] Complete LlamaServerBackend HTTP (`src/backends/llama_server.py`)
- [x] Wire CachingBackend init (`src/llm_primitives.py`)
- [x] Connect role→backend routing (`src/llm_primitives.py`)
- [x] Fix real mode initialization (`src/api.py`)

**Note**: All infrastructure is complete. Real inference requires starting llama-server instances.
To test: `llama-server -m MODEL.gguf --host 0.0.0.0 --port 8080` then call API with `real_mode=True`.

### Phase 2: RLM Enhancements (COMPLETE - 2026-01-14)
- [x] Forced exploration validation (`src/repl_environment.py` - REPLConfig.require_exploration_before_final)
- [x] Async `llm_batch_async()` (`src/llm_primitives.py`)
- [x] Configurable recursion depth (`src/llm_primitives.py` - LLMPrimitivesConfig.max_recursion_depth)
- [x] Per-query cost tracking (`src/llm_primitives.py` - QueryCost, start_query/end_query)

**Implementation:**
- Forced exploration: tracks peek/grep/llm_call before FINAL(); opt-in via config
- Async batch: `llm_batch_async()` using asyncio.gather for parallel execution
- Recursion depth: max 5 levels by default, RecursionError on exceed
- Cost tracking: QueryCost dataclass, token estimation, per-query cost in dollars

### Phase 3: Escalation Integration (COMPLETE - 2026-01-14)
- [x] Error classification (`src/api.py` - `_classify_error()`)
- [x] Wire FailureRouter into Root LM loop (`src/api.py`)
- [x] Role switching on escalation (`src/api.py`)
- [x] Gate execution integration (`src/api.py` - FailureContext supports gate_name)

**Implementation:**
- Added `_classify_error()` helper in api.py to map errors to ErrorCategory
- Added `_build_escalation_prompt()` for escalated role context
- Root LM loop now tracks current_role, consecutive_failures, and role_history
- FailureRouter consulted on errors, returns RoutingDecision (retry/escalate/fail)
- Role switching on "escalate" action with escalation prompt
- Escalations logged via `progress_logger.log_escalation()`
- 26 API tests + 51 failure_router tests pass

### Phase 4: Formalizer Integration
- [ ] Formalizer routing (`src/dispatcher.py`)
- [ ] Create formalizer module (`src/formalizer.py`)
- [ ] IR → REPL context injection (`src/api.py`)

### Phase 5: Tool/Script Completion
- [ ] MCP client implementation (`src/tool_registry.py`)
- [ ] Script `invoke()` method (`src/script_registry.py`)
- [ ] Script `find_scripts()` method (`src/script_registry.py`)
- [ ] Tool result capture (`src/repl_environment.py`)

### Phase 6: Early Failure Detection
- [ ] Wire GenerationMonitor (`src/llm_primitives.py`)
- [ ] Add entropy thresholds to registry (`model_registry.yaml`)
- [ ] Early abort trigger (`src/llm_primitives.py`)

### Phase 7: Hyperparameter Tuning
- [ ] Sweep framework (`scripts/benchmark/sweep_hyperparams.py`)
- [ ] Temperature sweep per task type
- [ ] Expert count optimization

### Phase 8: Trajectory Visualization
- [ ] Enhanced SSE events (`src/api.py`)
- [ ] Trajectory logging (`src/llm_primitives.py`)
- [ ] Gradio visualization tab (`src/gradio_ui.py`)

---

## Resume Commands

### When PR #15225 Merges (MTP Support)

```bash
# 1. Check PR status
# https://github.com/ggml-org/llama.cpp/pull/15225

# 2. Update llama.cpp and rebuild
cd /mnt/raid0/llm/llama.cpp
git fetch origin master
git merge origin/master
cmake --build build --config Release -j 96

# 3. Test MTP on GLM-4.6
numactl --interleave=all /mnt/raid0/llm/llama.cpp/build/bin/llama-cli \
  -m /mnt/raid0/llm/lmstudio/models/unsloth/GLM-4.6-GGUF/GLM-4.6-Q4_K_S-00001-of-00005.gguf \
  --mtp 2 --override-kv glm4moe.expert_used_count=int:4 \
  -t 96 -n 100 -p "Write a Python quicksort:" --no-display-prompt

# 4. Read refactoring plan
cat /mnt/raid0/llm/claude/research/mtp_investigation.md | grep -A 100 "MTP Refactoring Plan"
```

---

## MTP Refactoring Plan (Ready for Implementation)

**Problem:** PR #15225 uses sequential token-by-token processing (defeats MTP benefit).

**Solution:** Batched drafting + parallel verification (like vLLM/SGLang).

**Expected Speedup:** 30-50% over PR #15225 baseline.

**Full details:** `research/mtp_investigation.md`

---

## DEPRECATED: AVX-512 VNNI Q8_0 Optimization

**Status:** NOT SUBMITTING - 8% speedup on small models, 0% on larger models. Bottleneck is elsewhere.

**Benchmark Results (2026-01-08):**
- Qwen2.5-Coder-0.5B Q8_0: 155 t/s vs 144 t/s = 8% speedup
- Qwen3-1.7B Q8_0: ~50 t/s both = 0% speedup
- DeepSeek-R1-8B Q8_0: ~13 t/s both = 0% speedup

**Code still in tree** (tests pass) but not worth PR overhead for 8% gain.

### Files Changed

**PR1 - VNNI optimization:**
- `ggml/src/ggml-cpu/arch/x86/quants.c` (added ~40 lines in `ggml_vec_dot_q8_0_q8_0`)

**PR2 - Shared helper header:**
- `ggml/src/ggml-cpu/arch/x86/avx512-helpers.h` (NEW - 42 lines)
- `ggml/src/ggml-cpu/arch/x86/repack.cpp` (removed ~25 lines, added include)

**PR3 - Use shared helper:**
- `ggml/src/ggml-cpu/arch/x86/quants.c` (added include, simplified VNNI code)

### Commit Message: PR2

```
ggml : add shared AVX-512 int8 dot product helpers

Move AVX-512F helper functions from repack.cpp to a new shared header
(arch/x86/avx512-helpers.h) to enable reuse across x86 quantization code.

Moved helpers:
- sum_i16_pairs_acc_int32x16: int16 pairwise sum with accumulator
- mul_sum_us8_pairs_acc_int32x16: unsigned×signed int8 dot product
- mul_sum_i8_pairs_acc_int32x16: signed×signed int8 dot product

The signed×signed helper uses the abs(x) * sign-adjusted(y) pattern to
convert for VNNI's dpbusd instruction, which expects unsigned×signed input.

This refactoring was motivated by PR #XXXXX (AVX-512 VNNI Q8_0 optimization)
which needed the same helper logic. Having these in a shared header avoids
duplication and ensures consistent implementation across quants.c and
repack.cpp.

No functional changes - existing code paths unchanged.
```

### Commit Message: PR3

```
ggml : use shared helper in AVX-512 VNNI Q8_0 vec_dot

Refactor the AVX-512 VNNI path in ggml_vec_dot_q8_0_q8_0 to use the
shared mul_sum_i8_pairs_acc_int32x16 helper from avx512-helpers.h.

This replaces 10 lines of inline signed×signed conversion logic with
a single helper call, improving readability while maintaining identical
generated code.

Before:
  const __m512i ax = _mm512_abs_epi8(qx);
  const __mmask64 blt0 = _mm512_movepi8_mask(qx);
  const __m512i sy = _mm512_mask_sub_epi8(qy, blt0, zero, qy);
  const __m512i sums = _mm512_dpbusd_epi32(zero, ax, sy);

After:
  const __m512i sums = mul_sum_i8_pairs_acc_int32x16(zero, qx, qy);

Depends on PR #YYYY (shared AVX-512 helpers).
```

### PR1 Commit Message Template (needs speed results)

```
ggml : add AVX-512 VNNI optimization for Q8_0 vec_dot

Add AVX-512 VNNI path to ggml_vec_dot_q8_0_q8_0, providing [X]x speedup
over AVX2 on VNNI-capable CPUs (Ice Lake, Zen 4, Zen 5, Sapphire Rapids).

Key features:
- Process 2 Q8_0 blocks per iteration using 512-bit registers
- Use _mm512_dpbusd_epi32 for efficient int8 dot product
- Handle signed×signed via abs(x) * sign-adjusted(y) pattern
- Use broadcast instructions for efficient scale vector creation

Tested on AMD EPYC 9655 (Zen 5):
- Qwen3-0.6B Q8_0: [BASELINE] t/s → [OPTIMIZED] t/s ([X]x speedup)
- Qwen3-1.7B Q8_0: [BASELINE] t/s → [OPTIMIZED] t/s ([X]x speedup)

test-quantize-fns: all 32 types pass
AddressSanitizer: clean
UndefinedBehaviorSanitizer: clean
```

### Submission Order

1. **PR1** first (standalone value)
2. **PR2** after PR1 merged/in-flight (references PR1 number)
3. **PR3** after PR2 merged (depends on PR2)

### Revert Commands (if needed)

```bash
# Revert all changes
cd /mnt/raid0/llm/llama.cpp
git checkout ggml/src/ggml-cpu/arch/x86/quants.c
git checkout ggml/src/ggml-cpu/arch/x86/repack.cpp
rm ggml/src/ggml-cpu/arch/x86/avx512-helpers.h
```

---

```bash
# 1. Formalizer evaluation (3 models)
./scripts/benchmark/bench_formalizers.sh \
  --model /mnt/raid0/llm/models/xLAM-2-1B-fc-r-Q4_K_M.gguf \
  --prompts benchmarks/prompts/v1/formalizer/

./scripts/benchmark/bench_formalizers.sh \
  --model /mnt/raid0/llm/models/xLAM-1b-fc-r.Q4_K_M.gguf \
  --prompts benchmarks/prompts/v1/formalizer/

./scripts/benchmark/bench_formalizers.sh \
  --model /mnt/raid0/llm/models/nexusraven-v2-13b.Q4_K_M.gguf \
  --prompts benchmarks/prompts/v1/formalizer/

# 2. Tree speculation benchmark
./scripts/benchmark/bench_tree_speculation.sh
```

### When YOLO Agent Available

```bash
# Set up path
export PATH="/mnt/raid0/llm/npm-global/bin:/mnt/raid0/llm/tools/devc/bin:$PATH"

# Launch devcontainer
devc /mnt/raid0/llm/claude

# Inside container - Orchestrator Integration (CODE COMPLETE - TEST ONLY):
claude --dangerously-skip-permissions -p \
  "Read research/orchestration_integration_handoff.md. All code is written. \
   Your job is to: 1) Start llama-server instances, 2) Run tests, \
   3) Fix any failures, 4) Run benchmarks until >50% cache hit rate."

# MathSmith Re-conversion: ✅ COMPLETE - Q4_K_M downloaded from mradermacher
# Path: /mnt/raid0/llm/models/MathSmith-Hard-Problem-Synthesizer-Qwen3-8B.Q4_K_M.gguf
```

### When Model Servers Running

```bash
# Start test server (after benchmark completes)
/mnt/raid0/llm/llama.cpp/build/bin/llama-server \
  -m /mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q8_0.gguf \
  --host 0.0.0.0 --port 8080 -c 4096 -np 4 -t 16

# Enable real inference mode in orchestrator
# See: research/orchestrator_handoff.md
```

### When llama.cpp PR #15225 Merges (MTP Support)

```bash
# 1. Check if PR is merged
# https://github.com/ggml-org/llama.cpp/pull/15225

# 2. Update and rebuild llama.cpp
cd /mnt/raid0/llm/llama.cpp
git pull origin master
cmake --build build --config Release -j 96

# 3. Test MTP on GLM-4.6
# See: research/mtp_investigation.md for full details
numactl --interleave=all /mnt/raid0/llm/llama.cpp/build/bin/llama-cli \
  -m /mnt/raid0/llm/lmstudio/models/lmstudio-community/glm-4-9b-0414-GGUF/glm-4-9b-0414-Q4_K_M.gguf \
  --mtp 2 -t 96 -n 100 \
  -p "Write a Python function to sort a list:"
```

---

## Completion Tracking

### Draft Model Benchmarks (Speed Tests)
- [x] Gemma-3-1B → Gemma-3-12B-IT (K=8,16,24) — WORKS (upstream b7684+)
- [x] Gemma-3-1B → Gemma-3-27B-IT-QAT (K=8,16,24) — WORKS (42-81% acceptance, PR #18720)
- [ ] Qwen3-1.7B → Qwen3-32B (K=8,16,24)
- [ ] Qwen3-0.6B → Qwen3-32B (K=8,16,24)
- [ ] Qwen3-1.7B → Qwen3-235B-A22B + MoE4 (K=8,16)
- [ ] jukofyork-0.75B → Qwen3-Coder-30B + MoE6 (K=8,16,24)
- [ ] jukofyork-0.75B → Qwen3-Coder-480B + MoE3 (if 30B works)
- [ ] Documentation updated (registry, RESULTS_SUMMARY, etc.)

### Formalizer Evaluation
- [ ] MathSmith-Qwen3-8B evaluated (problem formalization)
- [ ] xLAM-2-1B-fc-r evaluated (tool sequences)
- [ ] xLAM-1B-fc-r evaluated (tool sequences)
- [ ] NexusRaven-V2-13B evaluated (complex functions)
- [ ] Results compared (parsability, completeness, speed)
- [ ] Best model added to `model_registry.yaml`
- [ ] `research/formalizer_evaluation.md` written

### Tree Speculation
- [ ] Benchmark complete (`n_parallel` × `p_split` sweep)
- [ ] Optimal parameters identified
- [ ] Results added to `RESULTS_SUMMARY.md`
- [ ] `model_registry.yaml` updated with tree params

### RadixAttention (YOLO Agent) — ✅ COMPLETE (2026-01-07)
- [x] Phase A: Persistent server mode (`src/backends/llama_server.py`)
- [x] Phase B: Sticky slot routing (`src/prefix_cache.py` - PrefixRouter)
- [x] Phase C: Prompt canonicalization (`src/prefix_cache.py` - canonicalize_prompt)
- [x] Phase D: Radix tree cache (`src/radix_cache.py`)
- [x] Phase E: Slot persistence (`src/prefix_cache.py` - save/restore_hot_prefixes)
- [x] Unit tests: 46/46 passing (`tests/unit/test_prefix_cache.py`)
- [ ] Integration benchmark (requires running llama-server)

### Orchestrator Integration (CODE COMPLETE) — 9 Phases
- [x] Phase 1: Server infrastructure (manual startup commands in handoff)
- [x] Phase 2: LLM Primitives integration (`src/llm_primitives.py`)
- [x] Phase 3: Model server factory (`src/model_server.py`)
- [x] Phase 4: Registry update (`orchestration/model_registry.yaml`)
- [x] Phase 5: Integration tests (`tests/integration/test_cache_integration.py`)
- [x] Phase 6: Benchmark script (`scripts/benchmark/bench_cache_performance.py`)
- [x] Phase 7: API integration (`src/api.py` - real_mode param)
- [x] **Phase 8: Root LM Loop** (`src/api.py` - recursive pattern implemented)
- [x] Phase 9: E2E validation (`scripts/test_recursive_orchestration.py`)
- [ ] Cache hit rate >50% on RLM workloads (YOLO agent to verify)
- [ ] Root LM completes multi-turn tasks (YOLO agent to verify)

### MathSmith Re-conversion — ✅ COMPLETE (2026-01-08)
- [x] Downloaded Q4_K_M from mradermacher (no re-conversion needed)
- [x] Path: `/mnt/raid0/llm/models/MathSmith-Hard-Problem-Synthesizer-Qwen3-8B.Q4_K_M.gguf` (4.7GB)
- [ ] Verify speed (~40-60 t/s expected) — blocked on benchmark
- [ ] Run formalizer benchmark — blocked on benchmark
- [ ] Update model registry

### Orchestrator Real Mode
- [ ] Model servers started (ports 8080-8088)
- [ ] `llm_call()` verified with real inference
- [ ] `llm_batch()` verified with parallel calls
- [ ] End-to-end TaskIR → execution tested

### GLM-4.6 MTP Testing (Blocked on PR #15225)
- [ ] llama.cpp PR #15225 merged
- [ ] Rebuild llama.cpp with MTP support
- [ ] Test MTP on GLM-4.6 (n_mtp=2, n_mtp=3)
- [ ] Benchmark acceptance rates and throughput
- [ ] Compare results with vLLM baseline
- [ ] Update model_registry.yaml with MTP performance
- [ ] Correct GLM-4.6 entry (remove incorrect MoE optimization reference)

---

## Paged Attention Copy-on-Write (CoW)

**Blocked on**: Current paged attention PR must be submitted and reviewed first.

**Purpose**: Enable prefix sharing between sequences (e.g., shared system prompts in llama-server).

**Expected benefit**: Additional 10-30% memory savings in multi-sequence scenarios.

**Implementation plan**: See `handoffs/active/paged-attention.md` Section 9 for:
- Infrastructure already in place (ref_count, is_shared(), add_ref())
- What needs to be implemented (seq_cp, cpy_k, cpy_v modifications)
- Test cases to add
- Benchmark plan

**Resume command**:
```bash
cd /mnt/raid0/llm/llama.cpp-experimental
git checkout feature/paged-attention
cat handoffs/active/paged-attention.md  # Section 9 has full plan
```

---

## Notes

- **Benchmark ETA**: Check with `./run_benchmark.py --status` or `pgrep -af llama`
- **Formalizer models**: Already downloaded to `/mnt/raid0/llm/models/`
- **Tree speculation**: Script at `scripts/benchmark/bench_tree_speculation.sh`
- **RadixAttention**: Full implementation plan in `research/radix_attention_handoff.md`
- **MTP for GLM-4.6**: Self-speculative decoding using built-in heads. See `research/mtp_investigation.md`
