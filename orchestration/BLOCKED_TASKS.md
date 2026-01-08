# Blocked Tasks

**Last Updated**: 2026-01-08
**Blocking Resource**: PR #15225 (MTP loader)

---

## Quick Status

| Task | Blocked On | Priority | Handoff | Status |
|------|------------|----------|---------|--------|
| **MTP Refactoring** | PR #15225 merge | **HIGH** | `research/mtp_investigation.md` | ✅ PLAN READY |
| AVX-512 VNNI Q8_0 | — | — | — | ❌ NOT SUBMITTING (8% speedup) |
| Draft model benchmarks | Benchmark completion | HIGH | `research/draft_benchmark_handoff.md` | Blocked |
| Formalizer eval | Benchmark completion | HIGH | `research/formalizer_handoff.md` | Blocked |
| Tree speculation | Benchmark completion | HIGH | `research/kernel_dev_handoff.md` | Blocked |
| RadixAttention | — | — | `research/radix_attention_handoff.md` | ✅ COMPLETE |
| Orchestrator integration | Model servers | HIGH | `research/orchestration_integration_handoff.md` | ✅ CODE COMPLETE |
| MathSmith re-conversion | — | LOW | `research/mathsmith_reconversion_handoff.md` | ✅ COMPLETE |
| Orchestrator real mode | Model servers | LOW | `research/orchestrator_handoff.md` | Blocked |
| Frontend Architecture | — | — | `research/orchestrator_handoff.md` | ✅ COMPLETE |
| CLI Parity Features | — | — | `research/orchestrator_handoff.md` | ✅ COMPLETE |

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
- [ ] Gemma-3-1B → Gemma-3-12B-IT (K=8,16,24)
- [ ] Gemma-3-1B → Gemma-3-27B-IT-QAT (K=8,16,24)
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

## Notes

- **Benchmark ETA**: Check with `./run_benchmark.py --status` or `pgrep -af llama`
- **Formalizer models**: Already downloaded to `/mnt/raid0/llm/models/`
- **Tree speculation**: Script at `scripts/benchmark/bench_tree_speculation.sh`
- **RadixAttention**: Full implementation plan in `research/radix_attention_handoff.md`
- **MTP for GLM-4.6**: Self-speculative decoding using built-in heads. See `research/mtp_investigation.md`
