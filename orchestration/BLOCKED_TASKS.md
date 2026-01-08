# Blocked Tasks

**Last Updated**: 2026-01-08
**Blocking Resource**: Background benchmark (Qwen3-Coder-30B)

---

## Quick Status

| Task | Blocked On | Priority | Handoff | Status |
|------|------------|----------|---------|--------|
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

### When Benchmark Completes

```bash
# Check if benchmark is still running
pgrep -af llama-completion

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

---

## Completion Tracking

### Formalizer Evaluation
- [ ] xLAM-2-1B-fc-r evaluated
- [ ] xLAM-1B-fc-r evaluated
- [ ] NexusRaven-V2-13B evaluated
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

---

## Notes

- **Benchmark ETA**: Check with `./run_benchmark.py --status` or `pgrep -af llama`
- **Formalizer models**: Already downloaded to `/mnt/raid0/llm/models/`
- **Tree speculation**: Script at `scripts/benchmark/bench_tree_speculation.sh`
- **RadixAttention**: Full implementation plan in `research/radix_attention_handoff.md`
