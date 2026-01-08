# Draft Model & Formalizer Benchmark Handoff

**Created:** 2026-01-08
**Status:** Ready for execution (blocked by ongoing benchmarks)
**Priority:** HIGH

---

## Overview

This handoff covers pending speed benchmarks for draft models and formalizer evaluation. Quality scores already exist for target models - **only speed/acceptance testing needed**.

---

## Part 1: Draft Model Speed Tests

### 1.1 Gemma-3 Family (NEVER TESTED)

**Draft Model:** `/mnt/raid0/llm/models/gemma-3-1b-it-Q8_0.gguf` (1.0 GB)

| Target Model | Quality Score | Baseline t/s | Test K Values |
|--------------|---------------|--------------|---------------|
| Gemma-3-27B-IT-QAT | 93.3% | 2.0 t/s | 8, 16, 24 |
| Gemma-3-12B-IT | (available) | 9.4 t/s | 8, 16, 24 |

**Command Template:**
```bash
numactl --interleave=all /mnt/raid0/llm/llama.cpp/build/bin/llama-speculative \
  -m /mnt/raid0/llm/lmstudio/models/lmstudio-community/gemma-3-27B-it-qat-GGUF/gemma-3-27B-it-QAT-Q4_0.gguf \
  -md /mnt/raid0/llm/models/gemma-3-1b-it-Q8_0.gguf \
  --draft-max {K} -t 96 -n 100 \
  -p "Write a Python function to sort a list:"
```

### 1.2 Qwen3 Dense Family (BASELINE ONLY - NO SPEC DECODE TESTED)

**Draft Models:**
- `/mnt/raid0/llm/models/Qwen3-1.7B-Q8_0.gguf` (1.7 GB)
- `/mnt/raid0/llm/models/Qwen_Qwen3-0.6B-Q8_0.gguf` (768 MB)

| Target Model | Quality Score | Baseline t/s | Test K Values |
|--------------|---------------|--------------|---------------|
| Qwen3-32B-Q4_K_M | (available) | ~5 t/s | 8, 16, 24 |
| Qwen3-235B-A22B + MoE4 | 88% | 6.75 t/s | 8, 16 |

**NOTE:** Qwen3-235B was incorrectly marked as "SSM hybrid (no spec)" in docs. It's pure MoE - spec decode should work.

**Command Template (235B with MoE + spec decode):**
```bash
numactl --interleave=all /mnt/raid0/llm/llama.cpp/build/bin/llama-speculative \
  -m /mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-Q4_K_M-00001-of-00004.gguf \
  -md /mnt/raid0/llm/models/Qwen3-1.7B-Q8_0.gguf \
  --override-kv qwen3moe.expert_used_count=int:4 \
  --draft-max {K} -t 96 -n 100 \
  -p "Explain the CAP theorem:"
```

### 1.3 Qwen3-Coder Family (NEW DRAFT - jukofyork)

**Draft Model:** `/mnt/raid0/llm/models/Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf` (448 MB)
- Specifically designed for 480B with vocabulary transplant to fix BOS mismatch
- Should work with entire Qwen3-Coder family (same tokenizer)

| Target Model | Quality Score | Current Best | Test K Values |
|--------------|---------------|--------------|---------------|
| Qwen3-Coder-30B-A3B + MoE6 | 90% | 18.3 t/s | 8, 16, 24 |
| Qwen3-Coder-53B-A3B + MoE4 | (available) | 30.4 t/s | 8, 16, 24 |
| Qwen3-Coder-480B + MoE3 | (available) | 10.3 t/s | 8, 16, 24 |

**Command Templates:**

*30B (Quick test - verify draft compatibility first):*
```bash
numactl --interleave=all /mnt/raid0/llm/llama.cpp/build/bin/llama-speculative \
  -m /mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
  -md /mnt/raid0/llm/models/Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf \
  --override-kv qwen3moe.expert_used_count=int:6 \
  --draft-max {K} -t 96 -n 100 \
  -p "Write a Python function to merge two sorted lists:"
```

*53B (Quick test - same tokenizer as 30B):*
```bash
numactl --interleave=all /mnt/raid0/llm/llama.cpp/build/bin/llama-speculative \
  -m "/mnt/raid0/llm/lmstudio/models/mradermacher/Qwen3-Coder-53B-A3B-Instruct-TOTAL-RECALL-v2-MASTER-CODER-L-i1-GGUF/Qwen3-Coder-53B-A3B-Instruct-TOTAL-RECALL-v2-MASTER-CODER-L.i1-Q4_K_M.gguf" \
  -md /mnt/raid0/llm/models/Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf \
  --override-kv qwen3moe.expert_used_count=int:4 \
  --draft-max {K} -t 96 -n 100 \
  -p "Write a Python function to merge two sorted lists:"
```

*480B (Full test after 30B succeeds):*
```bash
numactl --interleave=all /mnt/raid0/llm/llama.cpp/build/bin/llama-speculative \
  -m /mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00008.gguf \
  -md /mnt/raid0/llm/models/Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf \
  --override-kv qwen3moe.expert_used_count=int:3 \
  --draft-max {K} -t 96 -n 100 \
  -p "Write a Python function to merge two sorted lists:"
```

### 1.4 Qwen3 Thinking Family (MoE - standard Qwen3 drafts)

**Draft Models:**
- `/mnt/raid0/llm/models/Qwen3-1.7B-Q8_0.gguf` (1.7 GB)
- `/mnt/raid0/llm/models/Qwen_Qwen3-0.6B-Q8_0.gguf` (768 MB)

| Target Model | Quality Score | Current Best | Test K Values |
|--------------|---------------|--------------|---------------|
| Qwen3-30B-A3B-Thinking-2507 + MoE6 | (available) | TBD | 8, 16, 24 |

**NOTE:** This is base Qwen3 MoE (not Qwen3-Coder), so use standard Qwen3 drafts, not jukofyork.

**Command Template:**
```bash
numactl --interleave=all /mnt/raid0/llm/llama.cpp/build/bin/llama-speculative \
  -m /mnt/raid0/llm/lmstudio/models/unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF/Qwen3-30B-A3B-Thinking-2507-Q4_K_S.gguf \
  -md /mnt/raid0/llm/models/Qwen3-1.7B-Q8_0.gguf \
  --override-kv qwen3moe.expert_used_count=int:6 \
  --draft-max {K} -t 96 -n 100 \
  -p "Solve this step by step: What is 17 * 23?"
```

---

## Part 2: Formalizer Evaluation

### Models to Benchmark

| Model | Path | Size | Purpose |
|-------|------|------|---------|
| MathSmith-Qwen3-8B | `/mnt/raid0/llm/models/MathSmith-Hard-Problem-Synthesizer-Qwen3-8B.Q4_K_M.gguf` | 4.7 GB | Problem formalization |
| xLAM-2-1B-fc-r | `/mnt/raid0/llm/models/xLAM-2-1B-fc-r-Q4_K_M.gguf` | 986 MB | Tool sequence (newest) |
| xLAM-1b-fc-r | `/mnt/raid0/llm/models/xLAM-1b-fc-r.Q4_K_M.gguf` | 873 MB | Tool sequence (legacy) |
| NexusRaven-V2-13B | `/mnt/raid0/llm/models/nexusraven-v2-13b.Q4_K_M.gguf` | 7.4 GB | Complex functions |

### Evaluation Script

```bash
/mnt/raid0/llm/claude/scripts/benchmark/bench_formalizers.sh \
  --model {MODEL_PATH} \
  --prompts /mnt/raid0/llm/claude/benchmarks/prompts/v1/formalizer/
```

### Test Cases (7 total with ground truth)
- `tool_formalization/t1_simple_sequence.txt`
- `tool_formalization/t2_parallel_tools.txt`
- `tool_formalization/t3_conditional_flow.txt`
- `architecture_formalization/t1_simple_app.txt`
- `architecture_formalization/t2_microservices.txt`
- `verification_formalization/t1_null_safety.txt`
- `verification_formalization/t2_invariant.txt`

### Success Criteria
- Parsability: 100% (valid JSON)
- Completeness: >70% of expected fields
- Speed: >10 t/s

---

## Part 3: Test Execution Order

### Batch 1: Quick Tests (~30 min)
1. Gemma-3-1B → Gemma-3-12B-IT (fastest target)
2. Qwen3-1.7B → Qwen3-32B
3. Qwen3-0.6B → Qwen3-32B
4. jukofyork draft → Qwen3-Coder-30B + MoE6 (verify draft compatibility)
5. Qwen3-1.7B → Qwen3-30B-A3B-Thinking + MoE6 (base Qwen3 MoE)
6. jukofyork draft → Qwen3-Coder-53B + MoE4 (same tokenizer as 30B)

### Batch 2: Large Model Tests (~1 hr)
7. Gemma-3-1B → Gemma-3-27B-IT-QAT
8. Qwen3-1.7B → Qwen3-235B-A22B + MoE4
9. jukofyork draft → Qwen3-Coder-480B + MoE3 (if 30B succeeds)

### Batch 3: Formalizers (~1.5 hr)
10. MathSmith-Qwen3-8B evaluation (problem formalization)
11. xLAM-2-1B evaluation (tool sequences)
12. xLAM-1b evaluation (tool sequences)
13. NexusRaven-V2-13B evaluation (complex functions)

---

## Part 4: Expected Outputs

### Speed Test Results Format
```
| Draft | Target | K | Acceptance % | t/s | Speedup |
```

### Registry Updates Needed
After testing, update `orchestration/model_registry.yaml`:
- Add `compatible_targets` with acceptance rates
- Update `performance.raw_tps` for drafts
- Add `benchmark_date`
- Remove incorrect "SSM" constraint from Qwen3-235B if spec works

### Files to Update

**Benchmark Results (raw data):**
- `benchmarks/results/runs/{timestamp}/` - timestamped run data
- `benchmarks/results/reviews/{model}_baseline.csv` - per-model review CSVs
- `benchmarks/results/index.jsonl` - benchmark index

**Reference Documentation (human-readable summaries):**
- `docs/reference/benchmarks/RESULTS.md` - canonical benchmark results

**Model Configuration:**
- `orchestration/model_registry.yaml` - draft performance data, compatible_targets
- `docs/reference/models/QUIRKS.md` - if new quirks/constraints discovered
- `CLAUDE.md` - only if critical new constraints discovered

**Task Tracking:**
- `handoffs/blocked/BLOCKED.md` - mark tasks complete
- `handoffs/active/formalizer-evaluation.md` - update formalizer status

**Progress Log:**
- `progress/YYYY-MM/YYYY-MM-DD.md` - daily summary

---

## Checklist

### Draft Model Speed Tests
- [ ] Gemma-3-1B → Gemma-3-12B-IT (K=8,16,24)
- [ ] Gemma-3-1B → Gemma-3-27B-IT-QAT (K=8,16,24)
- [ ] Qwen3-1.7B → Qwen3-32B (K=8,16,24)
- [ ] Qwen3-0.6B → Qwen3-32B (K=8,16,24)
- [ ] Qwen3-1.7B → Qwen3-235B-A22B + MoE4 (K=8,16)
- [ ] Qwen3-1.7B → Qwen3-30B-A3B-Thinking + MoE6 (K=8,16,24) - base Qwen3 MoE
- [ ] jukofyork-0.75B → Qwen3-Coder-30B + MoE6 (K=8,16,24) - compatibility test
- [ ] jukofyork-0.75B → Qwen3-Coder-53B + MoE4 (K=8,16,24) - same tokenizer as 30B
- [ ] jukofyork-0.75B → Qwen3-Coder-480B + MoE3 (K=8,16,24) - if 30B works

### Formalizer Evaluation
- [ ] MathSmith-Qwen3-8B evaluated (problem formalization)
- [ ] xLAM-2-1B-fc-r evaluated (tool sequences)
- [ ] xLAM-1b-fc-r evaluated (tool sequences)
- [ ] NexusRaven-V2-13B evaluated (complex functions)
- [ ] Results compared
- [ ] Best model per category documented

### Documentation Updates
- [ ] model_registry.yaml updated with draft performance data
- [ ] docs/reference/benchmarks/RESULTS.md updated
- [ ] benchmarks/results/runs/ data created
- [ ] benchmarks/results/reviews/ CSVs created
- [ ] handoffs/blocked/BLOCKED.md checklist updated
- [ ] handoffs/active/formalizer-evaluation.md status updated
- [ ] CLAUDE.md updated if new quirks found (rare)
- [ ] Progress log entry created

---

## Resume Command

```bash
# When ready to execute, start with Batch 1 quick tests:
cd /mnt/raid0/llm/claude
# Follow commands in Part 1 sections above
```
