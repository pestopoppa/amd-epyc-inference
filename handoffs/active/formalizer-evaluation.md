# Research Handoff: Formalizer Model Evaluation

**Created**: 2026-01-07
**Status**: Ready for Evaluation
**Prerequisites**: Background benchmark must complete first

---

## Overview

This handoff continues the formalizer investigation from `FORMALIZER_INVESTIGATION.md`. Infrastructure is ready, models are downloaded, evaluation can begin when benchmarks complete.

---

## What's Ready

### 1. Downloaded Models

| Model | Path | Size | Use Case |
|-------|------|------|----------|
| xLAM-2-1B-fc-r | `/mnt/raid0/llm/models/xLAM-2-1B-fc-r-Q4_K_M.gguf` | 941 MB | Tool formalization (v2) |
| xLAM-1b-fc-r | `/mnt/raid0/llm/models/xLAM-1b-fc-r.Q4_K_M.gguf` | 833 MB | Tool formalization (v1) |
| NexusRaven-V2-13B | `/mnt/raid0/llm/models/nexusraven-v2-13b.Q4_K_M.gguf` | 7.4 GB | Complex function calling |

### 2. Extended Schema

`orchestration/formalization_ir.schema.json` now includes:

```json
// New problem types
"problem_type": ["tool_orchestration", "architecture", "workflow", ...]

// Tool sequence for tool_orchestration problems
"tool_sequence": [{
  "step": 0,
  "tool": "download_file",
  "arguments": {"url": "..."},
  "depends_on": [],
  "output_var": "data",
  "error_handling": "fail"
}]

// Architecture spec for architecture problems
"architecture": {
  "components": [...],
  "interfaces": [...],
  "data_flow": [...],
  "quality_attributes": [...]
}
```

### 3. Benchmark Suite

**Location**: `benchmarks/prompts/v1/formalizer/`

```
formalizer/
├── tool_formalization/
│   ├── t1_simple_sequence.txt
│   ├── t2_parallel_tools.txt
│   └── t3_conditional_flow.txt
├── architecture_formalization/
│   ├── t1_simple_app.txt
│   └── t2_microservices.txt
├── verification_formalization/
│   ├── t1_null_safety.txt
│   └── t2_invariant.txt
└── ground_truth/
    ├── t1_simple_sequence.json
    ├── t2_parallel_tools.json
    ├── t3_conditional_flow.json
    ├── arch_t1_simple_app.json
    ├── verif_t1_null_safety.json
    └── verif_t2_invariant.json
```

### 4. Evaluation Scripts

**Single model**: `scripts/benchmark/bench_formalizers.sh`
**All models**: `scripts/benchmark/run_all_formalizers.sh` (NEW - 2026-01-09)

**Quick Start (run in background)**:
```bash
# Run all 3 formalizers in background (~10-20 minutes)
nohup ./scripts/benchmark/run_all_formalizers.sh > /dev/null 2>&1 &

# Check progress
tail -f logs/formalizer_eval/run.log

# View final comparison
cat logs/formalizer_eval/run.log | tail -20
```

**Single model (for debugging)**:
```bash
./scripts/benchmark/bench_formalizers.sh \
  --model /mnt/raid0/llm/models/xLAM-2-1B-fc-r-Q4_K_M.gguf \
  --prompts benchmarks/prompts/v1/formalizer/
```

**Scoring Criteria**:
- **Parsability** (0/1): Is output valid JSON?
- **Schema validity** (0/1): Does it match FormalizationIR schema?
- **Completeness** (0-1): Fraction of expected fields present
- **Speed**: Tokens per second

---

## Evaluation Plan

### Phase 1: Tool Formalizer Comparison

Run all three models on tool formalization tests:

```bash
# Test xLAM-2-1B (newest, recommended)
./scripts/benchmark/bench_formalizers.sh \
  --model /mnt/raid0/llm/models/xLAM-2-1B-fc-r-Q4_K_M.gguf \
  --prompts benchmarks/prompts/v1/formalizer/ \
  --output /mnt/raid0/llm/claude/logs/formalizer_eval/xLAM-2-1B

# Test xLAM-1B (older version)
./scripts/benchmark/bench_formalizers.sh \
  --model /mnt/raid0/llm/models/xLAM-1b-fc-r.Q4_K_M.gguf \
  --prompts benchmarks/prompts/v1/formalizer/ \
  --output /mnt/raid0/llm/claude/logs/formalizer_eval/xLAM-1B

# Test NexusRaven (larger, better at complex functions)
./scripts/benchmark/bench_formalizers.sh \
  --model /mnt/raid0/llm/models/nexusraven-v2-13b.Q4_K_M.gguf \
  --prompts benchmarks/prompts/v1/formalizer/ \
  --output /mnt/raid0/llm/claude/logs/formalizer_eval/NexusRaven
```

### Phase 2: Analysis

Create comparison matrix:

| Model | Parsability | Completeness | Speed | Recommendation |
|-------|-------------|--------------|-------|----------------|
| xLAM-2-1B | ?/7 | ? | ? t/s | ? |
| xLAM-1B | ?/7 | ? | ? t/s | ? |
| NexusRaven | ?/7 | ? | ? t/s | ? |

### Phase 3: Model Registry Update

If a model passes (>70% completeness, 100% parsability, >10 t/s):

Update `orchestration/model_registry.yaml`:
```yaml
tool_formalizer:
  tier: D  # Preprocessing tier
  description: Tool sequence formalizer
  model:
    name: xLAM-2-1B-fc-r-Q4_K_M
    path: /mnt/raid0/llm/models/xLAM-2-1B-fc-r-Q4_K_M.gguf
    quant: Q4_K_M
    size_gb: 0.9
  candidate_roles: [tool_formalizer]
```

Add routing rule:
```yaml
- if: "task_type == 'agentic' and tool_count > 3"
  preprocess: tool_formalizer
```

---

## Key Findings from Research

### xLAM-2-1B Advantages
- Outperforms GPT-4o on BFCL benchmark
- Only 1B params (fast preprocessing)
- Official GGUF from Salesforce
- Verified working on our hardware (91.2 t/s prompt, 9.4 t/s gen)

### NexusRaven Advantages
- Better at nested/composite functions
- Generates explanations (can disable to save tokens)
- 7% better than GPT-4 on complex function calling

### Why Not Use llama.cpp Native Tool Calling?
- Native tool calling (`--jinja` flag) is for execution, not formalization
- Formalizer converts vague tasks → structured plan BEFORE execution
- Different purpose: planning vs doing

---

## Success Criteria

A formalizer model is suitable if:

| Criterion | Threshold |
|-----------|-----------|
| Parsability | 100% (must output valid JSON) |
| Completeness | >70% of expected fields |
| Speed | >10 t/s (formalizers must not bottleneck) |
| Schema validity | >80% valid FormalizationIR |

---

## Files to Update After Evaluation

1. `research/formalizer_evaluation.md` - Results and comparison
2. `orchestration/model_registry.yaml` - Add winning model
3. `research/FORMALIZER_INVESTIGATION.md` - Mark complete
4. `orchestration/progress/PROGRESS_2026-01-XX.md` - Final report

---

## References

- Original investigation: `research/FORMALIZER_INVESTIGATION.md`
- Schema: `orchestration/formalization_ir.schema.json`
- Plan file: `/home/daniele/.claude/plans/twinkly-sniffing-crescent.md`
- xLAM paper: https://huggingface.co/Salesforce/xLAM-2-1b-fc-r
- NexusRaven: https://huggingface.co/Nexusflow/NexusRaven-V2-13B
