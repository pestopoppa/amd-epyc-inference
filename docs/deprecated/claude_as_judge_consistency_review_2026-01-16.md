# DEPRECATED: Claude-as-Judge Blind Re-Scoring

> **Deprecated**: 2026-01-30
> **Reason**: Superseded by on-the-fly benchmark dataset adapters (`scripts/benchmark/dataset_adapters.py`)
> which sample from 31,820 real public dataset questions. The static `prompts/v1/*.yaml` files
> referenced below are now debug-only suites.
>
> **Completed work preserved**: `benchmarks/results/reviews/BLIND_RESCORE_2026-01-16.md` (77 models scored)
> and `benchmarks/results/reviews/*.csv` (131 CSV files) remain valid historical records.
>
> **If rescoring is needed**: Create a fresh handoff using the new dataset adapters rather than
> reviving this document. Ground truth now comes from the datasets themselves.

---

**Date Created:** 2026-01-16
**Purpose:** BLIND re-score ALL benchmark results from scratch
**Agent:** YOLO Agent
**Status:** DEPRECATED (was NEEDS REFRESH)
**Updated:** 2026-01-30

---

## Mission

**Score ALL model responses from scratch.**

You will evaluate model responses using ONLY the prompt YAML reference answers as ground truth. The previous scores have been made unreadable - you cannot see them.

**Your task:**
1. Read prompt YAMLs for reference answers (ground truth)
2. Score each model's benchmark responses
3. Output your scores to a new file

---

## Directory Structure

> **WARNING (2026-01-30):** Benchmark suites have been rebuilt to use real public datasets
> sampled on-the-fly from HuggingFace cache. The static `prompts/v1/*.yaml` files are now
> **debug suites only** (hand-written approximations, NOT canonical ground truth).
>
> **For new rescoring**, use the on-the-fly dataset adapters in
> `scripts/benchmark/dataset_adapters.py` which sample from 31,820 real questions:
> - MMLU (14,042) — general knowledge
> - GSM8K (1,319) + MATH-500 (500) — math reasoning
> - HumanEval (164) + MBPP (500) — code generation
> - ARC-Challenge (1,172) + HellaSwag (10,042) — thinking/reasoning
> - IFEval (541) — instruction precision
> - OCRBench (1,000) + ChartQA (2,500) — vision-language
>
> Ground truth answers come from the datasets themselves (not hand-written expected fields).

```
/mnt/raid0/llm/claude/benchmarks/
├── prompts/v1/                    # DEBUG SUITES (hand-written, use for quick testing only)
│   ├── thinking.yaml              # 10 questions with reference_answer
│   ├── general.yaml               # 10 questions with reference_answer
│   ├── math.yaml                  # 10 questions with reference_answer
│   ├── agentic.yaml               # 10-11 questions with expected tool calls
│   ├── coder.yaml                 # 10 questions with reference_answer
│   ├── instruction_precision.yaml # 11 questions with strict format requirements
│   ├── long_context.yaml          # 5-9 questions with reference_answer
│   └── vl.yaml                    # Vision-language questions
├── prompts/debug/                 # EXPANDED DEBUG SUITES (40+ questions each, also hand-written)
│
├── scripts/benchmark/
│   └── dataset_adapters.py        # ON-THE-FLY sampling from real datasets (31,820 questions)
│
└── results/
    ├── runs/20251220_214317/      # MODEL RESPONSES - score these
    │   ├── {role}_{model}_baseline.json
    │   ├── {role}_{model}_moe{N}.json
    │   └── ...
    │
    └── reviews/
        └── BLIND_RESCORE_2026-01-16.md  # YOUR OUTPUT (create this)
```

**Note:** The `reviews/*.csv` files are intentionally unreadable. Do not attempt to read them.

---

## Scoring Rubric (0-3 Scale)

| Score | Meaning | Criteria |
|-------|---------|----------|
| **3** | Correct | Answer matches reference_answer in substance. Minor variations OK. |
| **2** | Partial | Correct approach but incomplete, OR correct answer but poor reasoning |
| **1** | Attempt | Wrong answer but shows understanding of the problem type |
| **0** | Fail | Empty, garbage, unrelated, or completely wrong |

---

## Known Quirks

### 1. VL Models on Text-Only Tasks
VL models output help menu when given text-only prompts.
**Score:** 0 for text-only suites (general, agentic)

### 2. Prompt Echo in Small Models
Draft models (0.5B-1.5B) often echo the prompt before answering.
**Score:** Score the answer AFTER the echo, not the echo itself

### 3. Repetition Loops
**Score:** 2 if first instance correct, 0-1 if wrong

### 4. Empty Responses
Check for empty string, whitespace, `<think>` tags with no content.
**Score:** 0

### 5. Truncated Responses
**Score:** 2 if answer was reached before truncation, 1 if not

---

## Completion Record

- 77 baseline models blind-rescored (2026-01-16)
- Results: `benchmarks/results/reviews/BLIND_RESCORE_2026-01-16.md`
- 131 CSV review files in `benchmarks/results/reviews/`
- Drift check completed with corrections documented
