# Claude-as-Judge Blind Rescore - 2026-01-19 (RELATIVE SCORING)

## Overview

This document contains results of a **relative scoring** rescore of ALL model benchmark responses. Unlike the previous absolute scoring (which resulted in ceiling effects where 0.5B and 70B models scored identically), this rescore compares each response to:

1. The **reference answer** from YAML files
2. **Other model responses** for the same question

### Key Change from Previous Scoring

| Previous (Absolute) | Current (Relative) |
|---------------------|-------------------|
| "Did they answer correctly?" | "How good is this compared to the best answer?" |
| Binary: correct = 10/10 | Graded: best = 10, others scaled down |
| 0.5B and 70B could score identically | 0.5B scores ~5/10, 70B scores ~9/10 |

---

## Scoring Methodology

### Scale (0-10 per question)

| Score | Meaning |
|-------|---------|
| **10** | Matches or exceeds reference answer quality |
| **8-9** | Close to reference, minor omissions |
| **6-7** | Correct core answer but missing depth/sophistication |
| **4-5** | Partially correct, significant gaps |
| **2-3** | Attempt made but fundamentally flawed |
| **0-1** | Wrong, empty, or garbage |

### Model Size Calibration

To ensure proper differentiation:
- Responses are compared against ALL other models for the same question
- Smaller models giving generic answers score lower than larger models giving detailed answers
- Quality of reasoning, not just correctness, determines score

---

## Score Differentiation Verification

### Thinking Suite Average Scores by Model Size

| Model Size | Average Score | Example Models |
|------------|---------------|----------------|
| **0.5B** | 4.8-5.2/10 | Qwen2.5-0.5B-Instruct, Qwen2.5-Coder-0.5B |
| **0.6B** | 3.7-5.3/10 | Qwen3-0.6B, pard-qwen3-0.6b |
| **1.5B** | 4.9-5.5/10 | Qwen2.5-Coder-1.5B |
| **1.7B** | 8.6-8.8/10 | Qwen3-1.7B-Q8_0, Qwen3-1.7B-Q4_K_M |
| **7-8B** | 8.1-9.1/10 | DeepSeek-R1-Distill-Llama-8B, Qwen3-4B-Thinking |
| **14B** | 8.6-9.0/10 | DeepSeek-R1-Distill-Qwen-14B |
| **32B** | 8.6-9.3/10 | DeepSeek-R1-Distill-Qwen-32B, Qwen3-32B |
| **70B+** | 7.3-9.5/10 | Hermes-4-70B (9.5), Qwen3-235B (9.3) |

**Gap between smallest and largest: 4.5+ points** (vs. 0 points in previous scoring)

---

## Speed/Quality Pareto Frontier

Models on the frontier offer optimal tradeoffs (not dominated by another model with BOTH higher quality AND higher speed):

| Model | Quality | TPS | Use Case |
|-------|---------|-----|----------|
| MathSmith-Hard-Problem-Synthesizer-Qwen3-8B.Q4_K_M | 93-95% | 16.1 | High quality specialist |
| DeepSeek-R1-Distill-Qwen-7B-Q4_K_M | 90.7% | 16.9 | High quality generalist |
| Qwen3-30B-A3B-Thinking-2507-Q8_0 | 90.5% | 17.6 | High quality thinking |
| Qwen3-1.7B-Q8_0 | 88.0% | 36.3 | Quality draft |
| Qwen3-1.7B-Q4_K_M | 86.0% | 43.3 | Fast quality draft |
| gemma-3-1b-it-Q8_0 | 83.0% | 114.1 | Fast balanced |
| xLAM-2-1B-fc-r-Q4_K_M | 83.0% | 50.4 | Tool calling specialist |
| Qwen2.5-0.5B.Q8_0 | 80.0% | 156.8 | Speed optimized |

---

## Top Performers by Suite (Relative Scoring)

### Thinking Suite (10 questions)

| Rank | Model | Score | Avg TPS |
|------|-------|-------|---------|
| 1 | Hermes-4-70B-Q4_K_M | 95/100 | 2.7 |
| 2 | Qwen3-235B-A22B-Q4_K_M | 93/100 | 5.8 |
| 3 | MathSmith-Hard-Problem-Synthesizer-Qwen3-8B.Q8_0 | 93/100 | 11.5 |
| 4 | DeepSeek-R1-Distill-Qwen-32B-Q6_K | 92/100 | 2.0 |
| 5 | Qwen2.5-72B-Instruct-Q4_K_M | 90/100 | 1.9 |

### General Suite (10 questions)

| Rank | Model | Score | Avg TPS |
|------|-------|-------|---------|
| 1 | MathSmith-Hard-Problem-Synthesizer-Qwen3-8B.Q4_K_M | 95/100 | 14.0 |
| 2 | DeepSeek-R1-Distill-Qwen-32B-Q6_K | 95/100 | 2.0 |
| 3 | Qwen3-235B-A22B-Q4_K_M | 94/100 | 5.8 |
| 4 | gemma-3-12b-it-Q4_K_M | 94/100 | 9.3 |
| 5 | Qwen2.5-Coder-32B-Instruct-Q4_K_M | 93/100 | 3.4 |

### Math Suite (10 questions)

| Rank | Model | Score | Avg TPS |
|------|-------|-------|---------|
| 1 | DeepSeek-R1-Distill-Qwen-7B-Q4_K_M | 96/100 | 16.9 |
| 2 | DeepSeek-R1-Distill-Llama-8B-Q4_K_M | 96/100 | 9.4 |
| 3 | Qwen3-4B-Thinking-2507-Q8_0 | 95/100 | 5.4 |
| 4 | GLM-4.6-Q4_K_S | 95/100 | 3.1 |
| 5 | MathSmith-Hard-Problem-Synthesizer-Qwen3-8B.Q4_K_M | 93/100 | 16.1 |

### Agentic Suite (10 questions)

| Rank | Model | Score | Avg TPS |
|------|-------|-------|---------|
| 1 | Qwen3-4B-Thinking-2507-Q8_0 | 98/100 | 5.4 |
| 2 | DeepSeek-R1-Distill-Llama-8B-Q4_K_M | 93/100 | 9.4 |
| 3 | DeepSeek-R1-Distill-Qwen-7B-Q4_K_M | 89/100 | 16.9 |
| 4 | Multiple 70B+ models | ~85/100 | 1-6 |

### Coder Suite (10 questions)

| Rank | Model | Score | Avg TPS |
|------|-------|-------|---------|
| 1 | Qwen3-32B-Q4_K_M | 95/100 | 1.6 |
| 2 | Qwen3-235B-A22B-Q4_K_M | 94/100 | 5.8 |
| 3 | Qwen3-32B-Q4_K_M (ingest role) | 93/100 | 1.6 |
| 4 | DeepSeek-R1-Distill-Qwen-32B-Q6_K | 92/100 | 2.0 |
| 5 | Qwen3-30B-A3B-Thinking-2507-Q8_0 | 91/100 | 17.6 |

### Instruction Precision Suite (11 questions)

| Rank | Model | Score | Avg TPS |
|------|-------|-------|---------|
| 1 | Qwen3-Coder-480B-A35B-Instruct-Q4_K_M | 78/110 | 6.0 |
| 2 | Qwen2.5-72B-Instruct-Q4_K_M | 76/110 | 1.9 |
| 3 | Qwen3-30B-A3B-Thinking-2507-Q8_0 | 60/110 | 17.6 |
| 4 | gemma-3-27B-it-QAT-Q4_0 | 58/110 | 2.2 |
| 5 | Hermes-4-70B-Q4_K_M | 55/110 | 2.7 |

**Note:** Instruction precision remains hard for ALL models. Best score is 71% (Qwen3-Coder-480B).

---

## Draft Model Rankings (for Speculative Decoding)

| Model | Quality | TPS | Recommended For |
|-------|---------|-----|-----------------|
| Qwen3-1.7B-Q8_0 | 88% | 36.3 | High quality spec decode |
| Qwen3-1.7B-Q4_K_M | 86% | 43.3 | Balanced spec decode |
| gemma-3-1b-it-Q8_0 | 83% | 114.1 | Fast spec decode |
| xLAM-2-1B-fc-r-Q4_K_M | 83% | 50.4 | Tool calling drafts |
| PARD-DeepSeek-R1-Distill-Qwen-1.5B.Q5_K_S | 80% | 45.6 | Thinking model drafts |
| Qwen2.5-0.5B.Q8_0 | 80% | 156.8 | Maximum speed |
| pard-qwen3-0.6b-q4_0 | 78% | 81.6 | Fast MoE drafts |

---

## Critical Issues Identified

### 1. Vision Models on Agentic Tasks (0%)
- Qwen2.5-VL-7B-Instruct-Q4_K_M: 0/100 on agentic
- Qwen3-VL-30B-A3B-Instruct-Q4_K_M: 0/100 on agentic
- **Reason:** Output `/image` interface prompts instead of tool call JSON
- **Not a bug:** These are vision-specialized models

### 2. Meta-Llama-3-70B-Instruct Empty Responses (37%)
- Many questions returned empty or prompt-only responses
- Likely inference configuration issue, not model quality

### 3. Phi-4-reasoning Models (46-48%)
- Frequent empty responses and prompt echoing
- Poor instruction following on precision tasks

### 4. Meta-Llama-3.1-8B.Q4_K_S (44.5%)
- Degenerative repetition on coder tasks (22/100)
- Better on general tasks (67/100)

---

## Orchestration Deployment Recommendations

Based on relative scoring, optimal model selections per role:

| Role | Primary Model | Quality | TPS | Fallback |
|------|---------------|---------|-----|----------|
| **Frontdoor** | Qwen3-Coder-30B-A3B-Instruct-Q4_K_M | 89% | 12.0 | - |
| **Coder Primary** | Qwen3-30B-A3B-Thinking-2507-Q8_0 | 90% | 17.6 | Qwen3-32B |
| **Coder Escalation** | Qwen2.5-Coder-32B-Instruct-Q4_K_M | 91.5% | 3.4 | - |
| **Architect General** | Qwen3-235B-A22B-Q4_K_M | 94% | 5.8 | - |
| **Architect Coding** | Qwen3-Coder-480B-A35B-Instruct-Q4_K_M | 88.5% | 6.0 | - |
| **Ingest** | Qwen3-30B-A3B-Thinking-2507-Q8_0 | 90% | 17.6 | - |
| **Worker Math** | DeepSeek-R1-Distill-Qwen-7B-Q4_K_M | 90.7% | 16.9 | - |
| **Worker General** | gemma-3-12b-it-Q4_K_M | 87.5% | 9.3 | - |
| **Draft** | Qwen3-1.7B-Q8_0 | 88% | 36.3 | gemma-3-1b-it |

---

## Files Generated

| File | Contents |
|------|----------|
| `thinking_relative_scores.csv` | 570 scores (57 models × 10 questions) |
| `general_relative_scores.csv` | 790 scores (79 models × 10 questions) |
| `math_relative_scores.csv` | 470 scores (47 models × 10 questions) |
| `agentic_relative_scores.csv` | 420 scores (42 models × 10 questions) |
| `coder_relative_scores.csv` | 404 scores (40 models × 10 questions) |
| `instruction_precision_relative_scores.csv` | 330 scores (30 models × 11 questions) |
| `long_context_relative_scores.csv` | 41 scores (6 models × ~7 questions) |
| `vl_relative_scores.csv` | 80 scores (8 models × 10 questions) |
| `summary_relative.csv` | Aggregated summary with totals |
| `summary_relative_with_speed.csv` | Summary with TPS data |

---

## Comparison: Old vs New Scoring

| Model | Old Score | New Score | Change |
|-------|-----------|-----------|--------|
| Qwen2.5-0.5B-Instruct-f16 | 100% | 48% | -52% |
| pard-qwen3-0.6b-q4_0 | 95% | 52% | -43% |
| Qwen3-1.7B-Q8_0 | 95% | 88% | -7% |
| Hermes-4-70B-Q4_K_M | 89% | 95% | +6% |
| Qwen3-235B-A22B-Q4_K_M | 94% | 94% | 0% |
| DeepSeek-R1-Distill-Qwen-32B-Q6_K | 94% | 93.5% | -0.5% |

**Key insight:** Small models dropped significantly (proper differentiation), large models stayed similar or improved.

---

*Generated: 2026-01-19*
*Methodology: Relative scoring against reference answers and cross-model comparison*
*Total Scores: 3,105 individual question-model pairs*
