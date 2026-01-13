# Vanilla TTT Feasibility Assessment

## Status: CLOSED - NO-GO (Research Paused)

**Created**: 2026-01-12
**Updated**: 2026-01-13
**Closed**: 2026-01-13
**Paper**: [End-to-End Test-Time Training for Long Context](https://arxiv.org/abs/2512.23675)
**Reference Repo**: https://github.com/test-time-training/e2e

---

## Executive Summary

**Goal**: Assess whether Vanilla TTT (test-time training on existing models) can improve long-context handling.

**FINAL RESULT: NO-GO**

### Key Findings

1. **Training Infrastructure Fixed**: Successfully patched llama.cpp to enable training on modern architectures
2. **TTT Performs WORSE Than Baseline**: On needle-in-haystack retrieval, TTT-adapted model failed while baseline succeeded
3. **Resource Requirements Prohibitive**: Requires F32 models and likely custom TTT-fine-tuned checkpoints

### Benchmark Results (2026-01-13)

| Metric | Baseline | TTT |
|--------|----------|-----|
| **Context tokens** | 3485 | 3481 |
| **Needle retrieved?** | **YES** | **NO** |
| **Answer** | Exact key extracted | "not specified in documentation" |
| **Total time** | ~30s | ~30s |
| **Training accuracy** | N/A | 90% |

**Critical Finding**: TTT trained to 90% next-token accuracy but FAILED to retrieve specific factual information, instead hallucinating that the information wasn't present.

---

## Technical Fixes Implemented

We fixed three cascading failures in llama.cpp training:

### Fix 1: SET_ROWS View Tensor Assertion
**File**: `ggml/src/ggml.c` (~line 6856)
```c
case GGML_OP_SET_ROWS:
    ignore_src[0] = true;  // source values
    ignore_src[1] = true;  // row indices
    ignore_src[2] = true;  // destination
    break;
```

### Fix 2: FLASH_ATTN_EXT No Backward Pass
**File**: `ggml/src/ggml.c` (~line 6863)
```c
case GGML_OP_FLASH_ATTN_EXT:
    ignore_src[0] = true;  // Q
    ignore_src[1] = true;  // K
    ignore_src[2] = true;  // V
    ignore_src[3] = true;  // mask
    break;
```

### Fix 3: Graph Overflow
**File**: `src/llama-context.cpp` (line 1953)
```cpp
// Changed from 8x to 32x multiplier
uint32_t res = std::max<uint32_t>(4096u, 32u*model.n_tensors());
```

**Result**: `llama-finetune` and `llama-ttt` now complete successfully on Llama-3.2-1B-F32.

---

## Why TTT Failed for Factual Retrieval

1. **FFN-only training doesn't help attention-based recall**: Attention patterns (frozen) are key for retrieval tasks
2. **Training on repetitive filler biases toward patterns**: Model learned general document structure, lost specific details
3. **Catastrophic forgetting**: Even single epoch with lr=1e-5 overwrote needle information
4. **Wrong task type**: TTT may benefit reasoning/pattern tasks, not factual extraction

---

## Why Research Should Pause

| Constraint | Impact |
|------------|--------|
| **FP32 models required** | 4.6GB for 1B model, ~100GB for 7B, impractical for production |
| **Custom checkpoints likely needed** | TTT-E2E paper used meta-trained models, not vanilla weights |
| **Training infrastructure** | Proper TTT requires training pipeline we don't have |
| **No benefit demonstrated** | Baseline outperformed TTT on our target task |

---

## Artifacts Preserved

| Artifact | Location |
|----------|----------|
| llama-ttt binary | `/mnt/raid0/llm/llama.cpp-experimental/build/bin/llama-ttt` |
| ttt.cpp source | `/mnt/raid0/llm/llama.cpp-experimental/examples/training/ttt.cpp` |
| F32 test model | `/mnt/raid0/llm/models/Llama-3.2-1B-Instruct-f32.gguf` |
| Patched ggml.c | `/mnt/raid0/llm/llama.cpp-experimental/ggml/src/ggml.c` |
| Feature branch | `feature/ttt-adaptation` in experimental worktree |
| Bug report | https://github.com/ggml-org/llama.cpp/issues/18805 |
| Needle test files | `/mnt/raid0/llm/tmp/needle_*.txt` |

---

## If Research Resumes

Prerequisites before reconsidering TTT:

1. **TTT-fine-tuned checkpoints available**: Meta-trained models designed for test-time adaptation
2. **Attention-inclusive training**: Current FFN-only approach insufficient for retrieval
3. **Quantization support**: F32 requirement is impractical for production
4. **Different task evaluation**: Test on reasoning/pattern tasks, not factual retrieval

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-12 | Focus on Vanilla TTT | No TTT-E2E checkpoints available |
| 2026-01-13 | Fixed training crashes | SET_ROWS, FLASH_ATTN_EXT, graph overflow |
| 2026-01-13 | Benchmark vs baseline | Direct comparison on needle task |
| 2026-01-13 | **NO-GO: Pause research** | TTT worse than baseline, requires FP32 + custom models |

---

*Status*: CLOSED - Research paused
*Reason*: Requires FP32 models and likely custom TTT-fine-tuned checkpoints
*Resume condition*: TTT-E2E checkpoints become available OR training infrastructure established
