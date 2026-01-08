# Chapter 08: Deprecated Approaches

## Introduction

Not all optimization attempts succeed. This chapter documents approaches we tested and abandoned, preserving the lessons learned. Understanding *why* these failed prevents future researchers from rediscovering the same dead ends.

## Track 3: EAGLE-1 (Self-Speculative)

### What It Is

EAGLE-1 uses a trained autoregression head to predict future tokens from hidden states, enabling self-speculative decoding without a separate draft model.

### Why We Tried It

- Eliminates need for compatible draft model
- Potentially higher acceptance rates (same model family)
- Published results showed 2-3x speedup

### What Happened

**Result**: 0% acceptance rate

**Root Cause**: EAGLE checkpoints are trained on specific model versions. Our GGUF-quantized models have different internal representations. The autoregression head produces garbage predictions.

**Attempted Fixes**:
- Different EAGLE checkpoints
- Various quantization levels
- Temperature adjustments

None worked. EAGLE requires exact checkpoint compatibility that GGUF conversion breaks.

### Lesson Learned

Self-speculative methods requiring trained components are fragile to quantization and format changes. Prefer methods that work with any model.

---

## Track 7: CAS-Spec (Layer Skipping)

### What It Is

Cascade Speculative Drafting (CAS-Spec) generates draft tokens by skipping early layers, then verifies with full model. The idea: early layers "draft," late layers "verify."

### Why We Tried It

- No external draft model needed
- Theoretically elegant (same weights, different depth)
- Paper reported 2.3x speedup

### What Happened

**Result**: 0.446% acceptance rate

**Root Cause**: Without trained exit classifiers, the layer-skipped outputs diverge too much from full-model outputs. The "draft" tokens are essentially random relative to what the full model would produce.

**Analysis**: CAS-Spec requires:
1. Trained exit classifiers per layer (we don't have these)
2. Calibrated confidence thresholds (model-specific)
3. Architecture that supports clean layer boundaries

Our GGUF models lack the necessary trained components.

### Lesson Learned

Layer-skipping methods need trained classifiers. Raw layer output without proper exit prediction is useless for speculation.

---

## Track 5: SSM Speculation

### What It Is

Applying speculative decoding or prompt lookup to SSM-hybrid models (Qwen3-Next series).

### Why We Tried It

- SSM models are fast and efficient
- Speculation provides 5-11x speedup on dense models
- Natural extension to test

### What Happened

**Result**: Corrupted output, model state invalid

**Root Cause**: SSM architectures maintain recurrent state across tokens. When draft tokens are rejected, the KV cache can be rolled back, but recurrent state cannot. The state becomes permanently corrupted.

**Technical Detail**:
```
Dense model rollback:
  Token 1 → KV[1] → Token 2 → KV[2] → Reject Token 2 → Restore KV[1] ✅

SSM model rollback:
  Token 1 → KV[1] + State[1] → Token 2 → KV[2] + State[2]
  → Reject Token 2 → Restore KV[1] but State still = State[2] ❌
```

### Lesson Learned

**NEVER use speculation with SSM models.** This is a fundamental architectural incompatibility. Use expert reduction (Track 2) only.

---

## Track 4: Medusa

### What It Is

Medusa adds multiple parallel prediction heads to the model, each predicting a different future token position.

### Why We Skipped It

- Requires training heads per model
- Training data and compute significant
- Heads don't transfer between model versions

### Alternative

External draft models (Track 1) achieve similar speedups without per-model training.

---

## Track 9: CLaSp/SWIFT

### What It Is

Similar to CAS-Spec - uses layer outputs for self-drafting with trained classifiers.

### Why We Skipped It

Same fundamental issue as CAS-Spec: requires trained exit classifiers we don't have.

---

## Track 10: Kangaroo

### What It Is

Trains a small adapter network that predicts when the draft model will be accepted.

### Why We Skipped It

- Requires adapter training per model pair
- Training overhead doesn't justify marginal gains over baseline Track 1
- Another component to maintain and update

---

## Summary: What Works vs What Doesn't

### Works (Production)

| Track | Method | Speedup | Key Requirement |
|-------|--------|---------|-----------------|
| 1 | External Draft | 5.9-11x | Compatible tokenizer |
| 2 | MoE Reduction | 21-52% | MoE architecture |
| 8 | Prompt Lookup | 2-12.7x | Grounded task (overlap) |

### Doesn't Work (Deprecated)

| Track | Method | Failure Mode | Alternative |
|-------|--------|--------------|-------------|
| 3 | EAGLE-1 | Checkpoint incompatibility | Use Track 1 |
| 7 | CAS-Spec | No trained classifiers | Use Track 1 |
| 5 | SSM Speculation | State corruption | Use Track 2 only |
| 4 | Medusa | Requires head training | Use Track 1 |
| 9 | CLaSp/SWIFT | Same as CAS-Spec | Use Track 1 |
| 10 | Kangaroo | Requires adapter training | Use Track 1 |

## Pattern Recognition

**Methods that work**:
- Use separate, complete models (Track 1)
- Exploit structural properties (MoE, n-gram overlap)
- Require no per-model training

**Methods that fail**:
- Require trained components we don't have
- Assume checkpoint/architecture compatibility that GGUF breaks
- Can't handle rollback (SSM)

## References

- speculative_decoding_research.md (Failed tracks section)
- CAS_SPEC_IMPLEMENTATION_PLAN.md
- track_reorganization_analysis.md

---

*Previous: [Chapter 07: RadixAttention](07-radix-attention.md)*
*Back to: [Index](INDEX.md)*
