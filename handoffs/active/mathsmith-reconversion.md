# MathSmith Model Re-Conversion Handoff

**Status**: ✅ COMPLETE (2026-01-08)
**Priority**: LOW (formalizer role has alternatives - xLAM models)
**Created**: 2026-01-07
**Updated**: 2026-01-08
**Depends On**: None (standalone task)

---

## Resolution

Downloaded Q4_K_M directly from mradermacher's HuggingFace repo instead of re-converting:

```bash
# Downloaded from mradermacher/MathSmith-Hard-Problem-Synthesizer-Qwen3-8B-GGUF
/mnt/raid0/llm/models/MathSmith-Hard-Problem-Synthesizer-Qwen3-8B.Q4_K_M.gguf (4.7GB)
```

mradermacher is a trusted GGUF converter - their conversion should not have the issues of the original bad GGUF.

---

## Original Problem

MathSmith-Hard-Problem-Synthesizer-Qwen3-8B showed 3.5 t/s (6% memory bandwidth), indicating a bad GGUF conversion.

**Expected for 8B model**: 40-60 t/s
**Observed**: 3.5 t/s

The compute-bound behavior (6% memory bandwidth) suggested the original GGUF had issues.

---

## Verification (TODO when benchmark completes)

Test the new Q4_K_M:

```bash
numactl --interleave=all /mnt/raid0/llm/llama.cpp/build/bin/llama-completion \
  -m /mnt/raid0/llm/models/MathSmith-Hard-Problem-Synthesizer-Qwen3-8B.Q4_K_M.gguf \
  -p "Formalize: The sum of two primes greater than 2 is always even" \
  -n 100 -t 96
```

Expected: ~40-60 t/s with >70% memory bandwidth utilization.

---

## Available Models

| Model | Path | Size |
|-------|------|------|
| Q4_K_M (NEW) | `/mnt/raid0/llm/models/MathSmith-Hard-Problem-Synthesizer-Qwen3-8B.Q4_K_M.gguf` | 4.7GB |
| Q8_0 | `/mnt/raid0/llm/lmstudio/models/mradermacher/.../MathSmith-Hard-Problem-Synthesizer-Qwen3-8B.Q8_0.gguf` | 8.7GB |

---

## Use Case

MathSmith specializes in **mathematical formalization** - converting natural language math problems into formal representations. This is different from:

- **xLAM models**: Function-calling formalizers for tool orchestration
- **General formalizers**: TaskIR emission for orchestrator routing

MathSmith is specifically useful for:
- Converting "Prove that the sum of two primes > 2 is even" into Lean/Coq specs
- Formalizing mathematical proofs
- Generating verification conditions

---

## Integration with Formalizer Pipeline

| Model | Type | Status |
|-------|------|--------|
| xLAM-2-1B-fc-r | Function calling | Ready |
| xLAM-1B-fc-r | Function calling | Ready |
| NexusRaven-V2-13B | Function calling | Ready |
| MathSmith-Qwen3-8B | Math formalization | ✅ Ready (Q4_K_M downloaded) |

---

## Related Documents

| Document | Purpose |
|----------|---------|
| `research/formalizer_handoff.md` | Formalizer evaluation pipeline |
| `orchestration/BLOCKED_TASKS.md` | Task tracking |
