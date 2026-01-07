# MathSmith Model Re-Conversion Handoff

**Status**: READY FOR IMPLEMENTATION
**Priority**: LOW (formalizer role has alternatives - xLAM models)
**Created**: 2026-01-07
**Depends On**: None (standalone task)

---

## Problem

MathSmith-Hard-Problem-Synthesizer-Qwen3-8B shows 3.5 t/s (6% memory bandwidth), indicating a bad GGUF conversion.

**Expected for 8B model**: 40-60 t/s
**Observed**: 3.5 t/s

The compute-bound behavior (6% memory bandwidth) suggests the GGUF conversion has issues - possibly wrong tensor types or missing optimizations. This is common with community conversions.

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

## Solution

Re-convert from HuggingFace source:

```bash
# 1. Download HF model
huggingface-cli download MathSmith/MathSmith-Hard-Problem-Synthesizer-Qwen3-8B \
  --local-dir /mnt/raid0/llm/hf/MathSmith-Hard-Problem-Synthesizer-Qwen3-8B

# 2. Convert to GGUF (F16 intermediate)
cd /mnt/raid0/llm/llama.cpp
python convert_hf_to_gguf.py \
  /mnt/raid0/llm/hf/MathSmith-Hard-Problem-Synthesizer-Qwen3-8B \
  --outfile /mnt/raid0/llm/models/MathSmith-Qwen3-8B-f16.gguf \
  --outtype f16

# 3. Quantize to Q4_K_M
./build/bin/llama-quantize \
  /mnt/raid0/llm/models/MathSmith-Qwen3-8B-f16.gguf \
  /mnt/raid0/llm/models/MathSmith-Qwen3-8B-Q4_K_M.gguf \
  Q4_K_M

# 4. Test speed
numactl --interleave=all ./build/bin/llama-completion \
  -m /mnt/raid0/llm/models/MathSmith-Qwen3-8B-Q4_K_M.gguf \
  -p "Formalize: The sum of two primes greater than 2 is always even" \
  -n 100 -t 96
```

---

## Expected Results

| Metric | Before | After |
|--------|--------|-------|
| Speed | 3.5 t/s | 40-60 t/s |
| Memory BW | 6% | 70-90% |
| Quantization | Unknown | Q4_K_M |

---

## Verification

After re-conversion, verify:

1. **Speed**: Should match other 8B models (~45 t/s)
2. **Quality**: Run formalizer benchmark
   ```bash
   ./scripts/benchmark/bench_formalizers.sh \
     --model /mnt/raid0/llm/models/MathSmith-Qwen3-8B-Q4_K_M.gguf \
     --prompts benchmarks/prompts/v1/formalizer/
   ```
3. **Memory bandwidth**: Should be >70% (memory-bound, not compute-bound)

---

## Integration with Formalizer Pipeline

Once converted, MathSmith can be added to the formalizer evaluation alongside:

| Model | Type | Status |
|-------|------|--------|
| xLAM-2-1B-fc-r | Function calling | Ready |
| xLAM-1B-fc-r | Function calling | Ready |
| NexusRaven-V2-13B | Function calling | Ready |
| MathSmith-Qwen3-8B | Math formalization | Needs re-conversion |

---

## Cleanup

After successful conversion, remove the bad GGUF:
```bash
rm /mnt/raid0/llm/models/MathSmith-Hard-Problem-Synthesizer-Qwen3-8B-Q4_K_M.gguf
```

---

## Related Documents

| Document | Purpose |
|----------|---------|
| `research/formalizer_handoff.md` | Formalizer evaluation pipeline |
| `orchestration/BLOCKED_TASKS.md` | Task tracking |
