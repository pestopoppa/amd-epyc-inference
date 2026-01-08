# Handoff: CPU Optimization Research

**Created**: 2026-01-05
**Status**: Active (Research Phase)
**Priority**: MEDIUM (T-MAC), HIGH (Tree Speculation)
**Source**: `research/cpu_optimization_findings.md`

---

## Overview

Two CPU optimization tracks identified during research:

1. **T-MAC** - LUT-based low-bit inference (MEDIUM priority)
2. **Tree Speculation** - Already in llama.cpp, needs benchmarking (HIGH priority)

---

## Track A: T-MAC Evaluation

### What It Is
T-MAC replaces dequantization with lookup tables for 1-4 bit inference. Published at EuroSys 2025.

- **Repository**: `/mnt/raid0/llm/T-MAC/`
- **Paper**: [arXiv:2407.00088](https://arxiv.org/abs/2407.00088)

### Current Assessment

| Aspect | Status |
|--------|--------|
| Quantization support | Partial - W4A16 from GPTQ, NOT Q4_K_M |
| llama.cpp version | Old (b2794, May 2024) |
| Model conversion | Required - HF → T-MAC GGUF |
| x86/AVX-512 support | Uncertain |
| Existing GGUF models | NOT compatible |

### Critical Warning
> "We cannot guarantee significant speedup (especially for 4-bit token generation) on all x86 platforms."

### Next Steps
1. [ ] Build T-MAC with `-DLLAMA_TMAC=ON`
2. [ ] Test with small 2-bit GPTQ model (not production)
3. [ ] Validate x86 performance before full conversion pipeline
4. [ ] If promising, reconvert production models

### Recommendation
Start with a small test to validate x86 performance. Best gains are at 1-2 bit which degrades quality. May not be worth the effort for 4-bit.

---

## Track B: Tree Speculation

### What It Is
Tree-based sampling drafts multiple token sequences in parallel, not just a single chain. Already integrated in llama.cpp.

### Current Status
- **Available**: Yes, in `llama-speculative` binary
- **Tested**: No comprehensive benchmarks yet
- **Expected Gain**: Higher effective K with same acceptance rate

### Available Flags
```bash
--draft-max N        # Max tokens to draft (enables tree exploration at higher N)
-td, --threads-draft N
-Cd, --cpu-mask-draft M
```

### Next Steps
1. [ ] Benchmark `--draft-max 32` vs current `--draft-max 24`
2. [ ] Measure acceptance rate at different tree widths
3. [ ] Profile memory bandwidth impact
4. [ ] Document optimal settings per model size

### Resume Command
```bash
# Test tree speculation with higher K
OMP_NUM_THREADS=1 numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-speculative \
  -m /mnt/raid0/llm/models/Qwen2.5-Coder-32B-Q4_K_M.gguf \
  -md /mnt/raid0/llm/models/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf \
  --draft-max 32 -t 96 -p "Implement a binary search:"
```

---

## NUMA Finding

During research, discovered system is in NPS1 mode (2 NUMA nodes), not NPS4 (8 nodes).

**Impact**: Cannot run 3+ draft models on separate NUMA domains without BIOS reconfiguration.

---

## References

- `research/cpu_optimization_findings.md` - Full research notes
- [T-MAC Paper](https://arxiv.org/abs/2407.00088)
- `llama.cpp/examples/speculative/speculative.cpp` - Tree sampling implementation
