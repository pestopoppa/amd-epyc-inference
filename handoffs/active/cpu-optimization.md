# Handoff: CPU Optimization Research

**Created**: 2026-01-05
**Status**: ✅ COMPLETE (T-MAC abandoned, Tree Speculation done)
**Priority**: —
**Source**: `research/cpu_optimization_findings.md`

---

## Overview

Two CPU optimization tracks identified during research:

1. **T-MAC** - LUT-based low-bit inference — ❌ ABANDONED
2. **Tree Speculation** - Already in llama.cpp — ✅ COMPLETE (K=24 optimal)

---

## Track A: T-MAC Evaluation

### Status: ❌ ABANDONED (2026-01-09)

**Reason:** Not viable for our use case.

| Issue | Why It's a Blocker |
|-------|-------------------|
| Old llama.cpp fork (May 2024) | Incompatible with current codebase |
| Only supports GPTQ format | Can't use our Q4_K_M GGUF models |
| Uncertain x86/AVX-512 support | Authors warn "no guarantee on x86" |
| Best gains at 1-2 bit | Quality degrades significantly |
| Model reconversion required | High effort, uncertain payoff |

**What It Was:** LUT-based low-bit inference (EuroSys 2025, arXiv:2407.00088)

**Why We're Not Pursuing:** Our existing speculative decoding already gives 11x speedup. T-MAC would require model reconversion, uses an outdated llama.cpp fork, and the authors explicitly warn about x86 performance.

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

### Results (2026-01-09)

| K Value | Acceptance | Decode Speed | Notes |
|---------|------------|--------------|-------|
| K=24 | 16.7% | ~3 t/s | **Optimal** |
| K=32 | 14.4% | 0.7 t/s | Worse - wastes compute |

**Conclusion:** K=24 remains optimal. K=32 wastes drafts without improving acceptance.

### Completed
1. [x] Benchmark `--draft-max 32` vs current `--draft-max 24` (2026-01-09)
2. [x] Measure acceptance rate at different tree widths (2026-01-09)
3. [ ] ~~Profile memory bandwidth impact~~ Not needed - K=32 clearly worse
4. [x] Document optimal settings (K=24 for Qwen2.5-Coder)

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
