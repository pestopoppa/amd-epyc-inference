# LLM Inference Optimization Research Report

## Executive Summary

This document tracks research progress on optimizing LLM inference on AMD EPYC 9655 "Turin" (96-core Zen 5) with 1.13TB RAM. Primary focus: speculative decoding techniques to maximize tokens/second.

**Last Updated:** [TIMESTAMP]

**Key Results:**
- [Summary of best results to date]

**Current Focus:**
- [Active research track]

---

## System Configuration

| Component | Specification |
|-----------|---------------|
| CPU | AMD EPYC 9655 "Turin" (96 cores, Zen 5) |
| RAM | 1.13 TB DDR5-5600 (12 channels, ~460 GB/s) |
| Storage | 2× Solidigm P44 Pro 2TB NVMe RAID0 |
| OS | Ubuntu |
| Inference Engine | llama.cpp (Zen 5 optimized build) |

**Critical Settings:**
- `OMP_NUM_THREADS=1` (prevent nested parallelism)
- `numactl --interleave=all` (saturate all memory channels)
- `-t 96` (physical cores only, no SMT)
- CPU Governor: `performance`
- THP: `always`

---

## Tested Models

### Successfully Tested

| Model | Format | Quant | Best t/s | Method | Date |
|-------|--------|-------|----------|--------|------|
| | | | | | |

### Pending Testing

| Model | Format | Location | Priority |
|-------|--------|----------|----------|
| | | | |

---

## Research Tracks

### Track 1: Adaptive Modular Pipeline
**Status:** [Active/Complete/Blocked]

**Methodology:**
- Multi-process architecture with NUMA-pinned instances
- Draft server (32 cores, NUMA 0) + Target server (64 cores, NUMA 1-3)
- Adaptive K based on content type

**Results:**
| Configuration | Model | K Value | Acceptance | t/s |
|---------------|-------|---------|------------|-----|
| | | | | |

**Findings:**
- [Key observations]

### Track 2: Monolithic Self-Drafting
**Status:** [Active/Complete/Blocked]

**Methodology:**
- MoE Top-1 gating for draft passes
- Single model in memory (bandwidth optimization)

**Results:**
| Configuration | Model | Draft K | t/s |
|---------------|-------|---------|-----|
| | | | |

**Findings:**
- [Key observations]

### Track 3: SSM Architecture
**Status:** Blocked (Architecture Incompatible)

**Notes:**
- SSM/Mamba models cannot use standard KV cache rollback
- Baseline performance only

---

## Successful Methodologies

### What Works

1. **NUMA Interleaving**
   - Essential for saturating 12-channel DDR5
   - 20-40% improvement over single-node allocation

2. **Adaptive Speculative Depth**
   - Code content: K=24, ~83% acceptance
   - Prose content: K=8, ~32% acceptance
   - Math/JSON: K=4-6

3. **Thread Limiting**
   - 96 threads (physical cores) optimal
   - SMT (192 threads) degrades performance

### What Doesn't Work

1. **Static Hugepages >150GB**
   - Caused system instability
   - THP preferred

2. **Cross-Family Draft Models**
   - Vocab mismatch causes silent failures
   - Must use same tokenizer family

---

## Literature References

### Speculative Decoding
- [SpecInfer: Accelerating Generative LLM Serving](https://arxiv.org/abs/2305.09781)
- [DISCO: Dynamic Speculative Decoding](https://arxiv.org/)
- [Medusa: Simple Framework for Accelerating LLM Generation](https://arxiv.org/abs/2401.10774)

### MoE Optimization
- [Switch Transformers](https://arxiv.org/abs/2101.03961)
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)

### SSM/Mamba
- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)
- [STree: Speculative Tree Decoding for SSMs](https://arxiv.org/)

### AMD Zen 5
- [Zen 5 AVX-512 Analysis](https://www.numberworld.org/blogs/2024_8_7_zen5_avx512_teardown/)
- [OpenBLAS Zen 5 Support](https://www.phoronix.com/news/OpenBLAS-0.3.29-Released)

---

## Future Research Directions

1. **Dynamic K with Reinforcement Learning**
   - Train K selection policy based on historical acceptance

2. **Token-Level Speculation Gating**
   - Per-token confidence thresholds

3. **Hybrid CPU-GPU Inference**
   - Draft on CPU, verify on GPU (if available)

4. **Quantization Experiments**
   - Compare Q2_K vs Q4_K_M vs Q5_K_M for draft models

---

## Appendix: Test Configurations

### Standard Benchmark Command
```bash
OMP_NUM_THREADS=1 numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-bench \
  -m MODEL_PATH -t 96 -p 512 -n 128
```

### Speculative Decoding Command
```bash
OMP_NUM_THREADS=1 numactl --interleave=all \
  /mnt/raid0/llm/llama.cpp/build/bin/llama-speculative \
  -m MAIN_MODEL --draft DRAFT_MODEL \
  --draft-max K -t 96
```

---

*Report generated for blog: "Maximizing LLM Inference on AMD EPYC Turin"*
