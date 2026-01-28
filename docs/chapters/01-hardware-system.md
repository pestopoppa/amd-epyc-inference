# Chapter 01: Hardware System

## Introduction

This project optimizes LLM inference on AMD's EPYC 9655 "Turin" processor. The system was chosen for its massive memory capacity (1.13 TB) and high memory bandwidth (~460 GB/s across 12 channels), which are critical bottlenecks for large language model inference on CPU.

## Hardware Specifications

| Component | Specification |
|-----------|---------------|
| CPU | AMD EPYC 9655 "Turin" (Zen 5 architecture) |
| Cores/Threads | 96 cores / 192 threads |
| RAM | 1.13 TB DDR5-5600 ECC (12 channels) |
| Memory Bandwidth | ~460 GB/s theoretical |
| Storage | 2× Solidigm P44 Pro 2TB NVMe in RAID0 |
| OS Drive | 120GB SSD (system only) |
| Architecture | True 512-bit AVX-512 (not double-pumped like Intel) |

### Why This Hardware Matters

**Memory Capacity**: At 1.13TB, we can load models up to ~500B parameters at Q4_K_M quantization entirely in RAM. This eliminates disk I/O bottlenecks that plague smaller systems.

**Memory Bandwidth**: The 12-channel DDR5 configuration provides approximately 460 GB/s of bandwidth. Since LLM inference is memory-bound during generation (reading weights for each token), this bandwidth directly determines maximum throughput.

**AVX-512**: Zen 5 implements true 512-bit AVX-512 units, unlike Intel's double-pumped approach. This provides genuine 2x vector width for SIMD operations in matrix multiplications.

## Runtime Optimizations

### Critical Environment Settings

```bash
# Prevent nested parallelism (severely degrades performance)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# NUMA interleaving for balanced memory access across 12 channels
numactl --interleave=all <command>

# Use physical cores only (hyperthreading hurts inference)
-t 96

# Enable Transparent Huge Pages
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled

# Set CPU governor to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### Why These Settings

**OMP_NUM_THREADS=1**: llama.cpp handles threading internally. OpenMP trying to parallelize on top of llama.cpp's threading causes thread contention and can reduce performance by 50% or more.

**numactl --interleave=all**: With 12 memory channels, NUMA effects are significant. Interleaving distributes data across all channels, maximizing bandwidth utilization.

**96 threads (physical cores only)**: Hyperthreading provides no benefit for compute-bound LLM inference and can actually hurt due to cache contention.

## Baseline Performance

Before any optimization, baseline token generation speeds on dense models:

| Model | Size (GGUF) | Prompt Processing | Token Generation |
|-------|-------------|-------------------|------------------|
| Qwen2.5-Coder-32B Q4_K_M | 19GB | 69.05 t/s | 2.89 t/s |
| Qwen2.5-72B Q4_K_M | 42GB | ~50 t/s | ~1.8 t/s |
| Qwen3-235B-A22B Q4_K_M | 131GB | ~30 t/s | ~3.6 t/s |
| Qwen3-Coder-480B Q4_K_M | 271GB | 34.66 t/s | 3.06 t/s |

**Key Observation**: Token generation is the bottleneck, not prompt processing. Generation requires reading the entire model for each token, making it memory-bandwidth bound. This is why speculative decoding (amortizing multiple tokens per read) provides such dramatic speedups.

## Storage Architecture

The system uses a split storage design:

- **OS Drive (120GB SSD)**: System files only. DO NOT write LLM data here.
- **RAID0 Array (/mnt/raid0/)**: 4TB striped array for all models, caches, and project files.

**Critical Rule**: All LLM-related files must reside on `/mnt/raid0/`. Writing large files to the OS drive causes system instability.

## References

### Hardware Documentation

1. AMD Corporation. (2024). *AMD EPYC 9655 Processor Specifications*. https://www.amd.com/en/products/cpu/amd-epyc-9655

2. AMD Corporation. (2024). *AMD Zen 5 Architecture White Paper*. https://www.amd.com/en/technologies/zen-architecture

### Software and Implementation

3. Gerganov, G., et al. (2024). *llama.cpp: LLM inference in C/C++*. GitHub. https://github.com/ggml-org/llama.cpp

4. llama.cpp Contributors. (2024). *CPU Performance Discussion: EPYC and Threadripper*. GitHub Discussions. https://github.com/ggml-org/llama.cpp/discussions/4167

### System Optimization

5. Drepper, U. (2007). *What Every Programmer Should Know About Memory*. Red Hat, Inc. https://people.freebsd.org/~lstewart/articles/cpumemory.pdf

6. Linux Kernel Documentation. *Transparent Hugepages*. https://www.kernel.org/doc/Documentation/vm/transhuge.txt

7. Linux Kernel Documentation. *NUMA Memory Policy*. https://www.kernel.org/doc/Documentation/admin-guide/mm/numa_memory_policy.rst

---

*Next: [Chapter 02: Runtime Environment](02-runtime-environment.md)*
