# AVX-512 Kernel Development - Agent Handoff

**Purpose**: Autonomous development of Zen 5-optimized AVX-512 kernels for ggml.
**Mode**: YOLO (no interactive prompts, continuous progress logging)
**Critical Constraint**: DO NOT load or test actual LLM models - development and unit tests only.

---

## 1. Objective

Optimize ggml AVX-512 kernels for AMD EPYC 9655 (Zen 5) to improve inference throughput.

**Target**: 20-50% improvement in specific matrix operations.

**Why this matters**: Zen 5 has true 512-bit AVX-512 (single cycle), while current ggml kernels may be tuned for Intel or Zen 4 (double-pumped).

---

## 2. Hardware Context

```
CPU:              AMD EPYC 9655 "Turin" (96 cores, 192 threads)
Architecture:     Zen 5
AVX-512:          TRUE 512-bit execution (not double-pumped like Zen 4)
Memory:           1.13 TB DDR5-5600
Memory Bandwidth: ~460 GB/s theoretical
L3 Cache:         384 MB (32 MB per CCD)
```

**Key Zen 5 Features to Exploit**:
- Full-width 512-bit FMA units
- Improved branch prediction
- Better prefetching
- AVX-512 VNNI for int8 dot products

---

## 3. Codebase Location

```
/mnt/raid0/llm/llama.cpp/          # Main llama.cpp fork
/mnt/raid0/llm/llama.cpp/ggml/     # ggml tensor library (target for optimization)
/mnt/raid0/llm/llama.cpp/ggml/src/ggml-cpu/  # CPU-specific implementations
```

**Key files to study**:
- `ggml/src/ggml-cpu/ggml-cpu.c` - Main CPU backend
- `ggml/src/ggml-cpu/ggml-cpu-quants.c` - Quantized operations
- `ggml/src/ggml-cpu/amx/` - Intel AMX (reference for SIMD patterns)
- `ggml/include/ggml.h` - Core data structures

---

## 4. What You CAN Do

1. **Read and analyze ggml source code**
2. **Profile existing kernels** using synthetic data (not real models)
   ```bash
   # Create synthetic test tensors
   # Benchmark specific operations in isolation
   ```
3. **Write optimized kernel variants** in a separate directory
4. **Create unit tests** that validate correctness with synthetic data
5. **Run micro-benchmarks** on isolated operations
6. **Document findings** in progress log

---

## 5. What You CANNOT Do

1. **DO NOT load any GGUF model files**
2. **DO NOT run llama-cli, llama-speculative, or llama-bench with models**
3. **DO NOT access `/mnt/raid0/llm/models/` directory**
4. **DO NOT run any inference**

**Why**: Production benchmark is running and model loading would interfere.

---

## 6. Development Approach

### Phase 1: Analysis (No code changes)

1. Read ggml-cpu.c and identify AVX-512 codepaths
2. Identify operations that dominate inference:
   - `ggml_vec_dot_q4_K_q8_K` - Q4_K matmul
   - `ggml_compute_forward_mul_mat` - General matmul
   - Attention computation
3. Document current SIMD strategy and memory access patterns

### Phase 2: Profiling Setup

Create synthetic benchmark that:
- Allocates test tensors matching typical inference shapes
- Runs target operations in isolation
- Measures cycles, instructions, cache behavior

```c
// Example: Profile Q4_K dot product with synthetic data
void profile_q4k_dot() {
    // Allocate synthetic quantized tensors
    // Run operation 1000x
    // Measure with perf counters
}
```

### Phase 3: Optimization Candidates

Priority targets:
1. **Memory prefetching** - Zen 5 has different prefetch behavior than Intel
2. **Register allocation** - Maximize ZMM register usage
3. **Loop unrolling** - Match Zen 5 decode width
4. **VNNI exploitation** - For int8 operations

### Phase 4: Validation

- Create unit tests comparing optimized vs original output
- Ensure bit-exact results (or acceptable numerical tolerance)
- Measure improvement on synthetic benchmark

---

## 7. Progress Logging

**MANDATORY**: Log all progress to:
```
/mnt/raid0/llm/claude/research/kernel_dev_progress.log
```

**Log format**:
```
[2026-01-05 15:30:00] PHASE: Analysis
[2026-01-05 15:30:00] ACTION: Reading ggml-cpu.c AVX-512 paths
[2026-01-05 15:45:00] FINDING: Q4_K dot product uses 16-wide unroll
[2026-01-05 16:00:00] HYPOTHESIS: Could benefit from 32-wide unroll on Zen 5
...
```

Log every:
- Phase transition
- Significant finding
- Hypothesis
- Experiment result
- Blocker encountered
- Decision made

---

## 8. Success Criteria

- [ ] Identified at least 3 optimization opportunities
- [ ] Created synthetic benchmark for target operations
- [ ] Implemented at least 1 optimized kernel variant
- [ ] Validated correctness with unit tests
- [ ] Measured improvement (target: 20%+ on micro-benchmark)
- [ ] Documented all findings in progress log

---

## 9. Failure Criteria (When to Stop)

- No measurable improvement after 50 optimization attempts
- Hardware limitation identified (e.g., memory bandwidth is true ceiling)
- Kernel modifications break correctness
- Changes would require invasive ggml refactoring

---

## 10. Workspace Setup

```bash
# Create isolated workspace
mkdir -p /mnt/raid0/llm/kernel-dev
cd /mnt/raid0/llm/kernel-dev

# Copy ggml source for experimentation (don't modify main llama.cpp)
cp -r /mnt/raid0/llm/llama.cpp/ggml ./ggml-experimental

# Create log file
touch /mnt/raid0/llm/claude/research/kernel_dev_progress.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] SESSION: Kernel development started" >> /mnt/raid0/llm/claude/research/kernel_dev_progress.log
```

---

## 11. Reference Materials

- **ggml documentation**: `/mnt/raid0/llm/llama.cpp/ggml/README.md`
- **AMD Zen 5 optimization guide**: Search for AMD Software Optimization Guide
- **AVX-512 intrinsics**: Intel Intrinsics Guide (applies to AMD too)
- **R&D Plan**: `/home/daniele/.claude/plans/twinkly-sniffing-crescent.md`
- **Research findings**: `/mnt/raid0/llm/claude/research/cpu_optimization_findings.md`

---

## 12. Quick Start

```bash
# 1. Set up workspace
mkdir -p /mnt/raid0/llm/kernel-dev && cd /mnt/raid0/llm/kernel-dev

# 2. Start logging
LOG=/mnt/raid0/llm/claude/research/kernel_dev_progress.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] SESSION: Started" >> $LOG

# 3. Begin analysis
cat /mnt/raid0/llm/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c | head -500
```

**Begin with Phase 1: Analysis. Read the code. Log everything.**
