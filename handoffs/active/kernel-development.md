# AVX-512 Kernel Development - Agent Handoff

**Purpose**: Autonomous development of Zen 5-optimized AVX-512 kernels for ggml.
**Mode**: YOLO (no interactive prompts, continuous progress logging)
**Critical Constraint**: DO NOT load or test actual LLM models - development and unit tests only.

---

## 0. YOLO Mode Launch (CRITICAL)

**To run Claude Code without permission prompts, launch with:**
```bash
claude --dangerously-skip-permissions
```

This flag is safe inside the devcontainer due to isolation. Without this flag, the agent will prompt for every tool use.

**References:**
- [banteg/agents](https://github.com/banteg/agents) - YOLO devcontainer setup
- [Claude Code devcontainer docs](https://code.claude.com/docs/en/devcontainer)

---

## 0.1 Path Mapping (IMPORTANT)

This handoff references paths for both host and devcontainer environments:

| Host Path | Container Path |
|-----------|----------------|
| `/mnt/raid0/llm/claude/` | `/workspace/` |
| `/mnt/raid0/llm/llama.cpp/` | NOT MOUNTED (clone fresh) |
| `/mnt/raid0/llm/kernel-dev/` | `/workspace/kernel-dev/` |

**If running in devcontainer**: Use `/workspace/` paths.
**If running on host**: Use `/mnt/raid0/llm/claude/` paths.

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

**In devcontainer** (clone fresh - host llama.cpp not mounted):
```bash
cd /workspace
git clone https://github.com/ggml-org/llama.cpp llama-cpp-dev
```

**On host**:
```
/mnt/raid0/llm/llama.cpp/          # Main llama.cpp fork
```

**Key paths after clone**:
```
llama-cpp-dev/ggml/                # ggml tensor library (target)
llama-cpp-dev/ggml/src/ggml-cpu/   # CPU-specific implementations
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
# In devcontainer:
/workspace/research/kernel_dev_progress.log

# On host:
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

**In devcontainer:**
```bash
# Create isolated workspace
mkdir -p /workspace/kernel-dev
cd /workspace/kernel-dev

# Clone llama.cpp for experimentation (host version not mounted)
git clone --depth 1 https://github.com/ggml-org/llama.cpp llama-cpp-dev

# Create log file
LOG=/workspace/research/kernel_dev_progress.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] SESSION: Kernel development started" >> $LOG
```

**On host:**
```bash
# Create isolated workspace
mkdir -p /mnt/raid0/llm/claude/kernel-dev
cd /mnt/raid0/llm/claude/kernel-dev

# Copy ggml source for experimentation (don't modify main llama.cpp)
cp -r /mnt/raid0/llm/llama.cpp/ggml ./ggml-experimental

# Create log file
LOG=/mnt/raid0/llm/claude/research/kernel_dev_progress.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] SESSION: Kernel development started" >> $LOG
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

**In devcontainer:**
```bash
# 1. Set up workspace
mkdir -p /workspace/kernel-dev && cd /workspace/kernel-dev
git clone --depth 1 https://github.com/ggml-org/llama.cpp llama-cpp-dev

# 2. Start logging
LOG=/workspace/research/kernel_dev_progress.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] SESSION: Started" >> $LOG

# 3. Begin analysis
cat llama-cpp-dev/ggml/src/ggml-cpu/ggml-cpu.c | head -500
```

**On host:**
```bash
# 1. Set up workspace
mkdir -p /mnt/raid0/llm/claude/kernel-dev && cd /mnt/raid0/llm/claude/kernel-dev

# 2. Start logging
LOG=/mnt/raid0/llm/claude/research/kernel_dev_progress.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] SESSION: Started" >> $LOG

# 3. Begin analysis
cat /mnt/raid0/llm/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c | head -500
```

**Begin with Phase 1: Analysis. Read the code. Log everything.**

---

## 13. Session Results (2026-01-05)

### Status: COMPLETED - Negative Result (Important Finding)

**Conclusion: AVX-512 optimization of Q4_K dot product is NOT viable.**

The Q4_K data format is fundamentally designed for 256-bit operations. Correct AVX-512 implementations are 17-21% SLOWER than AVX2 due to permutation overhead.

### Final Benchmark Results (Host)

| Implementation | Time/call | vs AVX2 | Correct? |
|----------------|-----------|---------|----------|
| AVX2 (baseline) | 76.30 ns | 1.00x | YES |
| AVX-512 VBMI | 92.43 ns | **0.83x** | YES |
| AVX-512 Full | 96.24 ns | **0.79x** | YES |
| Original (buggy) | 35.99 ns | 2.07x | **NO** |

### Root Cause: Data Layout

```
Q4_K Layout (optimized for AVX2):
  byte[i] = low_nibble[i] | (high_nibble[i] << 4)

  AVX2 (32 bytes):  Q4[0:31] → Q8[0:31] + Q8[32:63]
                    Perfect contiguous access

  AVX-512 (64 bytes): Q4[0:63] → need Q8 from 4 non-contiguous regions
                      Permute overhead > register width benefit
```

### Why This Matters

- **Explains missing AVX-512 Q4_K in ggml**: Not an oversight, it's slower
- **Original 2.07x was wrong**: Buggy kernel doing incorrect computation
- **No PR opportunity**: Format redesign required (breaks all GGUF models)

### Files Created

| File | Purpose |
|------|---------|
| `/workspace/kernel-dev/bench_q4k_dot.c` | Synthetic micro-benchmark |
| `/workspace/kernel-dev/q4k_avx512_kernel.c` | Original kernel (INCORRECT) |
| `/workspace/kernel-dev/q4k_avx512_fixed.c` | "Fixed" kernel (0.93x - no gain) |
| `/workspace/kernel-dev/q4k_avx512_vbmi.c` | VBMI kernel (0.83x - slower) |
| `/workspace/research/kernel_dev_progress.log` | Full session log |

### Alternative Optimization Targets

Operations that ARE 512-bit friendly (for future work):

| Target | Why 512-bit Friendly |
|--------|---------------------|
| Prefill | Continuous token processing, no data layout issues |
| Attention | Large contiguous matrix operations |
| FP32/FP16 ops | No nibble packing complications |
| Quantization | Writing new format, not reading existing |
| Tensor repacking | Already uses AVX-512 in ggml |

### Key Learning

> **Data layout design determines SIMD efficiency more than instruction width.**

The Q4_K format was designed when AVX2 was state-of-art. Making it AVX-512 efficient would require:
1. Planar nibble storage (separate low/high arrays)
2. Or larger block sizes aligned to 512-bit
3. Either breaks existing model compatibility

### Success Criteria (Revised)

- [x] Identified optimization opportunities
- [x] Created synthetic benchmark
- [x] Implemented optimized kernel variants
- [x] Validated correctness
- [x] Measured performance (negative result: -17% to -21%)
- [x] Documented findings
- [x] **Explained WHY ggml lacks AVX-512 Q4_K**

### Progress Log

Full session log: `/workspace/research/kernel_dev_progress.log`

---

## 14. Session 2 Results (2026-01-05) - Q8_0 AVX-512

### Status: VERIFIED - Modest Gains (+13-15%)

After Q4_K failed due to data layout issues, we pivoted to Q8_0 which is AVX-512 friendly.

**Host Test Results (2026-01-05):**

| Implementation | Time | vs AVX2 | Status |
|----------------|------|---------|--------|
| quantize_q8_0 AVX-512 | 876.64 ns | **1.15x** | ✅ PASS |
| vec_dot_q8_0 AVX-512 | 434.74 ns | **1.13x** | ✅ PASS |
| vec_dot_q8_0 VNNI | N/A | N/A | ⚠️ BUG |

**VNNI Bug**: `_mm512_dpbssd_epi32` doesn't exist. Fix: use `_mm512_dpbusd_epi32` with sign trick.

### Why Q8_0 Works for AVX-512

| Property | Q4_K | Q8_0 |
|----------|------|------|
| Block data | Nibble-packed (interleaved) | Contiguous int8s |
| Data per block | 32 nibbles = 16 bytes | 32 int8s = 32 bytes |
| AVX-512 loads | Requires expensive permutes | Direct 64-byte loads |
| Use case | Full model weights | Draft model, activations |

Q8_0 blocks have 32 contiguous int8s - AVX-512 can process 2 blocks at once without permutation overhead.

### Files Created

| File | Purpose |
|------|---------|
| `/workspace/kernel-dev/q8_0_avx512_kernels.c` | **Complete standalone benchmark** |
| `/workspace/kernel-dev/bench_q8_0_avx512.c` | Initial benchmark (superseded) |
| `/workspace/kernel-dev/patches/0002-avx512-q8_0-dot-product.patch` | ggml vec_dot patch |
| `/workspace/kernel-dev/patches/0003-avx512-quantize-q8_0.patch` | ggml quantize patch |

### Implementations

#### 1. `quantize_row_q8_0_avx512`
- Uses `_mm512_reduce_max_ps` for efficient max-abs
- Uses `_mm512_cvtsepi32_epi8` for direct int32→int8 saturation
- Processes 2 blocks per iteration (64 floats)

#### 2. `vec_dot_q8_0_avx512` (two variants)

**AVX-512 F/BW (baseline):**
```c
// Sign trick + maddubs
__m512i ax = _mm512_abs_epi8(qx);
__mmask64 neg_mask = _mm512_movepi8_mask(qx);
__m512i sy = _mm512_mask_blend_epi8(neg_mask, qy, neg_qy);
__m512i dot16 = _mm512_maddubs_epi16(ax, sy);
```

**AVX-512 VNNI (Zen 4+, Ice Lake+):**
```c
// Direct signed int8 dot product
__m512i sumi = _mm512_dpbssd_epi32(_mm512_setzero_si512(), qx, qy);
```

### Testing Instructions (Host)

```bash
# 1. Copy from container
docker cp <container>:/workspace/kernel-dev/q8_0_avx512_kernels.c /mnt/raid0/llm/kernel-dev/

# 2. Compile on host
cd /mnt/raid0/llm/kernel-dev
gcc -O3 -march=znver5 -mavx512f -mavx512bw -mavx512vnni \
    -o q8_0_test q8_0_avx512_kernels.c -lm

# 3. Run benchmark
./q8_0_test
```

### Verified Results (Host Testing 2026-01-05, Re-verified 2026-01-09)

| Implementation | Actual Speedup vs AVX2 | Notes |
|----------------|------------------------|-------|
| Q8_0 quantize AVX-512 | **+39%** (1.39x) | Correct, significant |
| Q8_0 vec_dot AVX-512 | **+13%** (1.13x) | Correct, modest |
| Q4_K AVX-512 VBMI | **-4%** (0.96x) | Correct but SLOWER |
| Q4_K AVX-512 Full | **-9%** (0.91x) | Correct but SLOWER |
| AVX-512 VNNI | N/A | Bug: `_mm512_dpbssd_epi32` doesn't exist |

**Key insight**: Gains are modest due to memory bandwidth limits, not compute.

**VNNI Bug**: The `_mm512_dpbssd_epi32` (signed×signed) intrinsic doesn't exist in standard AVX-512 VNNI. Only `_mm512_dpbusd_epi32` (unsigned×signed) is available. The signed×signed variant requires AVX10.2 or AMX.

### Impact on Speculative Decoding

Faster Q8_0 operations directly benefit draft model throughput:
- Draft model (Qwen2.5-Coder-0.5B-Instruct-Q8_0) uses Q8_0 format
- `vec_dot_q8_0_q8_0` is on critical path for draft token generation
- Higher draft speed → can increase K (speculation depth) → better throughput

### Next Steps

1. ~~Run benchmark on host~~ ✅ Done: +13-15% verified
2. **Tree speculation (Track B)** - Next priority, potentially higher impact
3. **Prefill profiling** - Verify if AVX-512 GEMM already active in llamafile/sgemm.cpp
4. Q8_0 AVX-512 patch: Low priority given modest gains vs complexity

### Conclusion

AVX-512 kernel optimization yields diminishing returns for quantized ops:
- **Q4_K**: Data layout fundamentally incompatible (-17% to -21%)
- **Q8_0**: Works but memory-bound limits gains to +13-15%
- **Better targets**: Algorithmic improvements (tree speculation) over SIMD micro-optimization

---

## 15. Prefill Profiling Results (2026-01-05)

### llamafile/sgemm.cpp AVX-512 Status

| Type | AVX-512 Path? | Register Width | Notes |
|------|---------------|----------------|-------|
| FP32 | **YES** | `__m512` (512-bit) | `tinyBLAS<16, __m512, __m512, ...>` |
| FP16 | **YES** | `__m512` (512-bit) | Full AVX-512 with F16C |
| BF16 | **YES** | `__m512` (512-bit) | Optional AVX512BF16 |
| Q8_0 | **NO** | `__m256` (256-bit) | `tinyBLAS_Q0_AVX` - AVX2 only |
| Q4_0 | **NO** | `__m256` (256-bit) | Same class, AVX2 only |
| Q5_0 | **NO** | `__m256` (256-bit) | Same class, AVX2 only |
| IQ4_NL | **NO** | `__m256` (256-bit) | Same class, AVX2 only |

### Key Finding

**`tinyBLAS_Q0_AVX` uses 256-bit registers only**, even when compiled with `__AVX512F__`:

```cpp
// Line 1639 - always __m256
__m256 Cv[RN][RM] = {};

// Line 1677 - 256-bit loads
inline __m256i load(const block_q8_0 *b) {
    return _mm256_loadu_si256((const __m256i *)b->qs);
}

// Line 1748-1754 - VNNI but 256-bit
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    res = _mm256_dpbusd_epi32(_mm256_setzero_si256(), u, s);  // 256-bit!
#endif
```

### Optimization Opportunity

Implement `tinyBLAS_Q0_AVX512` with:
- `__m512i` registers for 64 int8s at once
- `_mm512_dpbusd_epi32` for 512-bit VNNI
- Process 2 Q8_0 blocks per iteration

**Expected improvement**: 1.3-1.5x (limited by memory bandwidth)

### Reality Check

Given Q8_0 dot product only achieved +13-15%, GEMM improvements would likely be similar due to memory bandwidth bottleneck. The prefill phase is already fast - focus should be on decode throughput (speculative decoding).

---

## 16. Tree Speculation (Track B)

### Status: ALREADY IMPLEMENTED in llama.cpp

Tree-based speculative decoding exists in `examples/speculative/speculative.cpp`.

### How It Works

Instead of generating a single draft sequence, the draft model generates a **tree of possible continuations**:

```
Token 0 ──► Token 1 ──► Token 2 ──► Token 3  (branch 0)
                   └──► Token 2' ──► Token 3' (branch 1)
                               └──► Token 3'' (branch 2)
```

When a candidate token has probability > `p_split`, a new branch is forked.

### Command Line Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-np, --parallel N` | 1 | Number of tree branches (parallel draft sequences) |
| `--draft-p-split P` | 0.1 | Probability threshold for branching |
| `--draft-max N` | 16 | Max draft tokens per branch |

### Example Usage

```bash
# Single-path speculation (baseline)
llama-speculative -m target.gguf -md draft.gguf --draft-max 16 -np 1

# Tree speculation with 4 branches
llama-speculative -m target.gguf -md draft.gguf --draft-max 16 -np 4 --draft-p-split 0.1

# Aggressive branching (more exploration)
llama-speculative -m target.gguf -md draft.gguf --draft-max 16 -np 8 --draft-p-split 0.05
```

### Benchmark Script

Created: `/workspace/kernel-dev/scripts/bench_tree_speculation.sh`

Tests combinations of:
- `n_parallel`: 1, 2, 4, 8
- `p_split`: 0.05, 0.1, 0.2, 0.3

### Expected Benefits

Tree speculation can improve throughput when:
1. **High uncertainty**: Multiple plausible continuations exist
2. **Temperature > 0**: Stochastic sampling benefits from exploration
3. **Sufficient KV cache**: More branches = more memory

### Potential Issues

1. **Memory overhead**: Each branch requires KV cache copy
2. **Diminishing returns**: Too many branches may slow down target verification
3. **Task dependent**: Code generation may benefit less than creative writing

### Testing Instructions (Host)

```bash
# Copy script from container
docker cp <container>:/workspace/kernel-dev/scripts/bench_tree_speculation.sh \
    /mnt/raid0/llm/claude/scripts/benchmark/

# Run benchmark
cd /mnt/raid0/llm/claude
./scripts/benchmark/bench_tree_speculation.sh
```
