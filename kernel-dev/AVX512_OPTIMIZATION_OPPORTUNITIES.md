# AVX-512 Optimization Opportunities for ggml on AMD EPYC 9655 (Zen 5)

## Executive Summary

Analysis of ggml CPU backend reveals significant optimization opportunities for Zen 5's true 512-bit AVX-512 execution. The most critical quantized dot product operations (Q4_K, Q5_K, Q6_K) use **AVX2 only**, leaving substantial performance on the table.

**Key Finding**: Adding AVX-512 to Q4_K dot product could yield 1.8-2.5x speedup.

## Hardware Context

| Specification | Value |
|---------------|-------|
| CPU | AMD EPYC 9655 "Turin" |
| Cores/Threads | 96 / 192 |
| Architecture | Zen 5 |
| AVX-512 | TRUE 512-bit (single cycle) |
| AVX-512 VNNI | Supported |
| Memory | 1.13 TB DDR5-5600 |
| Memory Bandwidth | ~460 GB/s theoretical |

**Zen 5 vs Zen 4**: Zen 5 has true single-cycle 512-bit execution, not double-pumped like Zen 4. This means AVX-512 operations should achieve full throughput.

## Current AVX-512 Usage in ggml

| File | AVX-512 Intrinsics | Purpose |
|------|-------------------|---------|
| arch/x86/repack.cpp | 1,294 | Tensor repacking for optimized matmul |
| amx/mmq.cpp | 467 | Intel AMX matrix multiply |
| vec.h | 29 | Activation functions (exp, silu) |
| simd-mappings.h | 30 | SIMD abstraction layer |
| llamafile/sgemm.cpp | 13 | GEMM operations |
| **arch/x86/quants.c** | **0** | **Quantized operations - NONE!** |

## Priority 1: Q4_K Dot Product (HIGH IMPACT)

### Current State

Location: `arch/x86/quants.c:1741`
```c
void ggml_vec_dot_q4_K_q8_K(int n, float * GGML_RESTRICT s, ...) {
    // Uses __m256i (AVX2) only
    // No AVX-512 implementation exists
}
```

### Proposed Change

Add AVX-512 F and AVX-512 VNNI implementations:

| Version | Register Width | Expected Speedup | Notes |
|---------|---------------|------------------|-------|
| AVX2 (current) | 256-bit | baseline | Uses `__m256i` |
| AVX-512 F | 512-bit | 1.8-2.0x | Uses `__m512i` |
| AVX-512 VNNI | 512-bit + VNNI | 2.0-2.5x | Uses `_mm512_dpbusd_epi32` |

### Implementation Files

Created in `/workspace/kernel-dev/`:
- `q4k_avx512_kernel.c` - Complete AVX-512 kernel implementation
- `bench_q4k_dot.c` - Synthetic micro-benchmark

### Key Changes

1. **Inner loop width**: 32 bytes → 64 bytes per iteration
2. **Iterations reduced**: 4 → 2 per block
3. **Horizontal sum**: Cascaded extracts → `_mm512_reduce_add_ps`
4. **Scale broadcast**: `MM256_SET_M128I` → `_mm512_broadcast_i32x4`

## Priority 2: Other K-Quant Dot Products (MEDIUM IMPACT)

Same pattern applies to:
- `ggml_vec_dot_q5_K_q8_K` (line 1919)
- `ggml_vec_dot_q6_K_q8_K` (line 2085)
- `ggml_vec_dot_q2_K_q8_K` (line 1493)
- `ggml_vec_dot_q3_K_q8_K` (line 1603)

All use AVX2 only. Similar speedups expected.

## Priority 3: Q8 Quantization (MEDIUM IMPACT)

Location: `arch/x86/quants.c:290`
```c
void quantize_row_q8_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    // Uses __m256 (AVX2) only
}
```

AVX-512 version would:
- Process 64 floats per iteration instead of 32
- Use `_mm512_cvtps_epi32` for float→int conversion
- Expected 1.5-2x speedup

## Priority 4: IQ (Integer Quantization) Dot Products (LOW IMPACT)

Various IQ dot products in `arch/x86/quants.c`:
- `ggml_vec_dot_iq2_xxs_q8_K`
- `ggml_vec_dot_iq2_xs_q8_K`
- `ggml_vec_dot_iq3_xxs_q8_K`
- etc.

Most use AVX2 only. Lower priority due to less common usage.

## Implementation Strategy

### Phase 1: Q4_K AVX-512 (This Analysis)
1. ✅ Analyze current implementation
2. ✅ Design AVX-512 kernel
3. ✅ Create benchmark harness
4. ⏳ Test on Zen 5 hardware
5. ⏳ Validate correctness
6. ⏳ Submit upstream patch

### Phase 2: Other K-Quants
1. Port Q5_K, Q6_K to AVX-512
2. Benchmark and validate
3. Submit patches

### Phase 3: Quantization Functions
1. Add AVX-512 to quantize_row_q8_0/q8_1
2. Benchmark and validate

## Build Instructions

```bash
# Compile with AVX-512 support
cmake -B build \
    -DCMAKE_C_FLAGS="-march=native -mavx512f -mavx512bw -mavx512vl -mavx512vnni" \
    -DCMAKE_CXX_FLAGS="-march=native -mavx512f -mavx512bw -mavx512vl -mavx512vnni"

cmake --build build -j$(nproc)
```

## Testing Methodology

### Correctness Testing
```c
// Compare AVX2 vs AVX-512 results
float result_avx2, result_avx512;
ggml_vec_dot_q4_K_q8_K_avx2(n, &result_avx2, x, y);
ggml_vec_dot_q4_K_q8_K_avx512(n, &result_avx512, x, y);
assert(fabsf(result_avx2 - result_avx512) < 1e-5f);
```

### Performance Testing
```bash
# Use synthetic benchmark (does not load models)
./bench_q4k_dot

# Or use perf for detailed analysis
perf stat -e cycles,instructions,cache-misses ./bench_q4k_dot
```

## Expected Impact on Inference

For a typical LLM inference workload:
- Dot products consume ~60-80% of compute time
- 2x speedup on dot products → ~1.4-1.6x overall speedup

This stacks with other optimizations:
- Speculative decoding: 8-12x speedup on decode
- MoE expert reduction: 21-87% speedup on MoE models
- Prompt lookup: 8-12x speedup on summarization

## Files Modified (Proposed Patch)

```
ggml/src/ggml-cpu/arch/x86/quants.c
├── Add: ggml_vec_dot_q4_K_q8_K_avx512f()
├── Add: ggml_vec_dot_q4_K_q8_K_avx512vnni()
└── Modify: ggml_vec_dot_q4_K_q8_K() to dispatch based on CPU features

ggml/src/ggml-cpu/arch/x86/CMakeLists.txt
└── Add: AVX-512 VNNI detection
```

## References

- [AMD EPYC 9004 Series Processors](https://www.amd.com/en/products/processors/server/epyc/9004-series.html)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- [ggml GitHub Repository](https://github.com/ggml-org/llama.cpp)
