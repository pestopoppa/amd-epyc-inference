/*
 * Q8_0 AVX-512 Kernel Implementations
 *
 * Standalone implementations for testing before ggml integration.
 * Contains:
 *   1. quantize_row_q8_0_avx512 - Activation quantization
 *   2. vec_dot_q8_0_avx512 - Dot product (VNNI variant)
 *
 * Build:
 *   gcc -O3 -march=znver5 -mavx512f -mavx512bw -mavx512vnni \
 *       -o q8_0_avx512_test q8_0_avx512_kernels.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>

#define QK8_0 32

typedef struct {
    uint16_t d;         // fp16 scale
    int8_t   qs[QK8_0]; // quants
} block_q8_0;

// FP16 ↔ FP32 conversion
static inline float fp16_to_fp32(uint16_t h) {
    union { uint32_t u; float f; } v;
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    if (exp == 0) {
        if (mant == 0) { v.u = sign; return v.f; }
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        exp++; mant &= ~0x400;
    } else if (exp == 31) {
        v.u = sign | 0x7f800000 | (mant << 13); return v.f;
    }
    v.u = sign | ((exp + 112) << 23) | (mant << 13);
    return v.f;
}

static inline uint16_t fp32_to_fp16(float f) {
    union { uint32_t u; float f; } v = { .f = f };
    uint32_t sign = (v.u >> 16) & 0x8000;
    int32_t exp = ((v.u >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (v.u >> 13) & 0x3ff;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7c00;
    return sign | (exp << 10) | mant;
}

// ============================================================================
// quantize_row_q8_0 implementations
// ============================================================================

void quantize_row_q8_0_scalar(const float * x, block_q8_0 * y, int64_t k) {
    int nb = k / QK8_0;
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            float v = fabsf(x[i * QK8_0 + j]);
            if (v > amax) amax = v;
        }
        float d = amax / 127.0f;
        y[i].d = fp32_to_fp16(d);
        float id = (amax != 0.0f) ? 127.0f / amax : 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            y[i].qs[j] = (int8_t)roundf(x[i * QK8_0 + j] * id);
        }
    }
}

#if defined(__AVX2__)
void quantize_row_q8_0_avx2(const float * x, block_q8_0 * y, int64_t k) {
    int nb = k / QK8_0;
    for (int i = 0; i < nb; i++) {
        __m256 v0 = _mm256_loadu_ps(x);
        __m256 v1 = _mm256_loadu_ps(x + 8);
        __m256 v2 = _mm256_loadu_ps(x + 16);
        __m256 v3 = _mm256_loadu_ps(x + 24);
        x += 32;

        const __m256 signBit = _mm256_set1_ps(-0.0f);
        __m256 maxAbs = _mm256_andnot_ps(signBit, v0);
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v1));
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v2));
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v3));

        __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1), _mm256_castps256_ps128(maxAbs));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
        float maxScalar = _mm_cvtss_f32(max4);

        float d = maxScalar / 127.f;
        y[i].d = fp32_to_fp16(d);
        float id = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
        __m256 mul = _mm256_set1_ps(id);

        v0 = _mm256_mul_ps(v0, mul);
        v1 = _mm256_mul_ps(v1, mul);
        v2 = _mm256_mul_ps(v2, mul);
        v3 = _mm256_mul_ps(v3, mul);

        v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
        v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
        v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
        v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);

        __m256i i0 = _mm256_cvtps_epi32(v0);
        __m256i i1 = _mm256_cvtps_epi32(v1);
        __m256i i2 = _mm256_cvtps_epi32(v2);
        __m256i i3 = _mm256_cvtps_epi32(v3);

        i0 = _mm256_packs_epi32(i0, i1);
        i2 = _mm256_packs_epi32(i2, i3);
        i0 = _mm256_packs_epi16(i0, i2);
        const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        i0 = _mm256_permutevar8x32_epi32(i0, perm);
        _mm256_storeu_si256((__m256i *)y[i].qs, i0);
    }
}
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
void quantize_row_q8_0_avx512(const float * x, block_q8_0 * y, int64_t k) {
    int nb = k / QK8_0;
    int i = 0;

    // Process 2 blocks per iteration
    for (; i + 1 < nb; i += 2) {
        // Block 0: Load 32 floats using 2x __m512
        __m512 v0_0 = _mm512_loadu_ps(x);
        __m512 v0_1 = _mm512_loadu_ps(x + 16);

        // Block 1: Load 32 floats using 2x __m512
        __m512 v1_0 = _mm512_loadu_ps(x + 32);
        __m512 v1_1 = _mm512_loadu_ps(x + 48);
        x += 64;

        // Compute max(abs(e)) for block 0
        __m512 abs0_0 = _mm512_abs_ps(v0_0);
        __m512 abs0_1 = _mm512_abs_ps(v0_1);
        __m512 maxAbs0 = _mm512_max_ps(abs0_0, abs0_1);
        float maxScalar0 = _mm512_reduce_max_ps(maxAbs0);

        // Compute max(abs(e)) for block 1
        __m512 abs1_0 = _mm512_abs_ps(v1_0);
        __m512 abs1_1 = _mm512_abs_ps(v1_1);
        __m512 maxAbs1 = _mm512_max_ps(abs1_0, abs1_1);
        float maxScalar1 = _mm512_reduce_max_ps(maxAbs1);

        // Quantize block 0
        float d0 = maxScalar0 / 127.f;
        y[i].d = fp32_to_fp16(d0);
        float id0 = (maxScalar0 != 0.0f) ? 127.f / maxScalar0 : 0.0f;
        __m512 mul0 = _mm512_set1_ps(id0);

        v0_0 = _mm512_mul_ps(v0_0, mul0);
        v0_1 = _mm512_mul_ps(v0_1, mul0);
        v0_0 = _mm512_roundscale_ps(v0_0, _MM_FROUND_TO_NEAREST_INT);
        v0_1 = _mm512_roundscale_ps(v0_1, _MM_FROUND_TO_NEAREST_INT);

        __m512i i0_0 = _mm512_cvtps_epi32(v0_0);
        __m512i i0_1 = _mm512_cvtps_epi32(v0_1);

        // Pack int32 → int8 with saturation
        __m128i b0_0 = _mm512_cvtsepi32_epi8(i0_0);
        __m128i b0_1 = _mm512_cvtsepi32_epi8(i0_1);
        _mm_storeu_si128((__m128i *)(y[i].qs), b0_0);
        _mm_storeu_si128((__m128i *)(y[i].qs + 16), b0_1);

        // Quantize block 1
        float d1 = maxScalar1 / 127.f;
        y[i + 1].d = fp32_to_fp16(d1);
        float id1 = (maxScalar1 != 0.0f) ? 127.f / maxScalar1 : 0.0f;
        __m512 mul1 = _mm512_set1_ps(id1);

        v1_0 = _mm512_mul_ps(v1_0, mul1);
        v1_1 = _mm512_mul_ps(v1_1, mul1);
        v1_0 = _mm512_roundscale_ps(v1_0, _MM_FROUND_TO_NEAREST_INT);
        v1_1 = _mm512_roundscale_ps(v1_1, _MM_FROUND_TO_NEAREST_INT);

        __m512i i1_0 = _mm512_cvtps_epi32(v1_0);
        __m512i i1_1 = _mm512_cvtps_epi32(v1_1);

        __m128i b1_0 = _mm512_cvtsepi32_epi8(i1_0);
        __m128i b1_1 = _mm512_cvtsepi32_epi8(i1_1);
        _mm_storeu_si128((__m128i *)(y[i + 1].qs), b1_0);
        _mm_storeu_si128((__m128i *)(y[i + 1].qs + 16), b1_1);
    }

    // Handle remaining odd block
    for (; i < nb; i++) {
        __m512 v0 = _mm512_loadu_ps(x);
        __m512 v1 = _mm512_loadu_ps(x + 16);
        x += 32;

        __m512 abs0 = _mm512_abs_ps(v0);
        __m512 abs1 = _mm512_abs_ps(v1);
        __m512 maxAbs = _mm512_max_ps(abs0, abs1);
        float maxScalar = _mm512_reduce_max_ps(maxAbs);

        float d = maxScalar / 127.f;
        y[i].d = fp32_to_fp16(d);
        float id = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
        __m512 mul = _mm512_set1_ps(id);

        v0 = _mm512_mul_ps(v0, mul);
        v1 = _mm512_mul_ps(v1, mul);
        v0 = _mm512_roundscale_ps(v0, _MM_FROUND_TO_NEAREST_INT);
        v1 = _mm512_roundscale_ps(v1, _MM_FROUND_TO_NEAREST_INT);

        __m512i i0 = _mm512_cvtps_epi32(v0);
        __m512i i1 = _mm512_cvtps_epi32(v1);

        __m128i b0 = _mm512_cvtsepi32_epi8(i0);
        __m128i b1 = _mm512_cvtsepi32_epi8(i1);
        _mm_storeu_si128((__m128i *)(y[i].qs), b0);
        _mm_storeu_si128((__m128i *)(y[i].qs + 16), b1);
    }
}
#endif

// ============================================================================
// vec_dot_q8_0 implementations
// ============================================================================

void vec_dot_q8_0_scalar(int n, float * s, const block_q8_0 * x, const block_q8_0 * y) {
    int nb = n / QK8_0;
    float sumf = 0.0f;
    for (int ib = 0; ib < nb; ++ib) {
        int sumi = 0;
        for (int j = 0; j < QK8_0; j++) {
            sumi += (int)x[ib].qs[j] * (int)y[ib].qs[j];
        }
        sumf += sumi * (fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d));
    }
    *s = sumf;
}

#if defined(__AVX2__)
static inline float hsum_float_8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

static inline __m256 mul_sum_i8_pairs_float_avx2(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i sum32 = _mm256_madd_epi16(dot, ones);
    return _mm256_cvtepi32_ps(sum32);
}

void vec_dot_q8_0_avx2(int n, float * s, const block_q8_0 * x, const block_q8_0 * y) {
    int nb = n / QK8_0;
    __m256 acc = _mm256_setzero_ps();

    for (int ib = 0; ib < nb; ++ib) {
        __m256 d = _mm256_set1_ps(fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d));
        __m256i qx = _mm256_loadu_si256((const __m256i *)x[ib].qs);
        __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);
        __m256 q = mul_sum_i8_pairs_float_avx2(qx, qy);
        acc = _mm256_fmadd_ps(d, q, acc);
    }
    *s = hsum_float_8(acc);
}
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
void vec_dot_q8_0_avx512_vnni(int n, float * s, const block_q8_0 * x, const block_q8_0 * y) {
    int nb = n / QK8_0;
    __m512 acc = _mm512_setzero_ps();

    int ib = 0;
    // Process 2 blocks per iteration
    for (; ib + 1 < nb; ib += 2) {
        __m256i qx0 = _mm256_loadu_si256((const __m256i *)x[ib].qs);
        __m256i qx1 = _mm256_loadu_si256((const __m256i *)x[ib + 1].qs);
        __m256i qy0 = _mm256_loadu_si256((const __m256i *)y[ib].qs);
        __m256i qy1 = _mm256_loadu_si256((const __m256i *)y[ib + 1].qs);

        __m512i qx = _mm512_inserti64x4(_mm512_castsi256_si512(qx0), qx1, 1);
        __m512i qy = _mm512_inserti64x4(_mm512_castsi256_si512(qy0), qy1, 1);

        // VNNI: signed int8 × signed int8 → int32
        __m512i sumi = _mm512_dpbssd_epi32(_mm512_setzero_si512(), qx, qy);
        __m512 sumf = _mm512_cvtepi32_ps(sumi);

        float d0 = fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d);
        float d1 = fp16_to_fp32(x[ib + 1].d) * fp16_to_fp32(y[ib + 1].d);

        __m512 scales = _mm512_insertf32x8(
            _mm512_castps256_ps512(_mm256_set1_ps(d0)),
            _mm256_set1_ps(d1), 1);

        acc = _mm512_fmadd_ps(scales, sumf, acc);
    }

    float sumf = _mm512_reduce_add_ps(acc);

    // Handle remaining
    for (; ib < nb; ++ib) {
        int sumi = 0;
        for (int j = 0; j < QK8_0; j++) {
            sumi += (int)x[ib].qs[j] * (int)y[ib].qs[j];
        }
        sumf += sumi * (fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d));
    }
    *s = sumf;
}
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
void vec_dot_q8_0_avx512(int n, float * s, const block_q8_0 * x, const block_q8_0 * y) {
    int nb = n / QK8_0;
    __m512 acc = _mm512_setzero_ps();

    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        __m256i qx0 = _mm256_loadu_si256((const __m256i *)x[ib].qs);
        __m256i qx1 = _mm256_loadu_si256((const __m256i *)x[ib + 1].qs);
        __m256i qy0 = _mm256_loadu_si256((const __m256i *)y[ib].qs);
        __m256i qy1 = _mm256_loadu_si256((const __m256i *)y[ib + 1].qs);

        __m512i qx = _mm512_inserti64x4(_mm512_castsi256_si512(qx0), qx1, 1);
        __m512i qy = _mm512_inserti64x4(_mm512_castsi256_si512(qy0), qy1, 1);

        // Sign trick for signed multiplication with maddubs
        __m512i ax = _mm512_abs_epi8(qx);
        __mmask64 neg_mask = _mm512_movepi8_mask(qx);
        __m512i neg_qy = _mm512_sub_epi8(_mm512_setzero_si512(), qy);
        __m512i sy = _mm512_mask_blend_epi8(neg_mask, qy, neg_qy);

        __m512i dot16 = _mm512_maddubs_epi16(ax, sy);
        __m512i dot32 = _mm512_madd_epi16(dot16, _mm512_set1_epi16(1));
        __m512 sumf_vec = _mm512_cvtepi32_ps(dot32);

        float d0 = fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d);
        float d1 = fp16_to_fp32(x[ib + 1].d) * fp16_to_fp32(y[ib + 1].d);

        __m512 scales = _mm512_insertf32x8(
            _mm512_castps256_ps512(_mm256_set1_ps(d0)),
            _mm256_set1_ps(d1), 1);

        acc = _mm512_fmadd_ps(scales, sumf_vec, acc);
    }

    float sumf = _mm512_reduce_add_ps(acc);

    for (; ib < nb; ++ib) {
        int sumi = 0;
        for (int j = 0; j < QK8_0; j++) {
            sumi += (int)x[ib].qs[j] * (int)y[ib].qs[j];
        }
        sumf += sumi * (fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d));
    }
    *s = sumf;
}
#endif

// ============================================================================
// Benchmark
// ============================================================================

static double get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

#define N_ELEMENTS (256 * QK8_0)
#define BENCH_ITERATIONS 1000000

int main(void) {
    printf("Q8_0 AVX-512 Kernel Benchmark\n");
    printf("==============================\n\n");

    int n_blocks = N_ELEMENTS / QK8_0;
    printf("Configuration:\n");
    printf("  Elements: %d\n", N_ELEMENTS);
    printf("  Blocks: %d\n", n_blocks);
    printf("  Iterations: %d\n\n", BENCH_ITERATIONS);

    // Allocate
    float *input = aligned_alloc(64, N_ELEMENTS * sizeof(float));
    block_q8_0 *x_scalar = aligned_alloc(64, n_blocks * sizeof(block_q8_0));
    block_q8_0 *x_avx2 = aligned_alloc(64, n_blocks * sizeof(block_q8_0));
    block_q8_0 *x_avx512 = aligned_alloc(64, n_blocks * sizeof(block_q8_0));
    block_q8_0 *y = aligned_alloc(64, n_blocks * sizeof(block_q8_0));

    // Initialize with random data
    srand(42);
    for (int i = 0; i < N_ELEMENTS; i++) {
        input[i] = (float)(rand() % 200 - 100) / 10.0f;
    }
    // Initialize y for dot product test
    quantize_row_q8_0_scalar(input, y, N_ELEMENTS);

    double t0, t1;
    float result;

    // ========= Quantization benchmarks =========
    printf("=== quantize_row_q8_0 ===\n\n");

    // Scalar (fewer iterations)
    t0 = get_time_ns();
    for (int i = 0; i < BENCH_ITERATIONS / 100; i++) {
        quantize_row_q8_0_scalar(input, x_scalar, N_ELEMENTS);
    }
    t1 = get_time_ns();
    double ns_quant_scalar = (t1 - t0) / (BENCH_ITERATIONS / 100);
    printf("Scalar:   %8.2f ns/call\n", ns_quant_scalar);

#if defined(__AVX2__)
    t0 = get_time_ns();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        quantize_row_q8_0_avx2(input, x_avx2, N_ELEMENTS);
    }
    t1 = get_time_ns();
    double ns_quant_avx2 = (t1 - t0) / BENCH_ITERATIONS;
    printf("AVX2:     %8.2f ns/call (%.2fx vs scalar) - baseline\n",
           ns_quant_avx2, ns_quant_scalar / ns_quant_avx2);
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
    t0 = get_time_ns();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        quantize_row_q8_0_avx512(input, x_avx512, N_ELEMENTS);
    }
    t1 = get_time_ns();
    double ns_quant_avx512 = (t1 - t0) / BENCH_ITERATIONS;
    printf("AVX-512:  %8.2f ns/call (%.2fx vs AVX2)\n",
           ns_quant_avx512, ns_quant_avx2 / ns_quant_avx512);

    // Verify correctness
    int errors = 0;
    for (int i = 0; i < n_blocks; i++) {
        if (x_avx2[i].d != x_avx512[i].d) errors++;
        for (int j = 0; j < QK8_0; j++) {
            if (abs(x_avx2[i].qs[j] - x_avx512[i].qs[j]) > 1) errors++;
        }
    }
    printf("  Correctness: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);
#endif

    // ========= Dot product benchmarks =========
    printf("\n=== vec_dot_q8_0 ===\n\n");

    // Scalar
    float ref_result;
    t0 = get_time_ns();
    for (int i = 0; i < BENCH_ITERATIONS / 100; i++) {
        vec_dot_q8_0_scalar(N_ELEMENTS, &ref_result, x_avx2, y);
    }
    t1 = get_time_ns();
    double ns_dot_scalar = (t1 - t0) / (BENCH_ITERATIONS / 100);
    printf("Scalar:      %8.2f ns/call (result: %.4f)\n", ns_dot_scalar, ref_result);

#if defined(__AVX2__)
    t0 = get_time_ns();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        vec_dot_q8_0_avx2(N_ELEMENTS, &result, x_avx2, y);
    }
    t1 = get_time_ns();
    double ns_dot_avx2 = (t1 - t0) / BENCH_ITERATIONS;
    printf("AVX2:        %8.2f ns/call (diff: %.4e) - baseline\n",
           ns_dot_avx2, fabs(result - ref_result));
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
    t0 = get_time_ns();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        vec_dot_q8_0_avx512(N_ELEMENTS, &result, x_avx2, y);
    }
    t1 = get_time_ns();
    double ns_dot_avx512 = (t1 - t0) / BENCH_ITERATIONS;
    printf("AVX-512:     %8.2f ns/call (diff: %.4e) %.2fx vs AVX2\n",
           ns_dot_avx512, fabs(result - ref_result), ns_dot_avx2 / ns_dot_avx512);
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    t0 = get_time_ns();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        vec_dot_q8_0_avx512_vnni(N_ELEMENTS, &result, x_avx2, y);
    }
    t1 = get_time_ns();
    double ns_dot_vnni = (t1 - t0) / BENCH_ITERATIONS;
    printf("AVX-512+VNNI:%8.2f ns/call (diff: %.4e) %.2fx vs AVX2\n",
           ns_dot_vnni, fabs(result - ref_result), ns_dot_avx2 / ns_dot_vnni);
#else
    printf("\n(VNNI not available on this CPU)\n");
#endif

    printf("\n=== Summary ===\n");
    printf("Q8_0 has 32 contiguous int8s per block - ideal for AVX-512\n");
    printf("With VNNI: _mm512_dpbssd_epi32 does signed dot product directly\n");
    printf("Expected: 1.3-2.0x improvement over AVX2 on Zen 5\n");

    free(input);
    free(x_scalar);
    free(x_avx2);
    free(x_avx512);
    free(y);

    return 0;
}
