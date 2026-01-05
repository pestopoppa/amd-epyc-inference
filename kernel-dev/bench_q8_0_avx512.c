/*
 * Q8_0 AVX-512 Dot Product Benchmark
 *
 * Tests AVX-512 optimization for Q8_0 dot product kernel.
 * Q8_0 format: 34-byte blocks (2-byte fp16 scale + 32 int8 values)
 *
 * Unlike Q4_K, Q8_0 has contiguous int8 data - ideal for AVX-512:
 * - AVX2: 32 int8s per load (1 block)
 * - AVX-512: 64 int8s per load (2 blocks)
 *
 * Build:
 *   gcc -O3 -march=znver5 -mavx512f -mavx512bw -mavx512vnni \
 *       -o bench_q8_0_avx512 bench_q8_0_avx512.c -lm
 *
 * Run:
 *   ./bench_q8_0_avx512
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>

#define QK8_0 32

// Q8_0 block: matches ggml format
typedef struct {
    uint16_t d;         // delta (fp16 scale as raw bits)
    int8_t   qs[QK8_0]; // quants
} block_q8_0;

// FP16 conversion (simplified - assumes little endian)
static inline float fp16_to_fp32(uint16_t h) {
    union { uint32_t u; float f; } v;
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;

    if (exp == 0) {
        if (mant == 0) {
            v.u = sign;
            return v.f;
        }
        // Denormalized
        while (!(mant & 0x400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x400;
    } else if (exp == 31) {
        v.u = sign | 0x7f800000 | (mant << 13);
        return v.f;
    }

    v.u = sign | ((exp + 112) << 23) | (mant << 13);
    return v.f;
}

static inline uint16_t fp32_to_fp16(float f) {
    union { uint32_t u; float f; } v = { .f = f };
    uint32_t sign = (v.u >> 16) & 0x8000;
    int32_t exp = ((v.u >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (v.u >> 13) & 0x3ff;

    if (exp <= 0) {
        return sign;
    } else if (exp >= 31) {
        return sign | 0x7c00;
    }
    return sign | (exp << 10) | mant;
}

// Horizontal sum for __m256
static inline float hsum_float_8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

// ============================================================================
// Reference scalar implementation
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

// ============================================================================
// AVX2 implementation (baseline - matches ggml)
// ============================================================================
static inline __m256 mul_sum_i8_pairs_float_avx2(const __m256i x, const __m256i y) {
    // Sign trick: |x| * sign(y, x) for signed int8 multiplication
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);

    // Multiply unsigned*signed → 16-bit, then sum pairs
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);

    // Sum adjacent pairs of 16-bit values to 32-bit
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i sum32 = _mm256_madd_epi16(dot, ones);

    return _mm256_cvtepi32_ps(sum32);
}

void vec_dot_q8_0_avx2(int n, float * s, const block_q8_0 * x, const block_q8_0 * y) {
    int nb = n / QK8_0;
    __m256 acc = _mm256_setzero_ps();

    for (int ib = 0; ib < nb; ++ib) {
        const __m256 d = _mm256_set1_ps(fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d));
        __m256i qx = _mm256_loadu_si256((const __m256i *)x[ib].qs);
        __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);

        const __m256 q = mul_sum_i8_pairs_float_avx2(qx, qy);
        acc = _mm256_fmadd_ps(d, q, acc);
    }

    *s = hsum_float_8(acc);
}

// ============================================================================
// AVX-512 implementation - Process 2 blocks at once
// ============================================================================
#if defined(__AVX512F__) && defined(__AVX512BW__)

static inline __m512 mul_sum_i8_pairs_float_avx512(const __m512i x, const __m512i y) {
#if defined(__AVX512VNNI__)
    // With VNNI: direct int8 dot product
    const __m512i zero = _mm512_setzero_si512();
    const __m512i summed = _mm512_dpbssd_epi32(zero, x, y);
    return _mm512_cvtepi32_ps(summed);
#else
    // Without VNNI: sign trick + maddubs
    const __m512i ax = _mm512_abs_epi8(x);

    // Sign y based on sign of x
    // _mm512_sign_epi8 doesn't exist, so we use mask + blend
    __mmask64 neg_mask = _mm512_movepi8_mask(x);  // 1 where x < 0
    __m512i neg_y = _mm512_sub_epi8(_mm512_setzero_si512(), y);
    __m512i sy = _mm512_mask_blend_epi8(neg_mask, y, neg_y);

    // Multiply unsigned*signed → 16-bit, sum pairs
    const __m512i dot = _mm512_maddubs_epi16(ax, sy);

    // Sum pairs to 32-bit
    const __m512i ones = _mm512_set1_epi16(1);
    const __m512i sum32 = _mm512_madd_epi16(dot, ones);

    return _mm512_cvtepi32_ps(sum32);
#endif
}

void vec_dot_q8_0_avx512(int n, float * s, const block_q8_0 * x, const block_q8_0 * y) {
    int nb = n / QK8_0;
    __m512 acc = _mm512_setzero_ps();

    int ib = 0;
    // Process 2 blocks per iteration
    for (; ib + 1 < nb; ib += 2) {
        // Load 64 int8s (2 blocks worth of quants)
        // But blocks are 34 bytes, not 32, so we can't load contiguously!
        // We must load each block's qs separately and combine

        __m256i qx0 = _mm256_loadu_si256((const __m256i *)x[ib].qs);
        __m256i qx1 = _mm256_loadu_si256((const __m256i *)x[ib + 1].qs);
        __m256i qy0 = _mm256_loadu_si256((const __m256i *)y[ib].qs);
        __m256i qy1 = _mm256_loadu_si256((const __m256i *)y[ib + 1].qs);

        // Combine into 512-bit vectors
        __m512i qx = _mm512_inserti64x4(_mm512_castsi256_si512(qx0), qx1, 1);
        __m512i qy = _mm512_inserti64x4(_mm512_castsi256_si512(qy0), qy1, 1);

        // Get scales for both blocks
        float d0 = fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d);
        float d1 = fp16_to_fp32(x[ib + 1].d) * fp16_to_fp32(y[ib + 1].d);

        // Compute dot products
        __m512 q = mul_sum_i8_pairs_float_avx512(qx, qy);

        // q now has 16 floats: [dot0_0..dot0_7, dot1_0..dot1_7]
        // We need to sum each half separately and scale
        __m256 q_lo = _mm512_castps512_ps256(q);
        __m256 q_hi = _mm512_extractf32x8_ps(q, 1);

        acc = _mm512_fmadd_ps(_mm512_set1_ps(d0), _mm512_castps256_ps512(q_lo), acc);
        __m512 acc_hi = _mm512_fmadd_ps(_mm512_set1_ps(d1), _mm512_castps256_ps512(q_hi), _mm512_setzero_ps());
        acc = _mm512_add_ps(acc, _mm512_castps256_ps512(_mm512_castps512_ps256(acc_hi)));

        // Simpler: just accumulate with right scales
        // Actually let's restructure this properly
    }

    // Handle remaining odd block with AVX2
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

// ============================================================================
// AVX-512 v2 - Cleaner implementation
// ============================================================================
void vec_dot_q8_0_avx512_v2(int n, float * s, const block_q8_0 * x, const block_q8_0 * y) {
    int nb = n / QK8_0;
    __m512 acc = _mm512_setzero_ps();

    int ib = 0;
    // Process 2 blocks per iteration - but accumulate separately
    for (; ib + 1 < nb; ib += 2) {
        // Load block 0
        __m256i qx0 = _mm256_loadu_si256((const __m256i *)x[ib].qs);
        __m256i qy0 = _mm256_loadu_si256((const __m256i *)y[ib].qs);
        float d0 = fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d);

        // Load block 1
        __m256i qx1 = _mm256_loadu_si256((const __m256i *)x[ib + 1].qs);
        __m256i qy1 = _mm256_loadu_si256((const __m256i *)y[ib + 1].qs);
        float d1 = fp16_to_fp32(x[ib + 1].d) * fp16_to_fp32(y[ib + 1].d);

        // Process block 0 with AVX2, but accumulate into 512
        __m256 q0 = mul_sum_i8_pairs_float_avx2(qx0, qy0);
        __m256 q1 = mul_sum_i8_pairs_float_avx2(qx1, qy1);

        // Scale and accumulate
        __m256 scaled0 = _mm256_mul_ps(_mm256_set1_ps(d0), q0);
        __m256 scaled1 = _mm256_mul_ps(_mm256_set1_ps(d1), q1);

        // Combine into 512 and add
        __m512 combined = _mm512_insertf32x8(_mm512_castps256_ps512(scaled0), scaled1, 1);
        acc = _mm512_add_ps(acc, combined);
    }

    // Reduce 512 to scalar
    float sumf = _mm512_reduce_add_ps(acc);

    // Handle remaining odd block
    for (; ib < nb; ++ib) {
        int sumi = 0;
        for (int j = 0; j < QK8_0; j++) {
            sumi += (int)x[ib].qs[j] * (int)y[ib].qs[j];
        }
        sumf += sumi * (fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d));
    }

    *s = sumf;
}

// ============================================================================
// AVX-512 v3 - True 512-bit compute with VNNI
// ============================================================================
#if defined(__AVX512VNNI__)
void vec_dot_q8_0_avx512_vnni(int n, float * s, const block_q8_0 * x, const block_q8_0 * y) {
    int nb = n / QK8_0;
    __m512 acc = _mm512_setzero_ps();

    int ib = 0;
    // Process 2 blocks per iteration
    for (; ib + 1 < nb; ib += 2) {
        // Load quants from 2 blocks
        __m256i qx0 = _mm256_loadu_si256((const __m256i *)x[ib].qs);
        __m256i qx1 = _mm256_loadu_si256((const __m256i *)x[ib + 1].qs);
        __m256i qy0 = _mm256_loadu_si256((const __m256i *)y[ib].qs);
        __m256i qy1 = _mm256_loadu_si256((const __m256i *)y[ib + 1].qs);

        // Get scales
        float d0 = fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d);
        float d1 = fp16_to_fp32(x[ib + 1].d) * fp16_to_fp32(y[ib + 1].d);

        // Combine into 512-bit registers
        __m512i qx = _mm512_inserti64x4(_mm512_castsi256_si512(qx0), qx1, 1);
        __m512i qy = _mm512_inserti64x4(_mm512_castsi256_si512(qy0), qy1, 1);

        // VNNI: signed int8 × signed int8 → int32
        __m512i zero = _mm512_setzero_si512();
        __m512i sumi = _mm512_dpbssd_epi32(zero, qx, qy);

        // sumi has 16 int32 values: [sum0_0..sum0_3, sum0_4..sum0_7, sum1_0..sum1_3, sum1_4..sum1_7]
        // Lower 8 belong to block 0, upper 8 belong to block 1
        __m512 sumf = _mm512_cvtepi32_ps(sumi);

        // Create scale vector: [d0,d0,d0,d0,d0,d0,d0,d0, d1,d1,d1,d1,d1,d1,d1,d1]
        __m512 scales = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_set1_ps(d0)),
                                           _mm256_set1_ps(d1), 1);

        acc = _mm512_fmadd_ps(scales, sumf, acc);
    }

    float sumf = _mm512_reduce_add_ps(acc);

    // Handle remaining odd block
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

#endif // AVX512F && AVX512BW

// ============================================================================
// Benchmark utilities
// ============================================================================
static double get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static void init_random_q8_0(block_q8_0 * blocks, int n_blocks) {
    for (int i = 0; i < n_blocks; i++) {
        // Random scale in reasonable range
        float scale = (float)(rand() % 100 + 1) / 100.0f;
        blocks[i].d = fp32_to_fp16(scale);

        // Random quants
        for (int j = 0; j < QK8_0; j++) {
            blocks[i].qs[j] = (int8_t)(rand() % 256 - 128);
        }
    }
}

#define BENCH_ITERATIONS 1000000
#define N_ELEMENTS (256 * QK8_0)  // 256 blocks = 8192 elements

int main(void) {
    printf("Q8_0 AVX-512 Dot Product Benchmark\n");
    printf("===================================\n\n");

    int n_blocks = N_ELEMENTS / QK8_0;
    printf("Test configuration:\n");
    printf("  Elements: %d\n", N_ELEMENTS);
    printf("  Blocks: %d\n", n_blocks);
    printf("  Block size: %zu bytes\n", sizeof(block_q8_0));
    printf("  Iterations: %d\n\n", BENCH_ITERATIONS);

    // Allocate aligned memory
    block_q8_0 *x = aligned_alloc(64, n_blocks * sizeof(block_q8_0));
    block_q8_0 *y = aligned_alloc(64, n_blocks * sizeof(block_q8_0));

    srand(42);
    init_random_q8_0(x, n_blocks);
    init_random_q8_0(y, n_blocks);

    float result_scalar, result_avx2, result_avx512, result_avx512_v2, result_vnni;
    double time_start, time_end;

    // Warm up and verify
    vec_dot_q8_0_scalar(N_ELEMENTS, &result_scalar, x, y);
    vec_dot_q8_0_avx2(N_ELEMENTS, &result_avx2, x, y);

    printf("Correctness check:\n");
    printf("  Scalar:  %.6f\n", result_scalar);
    printf("  AVX2:    %.6f (diff: %.6e)\n", result_avx2, fabs(result_avx2 - result_scalar));

#if defined(__AVX512F__) && defined(__AVX512BW__)
    vec_dot_q8_0_avx512(N_ELEMENTS, &result_avx512, x, y);
    vec_dot_q8_0_avx512_v2(N_ELEMENTS, &result_avx512_v2, x, y);
    printf("  AVX-512:    %.6f (diff: %.6e)\n", result_avx512, fabs(result_avx512 - result_scalar));
    printf("  AVX-512 v2: %.6f (diff: %.6e)\n", result_avx512_v2, fabs(result_avx512_v2 - result_scalar));

#if defined(__AVX512VNNI__)
    vec_dot_q8_0_avx512_vnni(N_ELEMENTS, &result_vnni, x, y);
    printf("  AVX-512 VNNI: %.6f (diff: %.6e)\n", result_vnni, fabs(result_vnni - result_scalar));
#endif
#endif

    printf("\nBenchmarking...\n\n");

    // Benchmark scalar
    time_start = get_time_ns();
    for (int i = 0; i < BENCH_ITERATIONS / 100; i++) {
        vec_dot_q8_0_scalar(N_ELEMENTS, &result_scalar, x, y);
    }
    time_end = get_time_ns();
    double ns_per_call_scalar = (time_end - time_start) / (BENCH_ITERATIONS / 100);
    printf("Scalar:       %.2f ns/call (%.2f M ops/sec)\n",
           ns_per_call_scalar, 1000.0 / ns_per_call_scalar);

    // Benchmark AVX2
    time_start = get_time_ns();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        vec_dot_q8_0_avx2(N_ELEMENTS, &result_avx2, x, y);
    }
    time_end = get_time_ns();
    double ns_per_call_avx2 = (time_end - time_start) / BENCH_ITERATIONS;
    printf("AVX2:         %.2f ns/call (%.2f M ops/sec) - baseline\n",
           ns_per_call_avx2, 1000.0 / ns_per_call_avx2);

#if defined(__AVX512F__) && defined(__AVX512BW__)
    // Benchmark AVX-512
    time_start = get_time_ns();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        vec_dot_q8_0_avx512(N_ELEMENTS, &result_avx512, x, y);
    }
    time_end = get_time_ns();
    double ns_per_call_avx512 = (time_end - time_start) / BENCH_ITERATIONS;
    printf("AVX-512:      %.2f ns/call (%.2f M ops/sec) - %.2fx vs AVX2\n",
           ns_per_call_avx512, 1000.0 / ns_per_call_avx512,
           ns_per_call_avx2 / ns_per_call_avx512);

    // Benchmark AVX-512 v2
    time_start = get_time_ns();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        vec_dot_q8_0_avx512_v2(N_ELEMENTS, &result_avx512_v2, x, y);
    }
    time_end = get_time_ns();
    double ns_per_call_avx512_v2 = (time_end - time_start) / BENCH_ITERATIONS;
    printf("AVX-512 v2:   %.2f ns/call (%.2f M ops/sec) - %.2fx vs AVX2\n",
           ns_per_call_avx512_v2, 1000.0 / ns_per_call_avx512_v2,
           ns_per_call_avx2 / ns_per_call_avx512_v2);

#if defined(__AVX512VNNI__)
    // Benchmark AVX-512 VNNI
    time_start = get_time_ns();
    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        vec_dot_q8_0_avx512_vnni(N_ELEMENTS, &result_vnni, x, y);
    }
    time_end = get_time_ns();
    double ns_per_call_vnni = (time_end - time_start) / BENCH_ITERATIONS;
    printf("AVX-512 VNNI: %.2f ns/call (%.2f M ops/sec) - %.2fx vs AVX2\n",
           ns_per_call_vnni, 1000.0 / ns_per_call_vnni,
           ns_per_call_avx2 / ns_per_call_vnni);
#else
    printf("\nNote: AVX-512 VNNI not available (no __AVX512VNNI__ defined)\n");
#endif

#else
    printf("\nNote: AVX-512F/BW not available\n");
#endif

    printf("\n");
    printf("Summary:\n");
    printf("  Q8_0 block structure: 34 bytes (2B scale + 32B quants)\n");
    printf("  AVX2 processes 1 block per iteration (32 int8s)\n");
    printf("  AVX-512 processes 2 blocks per iteration (64 int8s)\n");
    printf("  With VNNI: _mm512_dpbssd_epi32 does signed int8 dot product directly\n");

    free(x);
    free(y);
    return 0;
}
