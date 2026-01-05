// Synthetic micro-benchmark for Q4_K dot product
// Tests AVX2 vs potential AVX-512 implementations
// DO NOT run with actual models - synthetic data only

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

// From ggml-common.h
#define QK_K 256
#define K_SCALE_SIZE 12

typedef uint16_t ggml_half;

// block_q4_K structure
typedef struct {
    union {
        struct {
            ggml_half d;
            ggml_half dmin;
        };
        uint32_t dm;
    };
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];  // 128 bytes of 4-bit quants
} block_q4_K;

// block_q8_K structure
typedef struct {
    float   d;
    int8_t  qs[QK_K];       // 256 bytes
    int16_t bsums[QK_K/16]; // 16 int16 = 32 bytes
} block_q8_K;

// Simple FP16 conversion (lookup table approximation)
static float ggml_fp16_to_fp32(ggml_half h) {
    // Simplified - just treat as float for synthetic benchmark
    return (float)h / 65536.0f;
}

// ============================================================================
// REFERENCE: Current AVX2 implementation (from arch/x86/quants.c)
// ============================================================================

static const uint32_t kmask1 = 0x3f3f3f3f;
static const uint32_t kmask2 = 0x0f0f0f0f;
static const uint32_t kmask3 = 0x03030303;

// Scale shuffle patterns
static const uint8_t k4_shuffle[8][32] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
    {2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3},
    {4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5},
    {6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7},
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
    {2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3},
    {4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5},
    {6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7},
};

static inline __m256i get_scale_shuffle_k4(int i) {
    return _mm256_loadu_si256((const __m256i*)k4_shuffle[i]);
}

// horizontally add 8 floats
static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

// Current AVX2 implementation
void ggml_vec_dot_q4_K_q8_K_avx2(int n, float * restrict s,
                                  const block_q4_K * restrict x,
                                  const block_q8_K * restrict y) {
    const int nb = n / QK_K;
    uint32_t utmp[4];

    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * ggml_fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * ggml_fp16_to_fp32(x[i].dmin);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t * restrict q4 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = MM256_SET_M128I(sc128, sc128);

        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < QK_K/64; ++j) {
            const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+0));
            const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+1));

            const __m256i q4bits = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            const __m256i q8l = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
            p16l = _mm256_madd_epi16(scale_l, p16l);

            const __m256i q8h = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
            p16h = _mm256_madd_epi16(scale_h, p16h);
            const __m256i sumj = _mm256_add_epi32(p16l, p16h);

            sumi = _mm256_add_epi32(sumi, sumj);
        }

        __m256 vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

    *s = hsum_float_8(acc) + _mm_cvtss_f32(acc_m);
}

// ============================================================================
// EXPERIMENTAL: AVX-512 implementation
// ============================================================================

#ifdef __AVX512F__

// horizontally add 16 floats
static inline float hsum_float_16(const __m512 x) {
    return _mm512_reduce_add_ps(x);
}

// AVX-512 scale shuffle patterns (64 bytes)
static const uint8_t k4_shuffle_512[8][64] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
     0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
    {2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
     2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3},
    {4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
     4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5},
    {6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
     6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7},
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
     0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
    {2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
     2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3},
    {4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
     4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5},
    {6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
     6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7},
};

// AVX-512 version - process 64 bytes at a time instead of 32
void ggml_vec_dot_q4_K_q8_K_avx512(int n, float * restrict s,
                                    const block_q4_K * restrict x,
                                    const block_q8_K * restrict y) {
    const int nb = n / QK_K;
    uint32_t utmp[4];

    const __m512i m4 = _mm512_set1_epi8(0xF);
    __m512 acc = _mm512_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * ggml_fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * ggml_fp16_to_fp32(x[i].dmin);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t * restrict q4 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        // Scale handling (same as AVX2 for now)
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        // Broadcast scales to 512 bits (4x the 128-bit scales)
        const __m512i scales = _mm512_broadcast_i32x4(sc128);

        __m512i sumi = _mm512_setzero_si512();

        // Process 2 iterations at a time (64 bytes of q4, 128 bytes of q8)
        for (int j = 0; j < QK_K/64; j += 2) {
            // Load 64 bytes of Q4 data (covers 2 original iterations)
            const __m512i q4bits = _mm512_loadu_si512((const __m512i*)q4); q4 += 64;

            // Split into low and high nibbles
            const __m512i q4l = _mm512_and_si512(q4bits, m4);
            const __m512i q4h = _mm512_and_si512(_mm512_srli_epi16(q4bits, 4), m4);

            // Load 128 bytes of Q8 data
            const __m512i q8l = _mm512_loadu_si512((const __m512i*)q8); q8 += 64;
            const __m512i q8h = _mm512_loadu_si512((const __m512i*)q8); q8 += 64;

            // Shuffle scales for these iterations
            const __m512i scale_l = _mm512_shuffle_epi8(scales,
                _mm512_loadu_si512((const __m512i*)k4_shuffle_512[2*j+0]));
            const __m512i scale_h = _mm512_shuffle_epi8(scales,
                _mm512_loadu_si512((const __m512i*)k4_shuffle_512[2*j+1]));

            // Multiply-add with VNNI if available
#ifdef __AVX512VNNI__
            __m512i p32l = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q4l, q8l);
            __m512i p32h = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q4h, q8h);
            // Apply scales (simplified - would need proper scale application)
            p32l = _mm512_madd_epi16(scale_l, _mm512_packs_epi32(p32l, p32l));
            p32h = _mm512_madd_epi16(scale_h, _mm512_packs_epi32(p32h, p32h));
#else
            // Standard AVX-512 path
            __m512i p16l = _mm512_maddubs_epi16(q4l, q8l);
            p16l = _mm512_madd_epi16(scale_l, p16l);

            __m512i p16h = _mm512_maddubs_epi16(q4h, q8h);
            p16h = _mm512_madd_epi16(scale_h, p16h);

            const __m512i sumj = _mm512_add_epi32(p16l, p16h);
            sumi = _mm512_add_epi32(sumi, sumj);
#endif
        }

        __m512 vd = _mm512_set1_ps(d);
        acc = _mm512_fmadd_ps(vd, _mm512_cvtepi32_ps(sumi), acc);
    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

    *s = hsum_float_16(acc) + _mm_cvtss_f32(acc_m);
}

#endif // __AVX512F__

// ============================================================================
// Benchmark harness
// ============================================================================

void init_synthetic_data(block_q4_K *x, block_q8_K *y, int nb) {
    srand(42);
    for (int i = 0; i < nb; i++) {
        x[i].d = rand() % 256;
        x[i].dmin = rand() % 256;
        for (int j = 0; j < K_SCALE_SIZE; j++) {
            x[i].scales[j] = rand() % 256;
        }
        for (int j = 0; j < QK_K/2; j++) {
            x[i].qs[j] = rand() % 256;
        }

        y[i].d = 1.0f + (rand() % 100) / 100.0f;
        for (int j = 0; j < QK_K; j++) {
            y[i].qs[j] = (rand() % 256) - 128;
        }
        for (int j = 0; j < QK_K/16; j++) {
            y[i].bsums[j] = rand() % 1000;
        }
    }
}

double get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

int main(int argc, char **argv) {
    const int n = 4096;  // QK_K * nb
    const int nb = n / QK_K;  // 16 blocks
    const int iterations = 100000;

    printf("Q4_K Dot Product Micro-Benchmark\n");
    printf("================================\n");
    printf("Vector size: %d elements (%d blocks)\n", n, nb);
    printf("Iterations: %d\n\n", iterations);

    // Allocate aligned memory
    block_q4_K *x = aligned_alloc(64, nb * sizeof(block_q4_K));
    block_q8_K *y = aligned_alloc(64, nb * sizeof(block_q8_K));

    if (!x || !y) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    init_synthetic_data(x, y, nb);

    float result_avx2 = 0;
    float result_avx512 = 0;

    // Warm up
    for (int i = 0; i < 1000; i++) {
        ggml_vec_dot_q4_K_q8_K_avx2(n, &result_avx2, x, y);
    }

    // Benchmark AVX2
    double start = get_time_ns();
    for (int i = 0; i < iterations; i++) {
        ggml_vec_dot_q4_K_q8_K_avx2(n, &result_avx2, x, y);
    }
    double end = get_time_ns();
    double avx2_time = (end - start) / iterations;

    printf("AVX2 Implementation:\n");
    printf("  Time per call: %.2f ns\n", avx2_time);
    printf("  Throughput: %.2f M ops/sec\n", 1e9 / avx2_time / 1e6);
    printf("  Result: %f\n\n", result_avx2);

#ifdef __AVX512F__
    // Warm up
    for (int i = 0; i < 1000; i++) {
        ggml_vec_dot_q4_K_q8_K_avx512(n, &result_avx512, x, y);
    }

    // Benchmark AVX-512
    start = get_time_ns();
    for (int i = 0; i < iterations; i++) {
        ggml_vec_dot_q4_K_q8_K_avx512(n, &result_avx512, x, y);
    }
    end = get_time_ns();
    double avx512_time = (end - start) / iterations;

    printf("AVX-512 Implementation:\n");
    printf("  Time per call: %.2f ns\n", avx512_time);
    printf("  Throughput: %.2f M ops/sec\n", 1e9 / avx512_time / 1e6);
    printf("  Result: %f\n\n", result_avx512);

    printf("Speedup: %.2fx\n", avx2_time / avx512_time);
    printf("Result match: %s\n", (fabsf(result_avx2 - result_avx512) < 0.01f) ? "YES" : "NO");
#else
    printf("AVX-512 not available on this system\n");
#endif

    free(x);
    free(y);

    return 0;
}
