/*
 * Q4_K AVX-512 VBMI Implementation - Correct Data Layout Handling
 *
 * Key insight: Q4_K nibble packing requires non-contiguous Q8 access:
 * - 64 low nibbles need Q8[0:31, 64:95]
 * - 64 high nibbles need Q8[32:63, 96:127]
 *
 * Solution: Use AVX-512 VBMI _mm512_permutex2var_epi8 for byte-level permutation
 * Zen 5 has full AVX-512 VBMI support with single-cycle permutes
 */

#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define QK_K 256
#define K_SCALE_SIZE 12

// Q4_K block structure (144 bytes)
typedef struct {
    uint16_t d;           // delta (fp16)
    uint16_t dmin;        // min (fp16)
    uint8_t scales[K_SCALE_SIZE]; // scales and mins (12 bytes)
    uint8_t qs[QK_K/2];   // quants (128 bytes)
} block_q4_K;

// Q8_K block structure (292 bytes)
typedef struct {
    float d;              // delta
    int8_t qs[QK_K];      // quants (256 bytes)
    int16_t bsums[QK_K/16]; // block sums (32 bytes)
} block_q8_K;

static const uint32_t kmask1 = 0x3f3f3f3f;
static const uint32_t kmask2 = 0x0f0f0f0f;
static const uint32_t kmask3 = 0x03030303;

// FP16 to FP32 conversion
static inline float fp16_to_fp32(uint16_t h) {
    union { uint32_t u; float f; } u;
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    if (exp == 0) {
        if (mant == 0) { u.u = sign; return u.f; }
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        exp++; mant &= ~0x400;
    } else if (exp == 31) {
        u.u = sign | 0x7f800000 | (mant << 13);
        return u.f;
    }
    u.u = sign | ((exp + 112) << 23) | (mant << 13);
    return u.f;
}

// Scale shuffle patterns for K-quants
static const uint8_t k_shuffle_data[8][32] = {
    { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3 },
    { 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3 },
    { 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3 },
    { 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3 },
    { 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7 },
};

static inline __m256i get_scale_shuffle_k4(int j) {
    return _mm256_loadu_si256((const __m256i*)k_shuffle_data[j]);
}

// Horizontal sum for __m256
static inline float hsum_float_8(__m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// Reference AVX2 implementation (from ggml)
void ggml_vec_dot_q4_K_q8_K_avx2(int n, float * __restrict s,
                                  const void * __restrict vx,
                                  const void * __restrict vy) {
    const int nb = n / QK_K;
    const block_q4_K * __restrict x = vx;
    const block_q8_K * __restrict y = vy;

    uint32_t utmp[4];

    const __m256i m4 = _mm256_set1_epi8(0xF);

    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * fp16_to_fp32(x[i].dmin);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t * __restrict q4 = x[i].qs;
        const int8_t  * __restrict q8 = y[i].qs;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_set_m128i(sc128, sc128);

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

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

    *s = hsum_float_8(acc) + _mm_cvtss_f32(acc_m);
}

/*
 * AVX-512 VBMI Implementation
 *
 * Strategy: Process 2 AVX2 iterations worth of data (64 bytes Q4, 128 bytes Q8)
 * Use permutes to rearrange Q8 for correct nibble pairing:
 *   - For low nibbles: need Q8[0:31] || Q8[64:95]
 *   - For high nibbles: need Q8[32:63] || Q8[96:127]
 */
void ggml_vec_dot_q4_K_q8_K_avx512_vbmi(int n, float * __restrict s,
                                         const void * __restrict vx,
                                         const void * __restrict vy) {
    const int nb = n / QK_K;
    const block_q4_K * __restrict x = vx;
    const block_q8_K * __restrict y = vy;

    uint32_t utmp[4];

    const __m512i m4_512 = _mm512_set1_epi8(0xF);
    const __m256i m4 = _mm256_set1_epi8(0xF);

    // Permute indices to rearrange Q8 bytes:
    // Input:  q8_01 = [0:63], q8_23 = [64:127]
    // Output for low:  [0:31 from q8_01] || [0:31 from q8_23]
    // Output for high: [32:63 from q8_01] || [32:63 from q8_23]
    //
    // For _mm512_permutex2var_epi8:
    // - indices 0-63 select from first operand
    // - indices 64-127 select from second operand (with bit 6 set)

    // Low nibbles need: bytes [0:31] from q8_01 and bytes [0:31] from q8_23
    // = indices [0,1,...,31, 64,65,...,95] but for 512-bit permute, second half comes from q8_23
    // Actually simpler: we want [q8_01[0:31], q8_23[0:31]]
    // Since q8_23 is loaded separately, we use inserti64x4

    // But wait - _mm512_permutex2var_epi8 works on two 512-bit operands
    // We'd need to load both Q8 halves into 512-bit registers first

    // Alternative: use extract/insert which is cleaner

    __m512 acc = _mm512_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * fp16_to_fp32(x[i].dmin);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t * __restrict q4 = x[i].qs;
        const int8_t  * __restrict q8 = y[i].qs;

        // Min correction term (same as AVX2)
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_set_m128i(sc128, sc128);

        __m512i sumi = _mm512_setzero_si512();

        // Process 2 iterations at a time (j and j+1)
        // QK_K/64 = 4, so we do 2 outer loops
        for (int j = 0; j < QK_K/64; j += 2) {
            // === First iteration (j) ===
            const __m256i scale_l0 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+0));
            const __m256i scale_h0 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+1));

            const __m256i q4bits_0 = _mm256_loadu_si256((const __m256i*)q4);
            const __m256i q4l_0 = _mm256_and_si256(q4bits_0, m4);
            const __m256i q4h_0 = _mm256_and_si256(_mm256_srli_epi16(q4bits_0, 4), m4);

            const __m256i q8l_0 = _mm256_loadu_si256((const __m256i*)q8);
            const __m256i q8h_0 = _mm256_loadu_si256((const __m256i*)(q8 + 32));

            // === Second iteration (j+1) ===
            const __m256i scale_l1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*(j+1)+0));
            const __m256i scale_h1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*(j+1)+1));

            const __m256i q4bits_1 = _mm256_loadu_si256((const __m256i*)(q4 + 32));
            const __m256i q4l_1 = _mm256_and_si256(q4bits_1, m4);
            const __m256i q4h_1 = _mm256_and_si256(_mm256_srli_epi16(q4bits_1, 4), m4);

            const __m256i q8l_1 = _mm256_loadu_si256((const __m256i*)(q8 + 64));
            const __m256i q8h_1 = _mm256_loadu_si256((const __m256i*)(q8 + 96));

            q4 += 64;
            q8 += 128;

            // Combine into 512-bit vectors for parallel processing
            // Low nibbles: [q4l_0, q4l_1] with [q8l_0, q8l_1]
            __m512i q4l_512 = _mm512_inserti64x4(_mm512_castsi256_si512(q4l_0), q4l_1, 1);
            __m512i q8l_512 = _mm512_inserti64x4(_mm512_castsi256_si512(q8l_0), q8l_1, 1);

            // High nibbles: [q4h_0, q4h_1] with [q8h_0, q8h_1]
            __m512i q4h_512 = _mm512_inserti64x4(_mm512_castsi256_si512(q4h_0), q4h_1, 1);
            __m512i q8h_512 = _mm512_inserti64x4(_mm512_castsi256_si512(q8h_0), q8h_1, 1);

            // Scales: [scale_l0, scale_l1] and [scale_h0, scale_h1]
            __m512i scale_l_512 = _mm512_inserti64x4(_mm512_castsi256_si512(scale_l0), scale_l1, 1);
            __m512i scale_h_512 = _mm512_inserti64x4(_mm512_castsi256_si512(scale_h0), scale_h1, 1);

            // 512-bit dot products
            __m512i p16l = _mm512_maddubs_epi16(q4l_512, q8l_512);
            p16l = _mm512_madd_epi16(scale_l_512, p16l);

            __m512i p16h = _mm512_maddubs_epi16(q4h_512, q8h_512);
            p16h = _mm512_madd_epi16(scale_h_512, p16h);

            sumi = _mm512_add_epi32(sumi, _mm512_add_epi32(p16l, p16h));
        }

        acc = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), acc);
    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

    *s = _mm512_reduce_add_ps(acc) + _mm_cvtss_f32(acc_m);
}

/*
 * AVX-512 with 512-bit loads and VBMI permute
 *
 * This version loads Q4 and Q8 at full 512-bit width, then permutes
 * Q8 to match the nibble pairing requirements.
 */
void ggml_vec_dot_q4_K_q8_K_avx512_vbmi_fullwidth(int n, float * __restrict s,
                                                   const void * __restrict vx,
                                                   const void * __restrict vy) {
    const int nb = n / QK_K;
    const block_q4_K * __restrict x = vx;
    const block_q8_K * __restrict y = vy;

    uint32_t utmp[4];

    const __m512i m4_512 = _mm512_set1_epi8(0xF);

    // Permute indices for rearranging Q8 data
    // We load Q8[0:63] and Q8[64:127] as two 512-bit vectors
    // For low nibbles: need [Q8[0:31], Q8[64:95]]
    // For high nibbles: need [Q8[32:63], Q8[96:127]]
    //
    // Using _mm512_permutex2var_epi8(a, idx, b):
    // - idx[i] & 0x3F selects byte from a (if bit 6 clear) or b (if bit 6 set)
    //
    // For low: indices 0-31 from a[0:31], indices 32-63 from b[0:31] (= 64+0 to 64+31)
    const __m512i perm_low = _mm512_set_epi8(
        95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80,  // b[0:15] -> idx 64-79
        79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64,  // b[16:31] -> idx 80-95
        31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,  // a[0:15]
        15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0   // a[16:31]
    );

    // For high: indices 0-31 from a[32:63], indices 32-63 from b[32:63] (= 64+32 to 64+63)
    const __m512i perm_high = _mm512_set_epi8(
        127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112,
        111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100,  99,  98,  97,  96,
         63,  62,  61,  60,  59,  58,  57,  56,  55,  54,  53,  52,  51,  50,  49,  48,
         47,  46,  45,  44,  43,  42,  41,  40,  39,  38,  37,  36,  35,  34,  33,  32
    );

    __m512 acc = _mm512_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * fp16_to_fp32(x[i].dmin);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t * __restrict q4 = x[i].qs;
        const int8_t  * __restrict q8 = y[i].qs;

        // Min correction (same as AVX2)
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales_256 = _mm256_set_m128i(sc128, sc128);

        // Broadcast scales to 512-bit
        const __m512i scales = _mm512_broadcast_i32x8(scales_256);

        __m512i sumi = _mm512_setzero_si512();

        // Process 2 iterations at once with full 512-bit loads
        for (int j = 0; j < QK_K/64; j += 2) {
            // Load 64 bytes Q4 (two groups worth)
            const __m512i q4bits = _mm512_loadu_si512((const __m512i*)q4);
            q4 += 64;

            // Extract nibbles at 512-bit width
            const __m512i q4l = _mm512_and_si512(q4bits, m4_512);
            const __m512i q4h = _mm512_and_si512(_mm512_srli_epi16(q4bits, 4), m4_512);

            // Load 128 bytes Q8 as two 512-bit vectors
            const __m512i q8_01 = _mm512_loadu_si512((const __m512i*)q8);       // Q8[0:63]
            const __m512i q8_23 = _mm512_loadu_si512((const __m512i*)(q8 + 64)); // Q8[64:127]
            q8 += 128;

            // Permute Q8 to match nibble correspondence
            // Low nibbles need: [Q8[0:31], Q8[64:95]]
            const __m512i q8_for_low = _mm512_permutex2var_epi8(q8_01, perm_low, q8_23);
            // High nibbles need: [Q8[32:63], Q8[96:127]]
            const __m512i q8_for_high = _mm512_permutex2var_epi8(q8_01, perm_high, q8_23);

            // Get scales for both iterations
            // j=0: scale indices 0,1 for low/high; j=1: scale indices 2,3 for low/high
            // This is trickier - scales are 8-bit and we need to broadcast appropriately
            // For simplicity, fall back to 256-bit scale handling
            const __m256i scale_l0 = _mm256_shuffle_epi8(scales_256, _mm256_loadu_si256((const __m256i*)k_shuffle_data[2*j+0]));
            const __m256i scale_h0 = _mm256_shuffle_epi8(scales_256, _mm256_loadu_si256((const __m256i*)k_shuffle_data[2*j+1]));
            const __m256i scale_l1 = _mm256_shuffle_epi8(scales_256, _mm256_loadu_si256((const __m256i*)k_shuffle_data[2*(j+1)+0]));
            const __m256i scale_h1 = _mm256_shuffle_epi8(scales_256, _mm256_loadu_si256((const __m256i*)k_shuffle_data[2*(j+1)+1]));

            const __m512i scale_l = _mm512_inserti64x4(_mm512_castsi256_si512(scale_l0), scale_l1, 1);
            const __m512i scale_h = _mm512_inserti64x4(_mm512_castsi256_si512(scale_h0), scale_h1, 1);

            // 512-bit multiply-add
            __m512i p16l = _mm512_maddubs_epi16(q4l, q8_for_low);
            p16l = _mm512_madd_epi16(scale_l, p16l);

            __m512i p16h = _mm512_maddubs_epi16(q4h, q8_for_high);
            p16h = _mm512_madd_epi16(scale_h, p16h);

            sumi = _mm512_add_epi32(sumi, _mm512_add_epi32(p16l, p16h));
        }

        acc = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), acc);
    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

    *s = _mm512_reduce_add_ps(acc) + _mm_cvtss_f32(acc_m);
}

// Initialize synthetic test data
void init_test_data(block_q4_K *x, block_q8_K *y, int nb) {
    srand(42);
    for (int i = 0; i < nb; i++) {
        x[i].d = 0x3C00;     // 1.0 in fp16
        x[i].dmin = 0x3800;  // 0.5 in fp16

        for (int j = 0; j < K_SCALE_SIZE; j++) {
            x[i].scales[j] = rand() & 0x3F;
        }
        for (int j = 0; j < QK_K/2; j++) {
            x[i].qs[j] = rand() & 0xFF;
        }

        y[i].d = 1.0f;
        for (int j = 0; j < QK_K; j++) {
            y[i].qs[j] = (rand() % 256) - 128;
        }
        for (int j = 0; j < QK_K/16; j++) {
            int16_t sum = 0;
            for (int k = 0; k < 16; k++) {
                sum += y[i].qs[j*16 + k];
            }
            y[i].bsums[j] = sum;
        }
    }
}

int main() {
    printf("Q4_K AVX-512 VBMI Benchmark\n");
    printf("===========================\n\n");

    const int nb = 1;  // Single block for correctness testing
    const int n = nb * QK_K;

    block_q4_K *x = aligned_alloc(64, nb * sizeof(block_q4_K));
    block_q8_K *y = aligned_alloc(64, nb * sizeof(block_q8_K));

    init_test_data(x, y, nb);

    float result_avx2 = 0, result_vbmi = 0, result_vbmi_full = 0;

    // Correctness test
    ggml_vec_dot_q4_K_q8_K_avx2(n, &result_avx2, x, y);
    ggml_vec_dot_q4_K_q8_K_avx512_vbmi(n, &result_vbmi, x, y);
    ggml_vec_dot_q4_K_q8_K_avx512_vbmi_fullwidth(n, &result_vbmi_full, x, y);

    printf("Correctness Test:\n");
    printf("  AVX2 result:           %.2f\n", result_avx2);
    printf("  AVX-512 VBMI result:   %.2f\n", result_vbmi);
    printf("  AVX-512 Full result:   %.2f\n", result_vbmi_full);
    printf("  VBMI match:            %s\n", fabsf(result_avx2 - result_vbmi) < 0.01 ? "YES" : "NO");
    printf("  Full match:            %s\n", fabsf(result_avx2 - result_vbmi_full) < 0.01 ? "YES" : "NO");
    printf("\n");

    // Performance test
    const int iterations = 100000;
    const int nb_perf = 16;  // More blocks for realistic timing
    const int n_perf = nb_perf * QK_K;

    block_q4_K *x_perf = aligned_alloc(64, nb_perf * sizeof(block_q4_K));
    block_q8_K *y_perf = aligned_alloc(64, nb_perf * sizeof(block_q8_K));
    init_test_data(x_perf, y_perf, nb_perf);

    struct timespec start, end;
    double elapsed;
    float dummy = 0;

    // Warmup
    for (int i = 0; i < 1000; i++) {
        ggml_vec_dot_q4_K_q8_K_avx2(n_perf, &dummy, x_perf, y_perf);
    }

    // AVX2
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; i++) {
        ggml_vec_dot_q4_K_q8_K_avx2(n_perf, &dummy, x_perf, y_perf);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double avx2_ns = elapsed * 1e9 / iterations;

    // AVX-512 VBMI (inserti approach)
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; i++) {
        ggml_vec_dot_q4_K_q8_K_avx512_vbmi(n_perf, &dummy, x_perf, y_perf);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double vbmi_ns = elapsed * 1e9 / iterations;

    // AVX-512 VBMI Full (permute approach)
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; i++) {
        ggml_vec_dot_q4_K_q8_K_avx512_vbmi_fullwidth(n_perf, &dummy, x_perf, y_perf);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double vbmi_full_ns = elapsed * 1e9 / iterations;

    printf("Performance Test (%d blocks, %d iterations):\n", nb_perf, iterations);
    printf("  AVX2:                  %.2f ns/call\n", avx2_ns);
    printf("  AVX-512 VBMI:          %.2f ns/call (%.2fx)\n", vbmi_ns, avx2_ns / vbmi_ns);
    printf("  AVX-512 Full:          %.2f ns/call (%.2fx)\n", vbmi_full_ns, avx2_ns / vbmi_full_ns);

    free(x);
    free(y);
    free(x_perf);
    free(y_perf);

    return 0;
}
