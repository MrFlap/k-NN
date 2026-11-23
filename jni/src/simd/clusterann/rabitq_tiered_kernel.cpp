/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

// Bulk inner products of a ClusterANN tiered RaBitQ scan (SearchServicesKnnVectorFormats TieredRaBitQKernel):
//   coarse: out[i] = sum_{j<4} 2^j * popcount(planes[j] & bits[i])      (4-bit query x 1-bit code)
//   refine: out[i] = sum_d recs[i * stride + d] * q[d]                   (codes in [0,127], q in [-127,127])
// Both are exact integer results, so every ISA path returns what the Java loops return. Like int8_bulk_kernel.cpp the
// ISA comes from the flags of the library this unit is compiled into; aarch64 picks dotprod at run time.

#include <cstdint>
#include <cstring>

#include "simd/similarity_function/clusterann_rabitq.h"

#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>
#define RBQ_AVX512 1
#if defined(__AVX512VPOPCNTDQ__)
#define RBQ_AVX512_POPCNT 1
#endif
#if defined(__AVX512VNNI__)
#define RBQ_AVX512_VNNI 1
#endif
#elif defined(__AVX2__)
#include <immintrin.h>
#define RBQ_AVX2 1
#elif defined(__aarch64__)
#include <arm_neon.h>
#if defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif
#define RBQ_NEON 1
#define RBQ_DOTPROD_TARGET __attribute__((target("arch=armv8.2-a+dotprod")))
#endif

namespace knn_jni::simd::clusterann {
namespace {

// ------------------------------------------------------------------------------------------------------------ coarse

FORCE_INLINE uint64_t load64(const uint8_t* p) {
    uint64_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

// Scalar popcount over 8-byte words, then the byte tail.
int32_t coarseScalar(const uint8_t* planes, const uint8_t* row, int32_t bitBytes) {
    int64_t s = 0;
    for (int j = 0; j < 4; ++j) {
        const uint8_t* p = planes + static_cast<size_t>(j) * bitBytes;
        int64_t c = 0;
        int32_t b = 0;
        for (; b + 8 <= bitBytes; b += 8) c += __builtin_popcountll(load64(p + b) & load64(row + b));
        for (; b < bitBytes; ++b) c += __builtin_popcount(static_cast<uint32_t>(p[b] & row[b]));
        s += c << j;
    }
    return static_cast<int32_t>(s);
}

#if defined(RBQ_AVX512_POPCNT)
int32_t coarseRow(const uint8_t* planes, const uint8_t* row, int32_t bitBytes) {
    __m512i acc[4] = {_mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512()};
    int32_t b = 0;
    for (; b < bitBytes; b += 64) {
        const int32_t left = bitBytes - b;
        const __mmask64 m = left >= 64 ? ~0ULL : ((1ULL << left) - 1ULL);
        const __m512i r = _mm512_maskz_loadu_epi8(m, row + b);
        for (int j = 0; j < 4; ++j) {
            const __m512i p = _mm512_maskz_loadu_epi8(m, planes + static_cast<size_t>(j) * bitBytes + b);
            acc[j] = _mm512_add_epi64(acc[j], _mm512_popcnt_epi64(_mm512_and_si512(p, r)));
        }
    }
    int64_t s = 0;
    for (int j = 0; j < 4; ++j) s += _mm512_reduce_add_epi64(acc[j]) << j;
    return static_cast<int32_t>(s);
}
#elif defined(RBQ_NEON)
int32_t coarseRow(const uint8_t* planes, const uint8_t* row, int32_t bitBytes) {
    uint16x8_t acc[4] = {vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0)};
    int32_t b = 0;
    for (; b + 16 <= bitBytes; b += 16) {
        const uint8x16_t r = vld1q_u8(row + b);
        for (int j = 0; j < 4; ++j) {
            const uint8x16_t p = vld1q_u8(planes + static_cast<size_t>(j) * bitBytes + b);
            acc[j] = vpadalq_u8(acc[j], vcntq_u8(vandq_u8(p, r)));
        }
    }
    int64_t s = 0;
    for (int j = 0; j < 4; ++j) s += static_cast<int64_t>(vaddlvq_u16(acc[j])) << j;
    for (; b < bitBytes; ++b) {
        for (int j = 0; j < 4; ++j) {
            s += static_cast<int64_t>(__builtin_popcount(static_cast<uint32_t>(planes[static_cast<size_t>(j) * bitBytes + b] & row[b]))) << j;
        }
    }
    return static_cast<int32_t>(s);
}
#else
// x86 without VPOPCNTDQ (and generic): hardware popcnt on 64-bit words is the fastest portable form.
int32_t coarseRow(const uint8_t* planes, const uint8_t* row, int32_t bitBytes) {
    return coarseScalar(planes, row, bitBytes);
}
#endif

// ------------------------------------------------------------------------------------------------------------ refine

int32_t refineScalar(const int8_t* code, const int8_t* q, int32_t dim) {
    int32_t s = 0;
    for (int32_t d = 0; d < dim; ++d) s += static_cast<int32_t>(code[d]) * static_cast<int32_t>(q[d]);
    return s;
}

#if defined(RBQ_AVX512_VNNI)
// Codes are in [0,127], so they are valid unsigned operands of vpdpbusd against the signed query: exact.
int32_t refineRow(const int8_t* code, const int8_t* q, int32_t dim) {
    __m512i acc = _mm512_setzero_si512();
    int32_t d = 0;
    for (; d < dim; d += 64) {
        const int32_t left = dim - d;
        const __mmask64 m = left >= 64 ? ~0ULL : ((1ULL << left) - 1ULL);
        acc = _mm512_dpbusd_epi32(acc, _mm512_maskz_loadu_epi8(m, code + d), _mm512_maskz_loadu_epi8(m, q + d));
    }
    return _mm512_reduce_add_epi32(acc);
}
#elif defined(RBQ_AVX512)
int32_t refineRow(const int8_t* code, const int8_t* q, int32_t dim) {
    __m512i acc = _mm512_setzero_si512();
    int32_t d = 0;
    for (; d + 32 <= dim; d += 32) {
        const __m512i c = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(code + d)));
        const __m512i v = _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + d)));
        acc = _mm512_add_epi32(acc, _mm512_madd_epi16(c, v));
    }
    return _mm512_reduce_add_epi32(acc) + refineScalar(code + d, q + d, dim - d);
}
#elif defined(RBQ_AVX2)
// Codes are non-negative and at most 127, so maddubs (u8 x s8 -> pairwise s16) cannot saturate: 2 * 127 * 127 < 32767.
int32_t refineRow(const int8_t* code, const int8_t* q, int32_t dim) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i acc = _mm256_setzero_si256();
    int32_t d = 0;
    for (; d + 32 <= dim; d += 32) {
        const __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(code + d));
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + d));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(c, v), ones));
    }
    const __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    const __m128i s2 = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    const __m128i s3 = _mm_add_epi32(s2, _mm_shuffle_epi32(s2, 0xB1));
    return _mm_cvtsi128_si32(s3) + refineScalar(code + d, q + d, dim - d);
}
#elif defined(RBQ_NEON)
RBQ_DOTPROD_TARGET int32_t refineRowDotprod(const int8_t* code, const int8_t* q, int32_t dim) {
    int32x4_t acc = vdupq_n_s32(0);
    int32_t d = 0;
    for (; d + 16 <= dim; d += 16) acc = vdotq_s32(acc, vld1q_s8(code + d), vld1q_s8(q + d));
    return vaddvq_s32(acc) + refineScalar(code + d, q + d, dim - d);
}

int32_t refineRowNeon(const int8_t* code, const int8_t* q, int32_t dim) {
    int32x4_t acc = vdupq_n_s32(0);
    int32_t d = 0;
    for (; d + 16 <= dim; d += 16) {
        const int8x16_t c = vld1q_s8(code + d);
        const int8x16_t v = vld1q_s8(q + d);
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(c), vget_low_s8(v)));
        acc = vpadalq_s16(acc, vmull_high_s8(c, v));
    }
    return vaddvq_s32(acc) + refineScalar(code + d, q + d, dim - d);
}

bool hasDotprod() {
#if defined(__linux__)
    static const bool value = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#else
    static const bool value = true;  // every Apple aarch64 core has dotprod; non-Linux builds are local only
#endif
    return value;
}
#else
int32_t refineRow(const int8_t* code, const int8_t* q, int32_t dim) {
    return refineScalar(code, q, dim);
}
#endif

}  // namespace

const char* rabitqTieredKernelName() {
#if defined(RBQ_AVX512_POPCNT) && defined(RBQ_AVX512_VNNI)
    return "avx512 vpopcntq coarse, vpdpbusd refine";
#elif defined(RBQ_AVX512)
    return "avx512 popcnt coarse, vpmaddwd refine";
#elif defined(RBQ_AVX2)
    return "popcnt coarse, avx2 vpmaddubsw refine";
#elif defined(RBQ_NEON)
    return hasDotprod() ? "neon cnt coarse, sdot refine" : "neon cnt coarse, smull refine";
#else
    return "scalar";
#endif
}

HOT_SPOT void rabitqCoarseDots(const uint8_t* planes, const uint8_t* bits, int32_t bitBytes, int32_t n, int32_t* out) {
    for (int32_t i = 0; i < n; ++i) {
        out[i] = coarseRow(planes, bits + static_cast<size_t>(i) * bitBytes, bitBytes);
    }
}

HOT_SPOT void rabitqRefineDots(const int8_t* recs, int32_t stride, int32_t m, int32_t dim, const int8_t* q, int32_t* out) {
#if defined(RBQ_NEON)
    if (hasDotprod()) {
        for (int32_t i = 0; i < m; ++i) out[i] = refineRowDotprod(recs + static_cast<size_t>(i) * stride, q, dim);
        return;
    }
    for (int32_t i = 0; i < m; ++i) out[i] = refineRowNeon(recs + static_cast<size_t>(i) * stride, q, dim);
#else
    for (int32_t i = 0; i < m; ++i) out[i] = refineRow(recs + static_cast<size_t>(i) * stride, q, dim);
#endif
}

}  // namespace knn_jni::simd::clusterann
