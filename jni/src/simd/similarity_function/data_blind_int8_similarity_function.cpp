/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

// Bulk data-blind int8 scoring, ported from ClusterANN's int8_bulk_kernel.cpp. See data_blind_int8.h for the layout.
// Also binds the kernel to memory-optimized search as a SimilarityFunction (end of file).
//
// Dot product. The format's Java (DataBlindInt8Quantizer.score) computes
//   unsignedDot = Σ (q + 128)(c + 128)        and then
//   signedDot   = unsignedDot - 128*qSum - 16384*dim - 128*dSum.
// Here, per ISA:
//   AVX-512 VNNI  vpdpbusd(u8 codes, s8 query) = Σ (c + 128) q = signedDot + 128*qSum       -> one correction
//   AVX2          widen u8 / s8 to i16, vpmaddwd                                             -> same
//   NEON dotprod  udot(u8 codes, u8 query) with query = qSigned ^ 0x80 (= q + 128)            -> Java's 3 terms
//   scalar        Java's loop
// Every path yields the exact integer signedDot, so the epilogue is shared and bit-exact against Java.
//
// Epilogue reproduces DataBlindInt8Quantizer.score in double precision with the same operation order. This translation
// unit is compiled with -ffp-contract=off (create_simd_library), so no multiply-add is fused; Java never fuses.
//
// 8 rows in flight, one query load shared by all 8, L1 prefetch of the next 8 rows' slice while the current 8 are
// being accumulated.
//
// ISA selection. On x86 it comes from the compiler's feature macros, i.e. from the flags this TU is built with: the
// per-variant simd libraries are selected at load time by PlatformUtils, so the flags are the dispatch. On aarch64
// there is a single simd library and no load-time CPU check, so this TU is compiled at the baseline ISA and the
// dotprod (udot) path is compiled with a per-function target attribute and chosen at RUNTIME from HWCAP_ASIMDDP;
// hosts without dotprod take the scalar path instead of faulting with SIGILL.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "simd/similarity_function/data_blind_int8.h"
#include "simd/similarity_function/similarity_function.h"

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
#include <immintrin.h>
#define INT8_KERNEL_AVX512VNNI 1
#define INT8_KERNEL_NAME "avx512-vnni u8*s8 (vpdpbusd): 8 rows/iter, 64 dims/step, next-rows L1 prefetch"
#elif defined(__AVX2__)
#include <immintrin.h>
#define INT8_KERNEL_AVX2 1
#define INT8_KERNEL_NAME "avx2 widen+vpmaddwd: 8 rows/iter, 16 dims/step, next-rows L1 prefetch"
#elif defined(__aarch64__)
#include <arm_neon.h>
#if defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif
#include <cstdlib>
#define INT8_KERNEL_AARCH64_DISPATCH 1
#define INT8_KERNEL_NAME_NEON "neon udot u8*u8 (runtime dispatch): 8 rows/iter, 16 dims/step, next-rows L1 prefetch"
#define INT8_KERNEL_NAME_SCALAR "scalar u8*u8 (aarch64 without dotprod)"
// Functions using udot are compiled for armv8.2-a+dotprod individually; the rest of the TU stays at the baseline.
#define INT8_DOTPROD_TARGET __attribute__((target("arch=armv8.2-a+dotprod")))
#else
#define INT8_KERNEL_NAME "scalar u8*u8"
#endif

#define INT8_PREFETCH_L1(p) __builtin_prefetch((const void*)(p), 0, 3)

namespace knn_jni::simd::data_blind_int8 {
namespace {

constexpr int ROWS = 8;

// ------------------------------------------------------------------ 8-row and 1-row integer dot products
// dot8: out[k] = raw accumulator for rows[k]; toSigned() turns it into the exact signed dot for that ISA.
// `next` may be null when there is no next batch to prefetch.

#if defined(INT8_KERNEL_AVX512VNNI)

FORCE_INLINE void dot8(const uint8_t* const rows[ROWS], const uint8_t* const next[ROWS],
                       const int8_t* q, int dim, int64_t out[ROWS]) {
    __m512i acc[ROWS];
    for (int k = 0; k < ROWS; ++k) acc[k] = _mm512_setzero_si512();
    int d = 0;
    for (; d + 64 <= dim; d += 64) {
        if (next != nullptr) for (int k = 0; k < ROWS; ++k) INT8_PREFETCH_L1(next[k] + d);
        const __m512i qv = _mm512_loadu_si512((const void*)(q + d));
        for (int k = 0; k < ROWS; ++k) {
            acc[k] = _mm512_dpbusd_epi32(acc[k], _mm512_loadu_si512((const void*)(rows[k] + d)), qv);
        }
    }
    if (d < dim) {
        // dim - d is in [1, 63] here, so the shift never reaches 64.
        const __mmask64 m = (static_cast<uint64_t>(1) << (dim - d)) - 1u;
        const __m512i qv = _mm512_maskz_loadu_epi8(m, q + d);
        for (int k = 0; k < ROWS; ++k) {
            acc[k] = _mm512_dpbusd_epi32(acc[k], _mm512_maskz_loadu_epi8(m, rows[k] + d), qv);
        }
    }
    for (int k = 0; k < ROWS; ++k) out[k] = _mm512_reduce_add_epi32(acc[k]);
}

FORCE_INLINE int64_t dot1(const uint8_t* row, const int8_t* q, int dim) {
    __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
    int d = 0;
    for (; d + 128 <= dim; d += 128) {
        a0 = _mm512_dpbusd_epi32(a0, _mm512_loadu_si512((const void*)(row + d)), _mm512_loadu_si512((const void*)(q + d)));
        a1 = _mm512_dpbusd_epi32(a1, _mm512_loadu_si512((const void*)(row + d + 64)), _mm512_loadu_si512((const void*)(q + d + 64)));
    }
    for (; d + 64 <= dim; d += 64) {
        a0 = _mm512_dpbusd_epi32(a0, _mm512_loadu_si512((const void*)(row + d)), _mm512_loadu_si512((const void*)(q + d)));
    }
    if (d < dim) {
        const __mmask64 m = (static_cast<uint64_t>(1) << (dim - d)) - 1u;
        a1 = _mm512_dpbusd_epi32(a1, _mm512_maskz_loadu_epi8(m, row + d), _mm512_maskz_loadu_epi8(m, q + d));
    }
    return _mm512_reduce_add_epi32(_mm512_add_epi32(a0, a1));
}

// acc = Σ (c + 128) * q  ->  signedDot = acc - 128 * qSum
FORCE_INLINE int64_t toSigned(int64_t acc, int32_t qSum, int32_t /*dSum*/, int /*dim*/) {
    return acc - 128LL * qSum;
}

#elif defined(INT8_KERNEL_AVX2)

FORCE_INLINE int32_t hsum256_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v), hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    return _mm_cvtsi128_si32(lo);
}

// i16 products (|c+128| <= 255, |q| <= 127, so <= 32385) summed pairwise into i32 by vpmaddwd: exact.
FORCE_INLINE void dot8(const uint8_t* const rows[ROWS], const uint8_t* const next[ROWS],
                       const int8_t* q, int dim, int64_t out[ROWS]) {
    __m256i acc[ROWS];
    for (int k = 0; k < ROWS; ++k) acc[k] = _mm256_setzero_si256();
    int d = 0;
    for (; d + 16 <= dim; d += 16) {
        if (next != nullptr && (d & 63) == 0) for (int k = 0; k < ROWS; ++k) INT8_PREFETCH_L1(next[k] + d);
        const __m256i qv = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(q + d)));
        for (int k = 0; k < ROWS; ++k) {
            const __m256i cv = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(rows[k] + d)));
            acc[k] = _mm256_add_epi32(acc[k], _mm256_madd_epi16(cv, qv));
        }
    }
    for (int k = 0; k < ROWS; ++k) {
        int64_t s = hsum256_epi32(acc[k]);
        for (int t = d; t < dim; ++t) s += static_cast<int64_t>(rows[k][t]) * q[t];
        out[k] = s;
    }
}

FORCE_INLINE int64_t dot1(const uint8_t* row, const int8_t* q, int dim) {
    __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256();
    int d = 0;
    for (; d + 32 <= dim; d += 32) {
        const __m256i q0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(q + d)));
        const __m256i q1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(q + d + 16)));
        a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(row + d))), q0));
        a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(row + d + 16))), q1));
    }
    int64_t s = hsum256_epi32(_mm256_add_epi32(a0, a1));
    for (; d < dim; ++d) s += static_cast<int64_t>(row[d]) * q[d];
    return s;
}

FORCE_INLINE int64_t toSigned(int64_t acc, int32_t qSum, int32_t /*dSum*/, int /*dim*/) {
    return acc - 128LL * qSum;
}

#elif defined(INT8_KERNEL_AARCH64_DISPATCH)

// ---- dotprod path. NOT always_inline: these carry a target attribute the caller lacks, and GCC refuses to
// always_inline across a target mismatch. The hot loop is inside dot8 itself, so the call per 8 rows is irrelevant.
namespace neon_dotprod {

// udot is u8 x u8; feed query + 128 (= qSigned ^ 0x80) and apply Java's three-term correction.
INT8_DOTPROD_TARGET static void dot8(const uint8_t* const rows[ROWS], const uint8_t* const next[ROWS],
                                     const int8_t* q, int dim, int64_t out[ROWS]) {
    uint32x4_t acc[ROWS];
    for (int k = 0; k < ROWS; ++k) acc[k] = vdupq_n_u32(0);
    const uint8x16_t flip = vdupq_n_u8(0x80);
    int d = 0;
    for (; d + 16 <= dim; d += 16) {
        if (next != nullptr && (d & 63) == 0) for (int k = 0; k < ROWS; ++k) INT8_PREFETCH_L1(next[k] + d);
        const uint8x16_t qv = veorq_u8(vld1q_u8(reinterpret_cast<const uint8_t*>(q + d)), flip);
        for (int k = 0; k < ROWS; ++k) {
            acc[k] = vdotq_u32(acc[k], vld1q_u8(rows[k] + d), qv);
        }
    }
    for (int k = 0; k < ROWS; ++k) {
        int64_t s = vaddvq_u32(acc[k]);
        for (int t = d; t < dim; ++t) s += static_cast<int64_t>(rows[k][t]) * static_cast<uint8_t>(static_cast<uint8_t>(q[t]) ^ 0x80);
        out[k] = s;
    }
}

INT8_DOTPROD_TARGET static int64_t dot1(const uint8_t* row, const int8_t* q, int dim) {
    uint32x4_t a0 = vdupq_n_u32(0), a1 = vdupq_n_u32(0);
    const uint8x16_t flip = vdupq_n_u8(0x80);
    int d = 0;
    for (; d + 32 <= dim; d += 32) {
        a0 = vdotq_u32(a0, vld1q_u8(row + d), veorq_u8(vld1q_u8(reinterpret_cast<const uint8_t*>(q + d)), flip));
        a1 = vdotq_u32(a1, vld1q_u8(row + d + 16), veorq_u8(vld1q_u8(reinterpret_cast<const uint8_t*>(q + d + 16)), flip));
    }
    int64_t s = vaddvq_u32(vaddq_u32(a0, a1));
    for (; d < dim; ++d) s += static_cast<int64_t>(row[d]) * static_cast<uint8_t>(static_cast<uint8_t>(q[d]) ^ 0x80);
    return s;
}

}  // namespace neon_dotprod

// ---- scalar path: Java's loop, for aarch64 hosts without dotprod.
namespace scalar_fallback {

static void dot8(const uint8_t* const rows[ROWS], const uint8_t* const /*next*/[ROWS],
                 const int8_t* q, int dim, int64_t out[ROWS]) {
    for (int k = 0; k < ROWS; ++k) {
        int64_t s = 0;
        for (int d = 0; d < dim; ++d) s += static_cast<int64_t>(rows[k][d]) * static_cast<uint8_t>(static_cast<uint8_t>(q[d]) ^ 0x80);
        out[k] = s;
    }
}

static int64_t dot1(const uint8_t* row, const int8_t* q, int dim) {
    int64_t s = 0;
    for (int d = 0; d < dim; ++d) s += static_cast<int64_t>(row[d]) * static_cast<uint8_t>(static_cast<uint8_t>(q[d]) ^ 0x80);
    return s;
}

}  // namespace scalar_fallback

// Both aarch64 paths accumulate Σ (c + 128)(q + 128), so one correction serves both:
// Java: signedDot = unsignedDot - 128*qSum - 16384*dim - 128*dSum
FORCE_INLINE int64_t toSigned(int64_t acc, int32_t qSum, int32_t dSum, int dim) {
    return acc - 128LL * qSum - 16384LL * dim - 128LL * dSum;
}

// Decided once per process. OPENSEARCH_KNN_INT8_FORCE_SCALAR=1 forces the scalar path (testing the fallback on a
// dotprod host).
static bool detectDotprod() {
    const char* force = std::getenv("OPENSEARCH_KNN_INT8_FORCE_SCALAR");
    if (force != nullptr && force[0] == '1') return false;
#if defined(__APPLE__)
    // Every Apple Silicon core implements dotprod; macOS has no getauxval.
    return true;
#else
    return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#endif
}

static bool hasDotprod() {
    static const bool value = detectDotprod();
    return value;
}

struct NeonDotprodKernel {
    static void rows8(const uint8_t* const rows[ROWS], const uint8_t* const next[ROWS], const int8_t* q, int dim, int64_t out[ROWS]) {
        neon_dotprod::dot8(rows, next, q, dim, out);
    }
    static int64_t row1(const uint8_t* row, const int8_t* q, int dim) { return neon_dotprod::dot1(row, q, dim); }
};

struct ScalarKernel {
    static void rows8(const uint8_t* const rows[ROWS], const uint8_t* const next[ROWS], const int8_t* q, int dim, int64_t out[ROWS]) {
        scalar_fallback::dot8(rows, next, q, dim, out);
    }
    static int64_t row1(const uint8_t* row, const int8_t* q, int dim) { return scalar_fallback::dot1(row, q, dim); }
};

#else  // scalar: Java's loop

FORCE_INLINE void dot8(const uint8_t* const rows[ROWS], const uint8_t* const /*next*/[ROWS],
                       const int8_t* q, int dim, int64_t out[ROWS]) {
    for (int k = 0; k < ROWS; ++k) {
        int64_t s = 0;
        for (int d = 0; d < dim; ++d) s += static_cast<int64_t>(rows[k][d]) * static_cast<uint8_t>(static_cast<uint8_t>(q[d]) ^ 0x80);
        out[k] = s;
    }
}

FORCE_INLINE int64_t dot1(const uint8_t* row, const int8_t* q, int dim) {
    int64_t s = 0;
    for (int d = 0; d < dim; ++d) s += static_cast<int64_t>(row[d]) * static_cast<uint8_t>(static_cast<uint8_t>(q[d]) ^ 0x80);
    return s;
}

FORCE_INLINE int64_t toSigned(int64_t acc, int32_t qSum, int32_t dSum, int dim) {
    return acc - 128LL * qSum - 16384LL * dim - 128LL * dSum;
}

#endif

#if !defined(INT8_KERNEL_AARCH64_DISPATCH)
// x86 and generic: the ISA was fixed at compile time; one policy forwarding to the free functions above.
struct CompileTimeKernel {
    static FORCE_INLINE void rows8(const uint8_t* const rows[ROWS], const uint8_t* const next[ROWS], const int8_t* q, int dim, int64_t out[ROWS]) {
        dot8(rows, next, q, dim, out);
    }
    static FORCE_INLINE int64_t row1(const uint8_t* row, const int8_t* q, int dim) { return dot1(row, q, dim); }
};
#endif

// ------------------------------------------------------------------ epilogue: verbatim DataBlindInt8Quantizer.score
// Java:
//   final double dot = (double) signedDot * query.scale() * docScale;   // left to right, floats widened to double
//   EUCLIDEAN:              double s = query.sqNorm() + docSqNorm - 2.0 * dot;   // float + float first, then double
//                           if (s < 0) s = 0; return (float) (1.0 / (1.0 + s));
//   MAXIMUM_INNER_PRODUCT:  return dot >= 0 ? (float) (dot + 1.0) : (float) (1.0 / (1.0 - dot));
//   COSINE:                 return (float) Math.max(0.0, (1.0 + dot) / 2.0);
FORCE_INLINE float finish(int64_t signedDot, float qScale, float qSqNorm, float dScale, float dSqNorm, int32_t sim) {
    const double dot = static_cast<double>(signedDot) * static_cast<double>(qScale) * static_cast<double>(dScale);
    switch (sim) {
        case INT8_EUCLIDEAN: {
            const float normSum = qSqNorm + dSqNorm;   // Java adds the two floats in float precision
            double s = static_cast<double>(normSum) - 2.0 * dot;
            if (s < 0) s = 0;
            return static_cast<float>(1.0 / (1.0 + s));
        }
        case INT8_MAXIMUM_INNER_PRODUCT:
            return dot >= 0 ? static_cast<float>(dot + 1.0) : static_cast<float>(1.0 / (1.0 - dot));
        default:
            return static_cast<float>(std::max(0.0, (1.0 + dot) / 2.0));
    }
}

template <class Kernel>
FORCE_INLINE float scoreAll(const DataBlindInt8Format& format,
                            const int32_t* positions, int32_t n,
                            const int8_t* qSigned, int32_t qSum, float qScale, float qSqNorm,
                            int32_t sim, float* scores) {
    const int dim = format.dim;
    const uint8_t* rows[ROWS];
    const uint8_t* next[ROWS];
    int64_t dots[ROWS];
    float maxScore = -std::numeric_limits<float>::infinity();
    int32_t i = 0;
    for (; i + ROWS <= n; i += ROWS) {
        for (int k = 0; k < ROWS; ++k) rows[k] = format.code(positions[i + k]);
        const bool hasNext = (i + 2 * ROWS <= n);
        if (hasNext) for (int k = 0; k < ROWS; ++k) next[k] = format.code(positions[i + ROWS + k]);
        Kernel::rows8(rows, hasNext ? next : nullptr, qSigned, dim, dots);
        for (int k = 0; k < ROWS; ++k) {
            float dScale, dNorm;
            int32_t dSum;
            format.corrections(positions[i + k], dScale, dSum, dNorm);
            const float s = finish(toSigned(dots[k], qSum, dSum, dim), qScale, qSqNorm, dScale, dNorm, sim);
            scores[i + k] = s;
            if (s > maxScore) maxScore = s;
        }
    }
    for (; i < n; ++i) {
        float dScale, dNorm;
        int32_t dSum;
        format.corrections(positions[i], dScale, dSum, dNorm);
        const float s = finish(toSigned(Kernel::row1(format.code(positions[i]), qSigned, dim), qSum, dSum, dim),
                               qScale, qSqNorm, dScale, dNorm, sim);
        scores[i] = s;
        if (s > maxScore) maxScore = s;
    }
    return maxScore;
}

}  // namespace

const char* int8BulkKernelName() {
#if defined(INT8_KERNEL_AARCH64_DISPATCH)
    return hasDotprod() ? INT8_KERNEL_NAME_NEON : INT8_KERNEL_NAME_SCALAR;
#else
    return INT8_KERNEL_NAME;
#endif
}

HOT_SPOT float bulkScoreDataBlindInt8(const DataBlindInt8Format& format,
                                       const int32_t* positions, int32_t n,
                                       const int8_t* qSigned, int32_t qSum, float qScale, float qSqNorm,
                                       int32_t sim, float* scores) {
    if (n <= 0) {
        return -std::numeric_limits<float>::infinity();
    }
#if defined(INT8_KERNEL_AARCH64_DISPATCH)
    if (hasDotprod()) {
        return scoreAll<NeonDotprodKernel>(format, positions, n, qSigned, qSum, qScale, qSqNorm, sim, scores);
    }
    return scoreAll<ScalarKernel>(format, positions, n, qSigned, qSum, qScale, qSqNorm, sim, scores);
#else
    return scoreAll<CompileTimeKernel>(format, positions, n, qSigned, qSum, qScale, qSqNorm, sim, scores);
#endif
}

}  // namespace knn_jni::simd::data_blind_int8


// ------------------------------------------------------------------ memory-optimized search binding
// The search context holds the query code (signed, dim bytes) and, at the front of tmpBuffer, the query's scale, sum
// and squared norm; records are looked up by internal vector id, across mmap regions.
namespace knn_jni::simd::similarity_function {
namespace {

using knn_jni::simd::data_blind_int8::DataBlindInt8Format;

struct DataBlindInt8SimilarityFunction final : SimilarityFunction {
    const int32_t sim;

    explicit DataBlindInt8SimilarityFunction(const int32_t similarity) : sim(similarity) {}

    void calculateSimilarityInBulk(SimdVectorSearchContext* srchContext, int32_t* internalVectorIds, float* scores,
                                   int32_t numVectors) final {
        static thread_local std::vector<uint8_t*> rows;
        static thread_local std::vector<int32_t> positions;
        rows.resize(numVectors);
        srchContext->getVectorPointersInBulk(rows.data(), internalVectorIds, numVectors);
        for (int32_t i = static_cast<int32_t>(positions.size()); i < numVectors; ++i) {
            positions.push_back(i);
        }
        score(srchContext, rows.data(), positions.data(), numVectors, scores);
    }

    float calculateSimilarity(SimdVectorSearchContext* srchContext, int32_t internalVectorId) final {
        uint8_t* row = srchContext->getVectorPointer(internalVectorId);
        const int32_t position = 0;
        float result;
        score(srchContext, &row, &position, 1, &result);
        return result;
    }

    void score(SimdVectorSearchContext* srchContext, uint8_t* const* rows, const int32_t* positions, int32_t n,
               float* scores) const {
        float qScale, qSqNorm;
        int32_t qSum;
        const uint8_t* terms = srchContext->tmpBuffer.data();
        std::memcpy(&qScale, terms, sizeof(float));
        std::memcpy(&qSum, terms + 4, sizeof(int32_t));
        std::memcpy(&qSqNorm, terms + 8, sizeof(float));
        knn_jni::simd::data_blind_int8::bulkScoreDataBlindInt8(
            DataBlindInt8Format(rows, srchContext->dimension), positions, n,
            static_cast<const int8_t*>(srchContext->queryVectorSimdAligned), qSum, qScale, qSqNorm, sim, scores);
    }
};

}  // namespace

SimilarityFunction* dataBlindInt8SimilarityFunction(const NativeSimilarityFunctionType type) {
    static DataBlindInt8SimilarityFunction l2 {knn_jni::simd::data_blind_int8::INT8_EUCLIDEAN};
    static DataBlindInt8SimilarityFunction ip {knn_jni::simd::data_blind_int8::INT8_MAXIMUM_INNER_PRODUCT};
    static DataBlindInt8SimilarityFunction cosine {knn_jni::simd::data_blind_int8::INT8_COSINE};
    switch (type) {
        case NativeSimilarityFunctionType::DB_INT8_L2:
            return &l2;
        case NativeSimilarityFunctionType::DB_INT8_IP:
            return &ip;
        case NativeSimilarityFunctionType::DB_INT8_COSINE:
            return &cosine;
        default:
            throw std::runtime_error("Not a data-blind int8 similarity function type: " + std::to_string(static_cast<int32_t>(type)));
    }
}

}  // namespace knn_jni::simd::similarity_function
