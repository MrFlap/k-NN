/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OPENSEARCH_KNN_SIMD_DATA_BLIND_INT8_H
#define OPENSEARCH_KNN_SIMD_DATA_BLIND_INT8_H

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "platform_defs.h"

// Data-blind int8 scoring for memory-optimized search, ported from ClusterANN's clusterann_int8.h. The kernel asks the
// format struct for a vector's code and corrections by position, so the hot loop inlines the accessors.
namespace knn_jni::simd::data_blind_int8 {

// The vectors to score, as records of the Java DataBlindInt8VectorValues layout:
//   [code: dim unsigned bytes, each q + 128][scale (f32)][sum of q (i32)][squared norm (f32)][unused 4 bytes]
// rows[p] points at the p-th record; records may live anywhere (mmap regions or a straddle buffer).
struct DataBlindInt8Format {
    uint8_t* const* rows;
    int32_t dim;

    FORCE_INLINE DataBlindInt8Format(uint8_t* const* rowsIn, const int32_t dimension) : rows(rowsIn), dim(dimension) {}

    FORCE_INLINE const uint8_t* code(const int32_t p) const {
        return rows[p];
    }

    FORCE_INLINE void corrections(const int32_t p, float& docScale, int32_t& docSum, float& docSqNorm) const {
        std::memcpy(&docScale, rows[p] + dim, sizeof(float));
        std::memcpy(&docSum, rows[p] + dim + 4, sizeof(int32_t));
        std::memcpy(&docSqNorm, rows[p] + dim + 8, sizeof(float));
    }
};

// Mirrors the Java side: 0 EUCLIDEAN, 1 MAXIMUM_INNER_PRODUCT, 2 COSINE.
enum Int8Similarity : int32_t { INT8_EUCLIDEAN = 0, INT8_MAXIMUM_INNER_PRODUCT = 1, INT8_COSINE = 2 };

// Scores positions[0..n) of `format` against the query: scores[i] receives the similarity of the vector at
// positions[i]; the maximum written is returned (-infinity when n <= 0). Reproduces Java's DataBlindInt8Quantizer.score
// bit for bit: an exact integer dot, then the same double-precision epilogue.
//
//   qSigned : the query code with the +128 offset removed, values in [-127, 127], dim bytes
//   qSum    : sum of qSigned
//   qScale  : the query's maxAbs / 127
//   qSqNorm : the query's squared norm
float bulkScoreDataBlindInt8(const DataBlindInt8Format& format,
                             const int32_t* positions, int32_t n,
                             const int8_t* qSigned, int32_t qSum, float qScale, float qSqNorm,
                             int32_t sim, float* scores);

// Which variant this library was compiled with, for logs and tests.
const char* int8BulkKernelName();

}  // namespace knn_jni::simd::data_blind_int8

#endif  // OPENSEARCH_KNN_SIMD_DATA_BLIND_INT8_H
