/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OPENSEARCH_KNN_SIMD_CLUSTERANN_RABITQ_H
#define OPENSEARCH_KNN_SIMD_CLUSTERANN_RABITQ_H

#include <cstdint>

#include "platform_defs.h"

// Bulk inner products of a ClusterANN tiered RaBitQ scan; see src/simd/clusterann/rabitq_tiered_kernel.cpp.
namespace knn_jni::simd::clusterann {

// out[i] = sum_{j<4} 2^j * popcount(planes[j] & bits[i]); planes holds 4 * bitBytes bytes, bits n * bitBytes.
void rabitqCoarseDots(const uint8_t* planes, const uint8_t* bits, int32_t bitBytes, int32_t n, int32_t* out);

// out[i] = sum_d recs[i * stride + d] * q[d]; codes in [0, 127], q in [-127, 127].
void rabitqRefineDots(const int8_t* recs, int32_t stride, int32_t m, int32_t dim, const int8_t* q, int32_t* out);

const char* rabitqTieredKernelName();

}  // namespace knn_jni::simd::clusterann

#endif  // OPENSEARCH_KNN_SIMD_CLUSTERANN_RABITQ_H
