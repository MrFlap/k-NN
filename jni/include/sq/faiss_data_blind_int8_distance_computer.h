/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef KNNPLUGIN_JNI_FAISS_DATA_BLIND_INT8_DISTANCE_COMPUTER_H
#define KNNPLUGIN_JNI_FAISS_DATA_BLIND_INT8_DISTANCE_COMPUTER_H

#include "faiss/impl/DistanceComputer.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace knn_jni {

    // Graph-build distances for data-blind int8 codes (Java: DataBlindInt8Quantizer). One element is
    //   [code: dimension bytes, each q + 128][scale (f32)][sum of q (i32)][squared norm (f32)][unused 4 bytes]
    // Both sides of every comparison are stored elements, encoded the same way, so
    //   signedDot = sum((qa + 128)(qb + 128)) - 128 * sumA - 16384 * dimension - 128 * sumB
    //   dot       = signedDot * scaleA * scaleB
    // Faiss HNSW minimizes, so L2 returns the squared distance and inner product returns -dot.
    template <bool IsMaxIP>
    struct FaissDataBlindInt8DistanceComputer final : faiss::DistanceComputer {
        const int64_t oneElementByteSize;
        const int32_t dimension;
        const uint8_t* data;
        const uint8_t* query = nullptr;
        float queryScale = 0;
        int32_t querySum = 0;
        float querySqNorm = 0;

        FaissDataBlindInt8DistanceComputer(int64_t _oneElementByteSize, const void* _data, int32_t _dimension)
          : faiss::DistanceComputer(),
            oneElementByteSize(_oneElementByteSize),
            dimension(_dimension),
            data(static_cast<const uint8_t*>(_data)) {
        }

        void corrections(const uint8_t* element, float& scale, int32_t& sum, float& sqNorm) const {
            std::memcpy(&scale, element + dimension, sizeof(float));
            std::memcpy(&sum, element + dimension + 4, sizeof(int32_t));
            std::memcpy(&sqNorm, element + dimension + 8, sizeof(float));
        }

        float distance(const uint8_t* a, float scaleA, int32_t sumA, float sqNormA, const uint8_t* b) const {
            float scaleB, sqNormB;
            int32_t sumB;
            corrections(b, scaleB, sumB, sqNormB);
            uint64_t unsignedDot = 0;
            for (int32_t d = 0; d < dimension; ++d) {
                unsignedDot += static_cast<uint32_t>(a[d]) * static_cast<uint32_t>(b[d]);
            }
            const int64_t signedDot = static_cast<int64_t>(unsignedDot) - 128LL * sumA - 16384LL * dimension - 128LL * sumB;
            const double dot = static_cast<double>(signedDot) * scaleA * scaleB;
            if constexpr (IsMaxIP) {
                return static_cast<float>(-dot);
            } else {
                return static_cast<float>(std::max(0.0, static_cast<double>(sqNormA) + sqNormB - 2.0 * dot));
            }
        }

        void set_query(const float* x) final {
            // During graph build the query is a stored element, not a float vector.
            query = reinterpret_cast<const uint8_t*>(x);
            corrections(query, queryScale, querySum, querySqNorm);
        }

        float operator()(faiss::idx_t i) final {
            return distance(query, queryScale, querySum, querySqNorm, data + i * oneElementByteSize);
        }

        float symmetric_dis(faiss::idx_t i, faiss::idx_t j) final {
            const uint8_t* a = data + i * oneElementByteSize;
            float scaleA, sqNormA;
            int32_t sumA;
            corrections(a, scaleA, sumA, sqNormA);
            return distance(a, scaleA, sumA, sqNormA, data + j * oneElementByteSize);
        }
    };

}

#endif //KNNPLUGIN_JNI_FAISS_DATA_BLIND_INT8_DISTANCE_COMPUTER_H
