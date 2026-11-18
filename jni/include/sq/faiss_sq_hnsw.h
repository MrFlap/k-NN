/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef KNNPLUGIN_JNI_FAISS_SQ_HNSW_H
#define KNNPLUGIN_JNI_FAISS_SQ_HNSW_H

#include "faiss/IndexBinaryHNSW.h"
#include "faiss_sq_flat.h"
#include "faiss_hashprune.h"
#include "faiss/impl/FaissAssert.h"

namespace knn_jni {

    // Block distances over Lucene scalar-quantized codes for the HashPrune builder. Codes are gathered into contiguous
    // buffers, correction factors are decoded once per vector, and the integer code dot products run on the
    // ISA-selected SQ kernels in their 1x4 form (one row against four targets per pass). Distances equal
    // FaissSQFlat's symmetric_dis.
    class FaissSQBlockDistance final : public faiss_wrapper::hashprune::BlockDistance {
    public:
        explicit FaissSQBlockDistance(const FaissSQFlat& flat)
          : flat(flat),
            quantizedBytes(flat.quantizedVectorBytes),
            stride((flat.quantizedVectorBytes + 7) / 8 * 8),
            kernelBytes(flat.docBits == 4 ? flat.quantizedVectorBytes : flat.quantizedVectorBytes / flat.docBits),
            intervalScale(1.0f / static_cast<float>((1 << flat.docBits) - 1)),
            innerProduct(flat.metric_type == faiss::METRIC_INNER_PRODUCT) {
            if (!resolveSQMultiBitKernel(flat.docBits, SQMultiBitImpl::Auto, kernel)) {
                throw std::runtime_error("FaissSQBlockDistance: no kernel for docBits=" + std::to_string(flat.docBits));
            }
        }

        void setTargets(const faiss::HNSW::storage_idx_t* ids, size_t n) final {
            gather(ids, n, targetCodes, targetFactors);
            numTargets = n;
        }

        void distances(const faiss::HNSW::storage_idx_t* ids, size_t n, float* out) final {
            gather(ids, n, rowCodes, rowFactors);
            if (flat.docBits == 1) {
                oneBitDistances(n, out);
                return;
            }
            uint32_t dp[4];
            for (size_t i = 0; i < n; i++) {
                const uint8_t* row = &rowCodes[i * stride];
                float* rowOut = out + i * numTargets;
                size_t j = 0;
                for (; j + 4 <= numTargets; j += 4) {
                    kernel.dp4(row, target(j), target(j + 1), target(j + 2), target(j + 3), kernelBytes, dp);
                    for (int t = 0; t < 4; t++) rowOut[j + t] = score(rowFactors[i], targetFactors[j + t], dp[t]);
                }
                for (; j < numTargets; j++) {
                    rowOut[j] = score(rowFactors[i], targetFactors[j], kernel.dp(row, target(j), kernelBytes));
                }
            }
        }

    private:
        struct Factors {
            float lower, length, additional, sum;
        };

        const uint8_t* target(size_t j) const { return &targetCodes[j * stride]; }

        // 1-bit: inlined 4x4 register tile of AND+popcount, each loaded word feeds four dot products.
        void oneBitDistances(size_t n, float* out) const {
            const size_t words = quantizedBytes / 8;  // 1-bit codes are whole 64-bit words
            const auto* rows = reinterpret_cast<const uint64_t*>(rowCodes.data());
            const auto* targets = reinterpret_cast<const uint64_t*>(targetCodes.data());
            const size_t wstride = stride / 8;
            size_t i = 0;
            for (; i + 4 <= n; i += 4) {
                size_t j = 0;
                for (; j + 4 <= numTargets; j += 4) {
                    uint32_t acc[4][4];
                    oneBitTile(rows + i * wstride, targets + j * wstride, wstride, words, acc);
                    for (int r = 0; r < 4; r++)
                        for (int t = 0; t < 4; t++)
                            out[(i + r) * numTargets + j + t] = score(rowFactors[i + r], targetFactors[j + t], acc[r][t]);
                }
                for (; j < numTargets; j++)
                    for (int r = 0; r < 4; r++) out[(i + r) * numTargets + j] = oneBitScore(rows, i + r, targets, j, words, wstride);
            }
            for (; i < n; i++)
                for (size_t j = 0; j < numTargets; j++) out[i * numTargets + j] = oneBitScore(rows, i, targets, j, words, wstride);
        }

        // acc[r][t] = popcount(row r & target t) for a 4x4 tile of 1-bit codes.
        static void oneBitTile(const uint64_t* rows, const uint64_t* targets, size_t wstride, size_t words,
                               uint32_t acc[4][4]) {
            size_t w = 0;
#if defined(__aarch64__)
            // 16-byte blocks: vcnt per byte, pairwise-accumulated into u16 lanes (<= 16 per lane per block, so
            // codes up to ~65k bits cannot overflow).
            uint16x8_t sums[4][4];
            for (int r = 0; r < 4; r++)
                for (int t = 0; t < 4; t++) sums[r][t] = vdupq_n_u16(0);
            for (; w + 2 <= words; w += 2) {
                uint8x16_t a[4], b[4];
                for (int r = 0; r < 4; r++) a[r] = vld1q_u8(reinterpret_cast<const uint8_t*>(rows + r * wstride + w));
                for (int t = 0; t < 4; t++) b[t] = vld1q_u8(reinterpret_cast<const uint8_t*>(targets + t * wstride + w));
                for (int r = 0; r < 4; r++)
                    for (int t = 0; t < 4; t++) sums[r][t] = vpadalq_u8(sums[r][t], vcntq_u8(vandq_u8(a[r], b[t])));
            }
            for (int r = 0; r < 4; r++)
                for (int t = 0; t < 4; t++) acc[r][t] = vaddlvq_u16(sums[r][t]);
#else
            for (int r = 0; r < 4; r++)
                for (int t = 0; t < 4; t++) acc[r][t] = 0;
#endif
            for (; w < words; w++)
                for (int r = 0; r < 4; r++)
                    for (int t = 0; t < 4; t++) acc[r][t] += __builtin_popcountll(rows[r * wstride + w] & targets[t * wstride + w]);
        }

        float oneBitScore(const uint64_t* rows, size_t i, const uint64_t* targets, size_t j, size_t words, size_t wstride) const {
            uint32_t dp = 0;
            for (size_t w = 0; w < words; w++) dp += __builtin_popcountll(rows[i * wstride + w] & targets[j * wstride + w]);
            return score(rowFactors[i], targetFactors[j], dp);
        }

        void gather(const faiss::HNSW::storage_idx_t* ids, size_t n,
                    std::vector<uint8_t, NBytesAlignedAllocator<uint8_t, 8>>& codes, std::vector<Factors>& factors) const {
            codes.assign(n * stride, 0);
            factors.resize(n);
            for (size_t i = 0; i < n; i++) {
                const uint8_t* element = flat.quantizedVectorsAndCorrectionFactors.data() + ids[i] * flat.oneElementSize;
                std::memcpy(&codes[i * stride], element, quantizedBytes);
                Factors& f = factors[i];
                readCorrectionFactorsSafe(element + quantizedBytes, f.lower, f.length, f.additional, f.sum);
                f.length *= intervalScale;  // no-op for 1-bit
            }
        }

        // Same formula as FaissSQDistanceComputer::symmetric_dis.
        float score(const Factors& x, const Factors& z, uint32_t dp) const {
            const float s = x.lower * z.lower * flat.dimension + z.lower * x.length * x.sum + x.lower * z.length * z.sum
                          + x.length * z.length * static_cast<float>(dp);
            return innerProduct ? -(s + x.additional + z.additional - flat.centroidDp) : x.additional + z.additional - 2 * s;
        }

        const FaissSQFlat& flat;
        const size_t quantizedBytes;
        const size_t stride;
        const uint64_t kernelBytes;
        const float intervalScale;
        const bool innerProduct;
        SQMultiBitKernel kernel;
        std::vector<uint8_t, NBytesAlignedAllocator<uint8_t, 8>> targetCodes, rowCodes;
        std::vector<Factors> targetFactors, rowFactors;
        size_t numTargets = 0;
    };

    struct FaissSQHnsw final : faiss::IndexBinaryHNSW {
        FaissSQFlat* faiss_sq_flat;
        // Graph build knobs when deferGraphBuild is set; 1-bit codes get their own tuned preset.
        faiss_wrapper::hashprune::BuildParams buildParams;

        FaissSQHnsw(int32_t _m, FaissSQFlat* _faiss_sq_flat)
          : faiss::IndexBinaryHNSW(_faiss_sq_flat, _m, true),
            faiss_sq_flat(_faiss_sq_flat),
            buildParams(_faiss_sq_flat->docBits == 1 ? faiss_wrapper::hashprune::BuildParams::OneBit()
                                                     : faiss_wrapper::hashprune::BuildParams()) {
            // This has the ownership of FaissSQFlat, setting this true to make it free'd when this class is freed
            own_fields = true;
            // Set metric type that was given, not just blindly use default space type.
            metric_type = _faiss_sq_flat->metric_type;
            // Since HNSW is setting query pointer as `base + query_index * index_hnsw.code_size`
            // we should update code_size in here.
            code_size = _faiss_sq_flat->code_size;
        }

        faiss::DistanceComputer* get_distance_computer() const final {
            return faiss_sq_flat->get_distance_computer();
        }

        // When set (build_algorithm=hashprune), add() only registers the vectors and the graph is built at once by
        // buildDeferredGraph() before the index is written.
        bool deferGraphBuild = false;

        void add(faiss::idx_t n, const uint8_t* x) final {
            if (!deferGraphBuild) {
                faiss::IndexBinaryHNSW::add(n, x);
                return;
            }
            FAISS_THROW_IF_NOT_MSG(hnsw.levels.empty(), "FaissSQHnsw: cannot add vectors after the graph is built");
            storage->add(n, x);
            ntotal = storage->ntotal;
        }

        void buildDeferredGraph() {
            if (!deferGraphBuild || !hnsw.levels.empty()) {
                return;
            }
            faiss_wrapper::hashprune::BuildGraph(
                hnsw, ntotal,
                [this] { return get_distance_computer(); },
                [this] { return new FaissSQBlockDistance(*faiss_sq_flat); }, buildParams);
        }
    };

}

#endif // KNNPLUGIN_JNI_FAISS_SQ_HNSW_H
