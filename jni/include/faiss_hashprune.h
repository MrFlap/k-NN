// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

#ifndef OPENSEARCH_KNN_FAISS_HASHPRUNE_H
#define OPENSEARCH_KNN_FAISS_HASHPRUNE_H

#include "faiss/IndexHNSW.h"
#include "faiss/impl/DistanceComputer.h"
#include "faiss/impl/HNSW.h"

#include <functional>

namespace knn_jni {
namespace faiss_wrapper {
namespace hashprune {

/// Creates a distance computer over the stored vectors (lower is closer). One is created per thread.
using DistanceComputerFactory = std::function<faiss::DistanceComputer*()>;

/**
 * Distances between blocks of stored vectors (lower is closer), used for the dense parts of the build: points vs
 * partition leaders and all pairs inside a leaf. Implementations gather the vectors contiguously and use matrix or
 * batched kernels. One instance is created per thread.
 */
struct BlockDistance {
    virtual ~BlockDistance() = default;
    /// Fixes the columns of subsequent distances() calls.
    virtual void setTargets(const faiss::HNSW::storage_idx_t* ids, size_t n) = 0;
    /// out[i * numTargets + j] = distance(ids[i], target j).
    virtual void distances(const faiss::HNSW::storage_idx_t* ids, size_t n, float* out) = 0;
};
using BlockDistanceFactory = std::function<BlockDistance*()>;

/**
 * Tuning knobs. Defaults were tuned with Optuna on cohere-1M (768-d, M=16) for build time subject to recall at least
 * HNSW efConstruction=100 at the same search work; they hold for float and multi-bit storage. OneBit() is the
 * separately tuned preset for 1-bit scalar-quantized codes.
 */
struct BuildParams {
    int hashBits = 14;                   // sectors per node = 2^hashBits (max 32)
    int reservoirSize = 0;               // candidates kept per node before the final prune; 0 derives it from
    float reservoirDegreeRatio = 1.5f;   //   this ratio times the level-0 degree (2M), capped at 255
    int topFanout = 9;                   // leaders each point joins at the first partitioning level
    int subFanout = 2;                   // ... and at the second level; deeper levels use 1
    int leafNeighbors = 8;               // k of the bi-directed kNN graph built inside each leaf
    size_t leafTarget = 512;             // desired leaf size, sets the number of leaders
    size_t leafMax = 1024;               // leaves larger than this are partitioned again
    size_t maxLeaders = 250;             // leaders per partitioning step
    size_t rowTile = 256;                // rows per BlockDistance::distances call

    /// 1-bit codes: few large leaves and a reservoir equal to the degree, so the noisy-distance final prune is skipped.
    static BuildParams OneBit() {
        BuildParams p;
        p.topFanout = 2;
        p.subFanout = 3;
        p.leafNeighbors = 4;
        p.leafTarget = 2048;
        p.leafMax = 4096;
        p.reservoirDegreeRatio = 1.0f;
        return p;
    }
};

/**
 * Builds a complete HNSW graph over stored vectors 0..n-1 of an empty `hnsw`, PiPNN-style
 * (https://arxiv.org/abs/2602.21247) instead of by incremental insertion of every point:
 * points are partitioned into overlapping leaves (randomized ball carving), an exact bi-directed kNN graph is computed
 * inside each leaf, candidate edges are merged per node with HashPrune and capped with HNSW's neighbor-selection
 * heuristic. Levels >= 1 are built with regular HNSW insertion of the ~1/M promoted points only.
 *
 * Distances come from `newBlockDistance` for the dense phases and from DistanceComputer::symmetric_dis elsewhere;
 * without a block factory every distance uses symmetric_dis, so any storage works. Uses hnsw.efConstruction for the
 * upper levels.
 */
void BuildGraph(faiss::HNSW& hnsw, faiss::idx_t n, const DistanceComputerFactory& newDistanceComputer,
                const BlockDistanceFactory& newBlockDistance = nullptr, const BuildParams& params = {});

}  // namespace hashprune

/**
 * HNSW,Flat index whose graph is built by hashprune::BuildGraph. add() only buffers vectors; build() must run once
 * after the last add(). The result is a plain IndexHNSWFlat, so it is serialized, loaded and searched exactly like an
 * index built by faiss.
 */
struct IndexHNSWFlatHashPrune : faiss::IndexHNSWFlat {
    IndexHNSWFlatHashPrune(int d, int M, faiss::MetricType metric);

    void add(faiss::idx_t n, const float* x) override;

    /// Builds the graph over all added vectors. No-op if already built or empty.
    void build();

    hashprune::BuildParams buildParams;
};

}  // namespace faiss_wrapper
}  // namespace knn_jni

#endif  // OPENSEARCH_KNN_FAISS_HASHPRUNE_H
