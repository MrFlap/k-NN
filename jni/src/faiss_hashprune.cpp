// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

#include "faiss_hashprune.h"

#include "faiss/IndexFlat.h"
#include "faiss/MetricType.h"
#include "faiss/utils/distances.h"
#include "faiss/impl/AuxIndexStructures.h"
#include "faiss/impl/FaissAssert.h"

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <functional>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <utility>
#include <vector>

#ifndef FINTEGER
#define FINTEGER int  // faiss is built with FINTEGER=int
#endif

extern "C" {
int sgemm_(const char* transa, const char* transb, FINTEGER* m, FINTEGER* n, FINTEGER* k, const float* alpha,
           const float* a, FINTEGER* lda, const float* b, FINTEGER* ldb, float* beta, float* c, FINTEGER* ldc);
}

namespace knn_jni {
namespace faiss_wrapper {
namespace hashprune {

namespace {

using storage_idx_t = faiss::HNSW::storage_idx_t;

constexpr int kMaxDepth = 8;  // guards against duplicate-heavy data that never splits
constexpr uint32_t kSeed = 1234;

using Neighbor = std::pair<float, storage_idx_t>;

// Exceptions must not escape an OpenMP region: std::terminate would take the JVM down.
class FirstException {
public:
    template <typename F>
    void run(F&& f) {
        try {
            f();
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!error_) error_ = std::current_exception();
        }
    }

    void rethrow() const {
        if (error_) std::rethrow_exception(error_);
    }

private:
    std::mutex mutex_;
    std::exception_ptr error_;
};

// One object per OpenMP thread, created up front by `factory`.
template <typename T>
class PerThread {
public:
    explicit PerThread(const std::function<T*()>& factory) : objects_(omp_get_max_threads()) {
        for (auto& object : objects_) object.reset(factory());
    }

    T& get() { return *objects_[omp_get_thread_num()]; }

private:
    std::vector<std::unique_ptr<T>> objects_;
};
using ThreadComputers = PerThread<faiss::DistanceComputer>;
using ThreadBlocks = PerThread<BlockDistance>;

// Fallback BlockDistance for storages without a block kernel: pairwise symmetric_dis.
class PairwiseBlockDistance final : public BlockDistance {
public:
    explicit PairwiseBlockDistance(faiss::DistanceComputer* dc) : dc_(dc) {}

    void setTargets(const storage_idx_t* ids, size_t n) override { targets_.assign(ids, ids + n); }

    void distances(const storage_idx_t* ids, size_t n, float* out) override {
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < targets_.size(); j++) out[i * targets_.size() + j] = dc_->symmetric_dis(ids[i], targets_[j]);
        }
    }

private:
    std::unique_ptr<faiss::DistanceComputer> dc_;
    std::vector<storage_idx_t> targets_;
};

// Answers "query = stored vector q" through symmetric_dis, so faiss's HNSW insertion runs on any storage without a
// query encoding.
struct StoredQueryDistanceComputer : faiss::DistanceComputer {
    explicit StoredQueryDistanceComputer(faiss::DistanceComputer& base) : base(base) {}

    void set_query(const float*) override { FAISS_THROW_MSG("use setStoredQuery"); }
    void setStoredQuery(faiss::idx_t id) { query = id; }
    float operator()(faiss::idx_t i) override { return base.symmetric_dis(query, i); }
    float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override { return base.symmetric_dis(i, j); }

    faiss::DistanceComputer& base;
    faiss::idx_t query = -1;
};

struct Candidate {
    storage_idx_t id;
    float dist;
    uint32_t hash;
};

// Per-node bounded candidate set. Keeps at most one candidate per sector around the node and, once full, evicts the
// farthest. Sector bit j says whether the candidate is closer than the node to pivot j; with inner product this is
// exactly the random-hyperplane hash of the edge direction, and it needs only distances. Insertion order does not
// matter up to ties, so leaves can be processed in parallel.
class HashPrune {
public:
    HashPrune(faiss::idx_t n, ThreadComputers& computers, const BuildParams& params)
            : hashBits_(params.hashBits), reservoirSize_(params.reservoirSize), pivotDistances_(n * hashBits_),
              candidates_(n * reservoirSize_), fill_(n, 0), locks_(new std::atomic_flag[n]) {
        for (faiss::idx_t i = 0; i < n; i++) locks_[i].clear();
        std::mt19937 rng(kSeed);
        std::uniform_int_distribution<faiss::idx_t> pick(0, n - 1);
        std::vector<faiss::idx_t> pivots(hashBits_);
        for (auto& pivot : pivots) pivot = pick(rng);
#pragma omp parallel for
        for (faiss::idx_t i = 0; i < n; i++) {
            faiss::DistanceComputer& dc = computers.get();
            for (int j = 0; j < hashBits_; j++) pivotDistances_[i * hashBits_ + j] = dc.symmetric_dis(i, pivots[j]);
        }
    }

    HashPrune(const HashPrune&) = delete;
    HashPrune& operator=(const HashPrune&) = delete;

    void insert(storage_idx_t p, storage_idx_t q, float dist) {
        if (p == q) return;
        const uint32_t h = hash(p, q);
        while (locks_[p].test_and_set(std::memory_order_acquire)) {
        }
        Candidate* slots = &candidates_[static_cast<size_t>(p) * reservoirSize_];
        const int size = fill_[p];
        int farthest = -1;
        float farthestDist = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < size; i++) {
            if (slots[i].hash == h) {
                if (dist < slots[i].dist) slots[i] = {q, dist, h};
                locks_[p].clear(std::memory_order_release);
                return;
            }
            if (slots[i].dist > farthestDist) {
                farthestDist = slots[i].dist;
                farthest = i;
            }
        }
        if (size < reservoirSize_) {
            slots[size] = {q, dist, h};
            fill_[p]++;
        } else if (dist < farthestDist) {
            slots[farthest] = {q, dist, h};
        }
        locks_[p].clear(std::memory_order_release);
    }

    const Candidate* candidates(size_t p) const { return &candidates_[p * reservoirSize_]; }
    int size(size_t p) const { return fill_[p]; }

private:
    uint32_t hash(storage_idx_t p, storage_idx_t q) const {
        const float* dp = &pivotDistances_[static_cast<size_t>(p) * hashBits_];
        const float* dq = &pivotDistances_[static_cast<size_t>(q) * hashBits_];
        uint32_t h = 0;
        for (int j = 0; j < hashBits_; j++) h |= static_cast<uint32_t>(dq[j] < dp[j]) << j;
        return h;
    }

    const int hashBits_;
    const int reservoirSize_;
    std::vector<float> pivotDistances_;
    std::vector<Candidate> candidates_;
    std::vector<uint8_t> fill_;
    std::unique_ptr<std::atomic_flag[]> locks_;  // 1-byte spinlock per node; critical sections are a few dozen ns
};

// Keeps the k smallest (dist, id) pairs seen, sorted ascending.
inline void keepNearest(Neighbor* best, int k, float dist, storage_idx_t id) {
    if (!(dist < best[k - 1].first)) return;
    int i = k - 1;
    while (i > 0 && best[i - 1].first > dist) {
        best[i] = best[i - 1];
        i--;
    }
    best[i] = {dist, id};
}

const Neighbor kNoNeighbor{std::numeric_limits<float>::infinity(), -1};

// Randomized ball carving: sample leaders, send each point to its `fanout` nearest leaders, recurse on oversized
// buckets with `subFanout` (then 1). The top call (parallel = true) parallelizes over points, then over its buckets;
// deeper calls run serially inside one thread, since a bucket is too small to split across threads.
void partition(const std::vector<storage_idx_t>& ids, int fanout, int subFanout, std::mt19937& rng, ThreadBlocks& blocks,
               std::vector<std::vector<storage_idx_t>>& leaves, int depth, const BuildParams& params, bool parallel) {
    if (ids.size() <= params.leafMax) {
        if (ids.size() > 1) leaves.push_back(ids);
        return;
    }
    if (depth >= kMaxDepth) {
        for (size_t i = 0; i < ids.size(); i += params.leafMax) {
            leaves.emplace_back(ids.begin() + i, ids.begin() + std::min(ids.size(), i + params.leafMax));
        }
        return;
    }

    // At least 2x fanout leaders: with fanout >= leaders every point joins every bucket and nothing splits.
    const size_t numLeaders = std::min(params.maxLeaders, std::max<size_t>(2 * fanout, ids.size() / params.leafTarget));
    fanout = std::min<int>(fanout, numLeaders);
    std::vector<storage_idx_t> leaders;
    std::sample(ids.begin(), ids.end(), std::back_inserter(leaders), numLeaders, rng);

    std::vector<int32_t> assignment(ids.size() * fanout);
    const int64_t numTiles = (ids.size() + params.rowTile - 1) / params.rowTile;
    auto assignTiles = [&](int64_t begin, int64_t step) {
        BlockDistance& block = blocks.get();
        block.setTargets(leaders.data(), numLeaders);
        std::vector<float> dist(params.rowTile * numLeaders);
        std::vector<Neighbor> best(fanout);
        for (int64_t tile = begin; tile < numTiles; tile += step) {
            const size_t start = tile * params.rowTile, rows = std::min(params.rowTile, ids.size() - start);
            block.distances(&ids[start], rows, dist.data());
            for (size_t r = 0; r < rows; r++) {
                std::fill(best.begin(), best.end(), kNoNeighbor);
                const float* row = &dist[r * numLeaders];
                for (size_t l = 0; l < numLeaders; l++) keepNearest(best.data(), fanout, row[l], static_cast<storage_idx_t>(l));
                for (int f = 0; f < fanout; f++) assignment[(start + r) * fanout + f] = best[f].second;
            }
        }
    };
    if (parallel) {
        FirstException errors;
#pragma omp parallel
        errors.run([&] { assignTiles(omp_get_thread_num(), omp_get_num_threads()); });
        errors.rethrow();
    } else {
        assignTiles(0, 1);
    }

    std::vector<std::vector<storage_idx_t>> buckets(numLeaders);
    for (size_t i = 0; i < ids.size(); i++) {
        for (int f = 0; f < fanout; f++) {
            const int32_t leader = assignment[i * fanout + f];
            if (leader >= 0) buckets[leader].push_back(ids[i]);
        }
    }
    std::vector<int32_t>().swap(assignment);

    auto childDepth = [&](const std::vector<storage_idx_t>& bucket) {
        return bucket.size() == ids.size() ? kMaxDepth : depth + 1;  // no progress: stop splitting
    };
    if (!parallel) {
        for (const auto& bucket : buckets) partition(bucket, subFanout, 1, rng, blocks, leaves, childDepth(bucket), params, false);
        return;
    }
    // Leaves are collected per bucket and concatenated in bucket order, so the result does not depend on scheduling.
    std::vector<std::vector<std::vector<storage_idx_t>>> bucketLeaves(numLeaders);
    const uint32_t seed = rng();
    FirstException errors;
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t b = 0; b < static_cast<int64_t>(numLeaders); b++) {
        errors.run([&] {
            std::mt19937 bucketRng(seed + static_cast<uint32_t>(b));
            partition(buckets[b], subFanout, 1, bucketRng, blocks, bucketLeaves[b], childDepth(buckets[b]), params, false);
            std::vector<storage_idx_t>().swap(buckets[b]);
        });
    }
    errors.rethrow();
    for (auto& perBucket : bucketLeaves) {
        for (auto& leaf : perBucket) leaves.push_back(std::move(leaf));
    }
}

// Exact bi-directed kNN inside each leaf, streamed into the reservoirs. Each row tile is only compared with the points
// from its own start onward and updates both endpoints, so every pair is computed once.
void addLeafEdges(const std::vector<std::vector<storage_idx_t>>& leaves, ThreadBlocks& blocks, HashPrune& reservoirs,
                  const BuildParams& params) {
    FirstException errors;
#pragma omp parallel
    {
        std::vector<float> dist;
        std::vector<Neighbor> best;
#pragma omp for schedule(dynamic, 1)
        for (int64_t li = 0; li < static_cast<int64_t>(leaves.size()); li++) {
            errors.run([&] {
                BlockDistance& block = blocks.get();
                std::vector<storage_idx_t> leaf = leaves[li];
                std::sort(leaf.begin(), leaf.end());  // memory locality when gathering
                const size_t m = leaf.size();
                const int k = static_cast<int>(std::min<size_t>(static_cast<size_t>(params.leafNeighbors), m - 1));
                best.assign(m * k, kNoNeighbor);
                dist.resize(params.rowTile * m);
                for (size_t start = 0; start < m; start += params.rowTile) {
                    const size_t rows = std::min(params.rowTile, m - start), cols = m - start;
                    block.setTargets(&leaf[start], cols);
                    block.distances(&leaf[start], rows, dist.data());
                    for (size_t r = 0; r < rows; r++) {
                        const size_t i = start + r;
                        const float* row = &dist[r * cols];
                        for (size_t c = r + 1; c < cols; c++) {
                            const size_t j = start + c;
                            keepNearest(&best[i * k], k, row[c], static_cast<storage_idx_t>(j));
                            keepNearest(&best[j * k], k, row[c], static_cast<storage_idx_t>(i));
                        }
                    }
                }
                for (size_t i = 0; i < m; i++) {
                    for (int j = 0; j < k; j++) {
                        const auto [d, v] = best[i * k + j];
                        if (v < 0) continue;
                        reservoirs.insert(leaf[i], leaf[v], d);
                        reservoirs.insert(leaf[v], leaf[i], d);
                    }
                }
            });
        }
    }
    errors.rethrow();
}

// Assigns levels with faiss's distribution and inserts the points that reach level >= 1 with faiss's own HNSW
// insertion, highest level first as faiss does. Their level-0 links are overwritten by pruneLevel0.
void buildUpperLevels(faiss::HNSW& hnsw, faiss::idx_t n, ThreadComputers& computers) {
    hnsw.prepare_level_tab(n, false);

    std::vector<std::vector<storage_idx_t>> byLevel;
    for (faiss::idx_t i = 0; i < n; i++) {
        const int level = hnsw.levels[i] - 1;
        if (level == 0) continue;
        if (static_cast<int>(byLevel.size()) <= level) byLevel.resize(level + 1);
        byLevel[level].push_back(static_cast<storage_idx_t>(i));
    }

    std::vector<omp_lock_t> locks(n);
    for (auto& lock : locks) omp_init_lock(&lock);
    std::mt19937 rng(kSeed);
    FirstException errors;
    for (int level = static_cast<int>(byLevel.size()) - 1; level >= 1; level--) {
        auto& points = byLevel[level];
        std::shuffle(points.begin(), points.end(), rng);  // no dataset-order bias, as in faiss
#pragma omp parallel if (points.size() > 100)
        {
            faiss::VisitedTable visited(n);
            StoredQueryDistanceComputer query(computers.get());
#pragma omp for schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(points.size()); i++) {
                errors.run([&] {
                    query.setStoredQuery(points[i]);
                    hnsw.add_with_locks(query, level, points[i], locks, visited);
                });
            }
        }
    }
    for (auto& lock : locks) omp_destroy_lock(&lock);
    errors.rethrow();

    if (hnsw.entry_point < 0) {  // no point reached level >= 1
        hnsw.entry_point = 0;
        hnsw.max_level = 0;
    }
}

// Level 0: HNSW's neighbor-selection heuristic over each node's reservoir, the same rule faiss applies on insertion.
void pruneLevel0(faiss::HNSW& hnsw, faiss::idx_t n, ThreadComputers& computers, const HashPrune& reservoirs) {
    const int maxDegree = hnsw.nb_neighbors(0);
    FirstException errors;
#pragma omp parallel
    {
        std::vector<faiss::HNSW::NodeDistFarther> selected;
#pragma omp for schedule(dynamic, 1024)
        for (faiss::idx_t u = 0; u < n; u++) {
            errors.run([&] {
                faiss::DistanceComputer& dc = computers.get();
                const Candidate* candidates = reservoirs.candidates(u);
                const int size = reservoirs.size(u);
                std::priority_queue<faiss::HNSW::NodeDistFarther> queue;
                for (int i = 0; i < size; i++) queue.emplace(candidates[i].dist, candidates[i].id);
                selected.clear();
                if (size <= maxDegree) {
                    while (!queue.empty()) {
                        selected.push_back(queue.top());
                        queue.pop();
                    }
                } else {
                    faiss::HNSW::shrink_neighbor_list(dc, queue, selected, maxDegree);
                }
                size_t begin, end;
                hnsw.neighbor_range(u, 0, &begin, &end);
                for (size_t j = begin; j < end; j++) {
                    hnsw.neighbors[j] = j - begin < selected.size() ? selected[j - begin].id : -1;
                }
            });
        }
    }
    errors.rethrow();
}

}  // namespace

void BuildGraph(faiss::HNSW& hnsw, faiss::idx_t n, const DistanceComputerFactory& newDistanceComputer,
                const BlockDistanceFactory& newBlockDistance, const BuildParams& params) {
    FAISS_THROW_IF_NOT_MSG(params.hashBits >= 1 && params.hashBits <= 32, "hashBits must be in [1, 32]");
    FAISS_THROW_IF_NOT_MSG(params.reservoirSize >= 0 && params.reservoirSize <= 255, "reservoirSize must be in [0, 255]");
    FAISS_THROW_IF_NOT_MSG(params.reservoirSize > 0 || params.reservoirDegreeRatio > 0, "reservoirDegreeRatio must be > 0");
    FAISS_THROW_IF_NOT_MSG(params.topFanout >= 1 && params.subFanout >= 1 && params.leafNeighbors >= 1, "fanouts and leafNeighbors must be >= 1");
    FAISS_THROW_IF_NOT_MSG(params.leafTarget >= 1 && params.leafMax >= 2 && params.maxLeaders >= 2 && params.rowTile >= 1,
                           "invalid leaf, leader or tile size");
    FAISS_THROW_IF_NOT_MSG(hnsw.levels.empty(), "hashprune::BuildGraph requires an empty HNSW graph");
    FAISS_THROW_IF_NOT_MSG(n <= std::numeric_limits<storage_idx_t>::max(), "too many vectors for HNSW");
    if (n == 0) return;

    ThreadComputers computers(newDistanceComputer);
    BuildParams resolved = params;
    if (resolved.reservoirSize == 0) {
        resolved.reservoirSize = static_cast<int>(
            std::clamp(std::lround(params.reservoirDegreeRatio * hnsw.nb_neighbors(0)), 1L, 255L));
    }
    HashPrune reservoirs(n, computers, resolved);
    {
        std::vector<std::vector<storage_idx_t>> leaves;
        std::vector<storage_idx_t> all(n);
        std::iota(all.begin(), all.end(), 0);
        std::mt19937 rng(kSeed);
        ThreadBlocks blocks(newBlockDistance ? newBlockDistance : BlockDistanceFactory([&] {
            return new PairwiseBlockDistance(newDistanceComputer());
        }));
        partition(all, params.topFanout, params.subFanout, rng, blocks, leaves, 0, params, true);
        addLeafEdges(leaves, blocks, reservoirs, params);
    }  // leaves freed before the graph is allocated
    buildUpperLevels(hnsw, n, computers);
    pruneLevel0(hnsw, n, computers, reservoirs);
}

}  // namespace hashprune

namespace {

// Flat float vectors: gather both blocks and compute all inner products with one sgemm.
class FloatBlockDistance final : public hashprune::BlockDistance {
public:
    FloatBlockDistance(const float* x, int d, bool innerProduct) : x_(x), d_(d), innerProduct_(innerProduct) {}

    void setTargets(const faiss::HNSW::storage_idx_t* ids, size_t n) override {
        gather(ids, n, targets_, targetNorms_);
        numTargets_ = n;
    }

    void distances(const faiss::HNSW::storage_idx_t* ids, size_t n, float* out) override {
        gather(ids, n, rows_, rowNorms_);
        FINTEGER nt = numTargets_, nr = n, d = d_;
        float one = 1, zero = 0;
        // out (row-major n x nt) = rows * targets^T
        sgemm_("Transpose", "Not transpose", &nt, &nr, &d, &one, targets_.data(), &d, rows_.data(), &d, &zero, out, &nt);
        for (size_t i = 0; i < n; i++) {
            float* row = out + i * numTargets_;
            for (size_t j = 0; j < numTargets_; j++) {
                row[j] = innerProduct_ ? -row[j] : std::max(0.0f, rowNorms_[i] + targetNorms_[j] - 2 * row[j]);
            }
        }
    }

private:
    void gather(const faiss::HNSW::storage_idx_t* ids, size_t n, std::vector<float>& out, std::vector<float>& norms) {
        out.resize(n * d_);
        norms.resize(n);
        for (size_t i = 0; i < n; i++) {
            std::memcpy(&out[i * d_], x_ + static_cast<size_t>(ids[i]) * d_, d_ * sizeof(float));
        }
        if (!innerProduct_) faiss::fvec_norms_L2sqr(norms.data(), out.data(), d_, n);
    }

    const float* x_;
    const int d_;
    const bool innerProduct_;
    std::vector<float> targets_, targetNorms_, rows_, rowNorms_;
    size_t numTargets_ = 0;
};

}  // namespace

IndexHNSWFlatHashPrune::IndexHNSWFlatHashPrune(int d, int M, faiss::MetricType metric)
        : faiss::IndexHNSWFlat(d, M, metric) {}

void IndexHNSWFlatHashPrune::add(faiss::idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(hnsw.levels.empty(), "IndexHNSWFlatHashPrune: cannot add vectors after build()");
    storage->add(n, x);
    ntotal = storage->ntotal;
}

void IndexHNSWFlatHashPrune::build() {
    if (!hnsw.levels.empty() || ntotal == 0) return;
    FAISS_THROW_IF_NOT_MSG(metric_type == faiss::METRIC_L2 || metric_type == faiss::METRIC_INNER_PRODUCT,
                           "IndexHNSWFlatHashPrune supports L2 and inner product");
    const bool innerProduct = metric_type == faiss::METRIC_INNER_PRODUCT;
    const float* x = dynamic_cast<faiss::IndexFlat*>(storage)->get_xb();
    hashprune::BuildGraph(
        hnsw, ntotal,
        [this]() -> faiss::DistanceComputer* {
            // Same conversion faiss's HNSW uses: similarities become distances.
            faiss::DistanceComputer* dc = storage->get_distance_computer();
            return faiss::is_similarity_metric(metric_type) ? new faiss::NegativeDistanceComputer(dc) : dc;
        },
        [this, x, innerProduct] { return new FloatBlockDistance(x, d, innerProduct); }, buildParams);
}

}  // namespace faiss_wrapper
}  // namespace knn_jni
