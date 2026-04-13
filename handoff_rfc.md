Reordering LLD

Next Discussion:

* ReorderedFlatVectorsWriter
    * Parallelize?
* Points raised in discussion.
* BBQ .veb


RFC for algorithmic aspect of it (BP)
RFC for LLD, how we are doing it which covers BBQ and the ReorderedFlatVectorsWriter, defaults, GPU (P1).


SortMap is an implementation detail. FIgure out how a cx passes in a sortmap.

1. Introduction

This document presents a low level design for vector reordering. We find that bipartite reordering increases throughput in memory constrained environments by 70% at the cost of -11% median indexing throughput and total indexing+force merge time of +12.4%. We evaluate other methods of reordering like merge-based kmeans and flush-based kmeans and find that they do not deliver the same performance as bipartite reordering. We discuss the design for integrating bipartite reordering and recommend it on the merge path. We expand on the benchmarks in Reorder Experiment V1 Results to address the questions raised during discussion.

2. Problem Statement

Disk based vector reordering uses a rescoring step over full-precision vectors to improve search quality. The full-precision vectors are currently arranged in random order on disk, which leads to a subpar access pattern and page faults. When a page fault occurs, the kernel must perform disk I/O to load the required pages, which leads to performance regression and lower QPS. Vectors returned in the result set are usually close to each other and we can exploit this property by clustering nearby vectors together on disk.

3. Use Cases

*  Disk Based Binary Quantization (Lucene-on-Faiss)
    * POC implemented in this Use case
* SQ
    * Note: supported through .vec file.

4. Requirements

Functional Requirements

* Reordering of vector ords: The vector reordering module must reorder internal vector ords.
* Correctness: The reordered index must produce identical search results (recall, scores, DocIds) to the preexisting index with same documents.
* Data type support: Binary (32x, 16x, 8x).
* Space type support: L2, innerproduct, cosine similarity (handled as equivalent to innerproduct due to VectorTransformer).

Non-functional Requirements

* Search Throughput increase in memory-constrained environments: Vector reordering must increase throughput in representative benchmarking.
* Pagefault decrease: Vector reordering must decrease page faults in representative benchmarking. This is the mechanism for increasing throughput and decreasing latency.
* No Search Throughput Regression in memory-abundant environments: Vector reordering MUST NOT reduce throughput in memory-abundant environments.
    * It is challenging ex ante to understand if a customer’s workload will fit entirely in memory. Reordering decisions will happen during indexing/merging (discussed later in this document) and new documents can be ingested at any time. We must not introduce additional search latency if the customer’s workload can fit into memory.
* Bound index and merge latency increases: Vector reordering presents a tradeoff in doing extra work during indexing to cluster nearby vectors in order to make searches faster. The decision taken on a reordering algorithm should consider indexing regressions. The reordering strategy should not trip the java circuit breaker or slow down indexing more than 20%.

Out of Scope

* Reordering to benefit non-disk based cases (permutation computed from .vec).
* Improving other aspects of disk based vector search (like prefetch).

5. Design

5.1 Solution HLD

* Introduce a pluggable VectorReorderer module in the k-NN plugin that physically rearranges vectors on disk during segment merges so that spatially similar vectors are stored contiguously. The module hooks into the Lucene codec merge path — after vectors are merged but before the segment is finalized — and rewrites the .vec file, HNSW graph, and associated metadata according to a computed permutation. No changes to the search path or query API are required; the benefit is realized automatically through improved disk I/O locality.
* Decision point: Opt-in or default? Currently implemented as opt-in via index setting index.knn.reorder.enabled. We can switch the default of this setting from false to true if we want to make reordering default. If enabled the setting will reorder all vectors fields with the chosen algorithm (BP). Alternatively we can make reordering default without a setting. Let’s discuss default and what optimizations/extra experiments are necessary for this.
    * Options: Index level setting, on by default, turn off if cx is non-memory constrained environment or index heavy workload.

5.1.1 Proposed Solution

* Bipartite (BP) reordering using Lucene's BPVectorReorderer. BP recursively bisects the vector set, placing vectors closer to the left centroid in the left partition and vice versa. This produces a permutation where spatially similar vectors are adjacent on disk, reducing page faults during graph traversal.

5.1.2 Alternatives Considered

* K-Means clustering (k=1500): Cluster vectors via FAISS k-means, sort by (cluster_id, distance_to_centroid). Achieves slightly better query latency than BP but at 4× the merge cost.
* Merge-Aware K-Means: Cluster at flush time, persist centroids in .kcs sidecar, pool centroids at merge time instead of re-clustering. Eliminates iterative training at merge but centroid quality degrades through the cascade — query throughput regressed 21% vs baseline.

5.1.3 Solution Comparison

Metric	Baseline	BP	KMeans (k=1500)	Merge-Aware KMeans
Query throughput (ops/s)	7.97	12.27 (+54%)	12.71 (+59%)	6.32 (-21%)
P50 query latency (ms)	1,177	712 (-40%)	686 (-42%)	1,513 (+29%)
P99 query latency (ms)	2,462	1,785 (-28%)	1,650 (-33%)	—
Page faults	10.7M	8.3M (-22%)	9.1M (-15%)	—
Indexing throughput (docs/s)	6,352	5,666 (-11%)	4,560 (-28%)	—
Cumulative merge time (min)	438	477 (+9%)	2,119 (+383%)	—
recall@k	0.8	0.8	0.8	0.8

A	BP	KMeans	Merge-Aware KMeans
Time complexity	O(N·d·log N)	O(t·k²·d + N·k·d)	O(N·k·d) at merge
Space complexity	O(N)	O(N + k·d)	O(N + k·d) + .kcs sidecar

BP is the recommended algorithm: it delivers comparable query improvement to k-means with far less indexing overhead (+9% merge time vs +383%), requires no external dependencies beyond Lucene, and has no sidecar file management complexity.

5.1.4 Key Design Decisions

* Reorder during merge, not flush. Merging happens less frequently than flushes, and merge segments are larger — reordering amortizes better over more vectors. Force merge (the primary use case for disk-optimized indices) does not invoke the flush path. See §5.2.2 for full rationale.
* New Lucene codec. Reordering requires intercepting the merge write path, which is only possible through a custom codec. The codec delegates to the default KNN codec for all non-merge operations. See §5.2.4.
* BP over k-means. BP achieves 54% throughput improvement with only 9% merge overhead. K-means achieves marginally better query latency (59%) but at 383% merge overhead and requires FAISS native library calls during merge.

5.1.5 Open Questions

* Reorder by default vs opt-in: Should reordering be enabled automatically for disk-optimized indices, or require an explicit index setting? Enabling by default maximizes adoption but increases merge cost for all users. Current plan: opt-in via index setting index.knn.reorder.enabled.
* Large-scale impact (100M+ vectors): Tested on Cohere-10M and Cohere-113M. 113M results show consistent benefit but reorder time per merge increases. Need to validate that merge timeouts are not hit at larger scales.
* Reordering with SortMap: DropBox might want to use SortMap to cluster flat vectors so that .
    * Disable reordering when there’s a sort map?
* Nested case
    * Blocker?
* Automated design
    * Handle multiple indices
    * Memory constrained or not?

5.1.6 Potential Issues

* Indexing throughput regression: BP adds +12% to cumulative index/merge time and -11% to indexing throughput. For write-heavy workloads that rarely search, this is pure overhead. The opt-in setting mitigates this. Additionally, by keeping it a setting, cx can turn on reordering just before final merge, similar to knn.approximate_threshold.



5.2 Solution LLD

5.2.1 Reorder Algorithms

Bipartite Reordering (BP)

Recursive bisection algorithm from Lucene's BpVectorReorderer. Recursively splits vectors into halves, shuffling at each level to minimize inter-partition distances. Vectors in left half of partition are closer to left centroid than right centroid and vice versa. Produces a permutation where spatially similar vectors are adjacent.
Key properties:

* Memory: O(N) — only 2 × 4 × N bytes for sortedIds + biases metadata arrays. Vectors stay on disk via mmap. BpVectorReorderer.computeValueMap() accepts FloatVectorValues directly and calls vectors.copy() per thread (each gets its own mmap view).
* Compute: O(N * d * log N) — but memory-bound at scale. Random-access reads into mmap'd .vec cause page faults at deeper recursion levels where partition IDs are scattered. ~99s for 2M × 768-dim on 8 vCPU Graviton2.
* Scaling: Gets faster on small segments because the working set fits in page cache. ~1-2s for 50k vectors.
* Similarity: Passed directly to BpVectorReorderer.computeValueMap() — supports L2, IP, cosine natively.

K-Means Reordering

Clusters vectors via FAISS k-means, then sorts by (cluster_id, distance_to_centroid). Vectors in the same cluster are stored contiguously, and within each cluster, vectors are ordered by proximity to the centroid.
Key properties:

* Memory (optimized, mmap passthrough): O(1) Java heap — passes the mmap'd .vec address directly to FAISS JNI via MemorySegmentAddressExtractorJDK21. FAISS internally subsamples to num_clusters * 256 vectors (~1.5GB native, auto-freed). Pre-fault loop warms page cache before FAISS's random subsample access.
* Compute: O(num_iters * (num_clusters)^2*d + N * num_clusters * d + N log N) — 25 iterations on the subsample (num_clusters * 256 vectors) + one brute-force assignment pass over all N vectors + sorting vectors by cluster. ~89s for 2M × 768-dim for num_cluster=1500.
* Scaling: Does NOT get faster on small segments below the subsample threshold (k×256 = 384k for k=1500). A 50k-vector segment still pays full N×k×2d training cost (~10s). This is why kmeans adds disproportionate overhead during repeated merges with small-to-medium segments. However kmeans is faster on larger segments (in the offline reordering case) due to subsampling. However this offline reordering case is not realistic as we will discuss below.
* Cluster count: Configurable via reorder_kmeans_num_clusters. Higher k = more page fault reduction but more indexing time. Crossover with BP reordering time is ~1500 clusters for 2M-vector shards.

Merge-Aware K-Means (Decided against)

Clusters vectors at flush time via FAISS k-means, persists cluster summaries (.kcs sidecar), then at merge time pools source centroids and does a single assignment pass instead of re-clustering from scratch. Aims to eliminate the expensive iterative training phase at every merge level.

Benchmark result: Query throughput regressed 21% vs baseline (6.32 vs 7.97 ops/s). Centroid quality degrades through the cascade — flush-time centroids are too coarse (k=70 for small segments) and concatenation without global re-optimization produces poor spatial locality in the final permutation. We won’t consider this solution due to the poor spatial locality. See the appendix for more information about the algorithm.


5.2.2 Reorder during merge instead of flush

We choose to apply reordering during the merge path instead of flush for the following reasons:

* The number of vectors is larger during merge on average.
* Force merge does not call the flush path.
* Reordering is a fairly heavyweight operation and flush is called more frequently.



5.2.3 Reorder Implementation

The following diagram illustrates the indexing and search workflows. drawio link here. Also available as LLM-readable markdown diagram in the appendix.

Indexing

The reorder hooks into the Lucene codec merge path inside NativeEngines990KnnVectorsWriter. The key insight is that reordering must happen after the .vec/.vemf files are finalized (footers written, buffers flushed to disk) but before the segment is returned to the caller. This is why finish() is the hook point — not mergeOneField().
What changes in NativeEngines990KnnVectorsWriter:

* New field: VectorReorderStrategy reorderStrategy — injected via NativeEngines990KnnVectorsFormat. null means no reordering.
* New field: List<FieldInfo> fieldsToReorder — populated during mergeOneField(), consumed during finish().
* mergeOneField() — after writing .vec (via flatVectorsWriter.mergeOneField()) and .faiss (via NativeIndexWriter.mergeIndex()), checks if reorderStrategy != null && totalLiveDocs >= 10,000. If so, adds the field to fieldsToReorder instead of reordering immediately.
* finish() — calls flatVectorsWriter.finish() (writes .vec/.vemf codec footers), then flatVectorsWriter.close() (flushes IndexOutput buffers to disk). Then iterates fieldsToReorder and calls SegmentReorderService.reorderSegmentFiles() for each field. Errors are caught and logged — a failed reorder does not fail the merge.
* close() — calls IOUtils.close(flatVectorsWriter), which is idempotent since finish() already closed it.

What SegmentReorderService.reorderSegmentFiles() does:

1. Compute permutation: Opens the finalized .vec via Lucene99FlatVectorsReader (mmap), passes FloatVectorValues to strategy.computePermutation() → returns int[] newOrd2Old.
2. Rewrite .vec/.vemf: Opens .vec reader, writes vectors in permuted order to .vec.reorder via ReorderedFlatVectorsWriter. The new .vemf gets a ReorderedLucene99FlatVectorsFormatMeta codec header and a per-field doc→ord skip list index (for translating docId to physical vector position at search time). Atomic rename .vec.reorder → .vec, .vemf.reorder → .vemf.
3. Rewrite .faiss: Loads the HNSW index via FaissIndex.load(), calls FaissIndexReorderTransformer.transform() which remaps flat vectors, HNSW levels, offsets, and neighbor lists using both newOrd2Old and oldOrd2New. Atomic rename .faiss.reorder → .faiss.

IOContext handling: SegmentReorderService wraps the directory with a FilterDirectory that overrides createOutput() to use state.context (which is IOContext.MERGE during merge). This satisfies ConcurrentMergeScheduler's assertion that merge-path I/O uses the merge context.

Why reorder after mergeOneField() is called:

* Calculating the permutation requires entire merged list of vectors to be materialized (KnnVectorValues is a lazy iterator).
* Lucene flatVectorsWriter.mergeOneField() assumes ordinal == sequential doc order.
    * This is not true with reordering which is why we introduce doc_id → reordered_ord redirection.
* If we want to apply reordering before the entire .vec file is written, we have to write .vec/.vemf with ReorderedFlatVectorsWriter and skip flatVectorsWriter.mergeOneField.

TODO: delete case

Think through the writer vs the replacement approach. MMapping/random access approach.


Search Path (Reader Side)

No changes to the query path or search API. The reorder is transparent to search — the reader detects whether a segment was reordered and handles the docId→vector position translation automatically.
What changes in NativeEngines990KnnVectorsFormat.fieldsReader():

* Tries to open the .vemf file with ReorderedLucene99FlatVectorsReader111, which checks for the ReorderedLucene99FlatVectorsFormatMeta codec header.
* If the header matches (reordered segment): uses the doc2OrdIndex skip list to translate docId → physical vector position on every vectorValue(docId) call. The HNSW graph already has remapped neighbor IDs, so graph traversal reads vectors in spatially-local order from the .vec file.
* If the header doesn't match (CorruptIndexException): falls back to the standard Lucene99FlatVectorsReader. This handles non-reordered segments and backward compatibility transparently.

No runtime translation for HNSW traversal: The .faiss neighbor lists were remapped at reorder time, so the graph already references the new ordinals. When HNSW traversal visits neighbor j, it calls vectorValue(j) which does seek(j * vecSize) — and the vector at physical position j is the correct one because .vec was rewritten in the new order. The skip list is only needed for the docId→ord mapping (e.g., when a query needs to fetch the vector for a specific document).

5.2.4 Lucene Codec Changes

Since reordering updates the internal ordinals (representing location) we must

Files modified after reorder

File	Before Reorder	After Reorder
.vec	Vectors in doc-ID order, Lucene99FlatVectorsFormatData header	Vectors in permuted order, ReorderedLucene99FlatVectorsFormatData header
.vemf	Standard Lucene99FlatVectorsFormatMeta header, field metadata	ReorderedLucene99FlatVectorsFormatMeta header, field metadata + per-field doc→ord skip list
.faiss	HNSW neighbor lists reference original ordinals	Neighbor lists remapped via oldOrd2New[], entry point remapped

ReorderOrdMap

Maps between original and reordered ordinals:

* newOrd2Old[i]: position i in reordered .vec → original ordinal. Used when writing reordered vectors.
* oldOrd2New[j]: original ordinal j → position in reordered file. Used when remapping HNSW neighbor lists.

Memory: 2 × 4 × N bytes.

Dense case: Doc→Ord skip list (FixedBlockSkipListIndex)

After reordering, vector ordinals no longer match doc IDs. The skip list provides O(1) lookup from doc ID to reordered ordinal, stored inline in .vemf.
Format: fixed-width encoding where each doc ID maps to its ordinal using numBytes = 4 - (Integer.numberOfLeadingZeros(maxDoc) / 8) bytes. For 2M vectors: 3 bytes/entry = ~6MB. Loaded as long[] blocks for fast bit-extraction at read time.


Sparse case (implementation TODO):

Sparse case (Lucene's approach): Not every doc has a vector. Lucene uses two structures:

1. IndexedDISI — a bitset of which docs have vectors, supporting iteration and advance()
2. DirectMonotonicReader — an ord → docId mapping (monotonically increasing since ords are assigned in doc order)

The sparse case needs a third mapping layer after reordering: docId → sparseOrd → reorderedOrd.

Please see the appendix for design details as I don’t think this needs to be a focus of review and it’s slightly tedious (can run approach by Dooyong or someone else later).

Reader auto-detection

NativeEngines990KnnVectorsFormat.fieldsReader() tries the reordered reader first:

try {
return new ReorderedLucene99FlatVectorsReader111(state, scorer);
} catch (CorruptIndexException | NullPointerException e) {
return flatVectorsFormat.fieldsReader(state);  // standard, non-reordered
}


ReorderedLucene99FlatVectorsReader111 validates the ReorderedLucene99FlatVectorsFormatMeta codec header on open. If the segment wasn't reordered, the header check fails and the catch block falls back to the standard reader. No flag files needed.

FAISS index reorder

Binary scalar quantized (SQ) case

In the binary SQ HNSW configuration, the .faiss file is a nested structure: IBMP (IdMap) → IBHF (BinaryHnsw) → IBXF (BinaryFlat). The reorder transformer recurses through this nesting — FaissIdMapIndexReorderer rewrites the ordToDocs mapping in permuted order, then delegates to FaissBinaryHnswIndexReorderer which remaps the HNSW graph (levels, offsets, neighbor lists) via FaissHnswReorderer, then delegates to FaissIndexBinaryFlatReorderer which physically reorders the quantized byte vectors using newOrd2Old.

Type-specific reorderers via IndexTypeToFaissIndexReordererMapping:

FAISS Type	Reorderer	What's remapped
IXMP/IBMP	FaissIdMapIndexReorderer	ordToDocs mapping + recursive nested index
IHNF/IHNS	FaissHNSWIndexReorderer	Levels array, offsets, all neighbor lists (oldOrd2New[neighborId]), entry point
IBHF	FaissBinaryHnswIndexReorderer	Same as HNSW float
IXF2/IXFI	FaissIndexFloatFlatReorderer	Raw vector storage in permuted order
IBXF	FaissIndexBinaryFlatReorderer	Raw binary vector storage in permuted order

The HNSW reorder (FaissHnswReorderer) is the most complex — it rewrites the entire graph structure: levels array reordered by new ord, offsets recomputed, every neighbor ID in every neighbor list remapped via oldOrd2New[], sentinels preserved, valid neighbors sorted ascending, entry point remapped.

Design: Strategy Interface

public interface VectorReorderStrategy {
int[] computePermutation(FloatVectorValues vectors, int numThreads,
VectorSimilarityFunction similarityFunction);
}


Implementations:

* BipartiteReorderStrategy — wraps BpVectorReorderer, mmap-friendly, zero heap copy
* KMeansReorderStrategy — wraps FAISS k-means via JNI, mmap passthrough with pre-fault

Strategy is selected in BasePerFieldKnnVectorsFormat.getReorderStrategy() from dynamic index settings, passed through NativeEngines990KnnVectorsFormat → NativeEngines990KnnVectorsWriter. Right now we’re just using bipartite but this is pluggable.

5.2.5 Backward Compatibility

In order to maintain information about when reordering was applied for debugging purposes we should add ReorderStrategy key and [None, BP] value to FieldInfo. We are targeting OpenSearch version 3.7 for reordering, so we should consider cases for >2.17 (scalar quantization added), >3.3 (ADC+RR added which are new FieldInfo options), and >=3.7 (Reordering added).


* Merge case
    * <3.7 merged with 3.7 segment: Merge flat vectors from <3.7 segment with the flat vectors on the 3.7 segment. Then apply reordering on the merged vector values. Then new segment will have 3.7 FieldInfo .

Test Plan

* Unit tests
* Integration tests
    * Non-compound file format test: Index >10k vectors, force merge, verify reorder + search correctness.
    * Compound file format test: Index <10k vectors (Force compound file format), verify reorder + search correctness.
    * Mixed reorder test: Have one segment that’s reordered, one that’s not reordered , verify search correctness.
    * Above tests are across all supported mappings (any path that supports .vec rescoring — 8x,16x,32x compression, bbq/bbqflat).
* Backwards compatibility tests
    * Assert that >2.17 , <3.7 index when upgraded can still flush, merge, and upgrade segments through Restart Upgrade bwc tests.
    * Manually test upgrade on a 2.17->3.7 cluster and put these results in the PR containing BWC tests as an additional safeguard.



Benchmarking

Algorithm Comparison (Cohere-10M, 768-dim, 2 nodes, 5 shards, 50% memory constraint)

Metric	Baseline	BP	KMeans (k=1500)
Median query throughput (ops/s)	7.97	12.27 (+54%)	12.71 (+59%)
P50 query latency (ms)	1,177	712 (-40%)	686 (-42%)
P99 query latency (ms)	2,462	1,785 (-28%)	1,650 (-33%)
Total # major page faults	10.7M	8.3M (-22%)	9.1M (-15%)
Median indexing throughput (docs/s)	6,352	5,666 (-11%)	4,560 (-28%)
Cumulative indexing time (min)	116.4	146.2 (+26%)	158.9 (+37%)
Cumulative merge time (min)	438.3	477.4 (+9%)	2,118.7 (+383%)
recall@k	0.8	0.8	0.8
recall@1	0.95	0.95	0.95

BP offers the best mix of faster searches at less indexing increase. Total merge+indexing minutes is 554.7 (baseline) vs  623.6 (BP) [+12%] vs 2118.7 (kmeans). KMeans indexing/merge is prohibitively slow with worse spatial locality.


Multi-Segment Benchmark (50% constrained memory, 50 segments, Cohere-10M 768-dim)

Index without force merge — segments remain as produced by tiered merge policy. Tests reorder benefit when segments are not force-merged to 1 (since merges are non-deterministic we have to still merge to 10 segments/shard, or else numbers will be widely different).

Metric	Baseline	BP	BP vs Baseline
Segment count	53	54	—
Median query throughput (ops/s)	8.47	11.12	31%
P50 query latency (ms)	1,116	795	-29%
P99 query latency (ms)	2,356	1,950	-17%
P100 query latency (ms)	7,760	5,290	-32%
Page faults (engine 0)	5,347,727	4,807,577	-10%
Page faults (engine 1)	5,368,081	4,764,971	-11%
recall@k	0.81	0.81	—
recall@1	0.96	0.95	—

BP delivers +31% throughput and -29% P50 latency even without force merge. Latency improvement is consistent across all percentiles including P100 (-32%). Page fault reduction is more modest (-11%) than the force-merged case (-22%), likely because more segments means each segment is smaller and more likely to fit in page cache.

Filters Benchmark (50% constrained memory, force-merged, Cohere-10M 768-dim)

Filter selectivity = fraction of docs passing the filter. Lower selectivity = fewer docs to rescore = fewer page faults. Page cache dropped between runs.

Filter Selectivity	Baseline Throughput (ops/s)	BP Throughput (ops/s)	BP vs Baseline	Baseline Page Faults (total)	BP Page Faults (total)	PF Reduction
0.10%	671.81	649.61	-3%	24,188	16,841	-30%
1%	75.78	123.98	64%	185,289	189,579	2%
5%	15.79	18.61	18%	5,136,004	4,815,533	-6%
10%	10.15	12.72	25%	8,347,758	7,422,805	-11%
25%	8.45	12.15	44%	10,973,028	8,095,065	-26%
50%	8.01	12.85	60%	11,956,121	7,566,145	-37%
75%	7.92	13.47	70%	11,818,170	7,131,559	-40%
90%	7.99	13.76	72%	11,621,823	6,957,198	-40%
99%	8.01	13.94	74%	11,489,094	6,851,148	-40%

filtering cas

Key observations:

* BP benefit increases with filter selectivity. At very restrictive filters (0.1%), there are so few vectors to rescore that page faults are negligible and BP adds slight overhead (-3%). At 1%+, BP consistently wins.
* The sweet spot is 50-99% selectivity where BP delivers +60-74% throughput with ~40% fewer page faults. This matches the unfiltered case (which is effectively 100% selectivity).
* At 1% selectivity, BP throughput is +64% despite page faults being roughly equal — suggests the reordered layout improves cache line utilization even when total fault counts are similar.

Mixed Indexing + Search Workload (50% constrained memory, force-merged)

Concurrent search and bulk indexing against a force-merged Cohere-10M index.
Single-client search during indexing:

Metric	Baseline	BP	BP vs Baseline
Median query throughput (ops/s)	1,439.42	1,482.93	3%
Median indexing throughput (docs/s)	6,073.2	5,763.97	-5%
Page faults (engine 0)	63,630	341,362	437%
Page faults (engine 1)	68,286	293,735	330%

Note: the single-client search throughput is very high (~1,440-1,483 ops/s) indicating queries are served almost entirely from page cache with minimal contention. The page fault increase for BP is likely due to page faults when computing a permutation from the mmaped flatvectors file, which does not occur in baseline indexing+search run.

Multi-client saturating search during indexing:

Metric	Baseline	BP	BP vs Baseline
Median query throughput (ops/s)	8.47	11.12	31%
Page faults (engine 0)	5,347,727	4,807,577	-10%
Page faults (engine 1)	5,368,081	4,764,971	-11%
Total page faults	10,715,808	9,572,548	-11%

Under saturating load, BP delivers +31% throughput with 11% fewer page faults, consistent with the force-merged benchmark results. The benefit is smaller than the pure search case (+54%) because concurrent indexing competes for page cache and I/O bandwidth.


Full Memory Benchmark

Methodology: Unconstrained memory, force-merged , cohere-10M 768-dim. 10 runs performed and last 5 runs averaged.


Metric	Baseline	BP	BP vs Baseline
P50 Latency (ms)	121.97	110.31	-9.56%
P99 Latency (ms)	249.1	257.66	3.44%
P100 Latency (ms)	336.25	389.11	15.72%
Throughput (ops/s)	80.09	85.14	6.31%

BP improves P50 latency and throughput slightly even in full-memory environments. Tail latencies (P99, P100) regress modestly. This satisfies the NFR that reordering must not reduce throughput in memory-abundant environments.


Outstanding Task Breakdown

Tasks:

* Implement sparse case.
* Optimize BP reordering on indexing size
    * Experiment with higher number of threads.
    * Try moving reordering to before vec is finish, instead of having vec file finished and then applying reordering on top (potentially 50% savings on cost for reloading .vec file and transforming it).
* GPU Integration
    * This needs to be thought through. My current thoughts are to fit with the file-replacement-on-merge strategy, GPU should calculate the bipartite permutation and pass back a permutation file and the hnsw file. This fits with the planned approach of not sending back flatvectors/vec.
    * Question: In the quantized case, do we send quantized vectors to GPU, or full precision vectors?

Appendix

Optimal Number of K-Means Clusters (Cohere-10M, 768-dim, ~2M vectors/shard, 50% memory constraint)

Algorithm	k	Throughput (ops/s)	vs BP	Page Faults (total)	Reorder Time/shard	vec_load	permutation	faiss_id	vec_transform
KMeans	500	13.35	-14%	7,126,259	~70s	~31s	~22s	~11s	~6s
KMeans	1000	14.41	-8%	6,626,991	~97s	~32s	~48s	~11s	~6s
KMeans	1500	15.01	-4%	6,379,189	~138s	~32s	~89s	~11s	~6s
KMeans	2000	15.21	-3%	6,279,076	~192s	~32s	~142s	~11s	~7s
KMeans	2500	15.5	-1%	6,194,687	~260s	~31s	~211s	~11s	~6s
KMeans	3000	15.81	1%	6,067,741	~341s	~32s	~291s	~11s	~6s
BP	—	15.61	—	6,029,150	~150s	~28s	~99s	~10s	~14s

LLM-readable LLD Diagram

┌─────────────────────────────────────────────────────────────────────────┐
│                        MERGE CALL CHAIN (Lucene)                        │
│                                                                         │
│  SegmentMerger.mergeVectorValues()                                      │
│    └─ try (KnnVectorsWriter w = codec.fieldsWriter(state)) {            │
│          w.merge(mergeState);  // KnnVectorsWriter.merge() is final     │
│       }                                                                 │
│                                                                         │
│  KnnVectorsWriter.merge(mergeState)                                     │
│    ├─ for each field: mergeOneField(fieldInfo, mergeState)    ──────┐   │
│    ├─ finish()                                                ──┐   │   │
│    └─ try-with-resources: close()                            ─┐ │   │   │
│                                                               │ │   │   │
└───────────────────────────────────────────────────────────────┼─┼───┼───┘
│ │   │
┌───────────────────────────────────────────────────────────────┼─┼───┼───┐
│              NativeEngines990KnnVectorsWriter                 │ │   │   │
│                                                               │ │   │   │
│  mergeOneField(fieldInfo, mergeState)  ◄──────────────────────┼─┼───┘   │
│    ├─ flatVectorsWriter.mergeOneField()  → writes .vec data             │
│    ├─ NativeIndexWriter.mergeIndex()     → writes .faiss                │
│    └─ if reorderStrategy != null && docs >= 10k:                        │
│          fieldsToReorder.add(fieldInfo)   ← MARK FOR LATER              │
│                                                               │ │       │
│  finish()  ◄──────────────────────────────────────────────────┼─┘       │
│    ├─ flatVectorsWriter.finish()   → writes .vec/.vemf footers          │
│    ├─ flatVectorsWriter.close()    → flushes IndexOutput to disk        │
│    └─ for each field in fieldsToReorder:                                │
│          SegmentReorderService.reorderSegmentFiles()  ──────────────┐   │
│                                                               │     │   │
│  close()  ◄───────────────────────────────────────────────────┘     │   │
│    └─ IOUtils.close(flatVectorsWriter)  → idempotent, already closed│   │
│                                                                     │   │
└─────────────────────────────────────────────────────────────────────┼───┘
│
┌─────────────────────────────────────────────────────────────────────┼───-┐
│              SegmentReorderService                                  │    │
│                                                                     │    │
│  reorderSegmentFiles()  ◄───────────────────────────────────────────┘    │
│    ├─ computePermutationFromVecFile()                                    │
│    │    ├─ opens .vec via Lucene99FlatVectorsReader (mmap)               │
│    │    └─ strategy.computePermutation(vectorValues, threads)            │
│    │         └─ BipartiteReorderStrategy → BpVectorReorderer             │
│    │              returns int[] permutation (newOrd → oldOrd)            │
│    │                                                                     │
│    ├─ rewriteVecFile(readDir, writeDir, ...)                             │
│    │    ├─ opens .vec via Lucene99FlatVectorsReader                      │
│    │    ├─ writes reordered .vec/.vemf via ReorderedFlatVectorsWriter    │
│    │    │    (new codec header + ordMap skip list in .vemf)              │
│    │    └─ atomic rename .vec.reorder → .vec, .vemf.reorder → .vemf      │
│    │                                                                     │
│    └─ rewriteFaissFile(readDir, writeDir, ...)                           │
│         ├─ loads .faiss via FaissIndex.load()                            │
│         ├─ FaissIndexReorderTransformer.transform()                      │
│         │    (remaps HNSW neighbor lists with permutation)               │
│         └─ atomic rename .faiss.reorder → .faiss                         │
│                                                                          │
│  NOTE: writeDir wraps directory with FilterDirectory to override         │
│        createOutput() → uses state.context (IOContext.MERGE)             │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│              SEARCH TIME (Reader Side)                                   │
│                                                                          │
│  NativeEngines990KnnVectorsFormat.fieldsReader(state)                    │
│    ├─ try: ReorderedLucene99FlatVectorsReader111(state)                  │
│    │    reads .vemf → checks for "ReorderedLucene99FlatVectorsFormatMeta"│
│    │    if match: uses ordMap to translate docId ↔ physical vec position │
│    │                                                                     │
│    └─ catch CorruptIndexException:                                       │
│         Lucene99FlatVectorsReader(state)  ← standard, non-reordered      │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘


The key insight: reordering can't happen in mergeOneField() (no footers yet) or KNN80CompoundFormat.write() (skipped for large segments). It happens in finish(), after footers are written and streams flushed, but before close(). The reader side auto-detects reordered segments by checking the codec header in .vemf.


Sparse case support

Lucene's Lucene99FlatVectorsWriter detects dense vs sparse at write time (count == maxDoc → dense, otherwise sparse). For sparse fields, it writes two extra structures into the vector data file:

1. IndexedDISI — a compressed bitset of which doc IDs have vectors. Supports advance(docId) to find the next doc with a vector, and tracks the ordinal position during iteration.
2. DirectMonotonicReader (ord→doc) — a monotonically increasing mapping from ordinal to doc ID. Since ords are assigned in doc-ID order, this is monotonic and compresses well. Used by SparseOffHeapVectorValues.ordToDoc(ord) for the reverse lookup (e.g., when HNSW traversal needs to check if a neighbor's doc passes a filter).

The dense case writes neither — docsWithFieldOffset == -1 signals dense, and ordToDoc(ord) is just return ord.
What reordering changes for sparse fields:
Before reordering, the mapping chain is:

docId → (IndexedDISI iteration) → ord → seek(ord * vecSize) → vector

After reordering, vectors are physically stored in permuted order, so we need:

docId → (IndexedDISI iteration) → originalOrd → reorderedOrd → seek(reorderedOrd * vecSize) → vector

The skip list (FixedBlockSkipListIndex) currently maps docId → reorderedOrd directly, which works for dense fields because docId == originalOrd. For sparse fields, the skip list must be built differently:

* At write time (ReorderedFlatVectorsWriter): When iterating docAndOrds sorted by doc ID, the ord stored is the reordered ordinal (looked up via newOrd2Old inverse). But for sparse fields, the doc IDs are not contiguous — the skip list cannot be indexed by raw doc ID. Instead, the writer must either:
    * (a) Store a full maxDoc-sized array with sentinel values for docs without vectors (wasteful for very sparse fields), or
    * (b) Keep the IndexedDISI bitset for doc→originalOrd translation, then add a separate originalOrd → reorderedOrd mapping of size N (compact).
* At read time (ReorderedOffHeapFloatVectorValues111): The DenseOffHeapVectorValues iterator currently does nextDoc() → ++doc and index() → skipList.skipTo(doc).getOrd(). A sparse variant needs:
    * iterator() returns an IndexedDISI-backed iterator that skips docs without vectors
    * index() returns the reordered ordinal for the current doc
    * ordToDoc(reorderedOrd) needs to reverse the chain: reorderedOrd → originalOrd → docId

Current state: FixedBlockSkipListIndexBuilder and ReorderedFlatVectorsWriter only implement the dense path. The addField() method hardcodes ReorderedDenseFloatFlatFieldVectorsWriter. The isDense flag exists in DocIdOrdSkipListIndexBuilder but the sparse branch in pushLeafSubBlock is marked TODO and the non-leaf level building doesn't handle it.
What needs to change:

1. ReorderedFlatVectorsWriter.addField() — detect sparse vs dense (from the original .vemf metadata or by comparing vector count to maxDoc) and instantiate the appropriate writer.
2. ReorderedFlatVectorsWriter sparse writer — write the IndexedDISI bitset (can copy from original .vemf) + a compact originalOrd → reorderedOrd array of size N, or a docId → reorderedOrd skip list that only contains entries for docs with vectors.
3. ReorderedLucene99FlatVectorsReader111 — read the isDense byte from .vemf and instantiate either DenseOffHeapVectorValues (current) or a new SparseOffHeapVectorValues that chains IndexedDISI → ord remapping.
4. ReorderedOffHeapFloatVectorValues111 — add a SparseOffHeapVectorValues inner class analogous to Lucene's SparseOffHeapVectorValues, with ordToDoc() reversing through the reorder permutation.



Merge-Aware KMeans Algorithm

Key properties:

* Memory: Same as optimized k-means at flush (mmap passthrough, ~0 Java heap for vectors). At merge, only the pooled centroids live on heap (~12MB for 2×1500 centroids at 1024-dim). Assignment pass reads merged vectors via mmap sequentially.
* Compute (flush): Same as vanilla k-means — O(t·k²·d + N·k·d). Full FAISS k-means runs once per flush segment.
* Compute (merge): O(N·k·d) — single brute-force assignment pass only. The 74s training phase (25 iterations on subsample) is eliminated. Centroid merge itself is negligible (~0.6s weighted k-means on ~3000 points, or free concatenation when pool_size ≤ k_max).
* Scaling: Merge cost scales linearly with N (assignment pass only). Flush cost is the same as vanilla k-means. Theoretical cumulative cascade cost: ~30s vs ~115s for BP vs ~217s for vanilla k-means.
* Adaptive k: Flush segments use capK() (e.g., k=70 for 5k vectors). Centroids accumulate through merges via concatenation until pool_size exceeds k_max, at which point weighted k-means reduces back to k_max.
* New artifacts: .kcs sidecar file per segment containing centroids, linear sums, counts, per-vector assignments (~20MB for 2M vectors at k=1500, but many .kcs files when merges/flushes ongoing , saw peak of ~2gb in .kcs files during force merge from ~80 segments → 1 segment).
* Constraint: All source segments in a merge must have .kcs files. Cannot enable on an index with existing non-clustered segments.



file to hold memory

Usage:

vi memory.cpp # paste in below code
g++ memory.cpp # compile to a.out
./a.out 30 # hold 30 GB

#include <iostream>
#include <vector>
#include <cstring>
#include <string>
#include <cstdlib>      // for std::strtod
#include <sys/mman.h>   // for mlock
#include <cerrno>
#include <cstdint>

int main(int argc, char* argv[]) {
if (argc != 2) {
std::cerr << "Usage: " << argv[0] << " <size_in_GB>\n";
std::cerr << "\n";
std::cerr << "Examples:\n";
std::cerr << "  " << argv[0] << " 30       # allocate 30 GB\n";
std::cerr << "  " << argv[0] << " 16       # allocate 16 GB\n";
std::cerr << "  " << argv[0] << " 7.5      # allocate 7.5 GB\n";
std::cerr << "\n";
return 1;
}

    // Parse the size argument
    char* endptr = nullptr;
    double size_gb = std::strtod(argv[1], &endptr);

    if (endptr == argv[1] || *endptr != '\0' || size_gb <= 0.0) {
        std::cerr << "Error: Invalid size '" << argv[1] << "'. Must be a positive number.\n";
        std::cerr << "Example: " << argv[0] << " 30\n";
        return 1;
    }

    // Convert GB to bytes
    const uint64_t size_bytes = static_cast<uint64_t>(size_gb * 1024ULL * 1024ULL * 1024ULL);

    std::cout << "Requesting " << size_gb << " GB (" << size_bytes << " bytes) ...\n";

    std::vector<char> buffer;
    try {
        buffer.resize(size_bytes);
        std::cout << "Successfully allocated " << size_gb << " GB.\n";
    } catch (const std::bad_alloc& e) {
        std::cerr << "Allocation failed: " << e.what() << "\n";
        return 1;
    }

    // Touch every page to commit physical memory
    std::cout << "Touching memory to commit physical pages...\n";
    std::memset(buffer.data(), 0xAA, size_bytes);  // using 0xAA so it's clearly visible in memory
    std::cout << "Memory touched.\n";

    // Try to lock in RAM
    std::cout << "Attempting to lock memory (mlock)...\n";
    if (mlock(buffer.data(), size_bytes) == 0) {
        std::cout << "Memory successfully locked in physical RAM.\n";
    } else {
        std::cerr << "mlock failed: " << std::strerror(errno) << "\n";
        std::cerr << "(this is common if you're not root or memlock limit is low)\n";
        std::cerr << "Memory is still allocated and resident, but may swap under heavy pressure.\n";
    }

    std::cout << "\nHolding " << size_gb << " GB of resident memory.\n";
    std::cout << "Press Enter to release and exit...\n";
    std::cin.get();

    // Optional: unlock before exit
    munlock(buffer.data(), size_bytes);
    std::cout << "Memory released.\n";

    return 0;
}

--end document--



Stop reading here

Sync w Dooyong on Sparse



















Reordering LLD Take 2
^ above is actual document, will populate here shortly.




Other docs:
Reorder Talk with Finn
Reorder Experiment V1 Results
Reordering Next Steps
Reorder Results Feb 13rd


Additional Experiments


Premerge (Cohere-10M)


	Number Segments		P50 latency		P99 Latency		# Page Faults
Baseline							
Reordered							
Gain





Page faults per segment size

	Reordering Time (BP)	Reordering time (kmeans opt)	# Page fault (baseline)	# page fault (bp reordered)	# page fault (kmeans_opt reordered)
Segment size (768dimension)					
10k					
100k					
500k					
1M					
2M					
10M

Need to figure out the memory constrainment scenario for these different dcenarios.
Need to figure out how to grab the memory constraints of a particular


Deciding on optimal #centroids

* for cohere-10M , with 5 segments (2M docs/segment)



	500 centroids	1000 centroids	1500 centroids	2000 centroids	2500 centroids	3000 centroids
Segment size						
10k						
100k						
500k						
1M						
2M						
9M



Filtering Benchmarks

(not sure what we need here)

100% Memory Environment Benchmarks



Additional Datasets Needed

Cohere-768-IP

Already performed runs.

Cohere-768-L2

Gets us insight into l2 clusters.


Glove-200 (cosine sim)

Insight into cosine similarity.

Full memory:

============================================================
ALL RUNS
============================================================

BASELINE (Full Memory)
------------------------------------------------------------
Runs averaged: 10
P50 Latency:   120.24 ms
P99 Latency:   242.37 ms
P100 Latency:  332.34 ms
Throughput:    81.26 ops/s

BP_FULL_MEMORY
------------------------------------------------------------
Runs averaged: 10
P50 Latency:   109.23 ms
P99 Latency:   256.65 ms
P100 Latency:  394.98 ms
Throughput:    85.69 ops/s

COMPARISON (BP vs Baseline)
------------------------------------------------------------
P50 Latency:   -9.15%
P99 Latency:   +5.89%
P100 Latency:  +18.85%
Throughput:    +5.46%

============================================================
LAST 5 RUNS ONLY (skipping first 5)
============================================================

BASELINE (Full Memory)
------------------------------------------------------------
Runs averaged: 5
P50 Latency:   121.97 ms
P99 Latency:   249.10 ms
P100 Latency:  336.25 ms
Throughput:    80.09 ops/s

BP_FULL_MEMORY
------------------------------------------------------------
Runs averaged: 5
P50 Latency:   110.31 ms
P99 Latency:   257.66 ms
P100 Latency:  389.11 ms
Throughput:    85.14 ops/s

COMPARISON (BP vs Baseline)
------------------------------------------------------------
P50 Latency:   -9.56%
P99 Latency:   +3.44%
P100 Latency:  +15.72%
Throughput:    +6.31%

Conclusion: throughput increased , p50 latency decreased , tail latencies (p99+) increased.
(10 clients).

Previous benchmarking result:

Cohere-113M 1024 (12P, 0R)

Insight into large-scale segments (~9M docs / shard).



Indexing impact

https://us-east-2.console.aws.amazon.com/cloudwatch/home?region=us-east-2#metricsV2?graph=~(metrics~(~(~'AWS*2fEC2~'CPUUtilization~'InstanceId~'i-0186e9a85e7220261~(label~'i-CPUUtilization*20*28baseline*29~region~'us-east-2~id~'m1))~(~'...~'i-00d1b361d986fbab3~(region~'us-east-2~id~'m2~label~'CPUUtilization*20*28reorder*29)))~view~'timeSeries~stat~'Average~period~60~stacked~false~yAxis~(left~(min~0))~region~'us-east-2~title~'CPU*20utilization*20*28*25*29~start~'-PT3H~end~'P0D)

(we need to diagnose how to improve this a bit.)



* TODO: look into memory pressure!
    * This is a big one for indexing, don’t want to trip circuit breaker.



baseline:

Cumulative indexing time of primary shards,,117.22721666666666,min
Min cumulative indexing time across primary shards,,0.00035,min
Median cumulative indexing time across primary shards,,23.251691666666666,min
Max cumulative indexing time across primary shards,,24.1748,min
Cumulative indexing throttle time of primary shards,,0,min
Min cumulative indexing throttle time across primary shards,,0,min
Median cumulative indexing throttle time across primary shards,,0.0,min
Max cumulative indexing throttle time across primary shards,,0,min
....
Min Throughput,custom-vector-bulk,5633.61,docs/s
Mean Throughput,custom-vector-bulk,6920.06,docs/s
Median Throughput,custom-vector-bulk,6453.48,docs/s
Max Throughput,custom-vector-bulk,12323.18,docs/s
50th percentile latency,custom-vector-bulk,67.31159694027156,ms
90th percentile latency,custom-vector-bulk,104.07538099680096,ms
99th percentile latency,custom-vector-bulk,231.44232069607767,ms
99.9th percentile latency,custom-vector-bulk,15262.930328940536,ms
99.99th percentile latency,custom-vector-bulk,23408.349045191073,ms
100th percentile latency,custom-vector-bulk,30092.254862887785,ms
50th percentile service time,custom-vector-bulk,67.31159694027156,ms
90th percentile service time,custom-vector-bulk,104.07538099680096,ms
99th percentile service time,custom-vector-bulk,231.44232069607767,ms
99.9th percentile service time,custom-vector-bulk,15262.930328940536,ms
99.99th percentile service time,custom-vector-bulk,23408.349045191073,ms
100th percentile service time,custom-vector-bulk,30092.254862887785,ms

bipartite changes:

Cumulative indexing time of primary shards,,140.02518333333333,min
Min cumulative indexing time across primary shards,,0.0002666666666666667,min
Median cumulative indexing time across primary shards,,27.656908333333334,min
Max cumulative indexing time across primary shards,,28.936300000000003,min
Cumulative indexing throttle time of primary shards,,0,min
Min cumulative indexing throttle time across primary shards,,0,min
Median cumulative indexing throttle time across primary shards,,0.0,min
Max cumulative indexing throttle time across primary shards,,0,min
...
Min Throughput,custom-vector-bulk,4873.26,docs/s
Mean Throughput,custom-vector-bulk,6049.02,docs/s
Median Throughput,custom-vector-bulk,5607.48,docs/s
Max Throughput,custom-vector-bulk,11757.59,docs/s
50th percentile latency,custom-vector-bulk,80.51437058020383,ms
90th percentile latency,custom-vector-bulk,127.39779769908637,ms
99th percentile latency,custom-vector-bulk,280.5397276836433,ms
99.9th percentile latency,custom-vector-bulk,17799.284055040294,ms
99.99th percentile latency,custom-vector-bulk,31767.740441186506,ms
100th percentile latency,custom-vector-bulk,38114.37944788486,ms
50th percentile service time,custom-vector-bulk,80.51437058020383,ms
90th percentile service time,custom-vector-bulk,127.39779769908637,ms
99th percentile service time,custom-vector-bulk,280.5397276836433,ms
99.9th percentile service time,custom-vector-bulk,17799.284055040294,ms
99.99th percentile service time,custom-vector-bulk,31767.740441186506,ms
100th percentile service time,custom-vector-bulk,38114.37944788486,ms

140 vs 117 mins  (16.4% increase).
Force merge time is also similar amount of increase.

We see page faults:
Baseline:

Min Throughput,prod-queries,0.87,ops/s
Mean Throughput,prod-queries,7.81,ops/s
Median Throughput,prod-queries,7.93,ops/s
Max Throughput,prod-queries,8.08,ops/s
50th percentile latency,prod-queries,1238.6251710122451,ms
90th percentile latency,prod-queries,1908.0253963824362,ms
99th percentile latency,prod-queries,2471.689590543974,ms
99.9th percentile latency,prod-queries,2896.4302258510183,ms
99.99th percentile latency,prod-queries,3289.845771058468,ms
100th percentile latency,prod-queries,3347.830123035237,ms
50th percentile service time,prod-queries,1238.6251710122451,ms
90th percentile service time,prod-queries,1908.0253963824362,ms
99th percentile service time,prod-queries,2471.689590543974,ms
99.9th percentile service time,prod-queries,2896.4302258510183,ms
99.99th percentile service time,prod-queries,3289.845771058468,ms
100th percentile service time,prod-queries,3347.830123035237,ms
error rate,prod-queries,0.00,%
Mean recall@k,prod-queries,0.80,
Mean recall@1,prod-queries,0.95,

Changed:

Min Throughput,prod-queries,0.56,ops/s
Mean Throughput,prod-queries,13.15,ops/s
Median Throughput,prod-queries,13.56,ops/s
Max Throughput,prod-queries,14.04,ops/s
50th percentile latency,prod-queries,673.9561969880015,ms
90th percentile latency,prod-queries,1150.023075658828,ms
99th percentile latency,prod-queries,1621.6809119516988,ms
99.9th percentile latency,prod-queries,2208.845802697121,ms
99.99th percentile latency,prod-queries,3201.5044481696327,ms
100th percentile latency,prod-queries,3234.313726890832,ms
50th percentile service time,prod-queries,673.9561969880015,ms
90th percentile service time,prod-queries,1150.023075658828,ms
99th percentile service time,prod-queries,1621.6809119516988,ms
99.9th percentile service time,prod-queries,2208.845802697121,ms
99.99th percentile service time,prod-queries,3201.5044481696327,ms
100th percentile service time,prod-queries,3234.313726890832,ms
error rate,prod-queries,0.00,%
Mean recall@k,prod-queries,0.80,
Mean recall@1,prod-queries,0.95,


Difference in mediam throughput:
73% increase in throughput.




Kmeans

The crossover point where kmeans takes longer to reorder than bipartite reordering is at around 1500 clusters.
These are for shards of ~2M.

3000 clusters:
Median Throughput,prod-queries,15.81,ops/s
engine 0: 3013417 faults
engine 1: 3054324 faults

1500 clusters (slightly less indexing time compared to bipartite reordering).
Median Throughput,prod-queries,15.01,ops/s
engine 0: 3172030 faults
engine 1: 3207159 faults

1000 clusters (less indexing time)
Median Throughput,prod-queries,14.41,ops/s
engine 0: 3296463
engine 1: 3330528

500 clusters (much less indexing time):
(note: we probably need to disable page cache again)
Median throughput: Throughput,prod-queries,13.35,ops/s
engine 0: 3553666
engine 1: 3572593

2000 clusters (more indexing time):
Median Throughput,prod-queries,15.21,ops/s
engine 0: 3128136

Reorder RFC (Replacement-Free LLD Draft)



5.2.2 Reorder during merge instead of flush

We choose to apply reordering during the merge path instead of flush for the following reasons:

* The number of vectors is larger during merge on average.
* Force merge does not call the flush path.
* Reordering is a fairly heavyweight operation and flush is called more frequently.

5.2.3 Reorder Implementation

Reordering hooks into the Lucene codec merge path inside NativeEngines990KnnVectorsWriter. The implementation writes .vec in permuted order during merge, builds the FAISS graph with vectors already in their final positions, and writes reordered metadata inline. We compute the permutation directly from the source segments' mmap-backed FloatVectorValues (which support random access and copy()), then write .vec in permuted order in a single pass.

Index Settings

index.knn.advanced.reorder_strategy   → WHAT permutation to compute (bp, kmeans, none)


Example:

PUT /my-index
{
"settings": {
"index.knn": true,
"index.knn.advanced.reorder_strategy": "bp" # applies to all fields
}
}



Merge Call Chain

NativeEngines990KnnVectorsWriter.mergeOneField(fieldInfo, mergeState)
│
├─ [reorder enabled && FLOAT && totalLiveDocs >= 10k]
│   │
│   ├─ 1. Build MergeOrdMapping from MergeState metadata
│   │      └─ mergedOrdToDocId[], segmentStarts[], liveLocalOrds[][]
│   │         Single pass over source segment ordinals. No vector I/O.
│   │         Uses MergeState.docMaps[seg].get(sourceDocId) for doc ID translation
│   │         and FloatVectorValues.ordToDoc(localOrd) for ord→doc mapping (both O(1)).
│   │         Allocates mergedOrdToDocId at max possible size (sum of maxDocs),
│   │         fills in one iteration, trims at end.
│   │         If mergeState.needsIndexSort, sorts mergedOrdToDocId ascending.
│   │
│   ├─ 2. Build MergedRandomAccessFloatVectorValues
│   │      └─ Random-access FloatVectorValues composite over source segment mmap readers.
│   │         Each source: mergeState.knnVectorsReaders[i].getFloatVectorValues(fieldName)
│   │         returns OffHeapFloatVectorValues (mmap-backed, random access, copy() supported).
│   │         vectorValue(mergedOrd) → binary search segmentStarts → delegate to source.
│   │         Handles deleted docs via liveLocalOrds[][] indirection.
│   │         copy() clones each segment's mmap slice for BP thread safety.
│   │         Memory: zero heap for vectors — all reads go through OS page cache.
│   │
│   ├─ 3. Compute BP permutation
│   │      └─ strategy.computePermutation(mergedRandomAccess, numThreads, similarity)
│   │         BP calls copy() per thread, each gets independent mmap views.
│   │         Returns int[] permutation where permutation[newOrd] = oldMergedOrd.
│   │
│   ├─ 4. Write .vec in permuted order into shared IndexOutput
│   │      └─ IndexOutput vectorData = reorderAwareFlatVectorsWriter.getVectorDataOutput()
│   │         for newOrd in 0..N-1:
│   │             vec = mergedRA.vectorValue(permutation[newOrd])  // mmap read
│   │             vectorData.writeBytes(vec)
│   │
│   ├─ 5. Write .vemf metadata + skip list + ord→doc array
│   │      └─ ReorderedFieldMetaWriter writes into shared meta IndexOutput:
│   │         - Field metadata (fieldNumber, encoding, similarity, offset, length, dimension)
│   │         - Doc→ord skip list via FixedBlockSkipListIndexBuilder (search-time translation)
│   │         - Ord→doc int array (merge-time ordToDoc() for future merges)
│   │
│   ├─ 6. Set fieldInfo.putAttribute("knn_reordered", "true")
│   │      └─ Persisted in SegmentInfo. Reader uses this to dispatch per-field.
│   │
│   └─ 7. Build .faiss HNSW graph with ReorderedKNNFloatVectorValues
│          └─ KNNVectorValues wrapper that iterates in permuted order:
│             nextDoc() advances newOrd from 0 to N-1
│             docId()    = mergedOrdToDocId[permutation[newOrd]]
│             getVector() = mergedRA.vectorValue(permutation[newOrd])
│             HNSW graph built in final order — no post-hoc neighbor remapping.
│
└─ [else: below threshold or non-FLOAT]
├─ flatVectorsWriter.mergeOneField() — writes .vec in original order
└─ NativeIndexWriter.mergeIndex() — builds .faiss (no reorder)


Key Components


* MergeOrdMappingBuilder: Builds mergedOrdToDocId[], segmentStarts[], and liveLocalOrds[][] in a single pass over source segment metadata. No vector data is read. Handles deleted docs (skips merged doc ID = -1), index sort (sorts mergedOrdToDocId ascending), and segments with/without deletions (liveLocalOrds is null for segments without deletions where liveIdx == rawLocalOrd).
* MergedRandomAccessFloatVectorValues: A FloatVectorValues subclass providing random access and copy() over multiple source segments. Maps merged ord → (segment, local ord) using segmentStarts[] and liveLocalOrds[][] (for deleted docs). Each source segment's FloatVectorValues is mmap-backed — vectorValue() triggers page faults served from OS page cache. copy() clones each segment's mmap slice, giving each BP thread an independent cursor.
* ReorderAwareFlatVectorsWriter: A copy of Lucene99FlatVectorsWriter that exposes getMetaOutput() and getVectorDataOutput() for direct stream access. Uses standard codec headers (Lucene99FlatVectorsFormatMeta / Lucene99FlatVectorsFormatData). Flush fields go through the normal addField()/writeField() path. Reordered merge fields bypass mergeOneField() entirely — the orchestrator writes vectors and metadata directly into the shared streams. finish() writes the end-of-fields sentinel and codec footers as normal.
* ReorderedFieldMetaWriter: Writes reordered field metadata into the .vemf IndexOutput. Builds oldOrd2New from the permutation, creates (docId, reorderedOrd) pairs sorted by docId ascending, writes the doc→ord skip list via FixedBlockSkipListIndexBuilder. For sparse fields, fills sentinel values for docs without vectors. Then writes the ord→doc int array for support subsequent merges (to retrieve the original doc id for each ord).
* ReorderedKNNFloatVectorValues: A KNNVectorValues<float[]> wrapper consumed by NativeIndexWriter.mergeIndex(). Iterates in permuted order so the faiss HNSW graph is built with vectors in their final disk positions. The faiss ordToDocs ID map is populated as ordToDocs[faissOrd] = mergedOrdToDocId[permutation[faissOrd]].
* ordToDoc for Re-Merged Reordered Segments: When a reordered segment is used as a source in a subsequent merge, MergeOrdMappingBuilder calls segValues.ordToDoc(localOrd) to map vector positions back to doc IDs. In a reordered segment, vectors are stored in permuted order, so ordToDoc(ord) must return the correct doc ID — not identity. The ord→doc array is stored in .vemf at write time and read into the FieldEntry record by UnifiedFlatVectorsReader. DenseOffHeapVectorValues.ordToDoc() and SparseOffHeapVectorValues.ordToDoc() both perform a O(1) direct array lookup. For non-reordered segments, ordToDoc returns identity.



Reordered .vemf Field Format

A single .vemf file can contain both standard (flush) and reordered (merge) fields, distinguished by the FieldInfo attribute rather than the codec header:

[Standard Lucene99FlatVectorsFormatMeta codec header]

Field 0 (flush, standard — no knn_reordered attribute):
fieldNumber, encoding, similarity, offset, length, dimension
count (int), OrdToDocDISI config

Field 1 (merge, reordered — knn_reordered = "true"):
fieldNumber, encoding, similarity, offset, length, dimension
isDense (byte), maxDoc (int), numLevel (int), numDocsForGrouping (int), groupFactor (int)
FixedBlockSkipListIndex body (doc→ord, with sentinel for sparse docs without vectors)
[Reverse mapping] ordToDocCount (int), ordToDoc[0..count-1] (int each)

sentinel: -1
[Standard codec footer]



5.2.4 Lucene Codec Changes

Since reordering updates the internal ordinals (representing location) we must update the following files:

* Files modified after reorder

File	Before Reorder	After reorder
.vec	Vectors in doc-ID order, standard header	Vectors in permuted order, standard header
.vemf	Standard header, field metadata	Standard header, metadata + skip list + ord→doc array
.faiss	HNSW neighbor lists reference original ordinals	Built in final order, no remapping needed

ReorderOrdMap: Maps between original and reordered ordinals:

* newOrd2Old[i]: position i in reordered .vec → original ordinal. Used when writing reordered vectors.
* oldOrd2New[j]: original ordinal j → position in reordered file. Used when building the skip list.

Memory: 2 × 4 × N bytes.

Dense case (FixedBlockSkipListIndex): After reordering, vector ordinals no longer match doc IDs. The skip list provides O(1) lookup from doc ID to reordered ordinal, stored inline in .vemf. Format is a fixed-width encoding where each doc ID maps to its ordinal using bit packing for efficiency.

Ord→Doc array: Stored after the skip list in .vemf. Simple int array: ordToDoc[newOrd] = docId. Used by ordToDoc() when this segment is a source in a subsequent merge. Cost: 4N bytes.

Sparse case: Not every doc has a vector. After reordering, the mapping chain is: docId → skipList.skipTo(doc) → reorderedOrd → seek(reorderedOrd * vecSize) → vector. The skip list must distinguish docs with vectors from docs without.

Writer (ReorderedFieldMetaWriter): Detects sparse vs dense via isDense = (n == maxDoc + 1). For sparse fields, fills sentinel values in the skip list for docs without vectors. The sentinel is (1 << (8 * numBytesPerValue)) - 1 — the max value representable in the skip list's byte width. Docs with vectors get their reordered ord; docs without get the sentinel.

Reader (UnifiedFlatVectorsReader): Reads the isDense byte from .vemf. getFloatVectorValues() dispatches:

* Dense → ReorderedOffHeapFloatVectorValues111.load() (numVectors = maxDoc + 1)
* Sparse → ReorderedOffHeapFloatVectorValues111.loadSparse() (numVectors = ordToDocMap.length, separate maxDoc)

Vector values (SparseOffHeapVectorValues):

* ordToDoc(ord) → direct array lookup from the stored ord→doc map
* iterator().nextDoc() → scans forward, calling skipList.skipTo(doc) and skipping docs where getOrd() == sentinelOrd (no vector)
* iterator().index() → skipList.skipTo(doc).getOrd() — returns the reordered ord for the current doc
* getAcceptOrds(acceptDocs) → wraps acceptDocs to translate through ordToDocMap

Reader auto-detection

NativeEngines990KnnVectorsFormat.fieldsReader() tries readers in order:


// 1. Unified reader: standard headers, per-field dispatch via knn_reordered attribute
//    Handles reordered segments and non-reordered segments
try {
UnifiedFlatVectorsReader unified = new UnifiedFlatVectorsReader(state, scorer);
return new NativeEngines990KnnVectorsReader(state, unified);
} catch (CorruptIndexException | NullPointerException e) {
// Not standard headers — try legacy reader
}

// 2. Standard Lucene reader: older non-reordered segments
return new NativeEngines990KnnVectorsReader(state, flatVectorsFormat.fieldsReader(state));


UnifiedFlatVectorsReader reads standard codec headers and dispatches per-field:


boolean isReordered = "true".equals(info.getAttribute("knn_reordered"));
FieldEntry entry = isReordered
? FieldEntry.createReordered(meta, info)   // skip list + ord→doc array
: FieldEntry.createStandard(meta, info);   // OrdToDocDISI


Design: Strategy Interface

public interface VectorReorderStrategy {
int[] computePermutation(FloatVectorValues vectors, int numThreads,
VectorSimilarityFunction similarityFunction);
}


Implementations:

* BipartiteReorderStrategy — wraps BpVectorReorderer, mmap-friendly, zero heap copy
* KMeans is also implemented but was decided against due to poor indexing and merge performance during k-means computation.

Strategy is selected in BasePerFieldKnnVectorsFormat.getReorderStrategy() from dynamic index settings, passed through NativeEngines990KnnVectorsFormat → NativeEngines990KnnVectorsWriter. Right now only “none” and “bp” are present.


5.2.5 Backward Compatibility

* Reordered segments use standard codec headers with knn_reordered FieldInfo attribute. The UnifiedFlatVectorsReader dispatches per-field based on this attribute.
    * Use the fieldinfo to guard against bad reads on the .vemf file.
* Non-reordered segments use standard codec headers with no knn_reordered attribute. Handled by both UnifiedFlatVectorsReader (standard field path) and Lucene99FlatVectorsReader (final fallback).
* Mixed merges: A merge can have source segments from any combination of the above. MergeOrdMappingBuilder handles all cases via ordToDoc() — identity for non-reordered and array lookup for reordered.

5.2.6 Search Path (Reader Side)

No changes to the query path or search API. The reorder is transparent to search — the reader detects whether a segment was reordered and handles the docId→vector position translation automatically.

For reordered segments: The doc→ord skip list translates docId → physical vector position on every vectorValue(docId) call. The HNSW graph already has correct neighbor IDs (built in final order), so graph traversal reads vectors in spatially-local order from the .vec file.

For non-reordered segments: Standard Lucene path. ordinal == doc ID.


Test Plan

* Unit tests for each component (MergeOrdMappingBuilder, MergedRandomAccessFloatVectorValues, ReorderAwareFlatVectorsWriter, ReorderedFieldMetaWriter, ReorderedKNNFloatVectorValues)
* Integration tests
    * Non-compound file format test: Index >10k vectors, force merge, verify reorder + search correctness.
    * Compound file format test: Index <10k vectors, verify reorder + search correctness.
    * Mixed reorder test: One reordered segment, one not reordered, verify search correctness.
    * Two-index comparison: Same data indexed with reorder and without reorder, compare search results.
    * Tiered merge test: Multiple flush batches, verify correctness after intermediate and final merges.
    * Above tests across all supported mappings (8x, 16x, 32x compression, bbq/bbqflat).
* Backwards compatibility tests
    * Assert that pre-reorder indices when upgraded can still flush, merge, and search.
    * Mixed cluster test: reordered and legacy segments coexist.

Correctness Invariants

For any reordered segment, the following must hold:

1. Permutation is a valid bijection: newOrd2Old[] contains each value in [0, N) exactly once.
2. Skip list is a bijection: For every doc ID with a vector, skipTo(docId).getOrd() returns a unique ord in [0, N).
3. Vector content preserved: vectorValue(skipList.getOrd(docId)) == original vector for docId.
4. Permutation and skip list are consistent inverses: For every docId, newOrd2Old[skipList.getOrd(docId)] == docId (dense case).



Performance Results

Cumulative indexing time of primary shards,,131.44308333333333,min
Min cumulative indexing time across primary shards,,0.00023333333333333333,min
Median cumulative indexing time across primary shards,,25.374933333333335,min
Max cumulative indexing time across primary shards,,28.21855,min
Cumulative indexing throttle time of primary shards,,0,min
Min cumulative indexing throttle time across primary shards,,0,min
Median cumulative indexing throttle time across primary shards,,0.0,min
Max cumulative indexing throttle time across primary shards,,0,min
Cumulative merge time of primary shards,,400.7009,min


