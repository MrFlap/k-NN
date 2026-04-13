# Feature

There is currently reordering code present in the k-NN plugin. However , it is implemented as a separate task, accessible through the do_reorder_bp.sh script. 

There is detailed information about reordering present at /Users/finnrobl/Documents/k-NN-2/vector-reorder/PORT_REORDER_TO_KNN.md . Read that.

We now need to implement reordering during the segment creation process.

The contract should be:
- When a segment is made searchable, reordering should apply.
- Reordering should apply to every segment with size greater than 10k vectors.
- Reordering should use a variable number of CPU cores , currently just using a default constant. 

Current thought: 
Put it in the vectors flush path, as opposed to the merge path or the refresh path.

Please fill out an implementation plan. Start by going through the flush, merge, and refresh pipelines (may also need to look at the OpenSearch repo under /Users/finnrobl/Documents/OpenSearch).

Put information about each of the entrypoints of this function and when they are triggered in the lifetime of an opensearch cluster (e.g. after indexing but before search, during force merge, etc).

Put all of this information in the document, and then figure out additional tasks for integration. We will do bipartite reordering for now, but make it pluggable so I can do kmeans reordering instead.

Also include test plans.

---

## Pipeline Analysis: Flush, Merge, and Refresh in OpenSearch + k-NN

### 1. Flush Pipeline

**When triggered:** When the in-memory indexing buffer (Lucene's `IndexWriter` buffer) is written to a new on-disk segment. This happens:
- Automatically when the indexing buffer exceeds `indices.memory.index_buffer_size` (default 10% of heap)
- Explicitly via the `_flush` API
- Before a `_refresh` if there are buffered docs
- During shard recovery

**OpenSearch core entrypoint:**
- `IndexShard.flush(FlushRequest)` → `InternalEngine.flush(boolean force, boolean waitIfOngoing)` → Lucene `IndexWriter.flush()` → triggers codec's `KnnVectorsWriter.flush()`

**k-NN entrypoint:**
- `NativeEngines990KnnVectorsWriter.flush(int maxDoc, Sorter.DocMap sortMap)`
  - Iterates over each `NativeEngineFieldVectorsWriter` (one per knn_vector field)
  - For each field: trains quantization if needed, then calls `NativeIndexWriter.flushIndex()`
  - `NativeIndexWriter.flushIndex()` → `buildAndWriteIndex()` → writes `.faiss` / engine file via `NativeIndexBuildStrategy.buildAndWriteIndex()`
  - The `flatVectorsWriter.flush()` call writes the `.vec` flat vectors file (Lucene99FlatVectorsWriter)
  - Stat tracked: `KNNGraphValue.REFRESH_TOTAL_TIME_IN_MILLIS`

**Segment lifecycle context:** After flush, the segment exists on disk but is NOT yet searchable. It becomes searchable after a refresh opens a new `IndexSearcher` over the latest segments.

**Typical segment sizes at flush:** Small-to-medium. Depends on buffer size and indexing rate. Often 1k–100k+ vectors per flush depending on workload.

### 2. Merge Pipeline

**When triggered:** When Lucene's `MergePolicy` (typically `TieredMergePolicy`) decides to combine multiple smaller segments into a larger one. This happens:
- Automatically in the background via `ConcurrentMergeScheduler` after flushes
- Explicitly via `_forcemerge` API (force merge to N segments)
- During shard recovery

**OpenSearch core entrypoint:**
- Background: `InternalEngine.EngineMergeScheduler` (extends `OpenSearchConcurrentMergeScheduler`) runs merges on dedicated threads
- Explicit: `IndexShard.forceMerge(ForceMergeRequest)` → `InternalEngine.forceMerge()` → Lucene `IndexWriter.forceMerge()`
- Lucene calls codec's `KnnVectorsWriter.mergeOneField()` for each knn_vector field in the merged segment

**k-NN entrypoint:**
- `NativeEngines990KnnVectorsWriter.mergeOneField(FieldInfo fieldInfo, MergeState mergeState)`
  - Merges flat vectors via `flatVectorsWriter.mergeOneField()` (writes merged `.vec`)
  - Reads merged vector values from all source segments
  - Trains quantization if needed, then calls `NativeIndexWriter.mergeIndex()`
  - `NativeIndexWriter.mergeIndex()` → `buildAndWriteIndex()` → writes new `.faiss` engine file
  - Stat tracked: `KNNGraphValue.MERGE_TOTAL_TIME_IN_MILLIS`

**Segment lifecycle context:** Merge creates a new, larger segment from multiple existing segments. The old segments are deleted after the merge completes and a refresh makes the new segment visible.

**Typical segment sizes at merge:** Medium-to-large. Tiered merge policy creates segments up to `max_merged_segment` (default 5GB). Force merge can create very large segments (entire index in one segment).

### 3. Refresh Pipeline

**When triggered:** When a new `IndexSearcher` is opened over the latest committed segments, making recently flushed/merged segments searchable. This happens:
- Automatically every `index.refresh_interval` (default 1 second)
- Explicitly via `_refresh` API
- Implicitly before search if `?refresh=true`

**OpenSearch core entrypoint:**
- `IndexShard.refresh(String source)` → `InternalEngine.refresh(String source)` → Lucene `ReferenceManager.maybeRefresh()` → opens new `DirectoryReader`
- Refresh listeners are notified: `RefreshListeners`, `CheckpointRefreshListener`, `RemoteStoreRefreshListener`

**k-NN entrypoint:**
- Refresh does NOT invoke any k-NN codec writer methods. It only opens a new reader over existing segments.
- k-NN's `NativeEngines990KnnVectorsReader` is instantiated when a new `DirectoryReader` opens a segment for the first time. This loads the `.faiss` index into the native memory cache.

**Segment lifecycle context:** Refresh makes segments searchable. No new segment files are written during refresh.

### Summary Table

| Pipeline | Trigger | Writes new segment files? | k-NN codec method | Typical vector count | Searchable after? |
|----------|---------|--------------------------|-------------------|---------------------|-------------------|
| Flush | Buffer full, explicit `_flush` | Yes (new segment) | `NativeEngines990KnnVectorsWriter.flush()` | 1k–100k+ | After next refresh |
| Merge | Background merge policy, `_forcemerge` | Yes (merged segment) | `NativeEngines990KnnVectorsWriter.mergeOneField()` | 10k–millions | After next refresh |
| Refresh | Timer (1s default), explicit `_refresh` | No | None (reader only) | N/A | Immediately |

---

## Recommended Integration Point (RESOLVED 2026-02-25)

**Merge path only**, via `NativeEngines990KnnVectorsWriter.finish()`.

**Approach: Post-write reorder in `finish()` after `.vec`/`.vemf` footers are flushed to disk**

The reorder runs inside `finish()`, which is called by `KnnVectorsWriter.merge()` after all
`mergeOneField()` calls complete. The sequence:

1. `mergeOneField()` writes `.vec` data and `.faiss` normally, marks fields >= 10k vectors
   in `fieldsToReorder`
2. `finish()` calls `flatVectorsWriter.finish()` → writes `.vec`/`.vemf` codec footers
3. `finish()` calls `flatVectorsWriter.close()` → flushes `IndexOutput` buffers to disk
4. For each field in `fieldsToReorder`, `SegmentReorderService.reorderSegmentFiles()`:
   a. Opens finalized `.vec` via `Lucene99FlatVectorsReader` (mmap-backed `FloatVectorValues`)
   b. Computes BP permutation — zero-heap-copy, only `2*4*N` bytes for metadata arrays
   c. Rewrites `.vec` + `.vemf` via `ReorderedFlatVectorsWriter` (reordered layout with skip list)
   d. Rewrites `.faiss` via `FaissIndexReorderTransformer` (remapped HNSW neighbor lists)
   e. Atomic rename of reordered files over originals

**Why `finish()` and not `mergeOneField()`:**
During `mergeOneField()`, the `.vec`/`.vemf` files are open `IndexOutput` streams without
finalized codec footers. `Lucene99FlatVectorsReader` validates footers on open via
`CodecUtil.checkFooter()` / `CodecUtil.retrieveChecksum()`, failing with
`CorruptIndexException: checksum status indeterminate: remaining=0`.
Footers are only written by `Lucene99FlatVectorsWriter.finish()`. Additionally,
`IndexOutput` buffers must be flushed via `close()` before the data is readable on disk.

**Why not `KNN80CompoundFormat.write()`:**
Compound file creation is skipped for large force-merged segments (`compound=False`).
`finish()` runs for ALL merges regardless of compound file creation.

**Why not flush path:**
Flush segments are typically small. They get merged later, at which point reordering applies.
The `fieldsToReorder` list is only populated by `mergeOneField()`, not `flush()`.

**Why not refresh path:**
Refresh does not write segment files; it only opens readers.

---

## Implementation Plan

### TODO 1: Define `VectorReorderStrategy` interface (pluggable reordering)

Create a strategy interface so BP and KMeans (and future strategies) are interchangeable.
The interface accepts `FloatVectorValues` (not `float[][]`) so implementations can work with
mmap-backed vectors without loading everything into heap.

**File:** `k-NN/src/main/java/org/opensearch/knn/memoryoptsearch/faiss/reorder/VectorReorderStrategy.java`

```java
public interface VectorReorderStrategy {
    /**
     * Compute a permutation array mapping new ord -> old ord.
     * @param vectors FloatVectorValues — may be mmap-backed (from Lucene99FlatVectorsReader)
     *                or heap-backed (from FloatVectorValues.fromFloats())
     * @param numThreads number of CPU threads to use
     * @return permutation array where permutation[newOrd] = oldOrd
     */
    int[] computePermutation(FloatVectorValues vectors, int numThreads);
}
```

### TODO 2: Implement `BipartiteReorderStrategy` (mmap-friendly)

Wrap Lucene's `BpVectorReorderer` behind the new interface. The key insight is that
`BpVectorReorderer.computeValueMap()` accepts `FloatVectorValues` directly — it does NOT
require all vectors in a `float[][]` on heap. It accesses vectors via random-access
`vectorValue(int ord)` calls, and each thread gets its own `vectors.copy()` (another mmap view).

**Memory overhead:** Only `2 * 4 bytes * numVectors` for the `sortedIds` + `biases` arrays
(checked by `BpVectorReorderer.docRAMRequirements()`). The vectors themselves stay on disk
via mmap.

**File:** `k-NN/src/main/java/org/opensearch/knn/memoryoptsearch/faiss/reorder/bpreorder/BipartiteReorderStrategy.java`

```java
public class BipartiteReorderStrategy implements VectorReorderStrategy {
    @Override
    public int[] computePermutation(FloatVectorValues vectors, int numThreads) {
        // BpVectorReorderer reads vectors via vectorValue(ord) — works with mmap-backed FloatVectorValues
        // See: k-NN/src/main/java/org/apache/lucene/misc/index/BpVectorReorderer.java
        //   - PerThreadState calls vectors.copy() per thread (line ~107) — each gets own mmap view
        //   - Hot loop: vectors.vectorValue(ids[i]) (line ~370 in ComputeBiasTask.compute())
        //   - RAM check: docRAMRequirements() only accounts for int[] + float[] metadata, not vectors
        BpVectorReorderer reorderer = new BpVectorReorderer("vectors");
        reorderer.setMinPartitionSize(1);
        ForkJoinPool pool = new ForkJoinPool(numThreads);
        try {
            TaskExecutor executor = new TaskExecutor(pool);
            Sorter.DocMap map = reorderer.computeValueMap(
                vectors, VectorSimilarityFunction.EUCLIDEAN, executor
            );
            int[] permutation = new int[vectors.size()];
            for (int i = 0; i < permutation.length; i++) {
                permutation[i] = map.newToOld(i);
            }
            return permutation;
        } finally {
            pool.shutdown();
        }
    }
}
```

**Contrast with current `BpReorderer` (heap-based):**
The existing `bpreorder/BpReorderer.java` forces all vectors into heap:
```java
// Current approach — loads ALL vectors into Java heap:
FloatVectorValues fvv = FloatVectorValues.fromFloats(Arrays.asList(vectors), dim);
```
The new strategy avoids this by passing the `FloatVectorValues` from `Lucene99FlatVectorsReader`
directly, which is backed by mmap'd `.vec` file on disk.

### TODO 3: Implement `KMeansReorderStrategy`

Wrap existing `ClusterSorter` behind the new interface.

**File:** `k-NN/src/main/java/org/opensearch/knn/memoryoptsearch/faiss/reorder/kmeansreorder/KMeansReorderStrategy.java`

- Delegates to existing `ClusterSorter.clusterAndSort()`
- Uses `FaissKMeansService` (JNI) when available, falls back to `KMeansClusterer` (pure Java)
- Note: KMeans currently requires `float[][]` in heap (JNI and pure-Java both need contiguous
  vector data). Future optimization: stream vectors through JNI in batches.

### TODO 4: Create `SegmentReorderService`

Orchestrator that decides whether to reorder and applies the reorder to `.vec` and `.faiss` files.

**File:** `k-NN/src/main/java/org/opensearch/knn/memoryoptsearch/faiss/reorder/SegmentReorderService.java`

Responsibilities:
- Check if segment size > 10k vectors threshold (configurable constant)
- Open a `Lucene99FlatVectorsReader` on the just-written `.vec` file to get mmap-backed `FloatVectorValues`
- Call `VectorReorderStrategy.computePermutation(floatVectorValues, numThreads)` — no heap copy for BP
- Build `ReorderOrdMap` from the permutation
- Rewrite `.vec` file in reordered order using `ReorderedFlatVectorsWriter`
- Rewrite `.faiss` file using `FaissIndexReorderTransformer` + the appropriate `FaissIndexReorderer` (HNSW, flat, etc.)
- Write the docid-to-ord skip list index for reordered access

Key constants:
```java
private static final int MIN_VECTORS_FOR_REORDER = 10_000;
private static final int DEFAULT_REORDER_THREADS = 4; // configurable
```

### TODO 5: ~~Integrate into flush path~~ SKIPPED

Flush is NOT reordered. Flush segments are typically small and will be merged later.
The `fieldsToReorder` list is only populated by `mergeOneField()`, not `flush()`.

### TODO 6: Integrate into merge path via `finish()` ✅ DONE

**File modified:** `NativeEngines990KnnVectorsWriter.java`

The reorder cannot happen inside `mergeOneField()` because `.vec`/`.vemf` files don't have
finalized codec footers yet (see Error 1 in Manual Testing Results). Instead:

1. `mergeOneField()` marks fields >= 10k vectors in `fieldsToReorder`
2. `finish()` calls `flatVectorsWriter.finish()` → writes `.vec`/`.vemf` footers
3. `finish()` calls `flatVectorsWriter.close()` → flushes `IndexOutput` buffers to disk
4. `finish()` iterates `fieldsToReorder` and calls `SegmentReorderService.reorderSegmentFiles()`
   for each, which rewrites `.vec` + `.vemf` + `.faiss`

```java
// In NativeEngines990KnnVectorsWriter.mergeOneField(), at the end:
if (reorderStrategy != null && totalLiveDocs >= SegmentReorderService.MIN_VECTORS_FOR_REORDER) {
    fieldsToReorder.add(fieldInfo);
}

// In NativeEngines990KnnVectorsWriter.finish():
flatVectorsWriter.finish();  // writes footers
flatVectorsWriter.close();   // flushes to disk
for (FieldInfo fieldInfo : fieldsToReorder) {
    SegmentReorderService reorderService = new SegmentReorderService(
        segmentWriteState, fieldInfo, reorderStrategy
    );
    reorderService.reorderSegmentFiles();  // rewrites .vec + .vemf + .faiss
}
```

**Call chain during merge:**
```
SegmentMerger.mergeVectorValues()
  → try (KnnVectorsWriter writer = codec.fieldsWriter(state)) {
        writer.merge(mergeState);
     }
     // merge() internally:
     //   1. mergeOneField() per field  → writes .vec data, .faiss; marks fieldsToReorder
     //   2. finish()                   → writes .vec/.vemf footers
     //                                 → closes flatVectorsWriter (flushes to disk)
     //                                 → reorders .vec + .vemf + .faiss  ← HERE
     // try-with-resources:
     //   3. close()                    → cleanup (flatVectorsWriter already closed)
```

**Force merge note:** A `_forcemerge?max_num_segments=1` on a 10M vector index will:
1. Stream all vectors from source segments → write merged `.vec` (disk-to-disk, no heap)
2. Build FAISS HNSW graph in native memory
3. `finish()` writes footers, closes streams
4. Reorder: open merged `.vec` via mmap, compute BP permutation (~80MB heap for metadata),
   rewrite `.vec`, `.vemf`, and `.faiss` with permutation

### TODO 7: Wire up strategy selection ✅ DONE

The `VectorReorderStrategy` instance is selected via dynamic index settings and passed through
the codec stack:

1. `KNNSettings.java`: Two new dynamic index settings:
   - `index.knn.advanced.reorder_strategy` — string: `bp` (default), `kmeans`, `none`
   - `index.knn.advanced.reorder_kmeans_num_clusters` — int, default 256, min 1
2. `BasePerFieldKnnVectorsFormat.getReorderStrategy()`: reads settings, creates strategy instance
3. `NativeEngines990KnnVectorsFormat` constructor → accepts strategy
4. `NativeEngines990KnnVectorsWriter` constructor → receives strategy

Settings are dynamic and index-scoped — can be changed on a live index, takes effect on next merge.

**Usage:**
```json
PUT /my-index
{
  "settings": {
    "index.knn": true,
    "index.knn.advanced.reorder_strategy": "kmeans",
    "index.knn.advanced.reorder_kmeans_num_clusters": 500
  }
}
```

### TODO 7b: Auto-detect similarity function ✅ DONE (2026-03-03)

The `VectorReorderStrategy` interface accepts `VectorSimilarityFunction` as a parameter:
```java
int[] computePermutation(FloatVectorValues vectors, int numThreads, VectorSimilarityFunction similarityFunction);
```

`SegmentReorderService` reads the similarity from `fieldInfo.getVectorSimilarityFunction()` and
passes it through. Each strategy uses it appropriately:
- **BP**: passes directly to `BpVectorReorderer.computeValueMap(vectors, similarityFunction, executor)`
- **KMeans**: maps `MAXIMUM_INNER_PRODUCT` and `COSINE` → `METRIC_INNER_PRODUCT`, else → `METRIC_L2`

No configuration needed — similarity is auto-detected from segment metadata.

### TODO 8: Handle `.faiss` file reordering

The existing reorder infrastructure already handles this via:
- `FaissIndexReorderTransformer` - reads `.faiss`, applies permutation, writes reordered `.faiss`
- `IndexTypeToFaissIndexReordererMapping` - maps FAISS index types to their reorderers
- `FaissHNSWIndexReorderer` / `FaissHnswReorderer` - reorders HNSW graph neighbor lists
- `FaissIndexFloatFlatReorderer` / `FaissIndexBinaryFlatReorderer` - reorders flat index data
- `FaissIdMapIndexReorderer` - reorders ID map wrapper

These work on files. During flush/merge, we need to:
1. Let the normal flush/merge write the `.faiss` file
2. Read it back, apply the reorder permutation, write a new reordered `.faiss`
3. Replace the original with the reordered version (atomic rename)

### TODO 9: Handle `.vec` file reordering

Use existing `ReorderedFlatVectorsWriter` which:
- Reads vectors from the original `.vec` via `ReorderedLucene99FlatVectorsReader`
- Writes them in permuted order to a new `.vec` file
- Handles the Lucene codec header/footer correctly

### TODO 10: Thread pool for reordering

Reordering (especially BP) is CPU-intensive. Options:
- Use a dedicated `ThreadPool.Names.KNN_REORDER` thread pool (new)
- Use the existing `GENERIC` thread pool
- Use `ForkJoinPool` with configurable parallelism (current BP approach)

For now: use `ForkJoinPool` with `DEFAULT_REORDER_THREADS` parallelism, matching the existing BP implementation. The thread count should be a constant that can be changed later to a setting.

---

## Test Plan

### Unit Tests

**TODO 11a: `BpReordererTests`** ✅ DONE
- Test `BpReorderer.computePermutation(float[][])` produces a valid permutation (all ords present, no duplicates)
- Test with small input (2 vectors), identical vectors
- Test that BP groups well-separated clusters (>90% same-cluster adjacency)

**TODO 11b: `BipartiteReorderStrategyTests`**
- Test `BipartiteReorderStrategy.computePermutation(FloatVectorValues, numThreads)` — the new mmap-friendly interface
- Create `FloatVectorValues` via `FloatVectorValues.fromFloats()` (heap-backed, simulates mmap contract)
- Verify valid permutation (all ords present, no duplicates)
- Verify clustering quality: interleaved 2-cluster input → >90% same-cluster adjacency after reorder
- Test with `numThreads=1` and `numThreads=4` — both produce valid permutations
- Edge case: exactly 2 vectors (minimum for BP)

**TODO 11c: `KMeansReorderStrategyTests`**
- Test `KMeansReorderStrategy.computePermutation(FloatVectorValues, numThreads)` — the new interface
- Create `FloatVectorValues` via `FloatVectorValues.fromFloats()`
- Verify valid permutation (all ords present, no duplicates)
- Verify clustering quality: interleaved 3-cluster input → >80% same-cluster adjacency
- Test with custom k and niter parameters
- Edge case: k > numVectors (should cap k at numVectors)
- Must set `kmeans.useJni=false` to avoid JNI dependency in unit tests

**TODO 11d: `ClusterSorterTests`** ✅ DONE
- Test `ClusterSorter.sortByCluster()` groups by cluster, sorts by distance within cluster
- Test `KMeansClusterer.cluster()` produces valid assignments and non-negative distances
- Test `ClusterSorter.clusterAndSort()` end-to-end with pure Java path

**TODO 11e: `ReorderOrdMapTests`** ✅ DONE
- Test newOrd2Old → oldOrd2New inversion
- Test round-trip: `oldOrd2New[newOrd2Old[i]] == i`
- Edge cases: single element, identity, reverse

**TODO 12: `SegmentReorderServiceTests`**
- Test that segments below 10k threshold are NOT reordered
- Test that segments at/above 10k threshold ARE reordered
- Test that reordered `.vec` file contains same vectors in different order
- Test that reordered `.faiss` file is valid and searchable
- Test that the docid-to-ord mapping is consistent after reorder

**TODO 13: `NativeEngines990KnnVectorsWriterReorderTests`**
- Extend existing `NativeEngines990KnnVectorsWriterMergeTests`:
  - Add test case: merge producing >10k vectors → verify reorder was applied (`.vec` rewritten
    with `ReorderedLucene99FlatVectorsReader111` codec, `.faiss` neighbor lists remapped)
  - Add test case: merge producing <10k vectors → verify reorder was NOT applied
  - Add test case: merge with `reorderStrategy=null` → verify reorder was NOT applied
- Flush tests: verify flush does NOT trigger reorder regardless of vector count

### Integration Tests

**TODO 14: End-to-end reorder during force merge**
- Index >10k vectors into a k-NN index across multiple flushes
- Run `_forcemerge?max_num_segments=1`
- Verify the merged segment's `.vec` file is reordered (uses `ReorderedLucene99FlatVectorsReader111` codec)
- Verify the merged segment's `.faiss` file has remapped neighbor lists
- Run k-NN search queries and verify correct results (recall is maintained)
- Verify reorder works even when final segment has `compound=False`

**TODO 15: End-to-end reorder during background merge**
- Index vectors continuously to trigger background merges via `TieredMergePolicy`
- Verify that merged segments above 10k vectors are reordered
- Verify that merged segments below 10k vectors are NOT reordered
- Verify search correctness throughout

**TODO 16: Strategy pluggability test**
- Configure index with BP reorder strategy → verify BP is used
- Configure index with KMeans reorder strategy → verify KMeans is used
- Configure index with "none" → verify no reordering

**TODO 19: `SegmentReorderService` integration test (manual / `./gradlew run`)**

This tests the full orchestration: flush/merge → reorder → search. Requires a running OpenSearch
instance with the k-NN plugin because `SegmentReorderService` depends on real Lucene segment files
(`.vec`, `.faiss`, `.vemf`) written by the codec.

**Setup:**
```bash
# Copy non-reordered segments before starting
cp -r /Users/finnrobl/Documents/k-NN-2/index-backups/* /Users/finnrobl/Documents/k-NN-2/e2e_data/

# Start OpenSearch with k-NN plugin
cd /Users/finnrobl/Documents/k-NN-2/k-NN
./gradlew run -Ddata=/Users/finnrobl/Documents/k-NN-2/e2e_data
```

**Test steps:**
1. Create a k-NN index with reorder enabled (BP strategy)
2. Index >10k vectors (e.g., 50k from sift dataset)
3. Call `_forcemerge?max_num_segments=1` → triggers `mergeOneField()` → marks field → `finish()` reorders
4. Verify segment files were rewritten:
   - `.vec` file: open with `ReorderedLucene99FlatVectorsReader111`, verify vectors accessible via ordMap
   - `.faiss` file: load with `FaissIndex.load()`, verify HNSW graph neighbor lists reference valid ords
5. Run k-NN search queries, verify results match non-reordered baseline (same recall@10)
6. Verify that the final force-merged segment IS reordered even when `compound=False`

**Validation checks:**
- `ReorderOrdMap` round-trip: for every vector, `oldOrd2New[newOrd2Old[i]] == i`
- Vector content preserved: every vector in the reordered `.vec` exists in the original (just different order)
- FAISS graph valid: all neighbor IDs in HNSW graph are in range `[0, numVectors)`
- Search correctness: recall@10 identical to non-reordered index
- No orphaned `.reorder` temp files left in segment directory

### Performance Tests

**TODO 17: Reorder overhead benchmarks**
- Measure flush latency with and without reordering for various segment sizes (10k, 100k, 500k, 1M vectors)
- Measure merge latency with and without reordering
- Measure search latency improvement from reordering (p50, p99)
- Use existing OSB benchmark infrastructure (`do_reorder_bp.sh` pattern)

### Correctness Tests

**TODO 18: Verify search recall is preserved**
- Index a known dataset (e.g., sift-1M subset)
- Run exact k-NN search (brute force) to get ground truth
- Run approximate k-NN search on reordered index
- Verify recall@10 is identical to non-reordered index (reordering should not affect recall, only latency)

---

## Manual Testing Results (2026-02-24)

### Error 1: Cannot read `.vec`/`.vemf` during flush — files not yet finalized

**When:** `SegmentReorderService.computePermutationFromVecFile()` tries to open the `.vemf` file
with `Lucene99FlatVectorsReader` during `flush()`.

**Error:**
```
org.apache.lucene.index.CorruptIndexException: checksum status indeterminate: remaining=0;
please run checkindex for more details
(resource=BufferedChecksumIndexInput(MemorySegmentIndexInput(
  path=".../_0_NativeEngines990KnnVectorsFormat_0.vemf")))
    at org.apache.lucene.codecs.CodecUtil.checkFooter(CodecUtil.java:483)
    at org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsReader.readMetadata(Lucene99FlatVectorsReader.java:112)
    at org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsReader.<init>(Lucene99FlatVectorsReader.java:69)
    at o.o.knn.memoryoptsearch.faiss.reorder.SegmentReorderService.computePermutationFromVecFile(SegmentReorderService.java:126)
    at o.o.knn.memoryoptsearch.faiss.reorder.SegmentReorderService.reorderSegmentFiles(SegmentReorderService.java:87)
    at o.o.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsWriter.maybeReorderSegmentFiles(NativeEngines990KnnVectorsWriter.java:158)
    at o.o.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsWriter.flush(NativeEngines990KnnVectorsWriter.java:148)
    at org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat$FieldsWriter.flush(PerFieldKnnVectorsFormat.java:120)
    at org.apache.lucene.index.VectorValuesConsumer.flush(VectorValuesConsumer.java:76)
    at org.apache.lucene.index.IndexingChain.flush(IndexingChain.java:305)
```

**Root cause:** During `flush()`, the `.vec` and `.vemf` files are written by `flatVectorsWriter.flush()`
but their codec footer has not been written yet (Lucene writes the footer later in the flush pipeline).
`Lucene99FlatVectorsReader` validates the footer checksum on open, which fails because the file is
incomplete.

**Additionally:** Even if we could read the files, writing `.reorder` temp files into the segment
directory causes Lucene's compound file writer (`Lucene90CompoundFormat.write()`) to pick them up
and fail with:
```
CorruptIndexException: compound sub-files must have a valid codec header and footer:
file is too small (0 bytes) (resource=...vemf.reorder)
```

### Manual Test: Merge Path (2026-02-24 13:21)

**Setup:** 15k vectors in 3 clusters (interleaved), 3 segments, force merge to 1.

**Result:** Force merge succeeded. BP permutation computed in 187ms. Search returns results without errors.

**Problem:** Pre-write reorder approach (remapping vectors in `KNNVectorValues` supplier) breaks
the doc-ID-to-vector correspondence. Search returns wrong doc IDs because the FAISS graph stores
reordered vectors under original doc IDs.

**Correct approach (from `ReorderAllWithBP` script):**
The original script rewrites BOTH `.vec` and `.faiss` files post-write:
1. Read vectors from finalized `.vec` via `Lucene99FlatVectorsReader`
2. Compute BP permutation
3. Rewrite `.vec` via `ReorderedFlatVectorsWriter` — vectors stored in BP order, new codec header
   (`ReorderedLucene99FlatVectorsFormatMeta`) with ordMap baked into `.vemf` metadata
4. Rewrite `.faiss` via `FaissIndexReorderTransformer` — HNSW neighbor lists remapped
5. Replace original files

The contract:
- Doc IDs in FAISS graph: **unchanged**
- HNSW neighbor lists: **remapped** to point to reordered ords
- `.vec` vector storage: **reordered** (BP order)
- `.vemf` metadata: **new codec** with ordMap so `ReorderedLucene99FlatVectorsReader111` can translate

**Integration point for merge: `KNN80CompoundFormat.write()`**

This is the right hook because:
- It runs AFTER `SegmentMerger.merge()` completes (all `mergeOneField()` calls done, files finalized with footers)
- It runs BEFORE the compound file is created
- It already iterates over engine files (`.faiss`) and copies them out of the compound file
- We can insert the reorder step here: rewrite `.vec`/`.vemf`/`.faiss` in-place before compound file creation

Lifecycle:
```
SegmentMerger.merge()
  → mergeOneField() per field  → writes .vec, .vemf, .faiss (all finalized)
IndexWriter.createCompoundFile()
  → KNN80CompoundFormat.write()  ← INSERT REORDER HERE (files are finalized, not yet compounded)
    → copies .faiss → .faissc
    → delegate writes remaining files into .cfs
```

### Issue: `KNN80CompoundFormat.write()` not always called (2026-02-24 13:46)

**Observation:** After `_forcemerge?max_num_segments=1` with 15k vectors:
- `_0` (flush, ~10k): not reordered → falls back to standard reader ✅
- `_1` (intermediate merge, 14245): reordered in `KNN80CompoundFormat.write()` ✅
- `_2` (flush, ~755): not reordered, below threshold ✅
- `_3` (final forcemerge of `_1`+`_2`, 15000): **NOT reordered** ❌

Segment `_3` has `compound=False` — Lucene skipped compound file creation for the final
force-merged segment. `KNN80CompoundFormat.write()` was never called for `_3`.

**Root cause:** Lucene's `IndexWriter` only creates compound files when `MergePolicy.useCompoundFile()`
returns true. For force merge, the final segment may not be compounded (depends on segment size
and merge policy settings).

**Resolution options:**
1. Move reorder back into `mergeOneField()` but use the post-write approach — after
   `writer.mergeIndex()` writes the `.faiss`, rewrite it in place. The `.vec`/`.vemf` can also
   be rewritten since the files are finalized at that point (footers written by
   `flatVectorsWriter.mergeOneField()`). Need to verify that `.vec`/`.vemf` footers are written
   by the time `mergeOneField()` returns.

2. Keep reorder in `KNN80CompoundFormat.write()` AND also add it to a post-merge hook. The
   `IndexWriter` calls `SegmentMerger.merge()` → then optionally `createCompoundFile()`. We need
   a hook that runs after merge regardless of compound file creation.

3. Use both hooks: `KNN80CompoundFormat.write()` for segments that get compounded, and
   `NativeEngines990KnnVectorsWriter.finish()` or a merge listener for non-compounded segments.

### Error 2: `AssertionError: got context=DEFAULT` during merge reorder (2026-02-25 11:33)

**When:** `SegmentReorderService.rewriteVecFile()` calls `ReorderedFlatVectorsWriter` constructor,
which calls `directory.createOutput(metaFileName, IOContext.DEFAULT)`.

**Error:**
```
java.lang.AssertionError: got context=DEFAULT
    at org.apache.lucene.index.ConcurrentMergeScheduler$1.createOutput(ConcurrentMergeScheduler.java:307)
    at org.apache.lucene.store.TrackingDirectoryWrapper.createOutput(TrackingDirectoryWrapper.java:41)
    at o.o.knn.memoryoptsearch.faiss.reorder.ReorderedFlatVectorsWriter.<init>(ReorderedFlatVectorsWriter.java:46)
    at o.o.knn.memoryoptsearch.faiss.reorder.SegmentReorderService.rewriteVecFile(SegmentReorderService.java:141)
    at o.o.knn.memoryoptsearch.faiss.reorder.SegmentReorderService.reorderSegmentFiles(SegmentReorderService.java:78)
    at o.o.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsWriter.finish(NativeEngines990KnnVectorsWriter.java:212)
```

**Root cause:** During merge, Lucene's `ConcurrentMergeScheduler` wraps the segment directory
with a delegate that asserts all `createOutput()` calls use `IOContext.MERGE`, not `IOContext.DEFAULT`.
`ReorderedFlatVectorsWriter` hardcodes `IOContext.DEFAULT` in its constructor (line 46 and 77).

**Resolution:** In `SegmentReorderService.reorderSegmentFiles()`, wrap the directory with a
`FilterDirectory` that overrides `createOutput()` to use `state.context` (which is `IOContext.MERGE`
during merge). This avoids modifying `ReorderedFlatVectorsWriter` itself:

```java
final Directory writeDir = new FilterDirectory(directory) {
    @Override
    public IndexOutput createOutput(String name, IOContext context) throws IOException {
        return in.createOutput(name, state.context);
    }
};
```

The `writeDir` is used for all `createOutput` calls (`.vec.reorder`, `.vemf.reorder`, `.faiss.reorder`),
while the original `directory` is used for reads (`openInput`), deletes, and renames.

### Manual Test: `finish()` reorder with IOContext fix (2026-02-25 11:37)

**Setup:** 15k vectors in 3 clusters (interleaved), 3 segments of 5k each, force merge to 1.

**Test strategy:**
1. Create k-NN index (HNSW/faiss, 128-dim, l2)
2. Index 15k vectors in 3 batches of 5k, flush between each → 3 segments
3. Search before force merge → capture baseline doc IDs and scores
4. `_forcemerge?max_num_segments=1` → triggers `mergeOneField()` → `finish()` → reorder
5. Search after force merge → verify same doc IDs and scores
6. Check logs for:
   - `[Reorder] Marked field [my_vector] for reorder (15000 vectors)` — field was marked
   - `[Reorder] Starting reorder for field [my_vector] in segment _3` — reorder started
   - `[Reorder] Permutation computed for 15000 vectors` — BP ran
   - `[Reorder] Completed reorder for field [my_vector]` — reorder finished
   - `[ReorderedReader] Falling back to standard reader` for flush segments _0, _1, _2
   - NO fallback for merged segment _3 → reordered reader used ✅

### Manual Test Result: `finish()` reorder with IOContext fix (2026-02-25 11:39) ✅ PASSED

**Segments before force merge:**
```
_0: 5000 docs, compound=True  (flush)
_1: 5000 docs, compound=True  (flush)
_2: 5000 docs, compound=True  (flush)
```

**Segments after force merge:**
```
_0: 5000 docs, compound=True   (flush, pending deletion)
_1: 5000 docs, compound=True   (flush, pending deletion)
_2: 5000 docs, compound=True   (flush, pending deletion)
_3: 15000 docs, compound=False (force-merged, REORDERED)
```

**Reorder log output:**
```
[Reorder] Marked field [my_vector] for reorder (15000 vectors)
[Reorder] Starting reorder for field [my_vector] in segment _3
[Reorder] Permutation computed for 15000 vectors, field [my_vector]
[Reorder] Completed reorder for field [my_vector] in 462 ms
```

**Reader log output:**
- Segments _0, _1, _2: `[ReorderedReader] Falling back to standard reader` (expected — flush segments, not reordered)
- Segment _3: `[ReorderedReader] Successfully opened REORDERED .vemf for segment _3` ✅

**Search results — BEFORE force merge:**
```
doc_id=8103,  score=0.048115
doc_id=13890, score=0.047268
doc_id=12219, score=0.047071
doc_id=12927, score=0.046295
doc_id=9924,  score=0.046064
```

**Search results — AFTER force merge (reordered):**
```
doc_id=8103,  score=0.048115
doc_id=13890, score=0.047268
doc_id=12219, score=0.047071
doc_id=12927, score=0.046295
doc_id=9924,  score=0.046064
```

**Verdict:** Doc IDs and scores are identical before and after reorder. Reorder was applied to
the force-merged segment `_3` (which has `compound=False`), confirming the `finish()` approach
works for non-compounded segments. The reordered reader (`ReorderedLucene99FlatVectorsReader111`)
is used for the merged segment, while flush segments correctly fall back to the standard reader.

---

## Design Implications (RESOLVED 2026-02-25)

The post-write reorder approach (rewrite `.vec`/`.faiss` after they're written) **cannot work
during `flush()` or `mergeOneField()`** because:
1. `.vec`/`.vemf` files don't have finalized footers yet → can't open with `Lucene99FlatVectorsReader`
2. `IndexOutput` buffers may not be flushed to disk → data not readable via `openInput()`
3. During flush, any new files in the segment directory get included in the compound file → corruption

**Resolution: Reorder in `finish()` (merge path only)**

The `finish()` method is called by `KnnVectorsWriter.merge()` after all `mergeOneField()` calls.
Inside `finish()`:
1. `flatVectorsWriter.finish()` writes `.vec`/`.vemf` codec footers
2. `flatVectorsWriter.close()` flushes `IndexOutput` buffers to disk
3. Files are now readable via `Lucene99FlatVectorsReader`
4. `SegmentReorderService.reorderSegmentFiles()` rewrites `.vec` + `.vemf` + `.faiss`

This avoids the `KNN80CompoundFormat.write()` approach which fails for non-compounded segments.

---

## Merge Path Reorder Design (RESOLVED 2026-02-25)

### Why `mergeOneField()` doesn't work

During `mergeOneField()`, the `.vec`/`.vemf` `IndexOutput` streams are still open without
finalized codec footers. `Lucene99FlatVectorsReader` validates footers on open, failing with
`CorruptIndexException: checksum status indeterminate: remaining=0`.

The footers are written by `Lucene99FlatVectorsWriter.finish()`, which is called by
`KnnVectorsWriter.merge()` AFTER all `mergeOneField()` calls complete.

### Why `KNN80CompoundFormat.write()` doesn't work

Compound file creation is skipped for large force-merged segments (`compound=False`).
The final segment in `_forcemerge?max_num_segments=1` is often not compounded.

### Solution: `finish()` with early close

The Lucene call chain during merge:
```
KnnVectorsWriter.merge(mergeState)
  → mergeOneField() per field     // writes .vec data + .faiss
  → finish()                      // writes footers
SegmentMerger try-with-resources
  → close()                       // closes streams
```

Our `NativeEngines990KnnVectorsWriter.finish()`:
1. Calls `flatVectorsWriter.finish()` — writes `.vec`/`.vemf` codec footers
2. Calls `flatVectorsWriter.close()` — flushes `IndexOutput` buffers to disk
3. For each field in `fieldsToReorder`: opens finalized `.vec` via `Lucene99FlatVectorsReader`,
   computes permutation, rewrites `.vec` + `.vemf` + `.faiss`

The early `close()` is safe because `finish()` is the last method that writes to the streams.
Our `close()` method handles the already-closed `flatVectorsWriter` gracefully via `IOUtils.close()`.

---

## File Change Summary

| File | Action | Description |
|------|--------|-------------|
| `VectorReorderStrategy.java` | NEW ✅ | Strategy interface — accepts `FloatVectorValues` + `VectorSimilarityFunction` |
| `BipartiteReorderStrategy.java` (bpreorder/) | NEW ✅ | BP implementation, mmap-friendly, auto-detects similarity |
| `KMeansReorderStrategy.java` (kmeansreorder/) | NEW ✅ | KMeans implementation, materializes to heap, auto-detects similarity |
| `SegmentReorderService.java` | NEW ✅ | Orchestrator: threshold check, permutation, full `.vec` + `.vemf` + `.faiss` rewriting |
| `NativeEngines990KnnVectorsWriter.java` | MODIFY ✅ | `fieldsToReorder` list in `mergeOneField()`, reorder loop in `finish()` after footer flush |
| `NativeEngines990KnnVectorsFormat.java` | MODIFY ✅ | Accept/pass reorder strategy to writer; reader tries reordered first |
| `BasePerFieldKnnVectorsFormat.java` | MODIFY ✅ | Reads reorder settings, creates strategy via `getReorderStrategy()` |
| `KNNSettings.java` | MODIFY ✅ | Added `index.knn.advanced.reorder_strategy` and `index.knn.advanced.reorder_kmeans_num_clusters` settings |
| `KNN80CompoundFormat.java` | MODIFY ✅ | Removed all reorder logic, restored to original compound-file-only handling |
| `BipartiteReorderStrategy.java` (root reorder/) | DELETED ✅ | Stale duplicate using old `float[][]` interface |
| `KMeansReorderStrategy.java` (root reorder/) | DELETED ✅ | Stale duplicate using old `float[][]` interface |


---

## Design Change: Move Reorder from CompoundFormat to mergeOneField (2026-02-25)

### Problem

Reordering was in `KNN80CompoundFormat.write()`, which only runs when Lucene creates compound
files. Large force-merged segments (e.g. `_forcemerge?max_num_segments=1`) skip compound file
creation entirely (`compound=False`), so they were never reordered.

### Solution

Moved reorder to `NativeEngines990KnnVectorsWriter.finish()`, which runs for ALL merges
regardless of compound file creation. The reorder happens after `flatVectorsWriter.finish()`
writes the `.vec`/`.vemf` codec footers and `flatVectorsWriter.close()` flushes them to disk.

### How it works

1. During `mergeOneField()`: if the field has >= 10k vectors and a reorder strategy is set,
   the field is added to `fieldsToReorder`
2. In `finish()`: `flatVectorsWriter.finish()` writes `.vec`/`.vemf` footers, then
   `flatVectorsWriter.close()` flushes the `IndexOutput` buffers to disk
3. For each field in `fieldsToReorder`, `SegmentReorderService.reorderSegmentFiles()`:
   a. Opens the finalized `.vec` via `Lucene99FlatVectorsReader` (mmap-backed)
   b. Computes BP permutation from the `FloatVectorValues`
   c. Rewrites `.vec` + `.vemf` via `ReorderedFlatVectorsWriter` (reordered vector layout
      with doc-to-ord skip list in metadata)
   d. Rewrites `.faiss` via `FaissIndexReorderTransformer` (remapped HNSW neighbor lists)
   e. Atomic rename of reordered files over originals

### Why finish() and not mergeOneField()

During `mergeOneField()`, the `.vec`/`.vemf` files are open `IndexOutput` streams without
finalized codec footers. `Lucene99FlatVectorsReader` validates the footer on open via
`CodecUtil.checkFooter()` and `CodecUtil.retrieveChecksum()`, which fails with:

```
CorruptIndexException: checksum status indeterminate: remaining=0
```

The footers are only written by `Lucene99FlatVectorsWriter.finish()`, which is called by
`KnnVectorsWriter.merge()` AFTER all `mergeOneField()` calls complete. So we must wait
until `finish()` to read the `.vec` file.

Additionally, `flatVectorsWriter.close()` must be called to flush the `IndexOutput` buffers
to disk before we can open the files with `Lucene99FlatVectorsReader`.

### Call chain during merge

```
SegmentMerger.mergeVectorValues()
  → try (KnnVectorsWriter writer = codec.fieldsWriter(state)) {
        writer.merge(mergeState);
     }
     // merge() internally:
     //   1. mergeOneField() per field  → writes .vec data, .faiss
     //   2. finish()                   → writes .vec/.vemf footers
     //                                 → closes flatVectorsWriter (flushes to disk)
     //                                 → reorders .vec + .vemf + .faiss  ← HERE
     // try-with-resources:
     //   3. close()                    → cleanup (flatVectorsWriter already closed)
```

### Files changed

- `NativeEngines990KnnVectorsWriter.java`: Added `fieldsToReorder` list populated in
  `mergeOneField()`, reorder loop in `finish()` after `flatVectorsWriter.finish()` + `close()`
- `KNN80CompoundFormat.java`: Removed all reorder logic, restored to original
- `SegmentReorderService.java`: Updated `reorderSegmentFiles()` to do full rewrite of
  `.vec` + `.vemf` + `.faiss` (was previously only rewriting `.faiss`)

### What about flush?

Flush is NOT reordered. The `fieldsToReorder` list is only populated by `mergeOneField()`,
not by `flush()`. Flush segments are typically small and will be merged later, at which point
reordering applies.

---

## Manual Testing Results (2026-03-03): Dynamic Settings + Similarity Auto-Detection

### Test: KMeans reorder with 32x compression, inner product

**Setup:** 15k vectors (normalized, 3 clusters), 3 segments of 5k, force merge to 1.
Index settings: `reorder_strategy=kmeans`, `reorder_kmeans_num_clusters=50`, `space_type=innerproduct`,
`mode=on_disk`, `compression_level=32x`.

**Result:** ✅ PASSED — identical doc IDs and scores before and after merge.
Diagnostics confirmed: similarity=MAXIMUM_INNER_PRODUCT auto-detected, FAISS index type=IBMp (IdMap+Binary),
permutation non-identity, reorder completed in 174ms.

### Test: All strategy/space combinations with 32x compression

15k vectors, 3 segments, force merge to 1. All combinations tested:

| Strategy | Space Type | Result | Before Top-1 | After Top-1 |
|----------|-----------|--------|---------------|-------------|
| kmeans | l2 | ✅ identical | 6810=0.061473 | 6810=0.061473 |
| kmeans | cosinesimil | ✅ identical | 4581=0.991069 | 4581=0.991069 |
| bp | l2 | ✅ identical | 6810=0.061473 | 6810=0.061473 |
| bp | cosinesimil | ✅ identical | 4581=0.991069 | 4581=0.991069 |
| bp | innerproduct | ✅ identical | 4581=1.982139 | 4581=1.982139 |

All 5 combinations produce identical search results before and after merge+reorder.

### Settings verified on index

```json
"knn": {
  "advanced": {
    "reorder_kmeans_num_clusters": "50",
    "reorder_strategy": "kmeans"
  }
}
```

### JNI note for EC2 deployment

`FaissKMeansService` JNI native method must be compiled into `libopensearchknn_faiss.so`.
Rebuild with:
```bash
cmake -S jni -B jni/build/release -DCONFIG_NMSLIB=OFF -DCMAKE_BUILD_TYPE=Release
make -C jni/build/release -j$(nproc)
```
Ensure `java.library.path` points to the correct build directory.
Fallback: `-Dkmeans.useJni=false` uses pure-Java KMeans (slower but no native dependency).

---

## Lucene Codec Details for Reordered Vector Files

This section documents the custom Lucene codecs added/changed to write and read reordered `.vec`
and `.vemf` files, the vector ordinal mapping structures, the skip list index, and the FAISS
index reorder transformers.

### Overview: What Changes on Disk After Reorder

After `SegmentReorderService.reorderSegmentFiles()` runs, three files are rewritten:

| File | Before Reorder | After Reorder |
|------|---------------|---------------|
| `.vec` | Vectors stored in original doc-ID order, standard `Lucene99FlatVectorsFormatData` codec header | Vectors stored in BP/KMeans permuted order, `ReorderedLucene99FlatVectorsFormatData` codec header |
| `.vemf` | Standard `Lucene99FlatVectorsFormatMeta` codec header, field metadata only | `ReorderedLucene99FlatVectorsFormatMeta` codec header, field metadata + per-field doc→ord skip list index |
| `.faiss` | HNSW neighbor lists reference original ordinals | HNSW neighbor lists remapped to new ordinals via `oldOrd2New[]`, entry point remapped |

### 1. ReorderOrdMap — Vector Ordinal Mapping

**File:** `memoryoptsearch/faiss/reorder/ReorderOrdMap.java`

The core data structure that maps between original and reordered vector ordinals.

```
Original vectors : v0, v1, v2
After reordering : v2, v0, v1

newOrd2Old = [2, 0, 1]   — "new ord 0 was old ord 2"
oldOrd2New = [1, 2, 0]   — "old ord 0 is now at new ord 1"
```

- `newOrd2Old[i]`: given a position `i` in the reordered `.vec` file, returns the original ordinal.
  Used when writing reordered vectors (iterate new ords, fetch from old ords).
- `oldOrd2New[j]`: given an original ordinal `j`, returns its position in the reordered file.
  Used when remapping HNSW neighbor lists (neighbor IDs are old ords → translate to new ords).

The constructor takes `newOrd2Old[]` (the permutation from `VectorReorderStrategy.computePermutation()`)
and inverts it to produce `oldOrd2New[]`. Memory: `2 * 4 * N` bytes for both arrays.

### 2. ReorderedFlatVectorsWriter — Writing Reordered `.vec` and `.vemf`

**File:** `memoryoptsearch/faiss/reorder/ReorderedFlatVectorsWriter.java`

Writes the reordered `.vec` (vector data) and `.vemf` (metadata + skip list) files. Called by
`SegmentReorderService.rewriteVecFile()`.

**Codec headers written:**
- `.vemf`: `ReorderedLucene99FlatVectorsFormatMeta` (version 0)
- `.vec`: `ReorderedLucene99FlatVectorsFormatData` (version 0)

These custom codec names are what `ReorderedLucene99FlatVectorsReader111` checks on open to
distinguish reordered segments from standard Lucene99 segments.

**Write sequence per field (`ReorderedDenseFloatFlatFieldVectorsWriter.finish()`):**

1. **Field metadata** into `.vemf`:
   - `fieldInfo.number` (int)
   - `vectorEncoding` ordinal (int)
   - `similarityFunction` ordinal (int)
   - `vectorDataOffset` (vlong) — byte offset into `.vec` where this field's vectors start
   - `vectorDataLength` (vlong) — total bytes of vector data for this field
   - `dimension` (vint)

2. **Skip list metadata** into `.vemf`:
   - `isDense` flag (byte, always 1 for dense)
   - `maxDoc` (int) — highest doc ID
   - `numLevel` (int, currently 4)
   - `numDocsForGrouping` (int, currently 256)
   - `groupFactor` (int, currently 4)

3. **Fixed-block skip list index** into `.vemf` via `FixedBlockSkipListIndexBuilder`:
   - Encodes the doc→ord mapping inline in the metadata file
   - See section 3 below for details

4. **Vector data** into `.vec`:
   - Vectors written in reordered (permuted) order
   - Each vector: `dimension * 4` bytes, little-endian floats
   - `addValue(docID, vector)` records `(docID, ord)` pairs and writes the vector bytes
   - The `docAndOrds[]` array is sorted by docID ascending before building the skip list

**End-of-file markers:**
- `.vemf`: sentinel field number `-1` (int), then `CodecUtil.writeFooter()`
- `.vec`: `CodecUtil.writeFooter()`

### 3. Skip List Index — Doc-ID to Reordered Ordinal Mapping

After reordering, vector ordinals no longer match doc IDs (e.g., doc 5 might be at ordinal 1200
in the reordered `.vec`). The skip list index provides O(1) lookup from doc ID to reordered ordinal.

There are two implementations; the active one is `FixedBlockSkipListIndex`.

#### 3a. FixedBlockSkipListIndexBuilder / FixedBlockSkipListIndexReader (Active)

**Files:**
- `memoryoptsearch/faiss/reorder/FixedBlockSkipListIndexBuilder.java`
- `memoryoptsearch/faiss/reorder/FixedBlockSkipListIndexReader.java`

A compact fixed-width encoding where each doc ID maps to its ordinal using the minimum number
of bytes needed to represent the maximum ordinal value.

**Write format (builder):**
1. `maxDoc` (int) — written by builder constructor
2. `numBytes` (int) — bytes per ordinal value: `4 - (Integer.numberOfLeadingZeros(maxDoc) / 8)`.
   For 15k vectors, `numBytes = 2` (16 bits suffice for ords up to 65535).
3. For each `(doc, ord)` pair in doc-ID order: `ord` encoded in `numBytes` little-endian bytes
4. Padding to 8-byte alignment (0xFF fill bytes)

**Read format (reader):**
- Loads the entire ordinal array as `long[]` blocks for fast bit-extraction
- `skipTo(doc)` → sets current doc (O(1), just stores the doc ID)
- `getOrd()` → extracts the ordinal from the packed `long[]` using bit arithmetic:
  ```
  bitPos = doc * 8 * numBytesPerValue
  word   = blocks[bitPos >>> 6]
  shift  = bitPos % 64
  result = (word >>> shift) & mask
  ```
  Handles cross-word boundaries when the value spans two `long` words.

**Memory:** `numBytesPerValue * (maxDoc + 1)` bytes, padded to 8-byte boundary. For 15k vectors
with 2-byte ords: ~30KB.

#### 3b. DocIdOrdSkipListIndex / DocIdOrdSkipListIndexBuilder (Legacy, multi-level)

**Files:**
- `memoryoptsearch/faiss/reorder/DocIdOrdSkipListIndex.java`
- `memoryoptsearch/faiss/reorder/DocIdOrdSkipListIndexBuilder.java`

A multi-level skip list with bit-packed leaf blocks. Currently commented out in
`ReorderedFlatVectorsWriter` but the code is present.

**Structure:**
- **Level 0 (leaf blocks):** Groups of `numDocsForGrouping` (256) ordinals, bit-packed via
  `IntValuesBitPackingUtil`. Each block stores a 1-byte `bitsPerValue` header followed by
  packed ordinal values.
- **Level 1:** One entry per `groupFactor` (4) leaf blocks. Each entry contains:
  - `lowerLevelStartOffset` (vlong) — byte offset into level-0 data
  - Jump table: `groupFactor - 1` short values encoding accumulated leaf block sizes,
    allowing binary skip within a group. The first entry packs `leafBlockSizeUpto` in the
    low 4 bits (for the last block).
- **Levels 2+:** One entry per `groupFactor^level` leaf blocks, containing `lowerLevelStartOffset`
  pointing into the level below.

**Lookup:** `Reader.skipTo(doc)` descends from the highest level, narrowing the search range
by `numDocsForGrouping * groupFactor^level` docs at each level. At level 1, `findOrd(doc)`
uses the jump table to locate the correct leaf block, then calls
`IntValuesBitPackingUtil.getValue()` to extract the ordinal from the bit-packed block.

### 4. IntValuesBitPackingUtil — Bit-Packed Integer Encoding

**File:** `memoryoptsearch/faiss/reorder/IntValuesBitPackingUtil.java`

Used by the legacy `DocIdOrdSkipListIndexBuilder` to pack ordinal values into minimal bits.

- `bitsRequired(values)` → computes `PackedInts.bitsRequired(max(values))`
- `writePackedInts(values, buffer, output)` → writes 1-byte `bitsPerValue` header + packed bytes
- `pack(values, bitsPerValue, dest)` → packs int array into byte array, each value using exactly
  `bitsPerValue` bits, little-endian bit order
- `getValue(packed, offset, index, bitsPerValue)` → extracts a single value by computing
  `bitPos = index * bitsPerValue`, loading 5 bytes from that position, shifting and masking

### 5. ReorderedLucene99FlatVectorsReader111 — Reading Reordered `.vec` and `.vemf`

**File:** `memoryoptsearch/faiss/reorder/ReorderedLucene99FlatVectorsReader111.java`

Custom `FlatVectorsReader` that reads reordered vector files. Registered as the primary reader
in `NativeEngines990KnnVectorsFormat.fieldsReader()` — if the `.vemf` file has the
`ReorderedLucene99FlatVectorsFormatMeta` codec header, this reader is used; otherwise falls back
to the standard `Lucene99FlatVectorsReader`.

**Codec constants:**
```java
META_CODEC_NAME = "ReorderedLucene99FlatVectorsFormatMeta"
VECTOR_DATA_CODEC_NAME = "ReorderedLucene99FlatVectorsFormatData"
META_EXTENSION = "vemf"
VECTOR_DATA_EXTENSION = "vec"
VERSION_START = 0, VERSION_CURRENT = 0
```

**Initialization:**
1. `readMetadata(state)` → opens `.vemf`, validates codec header via `CodecUtil.checkIndexHeader()`,
   reads per-field entries via `readFields()`
2. `openDataInput(state, ...)` → opens `.vec`, validates codec header, calls
   `CodecUtil.retrieveChecksum()` to verify footer

**Per-field FieldEntry (record):**
Each field entry deserialized from `.vemf` contains:
- `vectorEncoding`, `similarityFunction`, `vectorDataOffset`, `vectorDataLength`, `dimension`
  — same as standard Lucene99 metadata
- `FixedBlockSkipListIndexReader doc2OrdIndex` — the doc→ord mapping, deserialized inline
  from the metadata stream

**Vector access:**
- `getFloatVectorValues(field)` → returns `ReorderedOffHeapFloatVectorValues111.DenseOffHeapVectorValues`
- `getRandomVectorScorer(field, target)` → returns a scorer backed by the same reordered values

**Fallback logic in `NativeEngines990KnnVectorsFormat.fieldsReader()`:**
```java
try {
    // Try reordered codec header
    return new ReorderedLucene99FlatVectorsReader111(state, scorer);
} catch (CorruptIndexException | NullPointerException e) {
    // Not a reordered segment — use standard reader
    return flatVectorsFormat.fieldsReader(state);
}
```

### 6. ReorderedOffHeapFloatVectorValues111 — Reordered Vector Values with Doc→Ord Translation

**File:** `memoryoptsearch/faiss/reorder/ReorderedOffHeapFloatVectorValues111.java`

Extends `FloatVectorValues` to provide vector access where vectors are stored in reordered
(permuted) order on disk but accessed by doc ID.

**Key method — `vectorValue(int targetOrd)`:**
Seeks to `targetOrd * byteSize` in the `.vec` slice and reads `dimension` floats. This accesses
vectors by their reordered ordinal directly (used by the HNSW graph scorer).

**DocIndexIterator — doc-ID-based iteration with ord translation:**
The iterator's `index()` method translates doc IDs to reordered ordinals via the skip list:
```java
public int index() {
    if (doc != ordDoc) {
        docIdOrdSkipListIndex.skipTo(doc);
        ordDoc = doc;
        ord = docIdOrdSkipListIndex.getOrd();
    }
    return ord;
}
```
This is the critical path: when the HNSW graph traversal visits a node (by doc ID), the iterator
translates it to the reordered ordinal so `vectorValue(ord)` reads the correct vector from the
permuted `.vec` file.

**VectorScorer:**
`scorer(float[] query)` creates a copy of the values (for thread safety), wraps the iterator,
and delegates scoring to `FlatVectorsScorer.getRandomVectorScorer()`.

### 7. FAISS Index Reorder Transformers — Remapping `.faiss` Files

**Files:**
- `memoryoptsearch/faiss/reorder/FaissIndexReorderTransformer.java` — abstract base, dispatch via `IndexTypeToFaissIndexReordererMapping`
- `memoryoptsearch/faiss/reorder/IndexTypeToFaissIndexReordererMapping.java` — maps FAISS index type strings to transformer classes
- `memoryoptsearch/faiss/reorder/FaissHNSWIndexReorderer.java` — HNSW float index (`IHNF`, `IHNS`)
- `memoryoptsearch/faiss/reorder/FaissBinaryHnswIndexReorderer.java` — HNSW binary index (`IBHF`)
- `memoryoptsearch/faiss/reorder/FaissHnswReorderer.java` — core HNSW graph reorder logic
- `memoryoptsearch/faiss/reorder/FaissIdMapIndexReorderer.java` — IdMap wrapper (`IXMP`, `IBMP`)
- `memoryoptsearch/faiss/reorder/FaissIndexFloatFlatReorderer.java` — flat float index (`IXF2`, `IXFI`)
- `memoryoptsearch/faiss/reorder/FaissIndexBinaryFlatReorderer.java` — flat binary index (`IBXF`)

**Dispatch:** `FaissIndexReorderTransformer.transform()` loads the FAISS index structure via
`FaissIndex.load(indexInput)`, looks up the transformer from `IndexTypeToFaissIndexReordererMapping`,
and calls `doTransform()`. The mapping supports all FAISS index types used by k-NN:

| FAISS Type | Class | Description |
|-----------|-------|-------------|
| `IXMP` | `FaissIdMapIndexReorderer` | Float IdMap (wraps HNSW or flat) |
| `IBMP` | `FaissIdMapIndexReorderer` | Binary IdMap |
| `IHNF` / `IHNS` | `FaissHNSWIndexReorderer` | Float HNSW (FP32 / SQ) |
| `IBHF` | `FaissBinaryHnswIndexReorderer` | Binary HNSW |
| `IXF2` / `IXFI` | `FaissIndexFloatFlatReorderer` | Float flat (FP32 / FP16) |
| `IBXF` | `FaissIndexBinaryFlatReorderer` | Binary flat |

#### 7a. FaissHnswReorderer — HNSW Graph Neighbor List Remapping

**File:** `memoryoptsearch/faiss/reorder/FaissHnswReorderer.java`

The most complex transformer. Rewrites the entire HNSW graph structure with remapped ordinals.

**What gets rewritten (in order):**

1. **assignProbas** — copied verbatim (level assignment probabilities, not ord-dependent)

2. **cumNumberNeighborPerLevel** — copied verbatim (structural, not ord-dependent)

3. **Levels array** — reordered: for each new ord `i`, writes the level of `newOrd2Old[i]`.
   This ensures the level array matches the new ordinal order.

4. **Offsets array** — recomputed: iterates new ords, accumulates neighbor list sizes from the
   old ords. Written as raw `long[]` (not DirectMonotonic — the original uses
   `DirectMonotonicReader` but the reordered version writes plain longs).

5. **Neighbors array** — the core remapping:
   - For each `oldOrd` in `newOrd2Old` order:
     - For each HNSW level of that vector:
       - Read neighbor IDs from the original file
       - Remap each neighbor: `newNeighborId = oldOrd2New[originalNeighborId]`
       - Preserve `-1` sentinel values (unfilled neighbor slots)
       - Sort valid neighbors ascending, push `-1` to end
       - Write remapped neighbor list

6. **Entry point** — remapped: `oldOrd2New[originalEntryPoint]`

7. **Scalar fields** — copied verbatim: `maxLevel`, `efConstruct`, `efSearch`, dummy field

#### 7b. FaissIdMapIndexReorderer — ID Map Remapping

**File:** `memoryoptsearch/faiss/reorder/FaissIdMapIndexReorderer.java`

Rewrites the `ordToDocs` mapping (FAISS's `id_map` that maps vector ordinals to external IDs/doc IDs).

1. Recursively transforms the nested index (HNSW or flat) via `transform(nestedIndex, ...)`
2. Rewrites the ID map: for each new ord, writes `ordToDocs[newOrd2Old[newOrd]]` — the doc ID
   that was at the old ordinal position

#### 7c. FaissIndexFloatFlatReorderer / FaissIndexBinaryFlatReorderer — Flat Vector Remapping

Rewrites the raw vector storage in permuted order. For each new ord, reads the vector at
`newOrd2Old[newOrd]` from the original and writes it to the output. This ensures the flat
vector storage in the `.faiss` file matches the reordered HNSW graph.

### 8. Integration Point: NativeEngines990KnnVectorsWriter

**File:** `index/codec/KNN990Codec/NativeEngines990KnnVectorsWriter.java`

The writer orchestrates the reorder during merge via two hooks:

**In `mergeOneField()`** — marks fields for reorder:
```java
if (reorderStrategy != null && totalLiveDocs >= SegmentReorderService.MIN_VECTORS_FOR_REORDER) {
    fieldsToReorder.add(fieldInfo);
}
```

**In `finish()`** — executes reorder after footers are written:
```java
flatVectorsWriter.finish();   // writes .vec/.vemf codec footers
flatVectorsWriter.close();    // flushes IndexOutput buffers to disk

for (FieldInfo fieldInfo : fieldsToReorder) {
    SegmentReorderService reorderService = new SegmentReorderService(
        segmentWriteState, fieldInfo, reorderStrategy
    );
    reorderService.reorderSegmentFiles();  // rewrites .vec + .vemf + .faiss
}
```

**In `fieldsReader()`** (`NativeEngines990KnnVectorsFormat.java`) — tries reordered reader first:
```java
try {
    return new ReorderedLucene99FlatVectorsReader111(state, scorer);
} catch (CorruptIndexException | NullPointerException e) {
    return flatVectorsFormat.fieldsReader(state);  // fallback to standard
}
```

### 9. SegmentReorderService — Orchestrator

**File:** `memoryoptsearch/faiss/reorder/SegmentReorderService.java`

Coordinates the full reorder of a segment's vector files:

1. **Compute permutation:** Opens finalized `.vec` via `Lucene99FlatVectorsReader` (mmap-backed),
   calls `strategy.computePermutation(floatVectorValues, numThreads, similarityFunction)`.
   The similarity function is auto-detected from `fieldInfo.getVectorSimilarityFunction()`.

2. **Build ReorderOrdMap:** `new ReorderOrdMap(permutation)` — inverts `newOrd2Old` → `oldOrd2New`.

3. **Rewrite `.vec` + `.vemf`:** Opens original `.vec` via `Lucene99FlatVectorsReader`, creates
   `ReorderedFlatVectorsWriter` for the `.reorder` temp files, iterates new ords writing
   `vectorValues.vectorValue(newOrd2Old[i])`, builds skip list via `FixedBlockSkipListIndexBuilder`.
   Atomic rename replaces originals.

4. **Rewrite `.faiss`:** Opens original via `FaissIndex.load()`, dispatches to the appropriate
   `FaissIndexReorderTransformer` which remaps HNSW neighbor lists, ID maps, and flat vectors.
   Atomic rename replaces original.

**IOContext handling:** Wraps the directory with a `FilterDirectory` that overrides `createOutput()`
to use `state.context` (which is `IOContext.MERGE` during merge), avoiding the
`ConcurrentMergeScheduler` assertion that all writes use merge context.

### 10. File Format Summary

**Reordered `.vemf` layout (per field):**
```
[CodecUtil header: "ReorderedLucene99FlatVectorsFormatMeta", version 0]
For each field:
  fieldNumber          : int
  vectorEncoding       : int (ordinal)
  similarityFunction   : int (ordinal)
  vectorDataOffset     : vlong
  vectorDataLength     : vlong
  dimension            : vint
  isDense              : byte (1)
  maxDoc               : int
  numLevel             : int (4)
  numDocsForGrouping   : int (256)
  groupFactor          : int (4)
  --- FixedBlockSkipListIndex ---
  maxDoc               : int (repeated)
  numBytesPerValue     : int
  ordinals[0..maxDoc]  : numBytesPerValue bytes each, little-endian
  padding              : 0xFF bytes to 8-byte alignment
  --- end skip list ---
sentinel: -1           : int
[CodecUtil footer]
```

**Reordered `.vec` layout:**
```
[CodecUtil header: "ReorderedLucene99FlatVectorsFormatData", version 0]
For each field:
  vectors[0..N-1]      : dimension * 4 bytes each, little-endian floats
                          stored in reordered (permuted) order
[CodecUtil footer]
```

---

## KMeans Reorder OOM Diagnosis (2026-03-03)

### Cluster Setup
- 2-node cluster, 32GB RAM total
- Cohere-10M dataset, 5 shards, force merged to 1 segment/shard (~2M vectors/shard)
- Separate benchmarking utility
- Parent circuit breaker limit: 14.2GB

### Symptoms

1. **Circuit breaker tripping on both HTTP and transport layers:**
```
CircuitBreakingException: [parent] Data too large, data for [<http_request>] would be [15320590184/14.2gb],
which is larger than the limit of [15300820992/14.2gb]
```

2. **FAISS clustering warning — too few points for k=1500:**
```
WARNING clustering 35496 points to 1500 centroids: please provide at least 58500 training points
```

3. **Corrupt `.faiss` files from truncated writes:**
```
CorruptIndexException: codec footer mismatch (file truncated?): actual footer=-581566464 vs expected footer=-1071082520
(resource=MemorySegmentIndexInput(path="..._d_165_target_field.faiss"))
```

4. **Shard failures from flush on corrupt files:**
```
FlushFailedEngineException → AllocationService failing shard
```

### Root Cause 1: KMeans materializes ALL vectors into Java heap

`KMeansReorderStrategy.computePermutation()` copies every vector into a `float[][]`:

```java
float[][] heapVectors = new float[n][];
for (int i = 0; i < n; i++) {
    float[] src = vectors.vectorValue(i);
    heapVectors[i] = new float[src.length];
    System.arraycopy(src, 0, heapVectors[i], 0, src.length);
}
```

Then `FaissKMeansService.storeVectors()` copies it AGAIN into native memory via
`JNICommons.storeVectorData()`.

For a 2M-vector shard at 1024 dims:
- `float[][]` on heap: 2M × 1024 × 4 bytes = **~8GB**
- Native copy via JNI: another **~8GB**
- Total: ~16GB for a single shard reorder, exceeding the 14.2GB circuit breaker

This saturates the heap, leaving no room for concurrent bulk writes, HTTP requests, or
transport operations — triggering the circuit breaker on everything else.

**Contrast with BP:** `BipartiteReorderStrategy` passes `FloatVectorValues` directly to
`BpVectorReorderer.computeValueMap()`, which reads vectors via mmap. Heap overhead is only
`2 * 4 * N` bytes for metadata arrays (~16MB for 2M vectors). KMeans has no such path —
both JNI and pure-Java require contiguous `float[][]`.

### Root Cause 2: k not capped for small segments

The warning `clustering 35496 points to 1500 centroids` means a segment with only 35,496
vectors is being clustered with k=1500. FAISS wants at least `k * min_points_per_centroid`
(default 39) = 58,500 training points. The clustering still runs but quality is poor and
memory is wasted on too many centroids.

This is either an intermediate merge segment before force merge completes, or a shard with
fewer vectors than expected.

### Root Cause 3: Corruption cascade from OOM

1. Circuit breaker trips during bulk write → shard operations fail mid-write
2. `.faiss` file gets truncated (incomplete write during flush/merge)
3. Next flush sees corrupt `.faiss` → `CorruptIndexException: codec footer mismatch`
4. `FlushFailedEngineException` → shard marked as failed

### Fix 1: Cap k based on segment size

In `KMeansReorderStrategy.computePermutation()`, reduce k when there aren't enough points:

```java
int minPointsPerCentroid = 39;
if (n < effectiveK * minPointsPerCentroid) {
    effectiveK = Math.max(1, n / minPointsPerCentroid);
    log.info("Reduced k from {} to {} for {} vectors (min {} points/centroid)", k, effectiveK, n, minPointsPerCentroid);
}
```

Additionally, return identity permutation when segment is too small for meaningful clustering:

```java
if (n < 2 * effectiveK) {
    log.warn("Skipping KMeans reorder: {} vectors < 2 * {} centroids", n, effectiveK);
    int[] identity = new int[n];
    for (int i = 0; i < n; i++) identity[i] = i;
    return identity;
}
```

### Fix 2: Stream vectors to native memory instead of heap-copying

Eliminate the `float[][]` by writing vectors directly to native memory from the mmap-backed
`FloatVectorValues`:

```java
int n = vectors.size();
int dim = vectors.vectorValue(0).length;
long addr = JNICommons.allocateVectorData((long) n * dim);
try {
    for (int i = 0; i < n; i++) {
        float[] vec = vectors.vectorValue(i);  // mmap read, no heap copy retained
        JNICommons.copyVectorToNative(addr, i, vec, dim);  // write directly to native
    }
    KMeansResult result = FaissKMeansService.kmeansWithDistances(addr, n, dim, effectiveK, niter, metricType);
    return ClusterSorter.sortByCluster(result.assignments(), result.distances(), metricType);
} finally {
    JNICommons.freeVectorData(addr);
}
```

This eliminates ~8GB of Java heap usage. Requires a new JNI method `copyVectorToNative`
(trivial: `memcpy` into offset within the allocated buffer).

**Alternative: Subsample for clustering, assign all vectors to nearest centroid:**

```java
int sampleSize = Math.min(n, effectiveK * 50);
float[][] sample = new float[sampleSize][];
int step = n / sampleSize;
for (int i = 0; i < sampleSize; i++) {
    sample[i] = vectors.vectorValue(i * step).clone();
}
// Cluster sample, then stream-assign all vectors to nearest centroid
```

Caps heap at `sampleSize * dim * 4` instead of `n * dim * 4`.

### Fix 3: Increase circuit breaker / heap for benchmarking (workaround)

```json
PUT _cluster/settings
{
  "transient": {
    "indices.breaker.total.limit": "98%"
  }
}
```

Or increase JVM heap: `-Xmx24g` per node (leaving 8GB for OS page cache + native memory).
This is a band-aid — the real fix is Fix 2.

### Fix 4: Combine clusters across merge iterations (longer-term)

Currently each merge re-materializes and re-clusters from scratch. For tiered merges where
a segment gets merged multiple times, this is wasteful. The cluster assignments from the
source segments could be carried forward and only the boundary vectors re-assigned.

### Action Items

- [ ] Implement Fix 1 (cap k) — immediate, prevents the FAISS warning and wasted memory
- [ ] Implement Fix 2 (stream to native) — eliminates the 8GB heap copy, required for large segments
- [ ] Validate with 2M-vector shard after fixes — confirm circuit breaker no longer trips
- [ ] Investigate the 35,496-vector segment — is this an intermediate merge or a small shard?

---

## Leveraging OffHeapVectorTransfer for KMeans Reorder (2026-03-03)

### Existing Infrastructure

`MemOptimizedNativeIndexBuildStrategy` already solves the "stream vectors to native memory
without heap-copying all of them" problem via `OffHeapVectorTransfer`:

1. `OffHeapVectorTransfer` batches vectors into a small Java-side `List<float[]>` (size
   controlled by `knn.vector_streaming_memory.limit`, default 1% of heap)
2. When the batch is full, `OffHeapFloatVectorTransfer.transfer()` calls
   `JNICommons.storeVectorData(addr, batch, capacity, append)` to copy the batch to native memory
3. The Java-side batch is cleared → GC'd. Vectors accumulate only in native memory.

For HNSW building, `append=false` is used (each batch overwrites the buffer, consumed by
`JNIService.insertToIndex()` incrementally). But `storeVectorData` with `append=true` grows
the native buffer, which is exactly what KMeans needs — all vectors contiguous in native memory.

### Key Difference from HNSW Build

- **HNSW:** Vectors consumed incrementally per batch → `append=false`, buffer reused
- **KMeans:** FAISS needs all vectors at once for clustering → `append=true`, buffer grows

But the heap savings are the same: only one batch of vectors lives on Java heap at a time
(~1% of heap), not the full `float[][]` (~8GB).

### Approach: Use OffHeapVectorTransfer with append=true

Replace the heap materialization in `KMeansReorderStrategy` with batched streaming:

```java
@Override
public int[] computePermutation(FloatVectorValues vectors, int numThreads,
                                VectorSimilarityFunction similarityFunction) throws IOException {
    int n = vectors.size();
    int dim = vectors.vectorValue(0).length;
    int effectiveK = Math.min(k, n);

    // Cap k for small segments
    int minPointsPerCentroid = 39;
    if (n < effectiveK * minPointsPerCentroid) {
        effectiveK = Math.max(1, n / minPointsPerCentroid);
    }
    if (n < 2 * effectiveK) {
        return identityPermutation(n);
    }

    int metricType = (similarityFunction == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT
        || similarityFunction == VectorSimilarityFunction.COSINE)
        ? KMeansClusterer.METRIC_INNER_PRODUCT
        : KMeansClusterer.METRIC_L2;

    int bytesPerVector = dim * Float.BYTES;

    // Stream vectors from mmap → native memory in batches via OffHeapVectorTransfer.
    // Only one batch lives on Java heap at a time (~1% of heap).
    // append=true grows the native buffer so all vectors are contiguous for FAISS kmeans.
    try (OffHeapFloatVectorTransfer transfer = new OffHeapFloatVectorTransfer(bytesPerVector, n)) {
        for (int i = 0; i < n; i++) {
            float[] vec = vectors.vectorValue(i);
            transfer.transfer(vec, true);  // append=true: accumulate in native memory
        }
        transfer.flush(true);

        long vectorAddress = transfer.getVectorAddress();
        KMeansResult result = FaissKMeansService.kmeansWithDistances(
            vectorAddress, n, dim, effectiveK, niter, metricType
        );
        return ClusterSorter.sortByCluster(result.assignments(), result.distances(), metricType);
    }
    // OffHeapVectorTransfer.close() calls freeVectorData() — no manual cleanup needed
}
```

### Memory Comparison

| Approach | Java Heap | Native Memory |
|----------|-----------|---------------|
| Current (`float[][]` + JNI copy) | ~8GB (2M × 1024 × 4) + batch overhead | ~8GB (JNI copy) |
| OffHeapVectorTransfer (append=true) | ~1% of heap (one batch) | ~8GB (accumulated) |

The native memory usage is the same — FAISS needs all vectors contiguous for kmeans. But the
Java heap drops from ~8GB to ~1% of heap (~150MB with 15GB heap). This keeps the circuit
breaker happy since it only tracks Java heap, not native memory.

### Caveat: Native Memory Pressure

The ~8GB native allocation still exists. On a 32GB 2-node cluster with 15GB JVM heap per node,
that leaves ~17GB for OS + native. A single 2M-vector shard reorder consuming 8GB native is
tight but feasible. Two concurrent shard reorders would be problematic.

Mitigation: reorder shards sequentially (already the case — `finish()` runs per-segment in the
merge thread, and merges are serialized per shard).

### Note on OffHeapVectorTransfer.transfer() Semantics

`transfer(vec, append)` adds `vec` to an internal `List<float[]>`. When the list hits
`transferLimit`, it calls `JNICommons.storeVectorData(addr, batch, capacity, append)` and
clears the list. The `float[]` references in the cleared list become eligible for GC.

With `append=true`, `storeVectorData` grows the native buffer to accommodate the new batch.
The returned address may change if reallocation occurs, but `OffHeapVectorTransfer` tracks
the latest address via `getVectorAddress()`.

With `append=false` (HNSW path), the buffer is overwritten — same address, same capacity.

---

## Decreasing Memory Requirements for KMeans

### Current Problem

`KMeansReorderStrategy` materializes all vectors into a `float[][]` on Java heap (~8GB for
2M × 1024-dim), then `FaissKMeansService.storeVectors()` copies them again into native memory
(~8GB). Total: ~16GB for a single shard reorder.

### FAISS KMeans Access Pattern Analysis

Inspecting `Clustering::train_encoded()` in `faiss/Clustering.cpp`, the algorithm performs
`niter` (default 25) iterations, each with two phases:

1. **Assignment:** `index.search(nx, x, 1, dis, assign)` — sequential scan of all vectors
   against `k` centroids
2. **Update:** `compute_centroids(d, k, nx, ..., x, ...)` — sequential scan accumulating
   vectors into centroid sums: `x + i * line_size`

Both phases access the vector pointer `x` as a `const uint8_t*` with sequential/strided reads.
FAISS does not require the data to be in heap or native-allocated memory — it just dereferences
the pointer. An mmap'd region works identically.

Additionally, FAISS has a built-in subsample: when `nx > k * max_points_per_centroid` (default
256), it subsamples to `k * 256` points before training. With k=1500, that's 384k vectors
(~1.5GB), not the full 2M.

### Option 1: mmap the .vec file and pass pointer directly to FAISS JNI (zero-copy)

The `.vec` file is already finalized on disk when `SegmentReorderService` runs (after
`flatVectorsWriter.finish()` + `close()`). Lucene's `MMapDirectory` maps it via
`MemorySegmentIndexInput`, which wraps a `java.lang.foreign.MemorySegment`.

Approach:
1. Open the `.vec` file via `directory.openInput()` → get `MemorySegmentIndexInput`
2. Extract the native address: `MemorySegment.address()` + offset to vector data start
3. Pass directly to `FaissKMeansService.kmeansWithDistances(addr, n, dim, k, niter, metric)`
4. FAISS reads vectors via page faults into OS page cache — no heap, no native allocation

Memory profile:

| | BP | KMeans (mmap) | KMeans (current) |
|---|---|---|---|
| Java heap | ~16MB | ~16MB (`int[n]` assign + `float[n]` dis) | ~8GB (`float[][]`) |
| Native alloc | 0 | 0 | ~8GB (JNI copy) |
| OS page cache | on-demand | on-demand | N/A |
| Total pressure | ~16MB | ~16MB | ~16GB |

KMeans with mmap would have essentially the same memory profile as BP.

**Caveats:**
- FAISS does `niter` full sequential scans. For 2M × 1024 × 4 = 8GB of vector data, that's
  25 × 8GB = 200GB of sequential reads. Sequential access pattern is readahead-friendly, but
  if the file exceeds available page cache, later iterations re-fault.
- With the built-in subsample (k=1500 → 384k points → ~1.5GB), only the subsample is scanned
  repeatedly. The full dataset is scanned once for the initial subsample copy.
- Requires extracting the raw mmap address from Lucene's `MemorySegmentIndexInput`, which
  uses `java.lang.foreign` APIs (Panama). The address must account for the codec header offset
  to point at the start of vector data, not the file header.
- The `.vec` file stores vectors in the original (pre-reorder) ordinal order. FAISS kmeans
  only needs to read them — ordinal order doesn't matter for clustering.

**Implementation sketch:**
```java
// In KMeansReorderStrategy or SegmentReorderService:
IndexInput vecInput = directory.openInput(vecDataFileName, IOContext.DEFAULT);
// Navigate past codec header to vector data start
long vectorDataOffset = fieldEntry.vectorDataOffset;

// Extract mmap address — requires Lucene internals access
// MemorySegmentIndexInput wraps MemorySegment which has .address()
long mmapAddress = getMmapAddress(vecInput) + vectorDataOffset;

KMeansResult result = FaissKMeansService.kmeansWithDistances(
    mmapAddress, n, dim, effectiveK, niter, metricType
);
```

### Option 2: OffHeapVectorTransfer with append=true (streaming to native)

Use the existing `OffHeapVectorTransfer` infrastructure from `MemOptimizedNativeIndexBuildStrategy`
to stream vectors from mmap → native memory in batches, avoiding the `float[][]` heap copy.

- `OffHeapFloatVectorTransfer` batches vectors into a small `List<float[]>` (size controlled
  by `knn.vector_streaming_memory.limit`, default 1% of heap)
- With `append=true`, `JNICommons.storeVectorData()` grows the native buffer
- Each batch is cleared after transfer → GC'd. Only one batch on heap at a time.

Memory profile:

| | KMeans (OffHeapTransfer) | KMeans (current) |
|---|---|---|
| Java heap | ~150MB (one batch) + ~16MB (results) | ~8GB |
| Native alloc | ~8GB (accumulated vectors) | ~8GB |
| Total pressure | ~8.2GB (mostly native) | ~16GB |

Eliminates the heap pressure (circuit breaker safe) but still allocates ~8GB native for the
contiguous vector buffer that FAISS needs.

### Option 3: Pure-Java KMeans on subsampled vectors + mmap assignment pass

Cluster a subsample on heap, then assign all vectors via streaming mmap reads:

1. Sample `k * 50` vectors into heap (~1500 × 50 × 1024 × 4 = ~300MB)
2. Run `KMeansClusterer.cluster()` on the sample (pure Java, no JNI)
3. Stream all vectors via `FloatVectorValues.vectorValue(i)` (mmap), compute nearest centroid
   and distance for each — only one vector on heap at a time
4. Build assignments + distances arrays (`int[n]` + `float[n]` = ~16MB)
5. Pass to `ClusterSorter.sortByCluster()`

Memory profile:

| | KMeans (subsample + mmap assign) |
|---|---|
| Java heap | ~300MB (sample) + ~16MB (results) |
| Native alloc | 0 |
| Total pressure | ~316MB |

This is the lowest memory option and doesn't require any JNI changes. The tradeoff is
clustering quality — subsampling may produce slightly worse centroids than training on all
vectors. However, FAISS itself subsamples when `nx > k * 256`, so this is consistent with
FAISS's own approach.

### Recommendation

Option 1 (mmap pointer) is the cleanest long-term solution — zero memory overhead, leverages
the existing mmap'd `.vec` file, and FAISS's sequential access pattern is readahead-friendly.
However, it requires extracting raw addresses from Lucene's `MemorySegmentIndexInput` which
couples to Lucene internals.

Option 3 (subsample + mmap assign) is the easiest to implement today — no JNI changes, no
Lucene internals, ~316MB total memory, and consistent with FAISS's own subsampling behavior.

Option 2 (OffHeapVectorTransfer) is a middle ground — uses existing infrastructure, eliminates
heap pressure, but still needs ~8GB native.

### Action Items

- [ ] Implement Option 3 (subsample + mmap assign) as immediate fix — lowest risk, no JNI changes
- [ ] Prototype Option 1 (mmap pointer passthrough) for long-term — benchmark page fault overhead
      vs native memory copy for `niter=25` iterations
- [ ] Cap k based on segment size regardless of which option is chosen (Fix 1 from OOM diagnosis)

---

## Implementation Plan: mmap Passthrough for KMeans (Option 1)

### Goal

Eliminate the ~8GB Java heap allocation in `KMeansReorderStrategy` by passing the mmap'd
`.vec` file address directly to FAISS `Clustering::train()` via JNI. FAISS only needs a
`const float*` pointer — it doesn't care whether it points to heap, native, or mmap'd memory.

### How FAISS KMeans Actually Uses the Vector Pointer

From `Clustering::train_encoded()` in `faiss/Clustering.cpp`:

1. **Subsample check** (line 314): if `nx > k * max_points_per_centroid` (default 256),
   FAISS calls `subsample_training_set()` which:
   - Allocates a NEW native buffer of size `k * 256 * dim * 4` bytes (~1.5GB for k=1500, dim=1024)
   - Copies `k * 256` randomly selected vectors from the input pointer into this buffer
   - Replaces `x` pointer with the new buffer — all subsequent iterations use the subsample
   - The original input pointer is only read once during this copy (random access pattern)

2. **Training loop** (25 iterations): operates entirely on the 1.5GB subsample buffer.
   Never touches the original input pointer again.

3. **Back in JNI** (`FaissKMeansService.cpp` line 56): after `clustering.train()` returns,
   `index->search(numVectors, vectors, 1, ...)` does one final sequential pass over ALL
   vectors to assign each to its nearest centroid.

**Total reads from the input pointer: 2 passes** — one random-access (subsample), one
sequential (final assignment). The 25 expensive iterations run on FAISS's internal 1.5GB copy.

### Memory Profile Comparison (2M vectors × 1024 dims)

| Phase | Current | mmap approach |
|-------|---------|---------------|
| Java `float[][]` | 8GB heap | 0 |
| JNI `storeVectors` native copy | 8GB native | 0 |
| FAISS internal subsample | 1.5GB native (inside train) | 1.5GB native (inside train) |
| Final assignment arrays | 16MB (int[2M] + float[2M]) | 16MB |
| **Total** | **~17.5GB** | **~1.5GB** (FAISS internal, auto-freed) |

### Existing Infrastructure

The k-NN plugin already extracts mmap addresses from Lucene's `MemorySegmentIndexInput`:

- `MemorySegmentAddressExtractorJDK21` — uses reflection to get `MemorySegment[]` from
  `IndexInput`, then calls `MemorySegment.address()` and `MemorySegment.byteSize()`
- `MMapFloatVectorValues` — stores `long[] addressAndSize` for native scoring code
- `AbstractMemorySegmentAddressExtractor.extractAddressAndSize(indexInput, baseOffset, requestSize)`
  — returns `long[]` of `[addr0, size0, addr1, size1, ...]` accounting for multi-segment mmap

The `.vec` file opened by `Lucene99FlatVectorsReader` is an `IndexInput` from `MMapDirectory`.
The vector data starts at `fieldEntry.vectorDataOffset` bytes from the start of the file
(after the codec header). The data is contiguous: `n * dim * Float.BYTES` bytes.

### Implementation Steps

#### Step 1: Add `computePermutationMMap` to `KMeansReorderStrategy`

New method that accepts an `IndexInput` + offset instead of `FloatVectorValues`:

```java
/**
 * Compute permutation using mmap'd vector data directly — zero Java heap for vectors.
 * Falls back to the heap-based path if mmap address extraction fails.
 */
public int[] computePermutationMMap(
    IndexInput vecData, long vectorDataOffset, long vectorDataLength,
    int numVectors, int dimension, int numThreads,
    VectorSimilarityFunction similarityFunction
) throws IOException {
    int effectiveK = capK(numVectors);
    if (effectiveK < 2) return identityPermutation(numVectors);

    int metricType = toFaissMetric(similarityFunction);

    // Extract mmap address from the IndexInput
    MemorySegmentAddressExtractor extractor = new MemorySegmentAddressExtractorJDK21();
    long[] addressAndSize = extractor.extractAddressAndSize(
        vecData, vectorDataOffset, vectorDataLength
    );

    if (addressAndSize == null || addressAndSize.length != 2) {
        // Multi-segment mmap or extraction failed — fall back to heap path
        log.warn("mmap address extraction failed, falling back to heap-based KMeans");
        // ... fall back to existing computePermutation() ...
    }

    // Single contiguous mmap segment — pass address directly to FAISS
    long mmapAddress = addressAndSize[0];

    KMeansResult result = FaissKMeansService.kmeansWithDistancesMMap(
        mmapAddress, numVectors, dimension, effectiveK, niter, metricType
    );
    return ClusterSorter.sortByCluster(result.assignments(), result.distances(), metricType);
}
```

#### Step 2: Add `kmeansWithDistancesMMap` JNI method

New native method that takes a raw pointer instead of a `std::vector<float>*`:

**Java side** (`FaissKMeansService.java`):
```java
/**
 * Run k-means on mmap'd vector data. The address points directly to contiguous float data
 * (not a std::vector wrapper). FAISS reads from this pointer; it is NOT freed by this method.
 */
public static native KMeansResult kmeansWithDistancesMMap(
    long mmapAddress, int numVectors, int dimension,
    int numClusters, int numIterations, int metricType
);
```

**JNI side** (`org_opensearch_knn_...FaissKMeansService.cpp`):
```cpp
JNIEXPORT jobject JNICALL Java_..._kmeansWithDistancesMMap(
    JNIEnv* env, jclass cls,
    jlong mmapAddress, jint numVectors, jint dimension,
    jint numClusters, jint numIterations, jint metricType)
{
    try {
        // mmapAddress points directly to float data (not std::vector*)
        float* vectors = reinterpret_cast<float*>(mmapAddress);

        faiss::ClusteringParameters cp;
        cp.niter = numIterations;
        cp.verbose = false;

        faiss::Clustering clustering(dimension, numClusters, cp);

        faiss::Index* index;
        if (metricType == 1) {
            index = new faiss::IndexFlatIP(dimension);
        } else {
            index = new faiss::IndexFlatL2(dimension);
        }

        // FAISS reads from `vectors` pointer:
        //   1. subsample_training_set: random-access read of k*256 vectors → copies to internal buffer
        //   2. training loop: operates on internal buffer only
        clustering.train(numVectors, vectors, *index);

        // Final assignment: one sequential pass over all numVectors
        std::vector<faiss::idx_t> assignments(numVectors);
        std::vector<float> distances(numVectors);
        index->search(numVectors, vectors, 1, distances.data(), assignments.data());

        delete index;

        // ... same Java object creation as existing kmeansWithDistances ...
    }
}
```

The only difference from the existing `kmeansWithDistances` is line 1: `reinterpret_cast<float*>(mmapAddress)`
instead of `reinterpret_cast<std::vector<float>*>(vectorsAddress)->data()`.

#### Step 3: Update `SegmentReorderService.computePermutationFromVecFile`

Pass the `IndexInput` and offset to the strategy instead of materializing `FloatVectorValues`:

```java
private int[] computePermutationFromVecFile(Directory directory, String segmentName) throws IOException {
    // ... existing code to build readState and open Lucene99FlatVectorsReader ...

    try (Lucene99FlatVectorsReader reader = new Lucene99FlatVectorsReader(readState, DefaultFlatVectorScorer.INSTANCE)) {
        FloatVectorValues vectorValues = reader.getFloatVectorValues(fieldInfo.name);
        if (vectorValues == null) {
            throw new IOException("No float vector values for field " + fieldInfo.name);
        }

        if (strategy instanceof KMeansReorderStrategy kmeansStrategy) {
            // Try mmap path — pass the raw .vec IndexInput + offset to avoid heap copy
            IndexInput vecInput = directory.openInput(vecDataFileName, IOContext.DEFAULT);
            // vectorDataOffset = codec header size. Lucene99FlatVectorsReader stores this
            // in FieldEntry but it's package-private. Compute it:
            //   CodecUtil.headerLength(codecName) = 16 + codecName.length bytes
            //   For "Lucene99FlatVectorsFormatData": header is ~43 bytes
            // Alternatively, read it from the .vemf metadata (vectorDataOffset field).
            long vectorDataOffset = readVectorDataOffsetFromMeta(directory, vecMetaFileName);
            long vectorDataLength = (long) vectorValues.size() * vectorValues.dimension() * Float.BYTES;

            try {
                return kmeansStrategy.computePermutationMMap(
                    vecInput, vectorDataOffset, vectorDataLength,
                    vectorValues.size(), vectorValues.dimension(),
                    numThreads, fieldInfo.getVectorSimilarityFunction()
                );
            } catch (Exception e) {
                log.warn("mmap KMeans failed, falling back to standard path", e);
            } finally {
                vecInput.close();
            }
        }

        // Default path (BP, or KMeans fallback)
        return strategy.computePermutation(vectorValues, numThreads, fieldInfo.getVectorSimilarityFunction());
    }
}
```

#### Step 4: Handle multi-segment mmap (edge case)

Lucene's `MMapDirectory` may split large files into multiple `MemorySegment` chunks (default
chunk size is `Integer.MAX_VALUE` ~2GB). For a 2M × 1024 × 4 = 8GB `.vec` file, there will
be ~4 chunks.

`extractAddressAndSize()` returns `[addr0, size0, addr1, size1, ...]` for multi-segment files.
FAISS needs a single contiguous pointer.

Options:
- **If single segment** (vectorDataLength < chunk size): pass `addressAndSize[0]` directly. ✅
- **If multi-segment**: fall back to `OffHeapVectorTransfer` (Option 2) or the existing heap
  path. This only happens for segments > ~2GB of vector data (~500k vectors at 1024-dim).

For the common case (segments up to ~500k vectors), single-segment mmap works. For larger
segments, the fallback is acceptable since FAISS subsamples to k*256 vectors anyway — the
full buffer is only read twice.

**Better option for multi-segment:** Use `IndexInput.slice()` to get a slice over the vector
data region, then check if the slice is backed by a single `MemorySegment`. Lucene's slicing
may consolidate the view. If not, fall back.

#### Step 5: Cap k for small segments

Regardless of mmap vs heap, add k-capping to `KMeansReorderStrategy`:

```java
private int capK(int numVectors) {
    int effectiveK = Math.min(k, numVectors);
    int minPointsPerCentroid = 39;  // FAISS default
    if (numVectors < effectiveK * minPointsPerCentroid) {
        effectiveK = Math.max(1, numVectors / minPointsPerCentroid);
    }
    return effectiveK;
}
```

### Files to Change

| File | Change |
|------|--------|
| `KMeansReorderStrategy.java` | Add `computePermutationMMap()`, add `capK()`, add `identityPermutation()` |
| `FaissKMeansService.java` | Add `native kmeansWithDistancesMMap(long, int, int, int, int, int)` |
| `FaissKMeansService.cpp` | Add JNI implementation — same as `kmeansWithDistances` but cast `jlong` directly to `float*` instead of `std::vector<float>*` |
| `SegmentReorderService.java` | In `computePermutationFromVecFile()`, detect `KMeansReorderStrategy`, extract mmap address via `MemorySegmentAddressExtractorJDK21`, call `computePermutationMMap()` |

### Test Plan

#### Unit Tests

**`KMeansReorderStrategyTests`:**
- Test `capK()`: verify k is reduced when `n < k * 39` (e.g., 1000 vectors with k=100 → k=25)
- Test `capK()`: verify k unchanged when `n >= k * 39`
- Test identity permutation returned when `n < 2 * effectiveK`

**`FaissKMeansServiceMMapTests`:**
- Allocate a native buffer via `JNICommons.storeVectorData()`, call `kmeansWithDistancesMMap()`
  with the raw data pointer (not the `std::vector*` wrapper), verify valid assignments returned
- This validates the JNI method works with a raw `float*` pointer
- Compare results with existing `kmeansWithDistances()` on same data — assignments should match

#### Integration Tests

**`SegmentReorderServiceMMapTest`:**
- Create a k-NN index with kmeans reorder strategy, index >10k vectors, force merge
- Verify reorder completes without `CircuitBreakingException`
- Verify search results identical before and after reorder
- Monitor heap usage: confirm Java heap stays under 500MB during reorder (vs ~8GB before)

**Memory pressure test:**
- Set JVM heap to 4GB (`-Xmx4g`) — would OOM with the old approach for any segment > ~500k vectors
- Index 1M vectors, force merge with kmeans reorder
- Verify reorder succeeds (proves mmap path is used, not heap materialization)

**Multi-segment mmap fallback test:**
- Index enough vectors that `.vec` file exceeds 2GB (>500k vectors at 1024-dim)
- Verify fallback to heap/OffHeapVectorTransfer path is triggered (check logs for warning)
- Verify reorder still succeeds

#### Benchmarks

- Compare reorder wall-clock time: mmap vs current heap approach on 2M × 1024-dim segment
- Expected: similar or slightly slower due to page faults during subsample (random access
  into 8GB file), but the final assignment pass is sequential and readahead-friendly
- The 25 training iterations are identical (both operate on FAISS's internal 1.5GB subsample)

---

## Fix: Pre-fault mmap Pages Before FAISS KMeans (2026-03-03)

### Problem

Initial mmap implementation was ~15× slower than the native-alloc path on a 123k × 768-dim
segment (168s vs ~10s). KMeans itself was not the bottleneck — the random page faults during
FAISS's subsampling pass were.

FAISS `subsample_training_set()` picks `k * 256` random vectors from the input pointer. With
mmap, each random access into a cold 362MB file triggers a kernel page fault + disk read. For
k=1500, that's up to 384k random 4KB page reads — catastrophically slow compared to sequential
I/O.

The old native-alloc path (`storeVectors()`) did one sequential copy into RAM first, so FAISS's
random subsample hit warm memory. The mmap path skipped that copy but paid for it in page faults.

### Fix

Added a pre-fault loop in the JNI `kmeansWithDistancesMMap()` before calling
`clustering.train()`:

```cpp
// Pre-fault: sequential touch at page stride to warm the page cache.
// Without this, FAISS's random-access subsampling triggers ~100k+ random page faults.
{
    volatile char sum = 0;
    const char* bytes = reinterpret_cast<const char*>(vectors);
    long nbytes = (long)numVectors * dimension * sizeof(float);
    for (long i = 0; i < nbytes; i += 4096) {
        sum += bytes[i];
    }
}
```

This sequentially touches one byte per 4KB page across the entire vector region. The OS
services these as efficient sequential reads with readahead. After the loop, all pages are
in the page cache and FAISS's random subsample hits warm memory — identical to the native-alloc
path.

### Cost

For a 362MB file: ~88k sequential page touches ≈ 100–200ms on SSD. Negligible compared to
the 168s of random page faults it eliminates.

### Why not `madvise(MADV_WILLNEED)`?

`madvise` is Linux-only and asynchronous (the kernel may not finish paging before FAISS starts
reading). The explicit touch loop is portable (works on macOS/Linux) and synchronous — when it
returns, all pages are guaranteed warm.

### Updated Memory + I/O Profile (123k × 768-dim, k=1500)

| Phase | Native-alloc (old) | mmap (before fix) | mmap (after fix) |
|-------|-------------------|-------------------|------------------|
| Java heap | ~360MB | 0 | 0 |
| Native alloc | ~360MB | 0 | 0 |
| Pre-fault I/O | N/A | N/A | ~200ms sequential |
| FAISS subsample | RAM (fast) | random page faults (168s) | RAM (fast) |
| FAISS training | ~1.1GB internal | ~1.1GB internal | ~1.1GB internal |
| Final assignment | RAM | sequential mmap (fast) | sequential mmap (fast) |
| **Total wall time** | **~10s** | **~168s** | **~10s** |

---

## Algorithms for Merging K-Means Centroids Efficiently (Research Notes, 2026-03-03)

Context: evaluating an architecture where clusters are computed during flush (background thread)
and a lightweight combine/merge of centroids happens during segment merge, avoiding full
re-clustering of all vectors.

### 1. Weighted Centroid Merging via Sufficient Statistics

The foundational technique. Each cluster is represented as a tuple `(sum, count)` — the vector
sum of all assigned points and the number of points. Two clusters merge in O(d):

```
merged_centroid = (sum_A + sum_B) / (count_A + count_B)
merged_count = count_A + count_B
```

To combine centroids from two flush segments:
1. Each flush writes k centroids with `(centroid, count, sum_of_squares)` — the sufficient statistics
2. At merge time, pool all centroids from source segments (e.g., 2 segments × k = 2k weighted centroids)
3. Run weighted k-means on the 2k centroids (treating each as a weighted point), reducing back to k
4. Reassign all vectors to the new k centroids via a single streaming pass over the merged `.vec`

The reassignment pass is the expensive part but is a single sequential scan — no random access.
The clustering step itself is trivial since you're clustering 2k points, not millions.

### 2. BIRCH Clustering Features (CF-tree)

BIRCH represents each cluster as a "Clustering Feature" triple: `CF = (N, LS, SS)` where
N = count, LS = linear sum (vector), SS = sum of squares (scalar or per-dim). These are fully
additive — merging two CFs is element-wise addition. The centroid is `LS/N`, and the
radius/diameter can be computed from SS.

Same idea as #1 but formalized. CFs are closed under addition, so merging is O(d) per cluster
pair. BIRCH uses a tree structure to decide which CFs to merge (closest pair by inter-cluster
distance), but for our case we'd pool and re-cluster.

Reference: Zhang et al., "BIRCH: A New Data Clustering Algorithm and Its Applications"
https://link.springer.com/article/10.1023/A:1009783824328

### 3. Hierarchical Merging of K-Means Solutions (Baudry et al.)

Run k-means independently on partitions, then merge clusters whose assigned point sets overlap
significantly. The merge criterion is based on Bhattacharyya distance or centroid distance
between clusters from different partitions.

For our case: each flush segment produces k clusters. At merge time, pool k₁ + k₂ centroids.
Compute pairwise distances between all centroids, then greedily merge the closest pairs until
back to k. The merged centroid uses the weighted average formula from #1.

Reference: "Clustering Large Datasets by Merging K-Means Solutions"
https://www.researchgate.net/publication/332081523_Clustering_Large_Datasets_by_Merging_K-Means_Solutions

### 4. Coreset-based Merging

A coreset is a small weighted point set that approximates the full dataset for clustering.
The centroids + counts from each flush segment are essentially a coreset. The merge operation:

1. Union the coresets from source segments
2. Run weighted k-means on the union (small — just the centroids, not the vectors)
3. Use the result as the new coreset for the merged segment

Theoretical guarantee: if each coreset is an ε-approximation, the merged result is a
(1+ε)-approximation. The practical version is just weighted k-means on the pooled centroids.

### 5. k*-means Split/Merge (MDL-based)

The k*-means algorithm uses minimum description length (MDL) to decide when to split or merge
clusters. Each cluster maintains two sub-centroids. The merge criterion compares the cost of
keeping two clusters separate vs. combining them. Relevant if the merge step should also adapt
k — e.g., if two flush segments had similar cluster structure, fewer total clusters may suffice.

Reference: Mahon & Lapata, "k*-means: A Parameter-free Clustering Algorithm"
https://arxiv.org/html/2505.11904v1

### Recommended Approach for Flush-then-Merge Architecture

**During flush:** Run k-means, store per-cluster sufficient statistics alongside the segment:

```java
class ClusterSummary {
    float[] centroid;    // LS / N
    float[] linearSum;   // LS = sum of all vectors in cluster
    int count;           // N
    float sumOfSquares;  // SS = sum of ||v - centroid||² for intra-cluster variance
}
```

**During merge:**
1. Pool all `ClusterSummary` objects from source segments (cheap — just reading metadata)
2. Run weighted k-means on the pooled centroids (k₁ + k₂ points, not millions of vectors) —
   the "lightweight combine" step, takes milliseconds
3. Single streaming pass over the merged `.vec` to assign each vector to its nearest new
   centroid and compute the permutation

Step 2 is where the centroid merging happens — k-means on ~2k weighted points instead of
millions of vectors. The weighted k-means update rule:

```
new_centroid_j = Σ(weight_i * centroid_i) / Σ(weight_i)  for all centroids_i assigned to cluster j
```

where `weight_i = count_i` from the source cluster summary.

Step 3 (the assignment pass) is the bottleneck but is a single sequential scan — much cheaper
than running full k-means on all vectors. Since we're reading from mmap'd `.vec` files, it's
readahead-friendly.

**Tradeoff vs. full k-means at merge time:** slightly worse cluster quality (centroids are
approximate since they were computed on partitions, not the full dataset), but dramatically
faster merge since we avoid the iterative k-means training loop on millions of vectors.

**Storage overhead:** ~`k * (d * 4 + d * 4 + 4 + 4)` bytes per segment for the cluster
summaries. With k=256 and d=1024: ~2MB per segment. Negligible compared to the `.vec` file.
