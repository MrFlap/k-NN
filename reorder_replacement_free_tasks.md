# Replacement-Free Reordering: Task Breakdown

## Related Documents

- [Design](reorder_skip_replacement.md) — Two-pass approach, doc ID mapping, sequence diagram
- [Implementation Setting](reorder_implementation_setting.md) — New index setting for switching implementations
- [Test Plan](reorder_skip_replacement_test_plan.md) — Verification strategy, invariants, two-index comparison
- [Original Reorder Integration](INTEGRATE_REORDER.md) — Replacement-based reorder design and pipeline analysis

---

## Task 1: Add `index.knn.advanced.reorder_implementation` Setting

### What
New dynamic index setting: `replacement` (default) | `replacement_free`.

### Files to Change
- `src/main/java/org/opensearch/knn/index/KNNSettings.java`
  - Add `INDEX_KNN_ADVANCED_REORDER_IMPLEMENTATION` constant and `Setting<String>` definition
  - Register in `getSettings()` list (~line 771)

### Unit Test
- `src/test/java/org/opensearch/knn/index/KNNSettingsTests.java`
- Test: valid values accepted (`replacement`, `replacement_free`), invalid values throw
  `IllegalArgumentException`, default is `replacement`
- Pattern: follow existing tests for `INDEX_KNN_ADVANCED_REORDER_STRATEGY_SETTING`

---

## Task 2: Thread Setting Through Codec Stack

### What
Pass the `replacementFree` boolean from settings → format → writer.

### Files to Change
- `src/main/java/org/opensearch/knn/index/codec/BasePerFieldKnnVectorsFormat.java`
  - `nativeEngineVectorsFormat()` (~line 160): read new setting, pass to constructor
- `src/main/java/org/opensearch/knn/index/codec/KNN990Codec/NativeEngines990KnnVectorsFormat.java`
  - Add `boolean replacementFree` field, accept in constructor, pass to writer
- `src/main/java/org/opensearch/knn/index/codec/KNN990Codec/NativeEngines990KnnVectorsWriter.java`
  - Add `boolean replacementFree` field, accept in constructor

### Unit Test
- Not directly unit-testable in isolation — this is pure wiring.
- **Alternative:** Verify via `NativeEngines990KnnVectorsFormatTests` that the format creates
  a writer with the correct `replacementFree` flag. Add a test that constructs the format with
  `replacementFree=true` and asserts the writer receives it (inspect via test accessor or mock).

---

## Task 3: Create `MergedRandomAccessFloatVectorValues`

### What
A `FloatVectorValues` that provides random access + `copy()` over multiple source segments'
mmap-backed readers. Maps merged ord → (segment, local ord) using `segmentStarts[]` and
`liveLocalOrds[][]`.

### File to Create
- `src/main/java/org/opensearch/knn/memoryoptsearch/faiss/reorder/MergedRandomAccessFloatVectorValues.java`

### Key Methods
```java
public class MergedRandomAccessFloatVectorValues extends FloatVectorValues {
    private final FloatVectorValues[] segmentValues;
    private final int[] segmentStarts;      // segmentStarts[seg] = first merged ord
    private final int[][] liveLocalOrds;    // null entry = no deletions in that segment
    private final int totalSize;
    private final int dimension;

    float[] vectorValue(int mergedOrd);     // binary search segmentStarts, delegate to source
    FloatVectorValues copy();               // clone each segment's values
    int size();
    int dimension();
}
```

### Code Pointers
- Source segment readers: `mergeState.knnVectorsReaders[i].getFloatVectorValues(fieldName)`
  returns `OffHeapFloatVectorValues` (mmap-backed, random access, `copy()` supported)
  — see `lucene/core/.../lucene95/OffHeapFloatVectorValues.java:127` (`DenseOffHeapVectorValues`)
- Existing composite pattern: `KnnVectorsWriter.MergedFloat32VectorValues` in
  `lucene/core/.../KnnVectorsWriter.java:305` (forward-only — ours adds random access)

### Unit Test
- `src/test/java/org/opensearch/knn/memoryoptsearch/faiss/reorder/MergedRandomAccessFloatVectorValuesTests.java`
- Create 3 heap-backed `FloatVectorValues` via `FloatVectorValues.fromFloats()` with known vectors
- Construct `MergedRandomAccessFloatVectorValues` with `segmentStarts = [0, 5, 12]`
- Test `vectorValue(0)` returns segment 0 vector 0
- Test `vectorValue(5)` returns segment 1 vector 0
- Test `vectorValue(11)` returns segment 1 vector 6
- Test `vectorValue(12)` returns segment 2 vector 0
- Test `copy()` returns independent instance (modify one, other unaffected)
- Test with `liveLocalOrds` (simulate deletions): `liveLocalOrds[1] = [0, 2, 4]` skips ords 1, 3
- Test `size()` and `dimension()` correct

---

## Task 4: Build Mapping Arrays from MergeState

### What
A utility that builds `mergedOrdToDocId[]`, `segmentStarts[]`, and `liveLocalOrds[][]` from
`MergeState.docMaps` + per-segment `FloatVectorValues.ordToDoc()`.

### File to Create
- `src/main/java/org/opensearch/knn/memoryoptsearch/faiss/reorder/MergeOrdMappingBuilder.java`

### Key Method
```java
public class MergeOrdMappingBuilder {
    public record MergeOrdMapping(
        int[] mergedOrdToDocId,
        int[] segmentStarts,
        int[][] liveLocalOrds,
        int totalLiveDocs
    ) {}

    public static MergeOrdMapping build(
        MergeState mergeState, FieldInfo fieldInfo
    ) throws IOException;
}
```

### Code Pointers
- `MergeState.docMaps[seg].get(sourceDocId)` — returns merged doc ID or -1 for deleted
  — see `lucene/core/.../MergeState.java:170` (`buildDeletionDocMaps`)
- `FloatVectorValues.ordToDoc(localOrd)` — dense returns `ord`, sparse does lookup
  — see `lucene/core/.../lucene95/OffHeapFloatVectorValues.java:148` and `:233`
- `mergeState.liveDocs[seg]` — `Bits` for deleted docs
- Index-sort handling: `mergeState.needsIndexSort` — sort `mergedOrdToDocId` ascending if true

### Unit Test
- `src/test/java/org/opensearch/knn/memoryoptsearch/faiss/reorder/MergeOrdMappingBuilderTests.java`
- Mock `MergeState` with 3 segments (sizes 5, 7, 3), no deletions:
  - Verify `segmentStarts = [0, 5, 12, 15]`
  - Verify `mergedOrdToDocId` is `[0, 1, ..., 14]` (dense, no sort)
  - Verify `liveLocalOrds` all null
- Mock with deletions in segment 1 (delete ords 1, 3):
  - Verify `segmentStarts` accounts for reduced count
  - Verify `liveLocalOrds[1] = [0, 2, 4, 5, 6]`
  - Verify `mergedOrdToDocId` skips deleted docs
- Mock with `needsIndexSort = true`:
  - Verify `mergedOrdToDocId` is sorted ascending
  - Verify `segmentStarts` rebuilt from sorted order

---

## Task 5: Create `ReorderAwareFlatVectorsWriter`

### What
Replacement for `Lucene99FlatVectorsWriter` that exposes `getMetaOutput()` and
`getVectorDataOutput()`. Replicates the same write logic (~200 lines).

### File to Create
- `src/main/java/org/opensearch/knn/index/codec/KNN990Codec/ReorderAwareFlatVectorsWriter.java`

### Key Differences from `Lucene99FlatVectorsWriter`
- Fields `meta` and `vectorData` are package-private (not private)
- Adds `IndexOutput getMetaOutput()` and `IndexOutput getVectorDataOutput()`
- Everything else identical: same codec headers, same `mergeOneField()`, same `writeMeta()`,
  same `finish()` (sentinel + footers), same `close()`

### Code Pointers
- Source to replicate: `lucene/core/.../lucene99/Lucene99FlatVectorsWriter.java:65-530`
- Codec constants: `Lucene99FlatVectorsFormat.META_CODEC_NAME`, `VECTOR_DATA_CODEC_NAME`,
  `META_EXTENSION`, `VECTOR_DATA_EXTENSION`, `VERSION_CURRENT`
- `writeMeta()` at line 296 — writes field metadata + `OrdToDocDISIReaderConfiguration`
- `writeVectorData()` at line 330 — forward-iterates `FloatVectorValues`, writes bytes

### Unit Test
- `src/test/java/org/opensearch/knn/index/codec/KNN990Codec/ReorderAwareFlatVectorsWriterTests.java`
- Test that `getMetaOutput()` and `getVectorDataOutput()` return non-null after construction
- Test that `mergeOneField()` writes the same bytes as `Lucene99FlatVectorsWriter`:
  - Create both writers with same `SegmentWriteState` (using `RAMDirectory`)
  - Call `mergeOneField()` on both with same mock `MergeState`
  - Call `finish()` + `close()` on both
  - Compare `.vec` and `.vemf` file contents byte-for-byte
- This is the critical correctness test — ensures our copy doesn't diverge from Lucene's

---

## Task 6: Write Reordered Metadata into `.vemf`

### What
A utility that writes the reordered field metadata + skip list into the `.vemf` `IndexOutput`,
using the same format as `ReorderedFlatVectorsWriter` but writing into the delegate's stream.

### File to Create
- `src/main/java/org/opensearch/knn/memoryoptsearch/faiss/reorder/ReorderedFieldMetaWriter.java`

### Key Method
```java
public class ReorderedFieldMetaWriter {
    /**
     * Write reordered field metadata + skip list into the meta IndexOutput.
     * Called instead of Lucene99FlatVectorsWriter.writeMeta() for reordered fields.
     */
    public static void writeReorderedMeta(
        IndexOutput meta,
        FieldInfo fieldInfo,
        long vectorDataOffset,
        long vectorDataLength,
        int[] mergedOrdToDocId,
        int[] permutation          // newOrd2Old
    ) throws IOException;
}
```

### Code Pointers
- Existing reordered meta format: `ReorderedFlatVectorsWriter.ReorderedDenseFloatFlatFieldVectorsWriter.finish()`
  at `src/.../reorder/ReorderedFlatVectorsWriter.java:140-196`
- Skip list builder: `FixedBlockSkipListIndexBuilder` at `src/.../reorder/FixedBlockSkipListIndexBuilder.java`
- Reader that consumes this format: `ReorderedLucene99FlatVectorsReader111.readFields()`
  at `src/.../reorder/ReorderedLucene99FlatVectorsReader111.java`

### Unit Test
- `src/test/java/org/opensearch/knn/memoryoptsearch/faiss/reorder/ReorderedFieldMetaWriterTests.java`
- Write metadata to a `ByteBuffersIndexOutput`, then read it back with
  `FixedBlockSkipListIndexReader`
- Verify: for known `permutation = [2, 0, 1]` and `mergedOrdToDocId = [0, 1, 2]`:
  - `skipTo(0).getOrd() == 1` (doc 0 → old ord 0 → new ord 1)
  - `skipTo(1).getOrd() == 2` (doc 1 → old ord 1 → new ord 2)
  - `skipTo(2).getOrd() == 0` (doc 2 → old ord 2 → new ord 0)
- Verify round-trip: write then read with `ReorderedLucene99FlatVectorsReader111` field entry parser

---

## Task 7: Create Reordered `KNNVectorValues` Wrapper

### What
A `KNNVectorValues<float[]>` that iterates in permuted order, yielding the correct doc ID
and vector for each new ord. Used by `NativeIndexWriter.mergeIndex()` to build the FAISS
graph in reordered order.

### File to Create
- `src/main/java/org/opensearch/knn/index/vectorvalues/ReorderedKNNFloatVectorValues.java`

### Key Behavior
```
nextDoc() advances newOrd from 0 to N-1
docId()   returns mergedOrdToDocId[permutation[newOrd]]
getVector() returns mergedRandomAccess.vectorValue(permutation[newOrd])
```

### Code Pointers
- `KNNVectorValues<T>` base class: `src/.../vectorvalues/KNNVectorValues.java`
- `KNNFloatVectorValues`: `src/.../vectorvalues/KNNFloatVectorValues.java`
- `KNNVectorValuesIterator`: `src/.../vectorvalues/KNNVectorValuesIterator.java`
- Consumer: `NativeIndexWriter.mergeIndex()` at `src/.../nativeindex/NativeIndexWriter.java:115`
  — calls `nextDoc()` + `getVector()` in a loop
- Test helper: `TestVectorValues.PreDefinedFloatVectorValues` at
  `src/test/.../vectorvalues/TestVectorValues.java:183`

### Unit Test
- `src/test/java/org/opensearch/knn/index/vectorvalues/ReorderedKNNFloatVectorValuesTests.java`
- Create with known vectors `[[1,0], [0,1], [1,1]]`, permutation `[2, 0, 1]`,
  docIds `[10, 20, 30]`
- Iterate: first `nextDoc()` → docId=30, getVector=[1,1] (newOrd 0 → oldOrd 2)
- Second `nextDoc()` → docId=10, getVector=[1,0] (newOrd 1 → oldOrd 0)
- Third `nextDoc()` → docId=20, getVector=[0,1] (newOrd 2 → oldOrd 1)
- Fourth `nextDoc()` → `NO_MORE_DOCS`
- Verify `dimension()` and `bytesPerVector()` correct

---

## Task 8: Implement `mergeOneFieldReplacementFree()`

### What
The core method in `NativeEngines990KnnVectorsWriter` that orchestrates the replacement-free
merge for a single field.

### File to Change
- `src/main/java/org/opensearch/knn/index/codec/KNN990Codec/NativeEngines990KnnVectorsWriter.java`

### Logic
```java
private void mergeOneFieldReplacementFree(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    // 1. Build mappings from MergeState
    MergeOrdMapping mapping = MergeOrdMappingBuilder.build(mergeState, fieldInfo);

    // 2. Build random-access composite over source segments
    MergedRandomAccessFloatVectorValues mergedRA = buildMergedRandomAccess(
        mergeState, fieldInfo, mapping);

    // 3. Compute permutation
    int[] permutation = reorderStrategy.computePermutation(
        mergedRA, SegmentReorderService.DEFAULT_REORDER_THREADS,
        fieldInfo.getVectorSimilarityFunction());
    ReorderOrdMap ordMap = new ReorderOrdMap(permutation);

    // 4. Write .vec in permuted order into delegate's stream
    ReorderAwareFlatVectorsWriter flatWriter = (ReorderAwareFlatVectorsWriter) flatVectorsWriter;
    IndexOutput vectorData = flatWriter.getVectorDataOutput();
    long vectorDataOffset = vectorData.alignFilePointer(Float.BYTES);
    ByteBuffer buffer = ByteBuffer.allocate(mergedRA.dimension() * Float.BYTES)
        .order(ByteOrder.LITTLE_ENDIAN);
    for (int newOrd = 0; newOrd < mapping.totalLiveDocs(); newOrd++) {
        int oldOrd = permutation[newOrd];
        float[] vec = mergedRA.vectorValue(oldOrd);
        buffer.asFloatBuffer().put(vec);
        vectorData.writeBytes(buffer.array(), buffer.array().length);
    }
    long vectorDataLength = vectorData.getFilePointer() - vectorDataOffset;

    // 5. Write reordered .vemf metadata + skip list
    ReorderedFieldMetaWriter.writeReorderedMeta(
        flatWriter.getMetaOutput(), fieldInfo, vectorDataOffset, vectorDataLength,
        mapping.mergedOrdToDocId(), permutation);

    // 6. Train quantization (order-independent)
    Supplier<KNNVectorValues<?>> reorderedSupplier = () -> new ReorderedKNNFloatVectorValues(
        mergedRA, permutation, mapping.mergedOrdToDocId());
    QuantizationState quantizationState = train(fieldInfo, reorderedSupplier, mapping.totalLiveDocs());

    // 7. Build .faiss with reordered supplier
    NativeIndexWriter writer = NativeIndexWriter.getWriter(
        fieldInfo, segmentWriteState, quantizationState, nativeIndexBuildStrategyFactory);
    writer.mergeIndex(reorderedSupplier, mapping.totalLiveDocs());
}
```

### Code Pointers
- Current `mergeOneField()`: same file, line ~161
- `flatVectorsWriter.mergeOneField()`: the call we're replacing for reordered fields
- `NativeIndexWriter.mergeIndex()`: `src/.../nativeindex/NativeIndexWriter.java:115`

### Unit Test
- Not directly unit-testable — this is the orchestrator that wires Tasks 3-7 together.
- **Alternative:** Integration test via the two-index comparison pattern from the test plan.
  Create a `NativeEngines990KnnVectorsWriter` with `replacementFree=true`, mock the
  `MergeState`, call `mergeOneField()`, then verify the output segment using `ReorderVerifier`
  invariants from the test plan.
- Can also add to existing `NativeEngines990KnnVectorsWriterMergeTests` as a new test case
  that exercises the replacement-free path.

---

## Task 9: Wire Up the Branch in `mergeOneField()`

### What
Add the `if (replacementFree)` branch in `NativeEngines990KnnVectorsWriter.mergeOneField()`.

### File to Change
- `src/main/java/org/opensearch/knn/index/codec/KNN990Codec/NativeEngines990KnnVectorsWriter.java`

### Logic
```java
@Override
public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    this.mergeState = mergeState;

    VectorDataType vectorDataType = extractVectorDataType(fieldInfo);
    int totalLiveDocs = countLiveDocs(fieldInfo, mergeState);

    if (replacementFree && reorderStrategy != null
        && totalLiveDocs >= SegmentReorderService.MIN_VECTORS_FOR_REORDER
        && vectorDataType == VectorDataType.FLOAT) {
        mergeOneFieldReplacementFree(fieldInfo, mergeState);
        return;
    }

    // Existing path unchanged
    flatVectorsWriter.mergeOneField(fieldInfo, mergeState);
    // ... rest of existing code ...
}
```

### Unit Test
- `src/test/java/org/opensearch/knn/index/codec/KNN990Codec/NativeEngines990KnnVectorsWriterMergeTests.java`
- Add test: `replacementFree=true`, totalLiveDocs >= 10k → verify `mergeOneFieldReplacementFree` called
- Add test: `replacementFree=true`, totalLiveDocs < 10k → verify standard path used
- Add test: `replacementFree=false` → verify standard path used regardless of doc count
- These can use mocks to verify which path was taken.

---

## Task 10: Two-Index Comparison Integration Test

### What
End-to-end test that creates two indices (one per implementation), indexes same vectors,
force merges, and compares search results.

### File to Create
- `src/test/java/org/opensearch/knn/index/ReorderImplementationComparisonIT.java`

### Code Pointers
- Existing integration test patterns: `src/test/java/org/opensearch/knn/index/` directory
- Index creation with settings: use `createKnnIndex()` helper with custom settings map
- Force merge: `forceMergeKnnIndex()`
- k-NN search: `searchKNNIndex()`

### Test Cases
1. BP strategy, 15k vectors, compare search results (top-10 doc IDs and scores)
2. Below threshold (5k vectors), verify neither index reorders
3. With deletions: index 15k, delete 2k, force merge, compare results
4. Run `ReorderVerifier` invariants on both indices independently

### Unit Test
- Not applicable — this is an integration test by nature (requires running OpenSearch).
- This is the capstone test that validates the entire feature end-to-end.

---

## Implementation Order

```
Task 1  (Setting)                    — standalone, no dependencies
Task 3  (MergedRandomAccessFVV)      — standalone, no dependencies
Task 4  (MergeOrdMappingBuilder)     — standalone, no dependencies
Task 5  (ReorderAwareFlatVectorsWriter) — standalone, no dependencies
Task 6  (ReorderedFieldMetaWriter)   — depends on FixedBlockSkipListIndexBuilder (existing)
Task 7  (ReorderedKNNFloatVectorValues) — depends on Task 3
Task 2  (Thread setting through codec) — depends on Task 1, Task 5
Task 8  (mergeOneFieldReplacementFree) — depends on Tasks 2-7
Task 9  (Wire up branch)             — depends on Task 8
Task 10 (Integration test)           — depends on Task 9
```

Tasks 1, 3, 4, 5, 6 can all be done in parallel. Each has its own unit test.
