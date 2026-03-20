# Replacement-Free Reordering: Write .vec in Reordered Order During Merge

## Goal

Eliminate the current double-write pattern where `.vec` is written in original order by
`Lucene99FlatVectorsWriter.mergeOneField()`, then read back and rewritten in reordered order
by `SegmentReorderService`. Instead, write `.vec` in reordered order from the start during merge.

## Current Flow (Replacement-Based)

```
mergeOneField()
  → flatVectorsWriter.mergeOneField(fieldInfo, mergeState)
      writes .vec in original ord order (forward-only MergedFloat32VectorValues)
      writes .vemf metadata (standard Lucene ord-to-doc mapping)
  → NativeIndexWriter.mergeIndex()
      builds .faiss HNSW graph in original ord order

finish()
  → flatVectorsWriter.finish()   // writes .vec/.vemf codec footers
  → flatVectorsWriter.close()    // flushes IndexOutput buffers to disk
  → for each field in fieldsToReorder:
      SegmentReorderService.reorderSegmentFiles():
        1. Open finalized .vec via Lucene99FlatVectorsReader (mmap)
        2. Compute BP permutation from FloatVectorValues
        3. Rewrite .vec + .vemf to .reorder temp files via ReorderedFlatVectorsWriter
        4. Rewrite .faiss to .reorder temp file via FaissIndexReorderTransformer
        5. Atomic rename .reorder files over originals
```

Problems:
- Double I/O for .vec (write original, read back, write reordered)
- Temp files + atomic renames
- .faiss HNSW neighbor lists must be remapped after the fact

## New Flow (Replacement-Free)

### Why MergedFloat32VectorValues Can't Be Used Directly

Lucene's `MergedFloat32VectorValues` (from `MergedVectorValues.mergeFloatVectorValues()`) is
**forward-only, single-pass**:

```java
public float[] vectorValue(int ord) throws IOException {
    if (ord != lastOrd) {
        throw new IllegalStateException("only supports forward iteration");
    }
    return current.values.vectorValue(current.index());
}
```

It's backed by a `DocIDMerger` that interleaves docs from multiple segments in doc-ID order.
No random access, no `copy()`. BP reorderer needs both.

However, the **per-segment** `FloatVectorValues` from each source segment's reader
(`OffHeapFloatVectorValues`) fully supports random access via mmap:

```java
// OffHeapFloatVectorValues — random access, mmap-backed, supports copy()
public float[] vectorValue(int targetOrd) throws IOException {
    slice.seek((long) targetOrd * byteSize);
    slice.readFloats(value, 0, value.length);
    return value;
}

public DenseOffHeapVectorValues copy() throws IOException {
    return new DenseOffHeapVectorValues(dimension, size, slice.clone(), ...);
}
```

### Two-Pass Approach

#### Pass 1: Compute Permutation (random-access mmap, no heap vectors)

Build a **random-access composite `FloatVectorValues`** over the source segments. Each source
segment's `FloatVectorValues` (from `mergeState.knnVectorsReaders[i].getFloatVectorValues()`)
supports random access via mmap. The composite maps merged ord → (segment, local ord):

```java
class MergedRandomAccessFloatVectorValues extends FloatVectorValues {
    final FloatVectorValues[] segmentValues;  // per-source-segment, mmap-backed
    final int[] segmentStarts;                // segmentStarts[i] = first merged ord for segment i
    // e.g. [0, 5000, 12000] for segments of size 5k, 7k

    float[] vectorValue(int mergedOrd) {
        int seg = findSegment(mergedOrd);              // binary search in segmentStarts
        int localOrd = mergedOrd - segmentStarts[seg];
        return segmentValues[seg].vectorValue(localOrd);
    }

    FloatVectorValues copy() {
        FloatVectorValues[] copies = new FloatVectorValues[segmentValues.length];
        for (int i = 0; i < copies.length; i++) copies[i] = segmentValues[i].copy();
        return new MergedRandomAccessFloatVectorValues(copies, segmentStarts);
    }
}
```

Feed this to `VectorReorderStrategy.computePermutation()`. BP works directly on mmap — each
thread calls `copy()` to get its own mmap view. Memory: only `2 * 4 * N` bytes for BP metadata
arrays, no vector data on heap.

**Handling deleted docs / doc ID mapping:**

The composite must account for deleted docs in source segments. During merge, only live docs
are included. The merged ord space is dense (0..totalLiveDocs-1), but source segments may have
gaps from deletions. Two options:

a) Build a per-segment live-ord-to-raw-ord mapping (iterate each segment's live docs once,
   build `int[]` mapping). Then `vectorValue(mergedOrd)` → find segment → find live-local-ord
   → map to raw-local-ord → call source `vectorValue(rawOrd)`.

b) Use `mergeState.liveDocs[i]` to skip deleted docs when building `segmentStarts`. Each
   segment contributes only its live doc count to the merged ord space.

#### Pass 2: Write .vec in Permuted Order + Build .faiss in Permuted Order

Once we have `int[] permutation` (where `permutation[newOrd] = oldMergedOrd`):

**Step 2a — Collect doc IDs:** Forward-iterate `MergedFloat32VectorValues` once to build
`int[] ordToDocId` mapping (mergedOrd → merged docID). This is needed for the skip list and
for the FAISS ID map. Alternatively, compute from `MergeState.docMaps` without iterating.

**Step 2b — Write .vec in permuted order:** For `newOrd = 0..N-1`:
- `oldMergedOrd = permutation[newOrd]`
- Read vector via `mergedRandomAccess.vectorValue(oldMergedOrd)` (mmap, random access)
- Write to the `.vec` IndexOutput

**Step 2c — Write .vemf metadata with skip list:** Write the reordered codec header
(`ReorderedLucene99FlatVectorsFormatMeta`), field metadata, and the doc→ord skip list
(via `FixedBlockSkipListIndexBuilder`). For each doc, the ord in the reordered file is
`oldOrd2New[mergedOrd]` where `mergedOrd` is the original position of that doc's vector.

**Step 2d — Build .faiss in permuted order:** Pass a reordered `KNNVectorValues` supplier
to `NativeIndexWriter.mergeIndex()`. The supplier yields vectors in permuted order, so the
HNSW graph is built with vectors already in their final reordered positions. No post-hoc
HNSW neighbor list remapping needed. The FAISS `ordToDocs` ID map maps `newOrd` →
`ordToDocId[permutation[newOrd]]`.

### Writing Into the Delegate's .vec/.vemf Streams vs. Separate Files

`Lucene99FlatVectorsWriter` opens `.vec` and `.vemf` `IndexOutput` streams in its constructor.
These are shared across all fields. `finish()` just writes the end-of-fields sentinel and
codec footers — it does NOT iterate over fields.

**Option A: Write into the same streams.** Requires access to the delegate's private `meta`
and `vectorData` IndexOutput fields. The `.vemf` would contain a mix of standard Lucene
metadata (for non-reordered fields) and reordered metadata (with skip lists). The reader
must handle both formats in the same file.

**Option B: Skip the delegate for reordered fields, write to separate files.** For reordered
fields, use `ReorderedFlatVectorsWriter` to write separate `.vec`/`.vemf` files. The delegate's
`.vec`/`.vemf` contain only non-reordered fields (or are empty if all fields are reordered).
`finish()` still writes footers to the delegate's streams (harmless). The reader checks for
reordered files first.

**Option C: Skip the delegate entirely during merge.** Write ALL fields (reordered and not)
through our own writer. The delegate's `.vec`/`.vemf` end up as valid but empty files (just
headers + sentinel + footers). Simplest but wastes two small files.

### Lucene99FlatVectorsWriter.finish() Behavior

```java
public void finish() throws IOException {
    finished = true;
    if (meta != null) {
        meta.writeInt(-1);           // end-of-fields sentinel
        CodecUtil.writeFooter(meta); // .vemf footer
    }
    if (vectorData != null) {
        CodecUtil.writeFooter(vectorData); // .vec footer
    }
}
```

Field-agnostic. Safe to call even if no fields were written via `mergeOneField()`.

### Memory Profile

| Component | Size |
|-----------|------|
| `int[] permutation` (newOrd2Old) | 4 * N bytes |
| `int[] oldOrd2New` (inverted) | 4 * N bytes |
| `int[] ordToDocId` | 4 * N bytes |
| `int[] segmentStarts` | 4 * numSourceSegments bytes |
| BP internal metadata | 2 * 4 * N bytes (sortedIds + biases) |
| Vector data on heap | 0 (all mmap) |
| **Total for 2M vectors** | **~40 MB** |

Compare to current replacement-based: same ~40 MB for permutation computation, but avoids
the second full .vec write pass (~8 GB I/O for 2M × 1024-dim).

### Sequence Diagram

```
NativeEngines990KnnVectorsFormat.fieldsWriter(state)
│
└─ new NativeEngines990KnnVectorsWriter(state,
       new ReorderAwareFlatVectorsWriter(state, scorer),  // replaces Lucene99FlatVectorsWriter
       ...)

NativeEngines990KnnVectorsWriter.mergeOneField(fieldInfo, mergeState)
│
├─ if (field needs reorder && totalLiveDocs >= 10k):
│   │
│   ├─ [Build mappings] From MergeState.docMaps + per-segment ordToDoc()
│   │   └─ Produces: mergedOrdToDocId[], segmentStarts[], liveLocalOrds[][]
│   │      No forward pass needed — pure metadata iteration
│   │
│   ├─ [Pass 1] Build MergedRandomAccessFloatVectorValues from source segment readers
│   │   └─ Each source: mergeState.knnVectorsReaders[i].getFloatVectorValues(fieldName)
│   │      (mmap-backed, random access, supports copy())
│   │      Uses segmentStarts[] and liveLocalOrds[][] for ord translation
│   │
│   ├─ [Pass 1] Compute permutation
│   │   └─ strategy.computePermutation(mergedRandomAccess, numThreads, similarity)
│   │      BP reads via vectorValue(ord) → mmap page fault → OS page cache
│   │
│   ├─ [Pass 2] Write .vec in permuted order directly into shared IndexOutput
│   │   └─ IndexOutput vectorData = reorderAwareFlatVectorsWriter.getVectorDataOutput()
│   │      for newOrd in 0..N-1:
│   │          oldOrd = permutation[newOrd]
│   │          vec = mergedRandomAccess.vectorValue(oldOrd)  // mmap read
│   │          vectorData.writeBytes(...)
│   │
│   ├─ [Pass 2] Write .vemf metadata + skip list into shared IndexOutput
│   │   └─ IndexOutput meta = reorderAwareFlatVectorsWriter.getMetaOutput()
│   │      Write field metadata, then FixedBlockSkipListIndexBuilder for doc→ord mapping
│   │
│   └─ [Pass 2] Build .faiss with reordered KNNVectorValues supplier
│       └─ NativeIndexWriter.mergeIndex(reorderedSupplier, totalLiveDocs)
│          HNSW graph built in permuted order — no remapping needed
│
├─ else (no reorder):
│   └─ reorderAwareFlatVectorsWriter.mergeOneField(fieldInfo, mergeState)  // standard path
│
└─ (continues to next field)

NativeEngines990KnnVectorsWriter.finish()
│
├─ reorderAwareFlatVectorsWriter.finish()  // writes sentinel + footers for ALL fields
└─ (no reorder loop needed — already done in mergeOneField)
```

## RESOLVED: Doc ID Mapping Without Forward Pass

We can compute `mergedOrd → mergedDocID` entirely from `MergeState.docMaps` and per-segment
`FloatVectorValues.ordToDoc()`, without iterating `MergedFloat32VectorValues`.

### How MergeState.DocMap Works

`DocMap` is `@FunctionalInterface` with `int get(int oldDocID)` — maps a source segment's
doc ID to the merged segment's doc ID. Returns `-1` for deleted docs.

In the common case (no index sort), `buildDeletionDocMaps` creates:

```java
// Per source segment i:
docMaps[i] = docID -> {
    if (liveDocs == null) {
        return docBase + docID;                       // no deletions: just rebase
    } else if (liveDocs.get(docID)) {
        return docBase + (int) delDocMap.get(docID);  // skip deleted, rebase
    } else {
        return -1;                                    // deleted
    }
};
```

Where `docBase` is the cumulative live doc count from prior segments.

### Per-Segment Ord-to-Doc Mapping

Each source segment's `FloatVectorValues` has `ordToDoc(int ord)`:
- **Dense** (`DenseOffHeapVectorValues`): `ordToDoc(ord) = ord` — vector ord equals doc ID
- **Sparse** (`SparseOffHeapVectorValues`): `ordToDoc(ord) = DirectMonotonicReader.get(ord)`

Both are O(1) random access, no iteration needed.

### Algorithm: Build All Mappings in One Pass Over Source Segment Metadata

```java
int mergedOrd = 0;
int[] mergedOrdToDocId = new int[totalLiveDocs];
// segmentStarts[seg] = first mergedOrd belonging to source segment seg
int[] segmentStarts = new int[numSourceSegments + 1];
// Per-segment: liveLocalOrds[seg][liveIdx] = rawLocalOrd (only needed if segment has deletions)
int[][] liveLocalOrds = new int[numSourceSegments][];

for (int seg = 0; seg < mergeState.knnVectorsReaders.length; seg++) {
    FloatVectorValues segValues = mergeState.knnVectorsReaders[seg]
        .getFloatVectorValues(fieldInfo.name);
    if (segValues == null) {
        segmentStarts[seg] = mergedOrd;
        continue;
    }

    segmentStarts[seg] = mergedOrd;
    DocMap docMap = mergeState.docMaps[seg];
    Bits liveDocs = mergeState.liveDocs[seg];
    boolean hasDeletions = (liveDocs != null);
    int[] segLiveOrds = hasDeletions ? new int[segValues.size()] : null;
    int liveCount = 0;

    for (int localOrd = 0; localOrd < segValues.size(); localOrd++) {
        int sourceDocId = segValues.ordToDoc(localOrd);  // O(1)
        int mergedDocId = docMap.get(sourceDocId);
        if (mergedDocId == -1) continue;  // deleted

        mergedOrdToDocId[mergedOrd] = mergedDocId;
        if (hasDeletions) {
            segLiveOrds[liveCount] = localOrd;
        }
        liveCount++;
        mergedOrd++;
    }

    if (hasDeletions) {
        liveLocalOrds[seg] = Arrays.copyOf(segLiveOrds, liveCount);
    }
}
segmentStarts[numSourceSegments] = mergedOrd;
```

This gives us:
- `mergedOrdToDocId[]` — the doc ID for each merged ord
- `segmentStarts[]` — segment boundaries in merged ord space
- `liveLocalOrds[][]` — per-segment mapping from live-index to raw local ord (only for
  segments with deletions; null for segments without deletions where liveIdx == rawLocalOrd)

### MergedRandomAccessFloatVectorValues Uses These Directly

```java
float[] vectorValue(int mergedOrd) {
    int seg = findSegment(mergedOrd, segmentStarts);  // binary search
    int liveIdx = mergedOrd - segmentStarts[seg];
    int rawLocalOrd = (liveLocalOrds[seg] != null)
        ? liveLocalOrds[seg][liveIdx]   // segment has deletions: indirect lookup
        : liveIdx;                       // no deletions: liveIdx == rawLocalOrd
    return segmentValues[seg].vectorValue(rawLocalOrd);  // mmap random access
}
```

### Index-Sort Complication

When `mergeState.needsIndexSort == true`, merged doc IDs are not in segment-sequential order —
they're interleaved by the sort. The `MergedFloat32VectorValues` uses a `SortedDocIDMerger`
(priority queue) to yield docs in ascending merged-doc-ID order.

Our segment-sequential construction above produces merged ords in a different order than what
`Lucene99FlatVectorsWriter.mergeOneField()` would produce. Fix: sort by `mergedOrdToDocId`
ascending after construction, then reassign merged ords:

```java
if (mergeState.needsIndexSort) {
    Integer[] sortIndices = IntStream.range(0, mergedOrd).boxed().toArray(Integer[]::new);
    Arrays.sort(sortIndices, Comparator.comparingInt(i -> mergedOrdToDocId[i]));
    // Apply permutation to all arrays (mergedOrdToDocId, segmentStarts, liveLocalOrds)
    // ... rebuild segmentStarts from the sorted order
}
```

This is rare (index sort is uncommon for k-NN workloads) and cheap (sorting int[] indices).

### No-Index-Sort Optimization (Common Case)

For dense segments without deletions: `mergedOrdToDocId[ord] = segmentDocBase + ord`.
Could detect this and skip the array, using arithmetic. But the array is only `4 * N` bytes
(~8 MB for 2M vectors), so the optimization is minor.

### Summary

No forward pass over `MergedFloat32VectorValues` needed. All mappings are computed from:
- `mergeState.docMaps[seg].get(docId)` — source doc ID → merged doc ID
- `segValues.ordToDoc(localOrd)` — source vector ord → source doc ID
- `mergeState.liveDocs[seg]` — which docs are deleted

Both are O(1) random access. The construction loop iterates source segment ords sequentially
(not vectors — just integer metadata), touching no vector data.

## Remaining Open Questions

1. **RESOLVED: Accessing delegate IndexOutput streams.** Replace `Lucene99FlatVectorsWriter`
   with our own `ReorderAwareFlatVectorsWriter` that extends `FlatVectorsWriter` and exposes
   `getMetaOutput()` / `getVectorDataOutput()`. This avoids reflection and gives full control.

   `Lucene99FlatVectorsWriter` is `public final` with `private final` fields — can't subclass
   or access without reflection. Instead, we replicate its logic (~200 lines of straightforward
   write code) in our own class with accessible fields. This follows the existing k-NN pattern
   of maintaining custom copies of Lucene codec writers (e.g., `Lucene99ScalarQuantizedVectorsWriter`
   is already a full copy in `src/main/java/org/apache/lucene/backward_codecs/`).

   **Changes:**
   - New class: `ReorderAwareFlatVectorsWriter extends FlatVectorsWriter`
     - Same `.vec`/`.vemf` file creation, codec headers, `mergeOneField()`, `flush()`,
       `finish()`, `close()` as `Lucene99FlatVectorsWriter`
     - Exposes `getMetaOutput()` and `getVectorDataOutput()` for reorder to write directly
   - `NativeEngines990KnnVectorsFormat.fieldsWriter()`: use `new ReorderAwareFlatVectorsWriter(state, scorer)`
     instead of `flatVectorsFormat.fieldsWriter(state)`
   - `NativeEngines990KnnVectorsWriter.mergeOneField()`: for reordered fields, call
     `((ReorderAwareFlatVectorsWriter) flatVectorsWriter).getVectorDataOutput()` to write
     vectors in permuted order directly into the shared `.vec` stream

2. **RESOLVED: Non-float vector types.** Not a concern — reorder only applies to FLOAT32.

3. **RESOLVED: Interaction with quantization.** Not a concern — quantization training is
   order-independent. Only the final index layout matters.

4. **TODO: Building .faiss in permuted order.** Pass a reordered `Supplier<KNNVectorValues<?>>`
   to `NativeIndexWriter.mergeIndex()` where vectors come out in permuted order. The FAISS
   `ordToDocs` ID map needs to map `newOrd → mergedDocId[permutation[newOrd]]`. Need to verify
   the `NativeIndexWriter` / JNI path can accept this without changes to the HNSW build or
   ID map construction.

5. **RESOLVED: Empty delegate files.** Not a concern — all fields write into the same
   `.vec`/`.vemf` files via `ReorderAwareFlatVectorsWriter`. No empty files.
