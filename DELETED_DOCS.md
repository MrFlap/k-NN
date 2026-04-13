# Deleted Docs Not Excluded During Replacement-Free Reorder Merge

## Summary

When a reordered segment has deleted documents and is used as a source in a subsequent merge,
`MergeOrdMappingBuilder` does not skip the deleted vectors. All vectors (including deleted
ones) are written into the new merged segment's `.vec` and FAISS index. Search still works
correctly because Lucene's `liveDocs` filters deleted docs at query time, but the merge
produces a segment with extra vectors that should have been purged.

## Observed Behavior

Test scenario:
1. Index 15k docs, force merge → 1 reordered segment (15000 vectors)
2. Delete 10 docs, flush
3. Index 15k more docs, force merge → merges reordered segment (with 10 deletions) + new segments

Expected: merged segment has 29990 vectors
Actual: merged segment has 30000 vectors (`docs.deleted=20` in `_cat/segments`)

## Root Cause

During the second merge, `MergeOrdMappingBuilder` iterates the reordered segment's vectors:

```
seg=1 liveDocs class=FixedBitSet len=15000 segSize=15000 deadDocs=[] totalDead=0
seg=1 segSize=15000 hasDeletions=true liveCount=15000 skipped=0
```

`mergeState.liveDocs[seg]` is non-null (`hasDeletions=true`) but the `FixedBitSet` has ALL
15000 bits set — zero dead docs. This means `docMap.get(sourceDocId)` never returns -1, so
all 15000 vectors pass through.

The `liveDocs` comes from `reader.getLiveDocs()` in Lucene's `MergeState` constructor:
```java
liveDocs[i] = reader.getLiveDocs();
```

For the reordered segment with 10 deleted docs, `reader.getLiveDocs()` returns a `FixedBitSet`
with all bits set. The deletions are tracked at the `IndexWriter` level (visible in
`_cat/segments` as `deleted=10`) but the `FixedBitSet` loaded by the segment reader does not
reflect them.

## Impact

- **Search correctness: NOT affected.** Deleted docs do not appear in search results. Lucene's
  query-time `liveDocs` filtering works correctly.
- **Merge efficiency: affected.** Deleted vectors are included in the merged segment, wasting
  disk space and HNSW graph capacity. A subsequent force merge would compact them.
- **Vector count: inflated.** The merged segment reports more vectors than expected. The extra
  vectors are marked as deleted in the segment metadata.

## Investigation Notes

Debug logging in `MergeOrdMappingBuilder` confirmed:
- `mergeState.liveDocs[seg]` is a `FixedBitSet` with `length=15000` and 0 dead bits
- `ordToDoc` correctly returns doc IDs from the `ordToDocMap` (read from `.vemf`)
- `docMap.get(sourceDocId)` returns valid merged IDs for all 15000 docs (never -1)
- The `UnifiedFlatVectorsReader` correctly identifies the segment as `reordered=true` with
  `ordToDocMap` of length 15000

The `FixedBitSet` with all bits set suggests `SegmentReader.getLiveDocs()` is not loading
the deletion information for this segment. This may be because:
1. The `.liv` file is not being read correctly for reordered segments
2. The deletions are tracked in a separate tombstone segment (`_e` with `docs=0, deleted=10`)
   and not reflected in the source segment's `liveDocs`
3. A Lucene-level issue with how `liveDocs` is computed for segments participating in a merge

## Reproduction

```bash
# Generate test data and run
python3 /tmp/test_delete_remerge.py
# Check segments
curl "localhost:9200/_cat/segments/test-delete-remerge?v"
```

## Status

Open — needs further investigation into why `reader.getLiveDocs()` returns all-live for a
segment with known deletions. This issue affects both dense and sparse reordered segments
and is independent of the sparse support changes.
