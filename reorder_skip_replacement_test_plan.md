# Test Plan: Replacement-Free Reordering Verification

## Goal

Verify that replacement-free reordering produces the same functional result as replacement-based
reordering. Specifically, after a segment is created with reordering, the following must be
identical between both approaches:

1. **Permutation:** `newOrd2Old[]` and `oldOrd2New[]` arrays
2. **Doc-to-ord skip list:** For every doc ID, `skipTo(docId).getOrd()` returns the same ord
3. **Vector content at each ord:** `vectorValue(ord)` returns the same vector
4. **FAISS graph structure:** HNSW neighbor lists reference the same (reordered) ords
5. **Search results:** Same doc IDs and scores for identical queries

## The Core Contract

After reordering, the reader (`ReorderedOffHeapFloatVectorValues111`) provides:

- `vectorValue(ord)` — reads vector at position `ord` in the reordered `.vec` file
- `iterator().index()` — translates doc ID → reordered ord via the skip list:
  ```java
  public int index() {
      docIdOrdSkipListIndex.skipTo(doc);
      return docIdOrdSkipListIndex.getOrd();
  }
  ```

The invariant: for every doc ID `d` that has a vector:
```
vectorValue(skipList.getOrd(d)) == original vector for doc d
```

Both replacement-based and replacement-free must produce the same skip list mapping and the
same vector layout in `.vec`.

## Approach 1: Side-by-Side Comparison Test

Run both reorder paths on the same input data, then compare the on-disk artifacts.

### Setup

1. Create a set of source segments with known vectors (e.g., 3 segments of 5k vectors each,
   interleaved clusters for non-trivial BP permutation)
2. Force merge to 1 segment using replacement-based reordering → capture artifacts
3. Force merge the same source segments using replacement-free reordering → capture artifacts
4. Compare

### What to Compare

**a) Permutation identity:**
Both paths call `strategy.computePermutation()` on the same `FloatVectorValues`. If the input
vectors are identical and the strategy is deterministic (BP with same thread count and seed),
the permutation must be identical.

- Extract `int[] permutation` from both paths (log it or expose via test hook)
- `assertArrayEquals(replacementPermutation, replacementFreePermutation)`

**b) Skip list consistency:**
Open the `.vemf` with `ReorderedLucene99FlatVectorsReader111`, iterate all doc IDs, verify
`skipTo(docId).getOrd()` matches between both segments.

```java
for (int docId = 0; docId < maxDoc; docId++) {
    replacementSkipList.skipTo(docId);
    replacementFreeSkipList.skipTo(docId);
    assertEquals(replacementSkipList.getOrd(), replacementFreeSkipList.getOrd());
}
```

**c) Vector content at each ord:**
For every ord `0..N-1`, read `vectorValue(ord)` from both segments' `.vec` files and compare.

```java
for (int ord = 0; ord < numVectors; ord++) {
    assertArrayEquals(
        replacementValues.vectorValue(ord),
        replacementFreeValues.vectorValue(ord),
        0.0f  // exact match, no tolerance
    );
}
```

**d) Round-trip doc→ord→vector consistency:**
For every doc ID, verify the full chain produces the same vector:

```java
for (int docId = 0; docId < maxDoc; docId++) {
    int ord = skipList.skipTo(docId).getOrd();
    float[] vec = vectorValues.vectorValue(ord);
    // This vector should equal the original vector for docId
    assertArrayEquals(originalVectors[docId], vec, 0.0f);
}
```

### Implementation

This can be a unit test that:
1. Uses `RAMDirectory` or `MMapDirectory` in a temp dir
2. Creates source segments via `IndexWriter` with known vectors
3. Runs merge with replacement-based writer → reads back artifacts
4. Resets, runs merge with replacement-free writer → reads back artifacts
5. Compares all four properties above

## Approach 2: Property-Based Invariant Test

Instead of comparing two implementations, verify that the replacement-free output satisfies
the reorder contract independently.

### Invariants to Check

**Invariant 1: Permutation is valid bijection**
```java
int[] newOrd2Old = ...;  // from the reorder
Set<Integer> seen = new HashSet<>();
for (int i = 0; i < n; i++) {
    assertTrue(newOrd2Old[i] >= 0 && newOrd2Old[i] < n);
    assertTrue(seen.add(newOrd2Old[i]));  // no duplicates
}
```

**Invariant 2: Skip list round-trip**
For every doc ID `d`, `skipList.getOrd(d)` returns an ord in `[0, N)`, and the mapping is
a bijection over docs that have vectors.
```java
Set<Integer> ords = new HashSet<>();
for (int docId = 0; docId < maxDoc; docId++) {
    skipList.skipTo(docId);
    int ord = skipList.getOrd();
    assertTrue(ord >= 0 && ord < numVectors);
    assertTrue(ords.add(ord));  // each doc maps to a unique ord
}
assertEquals(numVectors, ords.size());
```

**Invariant 3: Vector content preservation**
Every original vector appears exactly once in the reordered `.vec`, at the ord assigned by
the permutation.
```java
for (int docId = 0; docId < maxDoc; docId++) {
    skipList.skipTo(docId);
    int ord = skipList.getOrd();
    float[] reorderedVec = reorderedValues.vectorValue(ord);
    float[] originalVec = originalVectors[docId];
    assertArrayEquals(originalVec, reorderedVec, 0.0f);
}
```

**Invariant 4: newOrd2Old and skip list are consistent**
The skip list encodes `oldOrd2New` (doc→reorderedOrd). The permutation is `newOrd2Old`.
They must be inverses:
```java
for (int docId = 0; docId < maxDoc; docId++) {
    skipList.skipTo(docId);
    int reorderedOrd = skipList.getOrd();
    // In the dense case, docId == originalOrd
    assertEquals(docId, newOrd2Old[reorderedOrd]);
}
```

### Implementation

A self-contained unit test that:
1. Creates known vectors (e.g., 15k vectors in 3 clusters)
2. Runs the replacement-free merge path
3. Opens the resulting segment with `ReorderedLucene99FlatVectorsReader111`
4. Checks all four invariants above

## Approach 3: Search Result Equivalence Test

End-to-end test that verifies search correctness without inspecting internal structures.

### Steps

1. Index N vectors (e.g., 15k) across multiple segments
2. Run k-NN search queries, capture results (doc IDs + scores) as baseline
3. Force merge with replacement-free reordering
4. Run the same k-NN search queries
5. Assert identical doc IDs and scores

This is the highest-level test — it doesn't verify internal ord mappings but confirms the
user-visible behavior is correct. It's the same test pattern already used in the manual
testing documented in `INTEGRATE_REORDER.md`.

### Variant: Compare Against Replacement-Based

1. Index same vectors, force merge with replacement-based → search → capture results A
2. Index same vectors, force merge with replacement-free → search → capture results B
3. `assertEquals(resultsA, resultsB)`

## Recommended Test Harness

### Shared Test Fixture

Create a base class or helper that:
1. Generates deterministic test vectors (e.g., 3 clusters of 5k vectors each, 128-dim)
2. Creates source segments in a `Directory`
3. Provides a `MergeState` for the merge
4. Exposes the `FloatVectorValues` from source segments for permutation computation

Both replacement-based and replacement-free tests use this same fixture.

### Comparison Utility

```java
class ReorderVerifier {
    /**
     * Given original vectors (indexed by docId) and a reordered segment,
     * verify all reorder invariants.
     */
    static void verifyReorderedSegment(
        float[][] originalVectors,       // originalVectors[docId] = vector
        FloatVectorValues reorderedValues, // from ReorderedLucene99FlatVectorsReader111
        FixedBlockSkipListIndexReader skipList,
        int[] newOrd2Old                  // permutation, if available
    ) {
        int n = originalVectors.length;

        // Invariant 1: permutation is valid bijection
        if (newOrd2Old != null) {
            Set<Integer> seen = new HashSet<>();
            for (int i = 0; i < n; i++) {
                assertTrue(newOrd2Old[i] >= 0 && newOrd2Old[i] < n);
                assertTrue(seen.add(newOrd2Old[i]));
            }
        }

        // Invariant 2: skip list is bijection
        Set<Integer> ords = new HashSet<>();
        for (int docId = 0; docId < n; docId++) {
            skipList.skipTo(docId);
            int ord = skipList.getOrd();
            assertTrue(ord >= 0 && ord < n);
            assertTrue(ords.add(ord));
        }

        // Invariant 3: vector content preserved
        for (int docId = 0; docId < n; docId++) {
            skipList.skipTo(docId);
            int ord = skipList.getOrd();
            float[] reorderedVec = reorderedValues.vectorValue(ord);
            assertArrayEquals(originalVectors[docId], reorderedVec, 0.0f);
        }

        // Invariant 4: newOrd2Old and skip list consistent
        if (newOrd2Old != null) {
            for (int docId = 0; docId < n; docId++) {
                skipList.skipTo(docId);
                int reorderedOrd = skipList.getOrd();
                assertEquals(docId, newOrd2Old[reorderedOrd]);
            }
        }
    }

    /**
     * Compare two reordered segments (replacement-based vs replacement-free).
     */
    static void compareReorderedSegments(
        FloatVectorValues valuesA, FixedBlockSkipListIndexReader skipListA,
        FloatVectorValues valuesB, FixedBlockSkipListIndexReader skipListB,
        int numVectors, int maxDoc
    ) {
        // Same skip list mapping
        for (int docId = 0; docId < maxDoc; docId++) {
            skipListA.skipTo(docId);
            skipListB.skipTo(docId);
            assertEquals(skipListA.getOrd(), skipListB.getOrd());
        }

        // Same vector at each ord
        for (int ord = 0; ord < numVectors; ord++) {
            assertArrayEquals(valuesA.vectorValue(ord), valuesB.vectorValue(ord), 0.0f);
        }
    }
}
```

## Test Matrix

| Test | What it verifies | Level | Needs both paths? |
|------|-----------------|-------|-------------------|
| Permutation equality | Same `newOrd2Old[]` | Unit | Yes |
| Skip list equality | Same doc→ord mapping | Unit | Yes |
| Vector layout equality | Same `.vec` content | Unit | Yes |
| Permutation bijection | Valid permutation | Unit | No |
| Skip list bijection | Valid doc→ord mapping | Unit | No |
| Vector preservation | No vectors lost/corrupted | Unit | No |
| newOrd2Old ↔ skip list | Inverse consistency | Unit | No |
| Search result equality | Same k-NN results | Integration | Yes |

## Handling Non-Determinism

BP reordering uses `ForkJoinPool` which may produce different permutations with different
thread scheduling. To get deterministic comparison:

- **Option A:** Use `numThreads=1` for both paths — eliminates thread scheduling variance
- **Option B:** Extract the permutation from the replacement-based path, feed it to the
  replacement-free path as a fixed permutation (bypass `computePermutation()`)
- **Option C:** Don't compare permutations directly — only verify the invariants
  (bijection, vector preservation, skip list consistency) independently for each path,
  then compare search results

Option C is the most robust since it doesn't require deterministic permutations.

## Automated Comparison via Index Setting

See `reorder_implementation_setting.md` for the full setting design.

A new index-level setting `index.knn.advanced.reorder_implementation` (`replacement` |
`replacement_free`) controls which code path is used, orthogonal to the strategy setting.
This enables automated side-by-side testing with two indices on the same cluster.

### Integration Test: Two-Index Comparison

```java
public void testReplacementFreeMatchesReplacement() {
    float[][] vectors = generateClusteredVectors(15000, 128, 3);
    float[] query = randomVector(128);

    // Index A: replacement-based (current)
    createIndex("index-a", Map.of(
        "index.knn.advanced.reorder_strategy", "bp",
        "index.knn.advanced.reorder_implementation", "replacement"
    ));
    bulkIndex("index-a", vectors);
    forceMerge("index-a", 1);

    // Index B: replacement-free (new)
    createIndex("index-b", Map.of(
        "index.knn.advanced.reorder_strategy", "bp",
        "index.knn.advanced.reorder_implementation", "replacement_free"
    ));
    bulkIndex("index-b", vectors);
    forceMerge("index-b", 1);

    // Compare search results — must be identical
    var resultsA = knnSearch("index-a", query, 10);
    var resultsB = knnSearch("index-b", query, 10);
    assertEquals(resultsA.docIds(), resultsB.docIds());
    assertArrayEquals(resultsA.scores(), resultsB.scores(), 0.0f);

    // Run invariant checks on both independently
    verifyReorderInvariants("index-a", vectors);
    verifyReorderInvariants("index-b", vectors);
}
```

### Why This Works

- Both indices get the same vectors in the same order → same source segments
- Both use the same strategy (bp) → same permutation algorithm
- The only difference is the implementation path
- Search results must match because the reordered segment encodes the same logical mapping
- Non-determinism from BP threading doesn't matter: each index computes its own permutation
  independently, and we verify invariants (not permutation equality) per-index, then compare
  search results which are permutation-independent (same vectors, same HNSW graph quality)

### Test Variants

| Test | Strategy | Implementation A | Implementation B | Assertion |
|------|----------|-----------------|-----------------|-----------|
| BP basic | bp | replacement | replacement_free | Search results equal |
| KMeans basic | kmeans | replacement | replacement_free | Search results equal |
| With deletions | bp | replacement | replacement_free | Search results equal after delete+merge |
| Multiple fields | bp | replacement | replacement_free | Both fields correct |
| Below threshold | bp | replacement | replacement_free | Neither reordered (< 10k vectors) |
| Mixed fields | bp | replacement | replacement_free | One field reordered, one not |

### Iteration Workflow

1. Add the setting (default `replacement`) — zero behavior change, ship it
2. Implement replacement-free behind the setting — iterate freely
3. Run the two-index comparison test on every change
4. Run OSB benchmarks with both settings for performance comparison
5. Flip default to `replacement_free` after validation
6. Remove replacement path and setting once stable
