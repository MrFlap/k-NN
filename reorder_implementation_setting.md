# Reorder Implementation Setting

## New Index Setting

```
index.knn.advanced.reorder_implementation = replacement | replacement_free
```

Default: `replacement` (current behavior). Controls how the reorder permutation is applied
during merge, orthogonal to the existing strategy setting:

```
index.knn.advanced.reorder_strategy        → WHAT permutation to compute (bp, kmeans, none)
index.knn.advanced.reorder_implementation   → HOW to apply it (replacement, replacement_free)
```

## Usage

```json
PUT /my-index
{
  "settings": {
    "index.knn": true,
    "index.knn.advanced.reorder_strategy": "bp",
    "index.knn.advanced.reorder_implementation": "replacement_free"
  }
}
```

## Code Changes

### KNNSettings.java

```java
public static final String INDEX_KNN_ADVANCED_REORDER_IMPLEMENTATION = "index.knn.advanced.reorder_implementation";
public static final String INDEX_KNN_ADVANCED_REORDER_IMPLEMENTATION_DEFAULT = "replacement";

public static final Setting<String> INDEX_KNN_ADVANCED_REORDER_IMPLEMENTATION_SETTING = Setting.simpleString(
    INDEX_KNN_ADVANCED_REORDER_IMPLEMENTATION,
    INDEX_KNN_ADVANCED_REORDER_IMPLEMENTATION_DEFAULT,
    value -> {
        if (!value.equals("replacement") && !value.equals("replacement_free")) {
            throw new IllegalArgumentException(
                "[" + INDEX_KNN_ADVANCED_REORDER_IMPLEMENTATION + "] must be one of "
                + "[replacement, replacement_free] but was [" + value + "]"
            );
        }
    },
    IndexScope,
    Setting.Property.Dynamic
);
```

Register in `getSettings()` alongside the existing reorder settings.

### BasePerFieldKnnVectorsFormat.java

`nativeEngineVectorsFormat()` reads the new setting and passes it through:

```java
private NativeEngines990KnnVectorsFormat nativeEngineVectorsFormat() {
    final int approximateThreshold = getApproximateThresholdValue();
    final VectorReorderStrategy reorderStrategy = getReorderStrategy();
    final boolean replacementFree = "replacement_free".equals(
        mapperService.get().getIndexSettings()
            .getValue(KNNSettings.INDEX_KNN_ADVANCED_REORDER_IMPLEMENTATION_SETTING)
    );
    return new NativeEngines990KnnVectorsFormat(
        new Lucene99FlatVectorsFormat(FlatVectorScorerUtil.getLucene99FlatVectorsScorer()),
        approximateThreshold,
        nativeIndexBuildStrategyFactory,
        reorderStrategy,
        replacementFree
    );
}
```

### NativeEngines990KnnVectorsFormat.java

Accepts `boolean replacementFree`, passes to writer. When `replacementFree` is true,
creates `ReorderAwareFlatVectorsWriter` instead of `Lucene99FlatVectorsWriter`.

### NativeEngines990KnnVectorsWriter.mergeOneField()

Branches on the implementation mode:

```java
if (replacementFree && reorderStrategy != null && totalLiveDocs >= MIN_VECTORS_FOR_REORDER) {
    // New path: compute permutation, write .vec in permuted order directly
    mergeOneFieldReplacementFree(fieldInfo, mergeState);
} else {
    // Existing path: delegate writes .vec, mark for reorder in finish()
    flatVectorsWriter.mergeOneField(fieldInfo, mergeState);
    if (reorderStrategy != null && totalLiveDocs >= MIN_VECTORS_FOR_REORDER) {
        fieldsToReorder.add(fieldInfo);
    }
}
```

## Lifecycle

1. Add setting with default `replacement` — no behavior change
2. Implement replacement-free path behind `replacement_free` setting
3. Validate via automated comparison tests (two indices, same data, compare results)
4. Flip default to `replacement_free` after validation
5. Remove replacement path and setting once stable
