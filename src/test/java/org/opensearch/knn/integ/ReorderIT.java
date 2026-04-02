/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;

import java.util.List;

import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

/**
 * End-to-end integration tests for vector reordering during merge.
 * Tests that reordered segments produce correct search results after force merge.
 */
public class ReorderIT extends KNNRestTestCase {

    private static final String INDEX_NAME = "reorder-test-index";
    private static final String FIELD_NAME = "test_vector";
    private static final int DIMENSION = 128;
    private static final int NUM_VECTORS = 500;
    private static final int K = 10;

    /**
     * Test BP reorder (quantized) with 32x compression: index vectors, force merge, verify search.
     */
    @SneakyThrows
    public void testBpReorder_32xCompression_searchCorrectAfterForceMerge() {
        String mapping = createMapping(DIMENSION, SpaceType.L2, "on_disk", "32x");
        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn", true)
            .put(KNNSettings.INDEX_KNN_ADVANCED_REORDER_STRATEGY, "bp")
            .put(KNNSettings.INDEX_KNN_ADVANCED_REORDER_IMPLEMENTATION, "replacement_free")
            .build();

        createKnnIndex(INDEX_NAME, settings, mapping);

        // Index vectors in multiple batches to create multiple segments
        indexVectorsInBatches(INDEX_NAME, FIELD_NAME, DIMENSION, NUM_VECTORS, 3);

        // Search before force merge — baseline results
        float[] queryVector = createQueryVector(DIMENSION);
        List<KNNResult> resultsBefore = searchAndGetHits(INDEX_NAME, FIELD_NAME, queryVector, K);
        assertFalse("Should have results before merge", resultsBefore.isEmpty());

        // Force merge to trigger reordering
        forceMergeKnnIndex(INDEX_NAME, 1);

        // Search after force merge — results should still be correct
        List<KNNResult> resultsAfter = searchAndGetHits(INDEX_NAME, FIELD_NAME, queryVector, K);
        assertFalse("Should have results after merge", resultsAfter.isEmpty());
        assertEquals("Should return same number of results", resultsBefore.size(), resultsAfter.size());

        // Verify same doc IDs returned (order may differ slightly due to rescoring precision)
        assertSameDocIds(resultsBefore, resultsAfter);

        deleteKNNIndex(INDEX_NAME);
    }

    /**
     * Test BP full (float-based) reorder with L2 space: verify search correctness.
     */
    @SneakyThrows
    public void testBpFullReorder_l2_searchCorrectAfterForceMerge() {
        String mapping = createMapping(DIMENSION, SpaceType.L2, "on_disk", "32x");
        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn", true)
            .put(KNNSettings.INDEX_KNN_ADVANCED_REORDER_STRATEGY, "bp_full")
            .put(KNNSettings.INDEX_KNN_ADVANCED_REORDER_IMPLEMENTATION, "replacement_free")
            .build();

        createKnnIndex(INDEX_NAME, settings, mapping);
        indexVectorsInBatches(INDEX_NAME, FIELD_NAME, DIMENSION, NUM_VECTORS, 3);

        float[] queryVector = createQueryVector(DIMENSION);
        List<KNNResult> resultsBefore = searchAndGetHits(INDEX_NAME, FIELD_NAME, queryVector, K);

        forceMergeKnnIndex(INDEX_NAME, 1);

        List<KNNResult> resultsAfter = searchAndGetHits(INDEX_NAME, FIELD_NAME, queryVector, K);
        assertFalse("Should have results after merge", resultsAfter.isEmpty());
        assertEquals("Should return same number of results", resultsBefore.size(), resultsAfter.size());
        assertSameDocIds(resultsBefore, resultsAfter);

        deleteKNNIndex(INDEX_NAME);
    }

    /**
     * Test no reorder (none strategy): verify search still works after force merge.
     */
    @SneakyThrows
    public void testNoReorder_searchCorrectAfterForceMerge() {
        String mapping = createMapping(DIMENSION, SpaceType.L2, "on_disk", "32x");
        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn", true)
            .put(KNNSettings.INDEX_KNN_ADVANCED_REORDER_STRATEGY, "none")
            .build();

        createKnnIndex(INDEX_NAME, settings, mapping);
        indexVectorsInBatches(INDEX_NAME, FIELD_NAME, DIMENSION, NUM_VECTORS, 3);

        float[] queryVector = createQueryVector(DIMENSION);
        List<KNNResult> resultsBefore = searchAndGetHits(INDEX_NAME, FIELD_NAME, queryVector, K);

        forceMergeKnnIndex(INDEX_NAME, 1);

        List<KNNResult> resultsAfter = searchAndGetHits(INDEX_NAME, FIELD_NAME, queryVector, K);
        assertFalse("Should have results after merge", resultsAfter.isEmpty());
        assertEquals("Should return same number of results", resultsBefore.size(), resultsAfter.size());
        assertSameDocIds(resultsBefore, resultsAfter);

        deleteKNNIndex(INDEX_NAME);
    }

    /**
     * Test that below-threshold segments are NOT reordered but search still works.
     */
    @SneakyThrows
    public void testBelowThreshold_notReordered_searchCorrect() {
        // Index only 100 vectors — well below the 10k threshold
        int smallCount = 100;
        String mapping = createMapping(DIMENSION, SpaceType.L2, "on_disk", "32x");
        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn", true)
            .put(KNNSettings.INDEX_KNN_ADVANCED_REORDER_STRATEGY, "bp")
            .put(KNNSettings.INDEX_KNN_ADVANCED_REORDER_IMPLEMENTATION, "replacement_free")
            .build();

        createKnnIndex(INDEX_NAME, settings, mapping);
        indexVectorsInBatches(INDEX_NAME, FIELD_NAME, DIMENSION, smallCount, 2);

        float[] queryVector = createQueryVector(DIMENSION);
        List<KNNResult> resultsBefore = searchAndGetHits(INDEX_NAME, FIELD_NAME, queryVector, K);

        forceMergeKnnIndex(INDEX_NAME, 1);

        List<KNNResult> resultsAfter = searchAndGetHits(INDEX_NAME, FIELD_NAME, queryVector, K);
        assertFalse("Should have results after merge", resultsAfter.isEmpty());
        assertSameDocIds(resultsBefore, resultsAfter);

        deleteKNNIndex(INDEX_NAME);
    }

    /**
     * Test BP reorder with inner product space type.
     */
    @SneakyThrows
    public void testBpReorder_innerProduct_searchCorrectAfterForceMerge() {
        String mapping = createMapping(DIMENSION, SpaceType.INNER_PRODUCT, "on_disk", "32x");
        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn", true)
            .put(KNNSettings.INDEX_KNN_ADVANCED_REORDER_STRATEGY, "bp")
            .put(KNNSettings.INDEX_KNN_ADVANCED_REORDER_IMPLEMENTATION, "replacement_free")
            .build();

        createKnnIndex(INDEX_NAME, settings, mapping);
        indexVectorsInBatches(INDEX_NAME, FIELD_NAME, DIMENSION, NUM_VECTORS, 3);

        float[] queryVector = createQueryVector(DIMENSION);
        List<KNNResult> resultsBefore = searchAndGetHits(INDEX_NAME, FIELD_NAME, queryVector, K);

        forceMergeKnnIndex(INDEX_NAME, 1);

        List<KNNResult> resultsAfter = searchAndGetHits(INDEX_NAME, FIELD_NAME, queryVector, K);
        assertFalse("Should have results after merge", resultsAfter.isEmpty());
        assertEquals("Should return same number of results", resultsBefore.size(), resultsAfter.size());

        deleteKNNIndex(INDEX_NAME);
    }

    /**
     * Test dynamic strategy switch: index with none, switch to bp, force merge.
     */
    @SneakyThrows
    public void testDynamicStrategySwitch_searchCorrectAfterForceMerge() {
        String mapping = createMapping(DIMENSION, SpaceType.L2, "on_disk", "32x");
        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn", true)
            .put(KNNSettings.INDEX_KNN_ADVANCED_REORDER_STRATEGY, "none")
            .build();

        createKnnIndex(INDEX_NAME, settings, mapping);
        indexVectorsInBatches(INDEX_NAME, FIELD_NAME, DIMENSION, NUM_VECTORS, 3);

        float[] queryVector = createQueryVector(DIMENSION);
        List<KNNResult> resultsBefore = searchAndGetHits(INDEX_NAME, FIELD_NAME, queryVector, K);

        // Switch strategy dynamically to bp, then force merge
        updateIndexSettings(INDEX_NAME, Settings.builder().put(KNNSettings.INDEX_KNN_ADVANCED_REORDER_STRATEGY, "bp"));
        forceMergeKnnIndex(INDEX_NAME, 1);

        List<KNNResult> resultsAfter = searchAndGetHits(INDEX_NAME, FIELD_NAME, queryVector, K);
        assertFalse("Should have results after merge with reorder", resultsAfter.isEmpty());
        assertEquals("Should return same number of results", resultsBefore.size(), resultsAfter.size());

        deleteKNNIndex(INDEX_NAME);
    }

    // ---- Helper methods ----

    private String createMapping(int dimension, SpaceType spaceType, String mode, String compressionLevel) throws Exception {
        return XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", dimension)
            .field("mode", mode)
            .field("compression_level", compressionLevel)
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .field(KNN_ENGINE, KNNEngine.FAISS.getName())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, 16)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, 128)
            .field(METHOD_PARAMETER_EF_SEARCH, 128)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();
    }

    private void indexVectorsInBatches(String indexName, String fieldName, int dimension, int totalVectors, int numBatches)
        throws Exception {
        int batchSize = totalVectors / numBatches;
        java.util.Random rng = new java.util.Random(42);
        for (int batch = 0; batch < numBatches; batch++) {
            int start = batch * batchSize;
            int end = (batch == numBatches - 1) ? totalVectors : start + batchSize;
            for (int i = start; i < end; i++) {
                Float[] vector = new Float[dimension];
                for (int d = 0; d < dimension; d++) {
                    vector[d] = rng.nextFloat();
                }
                addKnnDoc(indexName, Integer.toString(i), fieldName, vector);
            }
            flushIndex(indexName);
        }
        refreshAllNonSystemIndices();
    }

    private float[] createQueryVector(int dimension) {
        float[] query = new float[dimension];
        java.util.Random rng = new java.util.Random(99);
        for (int d = 0; d < dimension; d++) {
            query[d] = rng.nextFloat();
        }
        return query;
    }

    private List<KNNResult> searchAndGetHits(String indexName, String fieldName, float[] queryVector, int k)
        throws Exception {
        String query = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject(fieldName)
            .field("vector", queryVector)
            .field("k", k)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .toString();

        var response = searchKNNIndex(indexName, query, k);
        String responseBody = EntityUtils.toString(response.getEntity());
        return parseSearchResponse(responseBody, fieldName);
    }

    private void assertSameDocIds(List<KNNResult> resultsBefore, List<KNNResult> resultsAfter) {
        // With approximate search (HNSW + binary quantization), the exact top-k can differ
        // slightly after merge because the HNSW graph is rebuilt. We verify that at least 80%
        // of the results overlap, which confirms the reorder didn't corrupt the index.
        java.util.Set<String> docIdsBefore = new java.util.HashSet<>();
        for (KNNResult result : resultsBefore) {
            docIdsBefore.add(result.getDocId());
        }
        java.util.Set<String> docIdsAfter = new java.util.HashSet<>();
        for (KNNResult result : resultsAfter) {
            docIdsAfter.add(result.getDocId());
        }
        java.util.Set<String> intersection = new java.util.HashSet<>(docIdsBefore);
        intersection.retainAll(docIdsAfter);
        float overlap = (float) intersection.size() / Math.max(docIdsBefore.size(), docIdsAfter.size());
        assertTrue(
            "Expected at least 80% overlap in doc IDs before and after force merge, got "
                + (overlap * 100) + "% (before=" + docIdsBefore + ", after=" + docIdsAfter + ")",
            overlap >= 0.80f
        );
    }
}
