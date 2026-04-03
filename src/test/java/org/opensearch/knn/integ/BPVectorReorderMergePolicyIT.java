/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import com.google.common.primitives.Floats;
import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.BeforeClass;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNJsonIndexMappingsBuilder;
import org.opensearch.knn.KNNJsonQueryBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;
import java.net.URL;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.index.KNNSettings.KNN_BP_VECTOR_REORDER_ENABLED;

/**
 * Integration tests for the BP vector reorder merge policy.
 * Verifies that enabling index.knn.bp_vector_reorder.enabled wraps the merge policy
 * with BPReorderingMergePolicy and that search results remain correct after force merge.
 */
@Log4j2
public class BPVectorReorderMergePolicyIT extends KNNRestTestCase {

    private static final String INDEX_NAME = "bp-reorder-test-index";
    private static final String FIELD_NAME = "test_vector";
    private static final int DIMENSION = 128;

    private static TestUtils.TestData testData;

    @BeforeClass
    public static void setUpClass() throws IOException {
        if (BPVectorReorderMergePolicyIT.class.getClassLoader() == null) {
            throw new IllegalStateException("ClassLoader of BPVectorReorderMergePolicyIT is null");
        }
        URL testIndexVectors = BPVectorReorderMergePolicyIT.class.getClassLoader().getResource("data/test_vectors_1000x128.json");
        URL testQueries = BPVectorReorderMergePolicyIT.class.getClassLoader().getResource("data/test_queries_100x128.csv");
        URL groundTruthValues = BPVectorReorderMergePolicyIT.class.getClassLoader().getResource("data/test_ground_truth_l2_100.csv");
        assert testIndexVectors != null;
        assert testQueries != null;
        assert groundTruthValues != null;
        testData = new TestUtils.TestData(testIndexVectors.getPath(), testQueries.getPath(), groundTruthValues.getPath());
    }

    /**
     * End-to-end test: create a KNN index with BP reorder enabled, ingest 1000 vectors across
     * multiple flushes to create multiple segments, force merge to 1 segment (triggering the
     * reorder merge policy), then verify KNN search recall is still above 0.9.
     */
    @SneakyThrows
    public void testFaissHnsw_withBPReorder_whenForceMerge_thenRecallIsAboveNinePointZero() {
        String indexName = INDEX_NAME + "-recall";

        // Create index with BP reorder enabled
        createKnnHnswIndexWithBPReorder(KNNEngine.FAISS, indexName, FIELD_NAME, DIMENSION);

        // Ingest test data
        ingestTestData(indexName, FIELD_NAME);

        // Force merge to 1 segment — this triggers BPReorderingMergePolicy
        forceMergeKnnIndex(indexName, 1);

        // Verify segment count is 1
        assertEquals(1, getTotalSegmentCount(indexName));

        // Verify recall is still good after reorder
        int k = 100;
        for (int i = 0; i < testData.queries.length; i++) {
            List<KNNResult> knnResults = runKnnQuery(indexName, FIELD_NAME, testData.queries[i], k);
            float recall = getRecall(
                Set.of(Arrays.copyOf(testData.groundTruthValues[i], k)),
                knnResults.stream().map(KNNResult::getDocId).collect(Collectors.toSet())
            );
            assertTrue("Recall: " + recall + " for query " + i, recall > 0.9);
        }
    }

    /**
     * Test that the BP reorder setting is persisted and readable from index settings.
     */
    @SneakyThrows
    public void testBPReorderSetting_whenEnabled_thenSettingIsPersisted() {
        String indexName = INDEX_NAME + "-setting";

        createKnnHnswIndexWithBPReorder(KNNEngine.FAISS, indexName, FIELD_NAME, DIMENSION);

        String settingValue = getIndexSettingByName(indexName, KNN_BP_VECTOR_REORDER_ENABLED);
        assertEquals("true", settingValue);
    }

    /**
     * Test that search works correctly with BP reorder on a small dataset with mixed content
     * (vector docs + non-vector docs), verifying the merge policy handles segments with
     * missing vector fields gracefully.
     */
    @SneakyThrows
    public void testBPReorder_withMixedContent_whenForceMerge_thenSearchSucceeds() {
        String indexName = INDEX_NAME + "-mixed";

        // Create index with BP reorder and a mapping that has both vector and keyword fields
        String mapping = KNNJsonIndexMappingsBuilder.builder()
            .fieldName(FIELD_NAME)
            .dimension(DIMENSION)
            .vectorDataType(VectorDataType.FLOAT.getValue())
            .method(
                KNNJsonIndexMappingsBuilder.Method.builder()
                    .methodName(METHOD_HNSW)
                    .spaceType(SpaceType.L2.getValue())
                    .engine(KNNEngine.FAISS.getName())
                    .build()
            )
            .build()
            .getIndexMapping();

        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn", true)
            .put(KNN_BP_VECTOR_REORDER_ENABLED, true)
            .build();

        createKnnIndex(indexName, settings, mapping);

        // Add vector docs
        for (int i = 0; i < 20; i++) {
            addKnnDoc(indexName, String.valueOf(i), FIELD_NAME, Floats.asList(testData.indexData.vectors[i]).toArray());
        }

        // Add a non-vector doc to create a mixed segment
        addNonKNNDoc(indexName, "non-vector-1", "description", "This doc has no vector");

        assertEquals(21, getDocCount(indexName));

        // Force merge — the reorder should handle the non-vector doc gracefully
        forceMergeKnnIndex(indexName, 1);

        assertEquals(1, getTotalSegmentCount(indexName));

        // Search should still work
        List<KNNResult> results = runKnnQuery(indexName, FIELD_NAME, testData.queries[0], 5);
        assertTrue("Expected at least 1 result, got " + results.size(), results.size() > 0);
    }

    /**
     * Test that multiple flushes create multiple segments, and force merge with BP reorder
     * correctly merges them while maintaining search correctness.
     */
    @SneakyThrows
    public void testBPReorder_withMultipleSegments_whenForceMerge_thenSearchCorrect() {
        String indexName = INDEX_NAME + "-multi-seg";

        createKnnHnswIndexWithBPReorder(KNNEngine.FAISS, indexName, FIELD_NAME, DIMENSION);

        // Ingest in batches with flushes to create multiple segments
        int batchSize = 200;
        for (int batch = 0; batch < 5; batch++) {
            int start = batch * batchSize;
            int end = Math.min(start + batchSize, testData.indexData.docs.length);
            for (int i = start; i < end; i++) {
                addKnnDoc(
                    indexName,
                    Integer.toString(testData.indexData.docs[i]),
                    FIELD_NAME,
                    Floats.asList(testData.indexData.vectors[i]).toArray()
                );
            }
            flushIndex(indexName);
        }

        int segmentsBefore = getTotalSegmentCount(indexName);
        log.info("Segments before force merge: {}", segmentsBefore);
        assertTrue("Expected multiple segments before merge, got " + segmentsBefore, segmentsBefore > 1);

        // Force merge — triggers BPReorderingMergePolicy
        forceMergeKnnIndex(indexName, 1);

        // Verify doc count preserved
        assertEquals(testData.indexData.docs.length, getDocCount(indexName));

        // Verify search correctness
        int k = 10;
        for (int i = 0; i < 10; i++) {
            List<KNNResult> knnResults = runKnnQuery(indexName, FIELD_NAME, testData.queries[i], k);
            assertEquals("Expected " + k + " results for query " + i, k, knnResults.size());
        }
    }

    /**
     * Test that without BP reorder enabled, the index works normally (control test).
     * Then compare with a BP reorder index to ensure both produce valid results.
     */
    @SneakyThrows
    public void testBPReorder_comparedToNonReorder_bothProduceValidResults() {
        String reorderIndex = INDEX_NAME + "-with-reorder";
        String normalIndex = INDEX_NAME + "-without-reorder";

        // Create both indices
        createKnnHnswIndexWithBPReorder(KNNEngine.FAISS, reorderIndex, FIELD_NAME, DIMENSION);
        createKnnHnswIndex(KNNEngine.FAISS, normalIndex, FIELD_NAME, DIMENSION);

        // Ingest same data into both
        for (int i = 0; i < testData.indexData.docs.length; i++) {
            Object[] vector = Floats.asList(testData.indexData.vectors[i]).toArray();
            String docId = Integer.toString(testData.indexData.docs[i]);
            addKnnDoc(reorderIndex, docId, FIELD_NAME, vector);
            addKnnDoc(normalIndex, docId, FIELD_NAME, vector);
        }

        // Force merge both
        forceMergeKnnIndex(reorderIndex, 1);
        forceMergeKnnIndex(normalIndex, 1);

        // Both should return the same top-k results
        int k = 10;
        for (int i = 0; i < 5; i++) {
            List<KNNResult> reorderResults = runKnnQuery(reorderIndex, FIELD_NAME, testData.queries[i], k);
            List<KNNResult> normalResults = runKnnQuery(normalIndex, FIELD_NAME, testData.queries[i], k);

            assertEquals(k, reorderResults.size());
            assertEquals(k, normalResults.size());

            // Both should return the same doc IDs (order may differ slightly due to reorder)
            Set<String> reorderIds = reorderResults.stream().map(KNNResult::getDocId).collect(Collectors.toSet());
            Set<String> normalIds = normalResults.stream().map(KNNResult::getDocId).collect(Collectors.toSet());
            assertEquals("Top-" + k + " results should match for query " + i, normalIds, reorderIds);
        }
    }

    // --- Helper methods ---

    private void createKnnHnswIndexWithBPReorder(KNNEngine engine, String indexName, String fieldName, int dimension) throws IOException {
        KNNJsonIndexMappingsBuilder.Method method = KNNJsonIndexMappingsBuilder.Method.builder()
            .methodName(METHOD_HNSW)
            .spaceType(SpaceType.L2.getValue())
            .engine(engine.getName())
            .build();

        String knnIndexMapping = KNNJsonIndexMappingsBuilder.builder()
            .fieldName(fieldName)
            .dimension(dimension)
            .vectorDataType(VectorDataType.FLOAT.getValue())
            .method(method)
            .build()
            .getIndexMapping();

        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.knn", true)
            .put(KNN_BP_VECTOR_REORDER_ENABLED, true)
            .build();

        createKnnIndex(indexName, settings, knnIndexMapping);
    }

    private void createKnnHnswIndex(KNNEngine engine, String indexName, String fieldName, int dimension) throws IOException {
        KNNJsonIndexMappingsBuilder.Method method = KNNJsonIndexMappingsBuilder.Method.builder()
            .methodName(METHOD_HNSW)
            .spaceType(SpaceType.L2.getValue())
            .engine(engine.getName())
            .build();

        String knnIndexMapping = KNNJsonIndexMappingsBuilder.builder()
            .fieldName(fieldName)
            .dimension(dimension)
            .vectorDataType(VectorDataType.FLOAT.getValue())
            .method(method)
            .build()
            .getIndexMapping();

        createKnnIndex(indexName, knnIndexMapping);
    }

    private void ingestTestData(String indexName, String fieldName) throws Exception {
        for (int i = 0; i < testData.indexData.docs.length; i++) {
            addKnnDoc(
                indexName,
                Integer.toString(testData.indexData.docs[i]),
                fieldName,
                Floats.asList(testData.indexData.vectors[i]).toArray()
            );
        }
        assertEquals(testData.indexData.docs.length, getDocCount(indexName));
    }

    private List<KNNResult> runKnnQuery(String indexName, String fieldName, float[] queryVector, int k) throws Exception {
        String query = KNNJsonQueryBuilder.builder()
            .fieldName(fieldName)
            .vector(ArrayUtils.toObject(queryVector))
            .k(k)
            .build()
            .getQueryString();
        Response response = searchKNNIndex(indexName, query, k);
        return parseSearchResponse(EntityUtils.toString(response.getEntity()), fieldName);
    }

    private float getRecall(Set<String> groundTruth, Set<String> results) {
        long intersection = groundTruth.stream().filter(results::contains).count();
        return (float) intersection / groundTruth.size();
    }
}
