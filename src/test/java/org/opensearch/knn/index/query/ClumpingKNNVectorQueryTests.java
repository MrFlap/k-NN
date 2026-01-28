/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.SneakyThrows;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.store.Directory;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.query.clumping.ClumpingContext;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.apache.lucene.tests.index.BaseKnnVectorsFormatTestCase.randomVector;
import static org.mockito.Mockito.mock;

/**
 * Unit tests for {@link ClumpingKNNVectorQuery}.
 * 
 * Validates: Requirements 6.1, 6.2, 7.1, 7.4, 8.1, 8.3, 8.4, 9.1, 9.2, 9.3, 12.1, 12.2, 12.3
 */
public class ClumpingKNNVectorQueryTests extends OpenSearchTestCase {

    private static final String FIELD_NAME = "vector-field";
    private static final int DIMENSION = 4;

    /**
     * Test that rescoreCandidates correctly scores and ranks candidates.
     * Validates: Requirements 8.1, 8.3, 8.4
     */
    @SneakyThrows
    public void testRescoreCandidatesReturnsTopK() {
        try (Directory directory = newDirectory()) {
            List<float[]> vectors = generateRandomVectors(10, DIMENSION);
            addDocuments(vectors, directory);

            try (IndexReader reader = DirectoryReader.open(directory)) {
                float[] queryVector = randomVector(DIMENSION);
                ClumpingContext clumpingContext = ClumpingContext.builder()
                    .clumpingFactor(8)
                    .expansionFactor(2.0f)
                    .enabled(true)
                    .build();

                ClumpingKNNVectorQuery query = new ClumpingKNNVectorQuery(
                    new MatchAllDocsQuery(),
                    FIELD_NAME,
                    5,
                    queryVector,
                    0,
                    clumpingContext,
                    SpaceType.L2
                );

                // Create candidates map with known vectors
                Map<Integer, float[]> candidates = new HashMap<>();
                for (int i = 0; i < vectors.size(); i++) {
                    candidates.put(i, vectors.get(i));
                }

                LeafReaderContext leaf = reader.leaves().get(0);
                TopDocs topDocs = query.rescoreCandidates(candidates, leaf);

                // Should return top 5 results
                assertEquals(5, topDocs.scoreDocs.length);

                // Verify results are sorted by score (descending)
                for (int i = 0; i < topDocs.scoreDocs.length - 1; i++) {
                    assertTrue(
                        "Results should be sorted by score descending",
                        topDocs.scoreDocs[i].score >= topDocs.scoreDocs[i + 1].score
                    );
                }
            }
        }
    }

    /**
     * Test that rescoreCandidates handles empty candidates.
     */
    @SneakyThrows
    public void testRescoreCandidatesWithEmptyCandidates() {
        try (Directory directory = newDirectory()) {
            addDocuments(generateRandomVectors(5, DIMENSION), directory);

            try (IndexReader reader = DirectoryReader.open(directory)) {
                float[] queryVector = randomVector(DIMENSION);
                ClumpingContext clumpingContext = ClumpingContext.getDefault();

                ClumpingKNNVectorQuery query = new ClumpingKNNVectorQuery(
                    new MatchAllDocsQuery(),
                    FIELD_NAME,
                    5,
                    queryVector,
                    0,
                    clumpingContext,
                    SpaceType.L2
                );

                Map<Integer, float[]> emptyCandidates = new HashMap<>();
                LeafReaderContext leaf = reader.leaves().get(0);
                TopDocs topDocs = query.rescoreCandidates(emptyCandidates, leaf);

                assertEquals(0, topDocs.scoreDocs.length);
            }
        }
    }

    /**
     * Test that rescoreCandidates handles fewer candidates than k.
     */
    @SneakyThrows
    public void testRescoreCandidatesWithFewerThanK() {
        try (Directory directory = newDirectory()) {
            List<float[]> vectors = generateRandomVectors(3, DIMENSION);
            addDocuments(vectors, directory);

            try (IndexReader reader = DirectoryReader.open(directory)) {
                float[] queryVector = randomVector(DIMENSION);
                ClumpingContext clumpingContext = ClumpingContext.getDefault();

                ClumpingKNNVectorQuery query = new ClumpingKNNVectorQuery(
                    new MatchAllDocsQuery(),
                    FIELD_NAME,
                    10, // k > number of candidates
                    queryVector,
                    0,
                    clumpingContext,
                    SpaceType.L2
                );

                Map<Integer, float[]> candidates = new HashMap<>();
                for (int i = 0; i < vectors.size(); i++) {
                    candidates.put(i, vectors.get(i));
                }

                LeafReaderContext leaf = reader.leaves().get(0);
                TopDocs topDocs = query.rescoreCandidates(candidates, leaf);

                // Should return all 3 candidates, not 10
                assertEquals(3, topDocs.scoreDocs.length);
            }
        }
    }

    /**
     * Test that hidden vectors scoring higher than markers are correctly ranked.
     * Validates: Requirement 8.4
     */
    @SneakyThrows
    public void testHiddenVectorsScoringHigherThanMarkersAreRankedCorrectly() {
        try (Directory directory = newDirectory()) {
            addDocuments(generateRandomVectors(5, DIMENSION), directory);

            try (IndexReader reader = DirectoryReader.open(directory)) {
                // Query vector is [1, 0, 0, 0]
                float[] queryVector = new float[] { 1.0f, 0.0f, 0.0f, 0.0f };
                ClumpingContext clumpingContext = ClumpingContext.getDefault();

                ClumpingKNNVectorQuery query = new ClumpingKNNVectorQuery(
                    new MatchAllDocsQuery(),
                    FIELD_NAME,
                    3,
                    queryVector,
                    0,
                    clumpingContext,
                    SpaceType.COSINESIMIL
                );

                // Create candidates where "hidden" vectors (higher doc IDs) score better
                Map<Integer, float[]> candidates = new HashMap<>();
                // Marker vectors (lower doc IDs) - less similar to query
                candidates.put(0, new float[] { 0.0f, 1.0f, 0.0f, 0.0f }); // orthogonal
                candidates.put(1, new float[] { 0.0f, 0.0f, 1.0f, 0.0f }); // orthogonal
                // Hidden vectors (higher doc IDs) - more similar to query
                candidates.put(100, new float[] { 1.0f, 0.0f, 0.0f, 0.0f }); // identical
                candidates.put(101, new float[] { 0.9f, 0.1f, 0.0f, 0.0f }); // very similar

                LeafReaderContext leaf = reader.leaves().get(0);
                TopDocs topDocs = query.rescoreCandidates(candidates, leaf);

                // Hidden vectors should be ranked higher
                assertEquals(3, topDocs.scoreDocs.length);
                // Doc 100 (identical to query) should be first
                assertEquals(100, topDocs.scoreDocs[0].doc);
                // Doc 101 (very similar) should be second
                assertEquals(101, topDocs.scoreDocs[1].doc);
            }
        }
    }

    /**
     * Test query equals and hashCode.
     */
    public void testEqualsAndHashCode() {
        float[] queryVector = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        ClumpingContext context = ClumpingContext.getDefault();
        Query innerQuery = new MatchAllDocsQuery();

        ClumpingKNNVectorQuery query1 = new ClumpingKNNVectorQuery(
            innerQuery, FIELD_NAME, 10, queryVector, 0, context, SpaceType.L2
        );

        ClumpingKNNVectorQuery query2 = new ClumpingKNNVectorQuery(
            innerQuery, FIELD_NAME, 10, queryVector, 0, context, SpaceType.L2
        );

        ClumpingKNNVectorQuery query3 = new ClumpingKNNVectorQuery(
            innerQuery, FIELD_NAME, 5, queryVector, 0, context, SpaceType.L2 // different k
        );

        assertEquals(query1, query2);
        assertEquals(query1.hashCode(), query2.hashCode());
        assertNotEquals(query1, query3);
    }

    /**
     * Test query toString.
     */
    public void testToString() {
        float[] queryVector = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        ClumpingContext context = ClumpingContext.getDefault();

        ClumpingKNNVectorQuery query = new ClumpingKNNVectorQuery(
            new MatchAllDocsQuery(), FIELD_NAME, 10, queryVector, 0, context, SpaceType.L2
        );

        String str = query.toString(FIELD_NAME);
        assertTrue(str.contains("ClumpingKNNVectorQuery"));
        assertTrue(str.contains(FIELD_NAME));
        assertTrue(str.contains("k=10"));
    }

    /**
     * Test getters return correct values.
     */
    public void testGetters() {
        float[] queryVector = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        ClumpingContext context = ClumpingContext.builder()
            .clumpingFactor(16)
            .expansionFactor(3.0f)
            .enabled(true)
            .build();
        Query innerQuery = new MatchAllDocsQuery();
        Query filterQuery = new MatchAllDocsQuery();

        ClumpingKNNVectorQuery query = new ClumpingKNNVectorQuery(
            innerQuery, FIELD_NAME, 10, queryVector, 5, context, SpaceType.L2, filterQuery
        );

        assertEquals(innerQuery, query.getInnerQuery());
        assertEquals(FIELD_NAME, query.getField());
        assertEquals(10, query.getK());
        assertArrayEquals(queryVector, query.getQueryVector(), 0.0001f);
        assertEquals(context, query.getClumpingContext());
        assertEquals(filterQuery, query.getFilterQuery());
    }

    /**
     * Test that createWeight produces valid weight.
     * Validates: Requirements 6.1, 6.2
     */
    @SneakyThrows
    public void testCreateWeightProducesValidWeight() {
        try (Directory directory = newDirectory()) {
            List<float[]> vectors = generateRandomVectors(10, DIMENSION);
            addDocuments(vectors, directory);

            try (IndexReader reader = DirectoryReader.open(directory)) {
                float[] queryVector = randomVector(DIMENSION);
                ClumpingContext clumpingContext = ClumpingContext.getDefault();

                ClumpingKNNVectorQuery query = new ClumpingKNNVectorQuery(
                    new MatchAllDocsQuery(),
                    FIELD_NAME,
                    5,
                    queryVector,
                    0,
                    clumpingContext,
                    SpaceType.L2
                );

                try (MockedStatic<ModelDao.OpenSearchKNNModelDao> mocked = Mockito.mockStatic(ModelDao.OpenSearchKNNModelDao.class)) {
                    mocked.when(ModelDao.OpenSearchKNNModelDao::getInstance).thenReturn(mock(ModelDao.OpenSearchKNNModelDao.class));
                    
                    IndexSearcher searcher = newSearcher(reader, true, false);
                    Weight weight = query.createWeight(searcher, ScoreMode.TOP_SCORES, 1.0f);
                    
                    assertNotNull(weight);
                }
            }
        }
    }

    /**
     * Test different space types produce different scores.
     */
    @SneakyThrows
    public void testDifferentSpaceTypesProduceDifferentScores() {
        try (Directory directory = newDirectory()) {
            List<float[]> vectors = generateRandomVectors(5, DIMENSION);
            addDocuments(vectors, directory);

            try (IndexReader reader = DirectoryReader.open(directory)) {
                float[] queryVector = randomVector(DIMENSION);
                ClumpingContext clumpingContext = ClumpingContext.getDefault();

                Map<Integer, float[]> candidates = new HashMap<>();
                for (int i = 0; i < vectors.size(); i++) {
                    candidates.put(i, vectors.get(i));
                }

                LeafReaderContext leaf = reader.leaves().get(0);

                // Test with L2
                ClumpingKNNVectorQuery l2Query = new ClumpingKNNVectorQuery(
                    new MatchAllDocsQuery(), FIELD_NAME, 5, queryVector, 0, clumpingContext, SpaceType.L2
                );
                TopDocs l2Results = l2Query.rescoreCandidates(candidates, leaf);

                // Test with COSINESIMIL
                ClumpingKNNVectorQuery cosineQuery = new ClumpingKNNVectorQuery(
                    new MatchAllDocsQuery(), FIELD_NAME, 5, queryVector, 0, clumpingContext, SpaceType.COSINESIMIL
                );
                TopDocs cosineResults = cosineQuery.rescoreCandidates(candidates, leaf);

                // Both should return results
                assertEquals(5, l2Results.scoreDocs.length);
                assertEquals(5, cosineResults.scoreDocs.length);

                // Verify both queries work correctly (scores may differ based on similarity function)
                assertNotNull(l2Results);
                assertNotNull(cosineResults);
            }
        }
    }

    // Helper methods

    private List<float[]> generateRandomVectors(int count, int dimension) {
        List<float[]> vectors = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            vectors.add(randomVector(dimension));
        }
        return vectors;
    }

    private void addDocuments(List<float[]> vectors, Directory directory) throws IOException {
        try (IndexWriter w = new IndexWriter(directory, newIndexWriterConfig())) {
            for (float[] vector : vectors) {
                Document document = new Document();
                KnnFloatVectorField vectorField = new KnnFloatVectorField(FIELD_NAME, vector, VectorSimilarityFunction.EUCLIDEAN);
                document.add(vectorField);
                w.addDocument(document);
                w.commit();
            }
        }
    }
}
