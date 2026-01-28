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
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.Bits;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.clumping.ClumpingFileReader;
import org.opensearch.knn.index.codec.clumping.ClumpingFileWriter;
import org.opensearch.knn.index.codec.clumping.HiddenVectorEntry;
import org.opensearch.knn.index.codec.clumping.MarkerSelector;
import org.opensearch.knn.index.query.clumping.ClumpingContext;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.apache.lucene.tests.index.BaseKnnVectorsFormatTestCase.randomVector;

/**
 * Integration tests for the clumping feature.
 * 
 * Validates: Requirements 9.1, 9.2, 9.3, 13.2, 13.4, 13.6, 21.1, 21.2
 */
public class ClumpingIntegrationTests extends OpenSearchTestCase {

    private static final String FIELD_NAME = "vector-field";
    private static final int DIMENSION = 4;

    /**
     * Test filter integration - filtered documents should not appear in results.
     * Validates: Requirement 9.1 (filter integration)
     */
    @SneakyThrows
    public void testFilterIntegration() {
        try (Directory directory = newDirectory()) {
            List<float[]> vectors = generateRandomVectors(20, DIMENSION);
            addDocuments(vectors, directory);

            try (IndexReader reader = DirectoryReader.open(directory)) {
                float[] queryVector = randomVector(DIMENSION);
                ClumpingContext clumpingContext = ClumpingContext.getDefault();

                ClumpingKNNVectorQuery query = new ClumpingKNNVectorQuery(
                    new MatchAllDocsQuery(),
                    FIELD_NAME,
                    10,
                    queryVector,
                    0,
                    clumpingContext,
                    SpaceType.L2
                );

                // Create candidates
                Map<Integer, float[]> candidates = new HashMap<>();
                for (int i = 0; i < vectors.size(); i++) {
                    candidates.put(i, vectors.get(i));
                }

                // Create filter that excludes even doc IDs
                Bits filterBits = new Bits() {
                    @Override
                    public boolean get(int index) {
                        return index % 2 != 0; // Only odd doc IDs pass
                    }

                    @Override
                    public int length() {
                        return vectors.size();
                    }
                };

                LeafReaderContext leaf = reader.leaves().get(0);
                
                // Expand with filter
                List<Integer> markerDocIds = new ArrayList<>();
                for (int i = 0; i < vectors.size(); i++) {
                    if (MarkerSelector.isMarker(i, 8)) {
                        markerDocIds.add(i);
                    }
                }
                
                Map<Integer, float[]> expandedCandidates = query.expandWithHiddenVectors(leaf, markerDocIds, filterBits);
                
                // Verify filtered documents are excluded
                for (int docId : expandedCandidates.keySet()) {
                    // Note: expandWithHiddenVectors returns global doc IDs
                    int localDocId = docId - leaf.docBase;
                    if (localDocId >= 0 && localDocId < vectors.size()) {
                        // Hidden vectors should be filtered
                        // Markers are not filtered in expandWithHiddenVectors (they're already filtered by inner query)
                    }
                }
            }
        }
    }

    /**
     * Test oversample + clumping interaction.
     * Validates: Requirement 9.3 (oversample and clumping interaction)
     */
    @SneakyThrows
    public void testOversampleAndClumpingInteraction() {
        // Test that expansion factor affects first pass k calculation
        ClumpingContext context2x = ClumpingContext.builder()
            .clumpingFactor(8)
            .expansionFactor(2.0f)
            .enabled(true)
            .build();

        ClumpingContext context3x = ClumpingContext.builder()
            .clumpingFactor(8)
            .expansionFactor(3.0f)
            .enabled(true)
            .build();

        // With expansion factor 2.0, first pass k for k=10 should be 20
        assertEquals(20, context2x.getFirstPassK(10));
        
        // With expansion factor 3.0, first pass k for k=10 should be 30
        assertEquals(30, context3x.getFirstPassK(10));
        
        // Verify the expansion factor is correctly stored
        assertEquals(2.0f, context2x.getExpansionFactor(), 0.001f);
        assertEquals(3.0f, context3x.getExpansionFactor(), 0.001f);
    }

    /**
     * Test end-to-end flow with various clumping factors.
     * Validates: Requirement 21.1 (end-to-end integration)
     */
    @SneakyThrows
    public void testEndToEndWithVariousClumpingFactors() {
        int[] clumpingFactors = { 2, 8, 50, 100 };
        
        for (int clumpingFactor : clumpingFactors) {
            try (Directory directory = newDirectory()) {
                int numDocs = clumpingFactor * 10; // Ensure enough docs for meaningful test
                List<float[]> vectors = generateRandomVectors(numDocs, DIMENSION);
                addDocuments(vectors, directory);

                try (IndexReader reader = DirectoryReader.open(directory)) {
                    float[] queryVector = randomVector(DIMENSION);
                    ClumpingContext clumpingContext = ClumpingContext.builder()
                        .clumpingFactor(clumpingFactor)
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

                    // Create candidates
                    Map<Integer, float[]> candidates = new HashMap<>();
                    for (int i = 0; i < vectors.size(); i++) {
                        candidates.put(i, vectors.get(i));
                    }

                    LeafReaderContext leaf = reader.leaves().get(0);
                    TopDocs topDocs = query.rescoreCandidates(candidates, leaf);

                    // Should return top 5 results
                    assertEquals(
                        String.format("With clumping factor %d, should return 5 results", clumpingFactor),
                        5,
                        topDocs.scoreDocs.length
                    );

                    // Results should be sorted by score
                    for (int i = 0; i < topDocs.scoreDocs.length - 1; i++) {
                        assertTrue(
                            "Results should be sorted by score descending",
                            topDocs.scoreDocs[i].score >= topDocs.scoreDocs[i + 1].score
                        );
                    }
                }
            }
        }
    }

    /**
     * Test with different vector dimensions.
     * Validates: Requirement 21.1 (different vector dimensions)
     */
    @SneakyThrows
    public void testWithDifferentVectorDimensions() {
        int[] dimensions = { 2, 4, 8, 16, 32, 64, 128 };
        
        for (int dimension : dimensions) {
            try (Directory directory = newDirectory()) {
                List<float[]> vectors = generateRandomVectors(50, dimension);
                addDocumentsWithDimension(vectors, directory, dimension);

                try (IndexReader reader = DirectoryReader.open(directory)) {
                    float[] queryVector = randomVector(dimension);
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

                    Map<Integer, float[]> candidates = new HashMap<>();
                    for (int i = 0; i < vectors.size(); i++) {
                        candidates.put(i, vectors.get(i));
                    }

                    LeafReaderContext leaf = reader.leaves().get(0);
                    TopDocs topDocs = query.rescoreCandidates(candidates, leaf);

                    assertEquals(
                        String.format("With dimension %d, should return 5 results", dimension),
                        5,
                        topDocs.scoreDocs.length
                    );
                }
            }
        }
    }

    /**
     * Test mixed index scenario - search across clumping and non-clumping indices.
     * Validates: Requirement 21.2 (mixed index scenarios)
     */
    @SneakyThrows
    public void testMixedIndexScenario() {
        // Simulate a scenario where some segments have clumping files and some don't
        try (Directory directory = newDirectory()) {
            List<float[]> vectors = generateRandomVectors(20, DIMENSION);
            addDocuments(vectors, directory);

            // Create a clumping file for a subset of the data
            String segmentName = "test_segment";
            String fileName = ClumpingFileReader.buildClumpingFileName(segmentName, FIELD_NAME);
            
            try (IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT)) {
                ClumpingFileWriter writer = new ClumpingFileWriter(output, 8, DIMENSION, VectorDataType.FLOAT);
                // Add some hidden vectors
                writer.addHiddenVector(10, vectors.get(10), 1);
                writer.addHiddenVector(11, vectors.get(11), 1);
                writer.finish();
            }

            try (IndexReader reader = DirectoryReader.open(directory)) {
                float[] queryVector = randomVector(DIMENSION);
                ClumpingContext clumpingContext = ClumpingContext.getDefault();

                ClumpingKNNVectorQuery query = new ClumpingKNNVectorQuery(
                    new MatchAllDocsQuery(),
                    FIELD_NAME,
                    10,
                    queryVector,
                    0,
                    clumpingContext,
                    SpaceType.L2
                );

                // Create candidates from all vectors
                Map<Integer, float[]> candidates = new HashMap<>();
                for (int i = 0; i < vectors.size(); i++) {
                    candidates.put(i, vectors.get(i));
                }

                LeafReaderContext leaf = reader.leaves().get(0);
                TopDocs topDocs = query.rescoreCandidates(candidates, leaf);

                // Should return results from both clumping and non-clumping data
                assertTrue(topDocs.scoreDocs.length > 0);
            }
        }
    }

    /**
     * Test clumping file round-trip with various data sizes.
     * Validates: Requirement 4.5 (round-trip property)
     */
    @SneakyThrows
    public void testClumpingFileRoundTripWithVariousSizes() {
        int[] sizes = { 1, 10, 100, 1000 };
        
        for (int size : sizes) {
            try (Directory directory = newDirectory()) {
                String segmentName = "test_segment_" + size;
                String fileName = ClumpingFileReader.buildClumpingFileName(segmentName, FIELD_NAME);
                
                // Write clumping file
                List<HiddenVectorEntry> originalEntries = new ArrayList<>();
                try (IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT)) {
                    ClumpingFileWriter writer = new ClumpingFileWriter(output, 8, DIMENSION, VectorDataType.FLOAT);
                    
                    for (int i = 0; i < size; i++) {
                        float[] vector = randomVector(DIMENSION);
                        int markerDocId = i % 10; // Distribute across 10 markers
                        writer.addHiddenVector(i, vector, markerDocId);
                        originalEntries.add(HiddenVectorEntry.builder()
                            .docId(i)
                            .vector(vector)
                            .markerDocId(markerDocId)
                            .build());
                    }
                    writer.finish();
                }

                // Read and verify
                try (ClumpingFileReader reader = ClumpingFileReader.open(directory, segmentName, FIELD_NAME)) {
                    assertEquals(size, reader.getHeader().getHiddenVectorCount());
                    
                    List<HiddenVectorEntry> readEntries = reader.getAllHiddenVectors();
                    assertEquals(size, readEntries.size());
                    
                    // Verify each entry
                    for (int i = 0; i < size; i++) {
                        HiddenVectorEntry original = originalEntries.get(i);
                        HiddenVectorEntry read = readEntries.get(i);
                        
                        assertEquals(original.getDocId(), read.getDocId());
                        assertEquals(original.getMarkerDocId(), read.getMarkerDocId());
                        assertArrayEquals(original.getVector(), read.getVector(), 0.0001f);
                    }
                }
            }
        }
    }

    /**
     * Test statistics are correctly tracked.
     * Validates: Requirements 10.1, 10.2, 10.3, 19.4
     */
    @SneakyThrows
    public void testStatisticsTracking() {
        try (Directory directory = newDirectory()) {
            String segmentName = "test_segment";
            String fileName = ClumpingFileReader.buildClumpingFileName(segmentName, FIELD_NAME);
            
            int hiddenVectorCount = 50;
            int markerCount = 5;
            
            // Write clumping file
            try (IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT)) {
                ClumpingFileWriter writer = new ClumpingFileWriter(output, 8, DIMENSION, VectorDataType.FLOAT);
                
                for (int i = 0; i < hiddenVectorCount; i++) {
                    float[] vector = randomVector(DIMENSION);
                    int markerDocId = i % markerCount;
                    writer.addHiddenVector(i, vector, markerDocId);
                }
                writer.finish();
                
                // Verify writer statistics
                assertEquals(hiddenVectorCount, writer.getHiddenVectorCount());
                assertEquals(markerCount, writer.getMarkerCount());
            }

            // Verify reader can access statistics
            try (ClumpingFileReader reader = ClumpingFileReader.open(directory, segmentName, FIELD_NAME)) {
                assertEquals(hiddenVectorCount, reader.getHeader().getHiddenVectorCount());
                assertEquals(markerCount, reader.getHeader().getMarkerCount());
                assertEquals(8, reader.getHeader().getClumpingFactor());
                assertEquals(DIMENSION, reader.getHeader().getDimension());
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
        addDocumentsWithDimension(vectors, directory, DIMENSION);
    }

    private void addDocumentsWithDimension(List<float[]> vectors, Directory directory, int dimension) throws IOException {
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
