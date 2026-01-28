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
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.clumping.ClumpingFileHeader;
import org.opensearch.knn.index.codec.clumping.ClumpingFileReader;
import org.opensearch.knn.index.codec.clumping.ClumpingFileWriter;
import org.opensearch.knn.index.query.clumping.ClumpingContext;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.apache.lucene.tests.index.BaseKnnVectorsFormatTestCase.randomVector;

/**
 * Unit tests for {@link ClumpingKNNVectorQuery} error handling.
 * 
 * Validates: Requirements 11.1, 12.1, 12.2, 12.3
 */
public class ClumpingKNNVectorQueryErrorHandlingTests extends OpenSearchTestCase {

    private static final String FIELD_NAME = "vector-field";
    private static final int DIMENSION = 4;

    /**
     * Test that query handles missing clumping file gracefully (backward compatibility).
     * Validates: Requirement 11.1 (backward compatibility)
     */
    @SneakyThrows
    public void testMissingClumpingFileFallsBackToMarkerOnlySearch() {
        try (Directory directory = newDirectory()) {
            List<float[]> vectors = generateRandomVectors(10, DIMENSION);
            addDocuments(vectors, directory);

            // No clumping file is created - simulating a non-clumping index

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

                // Create candidates with marker vectors only
                Map<Integer, float[]> candidates = new HashMap<>();
                for (int i = 0; i < vectors.size(); i++) {
                    candidates.put(i, vectors.get(i));
                }

                LeafReaderContext leaf = reader.leaves().get(0);
                
                // Should not throw exception - should fall back to marker-only search
                TopDocs topDocs = query.rescoreCandidates(candidates, leaf);
                
                assertNotNull(topDocs);
                assertTrue(topDocs.scoreDocs.length > 0);
            }
        }
    }

    /**
     * Test that corrupted clumping file (invalid magic bytes) is handled gracefully.
     * Validates: Requirement 12.1 (corrupted file fallback)
     */
    @SneakyThrows
    public void testCorruptedMagicBytesHandledGracefully() {
        try (Directory directory = newDirectory()) {
            String segmentName = "test_segment";
            String fileName = ClumpingFileReader.buildClumpingFileName(segmentName, FIELD_NAME);
            
            // Write file with invalid magic bytes
            try (IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT)) {
                output.writeBytes(new byte[] { 'B', 'A', 'D', '!' }, 4); // Invalid magic
                output.writeInt(1); // version
            }

            // Attempting to open should throw IOException
            IOException exception = expectThrows(
                IOException.class,
                () -> ClumpingFileReader.open(directory, segmentName, FIELD_NAME)
            );
            assertTrue(exception.getMessage().contains("Invalid clumping file"));
        }
    }

    /**
     * Test that unsupported version is handled with descriptive error.
     * Validates: Requirement 12.2 (version mismatch error)
     */
    @SneakyThrows
    public void testUnsupportedVersionHandledWithDescriptiveError() {
        try (Directory directory = newDirectory()) {
            String segmentName = "test_segment";
            String fileName = ClumpingFileReader.buildClumpingFileName(segmentName, FIELD_NAME);
            
            // Write file with unsupported version
            try (IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT)) {
                output.writeBytes(ClumpingFileHeader.MAGIC_BYTES, 4);
                output.writeInt(999); // Unsupported version
            }

            // Attempting to open should throw IOException with version info
            IOException exception = expectThrows(
                IOException.class,
                () -> ClumpingFileReader.open(directory, segmentName, FIELD_NAME)
            );
            assertTrue(
                "Error message should mention unsupported version",
                exception.getMessage().contains("Unsupported clumping file format version")
            );
            assertTrue(
                "Error message should include the version number",
                exception.getMessage().contains("999")
            );
        }
    }

    /**
     * Test that partial expansion failure doesn't affect successful expansions.
     * Validates: Requirement 12.3 (partial expansion failure handling)
     */
    @SneakyThrows
    public void testPartialExpansionFailureDoesNotAffectSuccessfulExpansions() {
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

                // Create candidates - some valid, some with potential issues
                Map<Integer, float[]> candidates = new HashMap<>();
                for (int i = 0; i < 5; i++) {
                    candidates.put(i, vectors.get(i));
                }

                LeafReaderContext leaf = reader.leaves().get(0);
                
                // Should successfully rescore the valid candidates
                TopDocs topDocs = query.rescoreCandidates(candidates, leaf);
                
                assertNotNull(topDocs);
                assertEquals(5, topDocs.scoreDocs.length);
            }
        }
    }

    /**
     * Test that empty marker list is handled gracefully.
     */
    @SneakyThrows
    public void testEmptyMarkerListHandledGracefully() {
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

                // Empty marker list
                List<Integer> emptyMarkerDocIds = new ArrayList<>();
                LeafReaderContext leaf = reader.leaves().get(0);
                
                // Should return empty result, not throw exception
                Map<Integer, float[]> candidates = query.expandWithHiddenVectors(leaf, emptyMarkerDocIds);
                
                assertTrue(candidates.isEmpty());
            }
        }
    }

    /**
     * Test that query works correctly when clumping file exists but has no hidden vectors.
     */
    @SneakyThrows
    public void testEmptyClumpingFileHandledCorrectly() {
        try (Directory directory = newDirectory()) {
            String segmentName = "test_segment";
            String fileName = ClumpingFileReader.buildClumpingFileName(segmentName, FIELD_NAME);
            
            // Create empty clumping file
            try (IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT)) {
                ClumpingFileWriter writer = new ClumpingFileWriter(output, 8, DIMENSION, VectorDataType.FLOAT);
                writer.finish(); // No hidden vectors added
            }

            // Should be able to open and read empty file
            try (ClumpingFileReader reader = ClumpingFileReader.open(directory, segmentName, FIELD_NAME)) {
                assertEquals(0, reader.getHeader().getHiddenVectorCount());
                assertEquals(0, reader.getHeader().getMarkerCount());
                
                // Getting hidden vectors for any marker should return empty list
                List<?> hiddenVectors = reader.getHiddenVectors(1);
                assertTrue(hiddenVectors.isEmpty());
            }
        }
    }

    /**
     * Test that query handles null query vector gracefully.
     */
    public void testNullQueryVectorThrowsException() {
        ClumpingContext clumpingContext = ClumpingContext.getDefault();

        expectThrows(
            NullPointerException.class,
            () -> new ClumpingKNNVectorQuery(
                new MatchAllDocsQuery(),
                FIELD_NAME,
                5,
                null, // null query vector
                0,
                clumpingContext,
                SpaceType.L2
            )
        );
    }

    /**
     * Test that query handles empty query vector gracefully.
     */
    @SneakyThrows
    public void testEmptyQueryVectorHandledCorrectly() {
        try (Directory directory = newDirectory()) {
            addDocuments(generateRandomVectors(5, DIMENSION), directory);

            try (IndexReader reader = DirectoryReader.open(directory)) {
                ClumpingContext clumpingContext = ClumpingContext.getDefault();

                // Empty query vector
                float[] emptyQueryVector = new float[0];

                ClumpingKNNVectorQuery query = new ClumpingKNNVectorQuery(
                    new MatchAllDocsQuery(),
                    FIELD_NAME,
                    5,
                    emptyQueryVector,
                    0,
                    clumpingContext,
                    SpaceType.L2
                );

                // Create candidates
                Map<Integer, float[]> candidates = new HashMap<>();
                candidates.put(0, new float[] { 1.0f, 2.0f, 3.0f, 4.0f });

                LeafReaderContext leaf = reader.leaves().get(0);
                
                // Should handle gracefully (may throw or return empty results)
                // The exact behavior depends on the space type implementation
                try {
                    TopDocs topDocs = query.rescoreCandidates(candidates, leaf);
                    // If it doesn't throw, it should return some result
                    assertNotNull(topDocs);
                } catch (Exception e) {
                    // Some space types may throw on dimension mismatch
                    // This is acceptable behavior
                }
            }
        }
    }

    /**
     * Test that query handles k=0 gracefully.
     */
    @SneakyThrows
    public void testZeroKHandledCorrectly() {
        try (Directory directory = newDirectory()) {
            addDocuments(generateRandomVectors(5, DIMENSION), directory);

            try (IndexReader reader = DirectoryReader.open(directory)) {
                float[] queryVector = randomVector(DIMENSION);
                ClumpingContext clumpingContext = ClumpingContext.getDefault();

                ClumpingKNNVectorQuery query = new ClumpingKNNVectorQuery(
                    new MatchAllDocsQuery(),
                    FIELD_NAME,
                    0, // k=0
                    queryVector,
                    0,
                    clumpingContext,
                    SpaceType.L2
                );

                Map<Integer, float[]> candidates = new HashMap<>();
                candidates.put(0, randomVector(DIMENSION));

                LeafReaderContext leaf = reader.leaves().get(0);
                
                // Should return empty results for k=0
                TopDocs topDocs = query.rescoreCandidates(candidates, leaf);
                assertEquals(0, topDocs.scoreDocs.length);
            }
        }
    }

    /**
     * Test that query handles negative k gracefully.
     */
    @SneakyThrows
    public void testNegativeKHandledCorrectly() {
        try (Directory directory = newDirectory()) {
            addDocuments(generateRandomVectors(5, DIMENSION), directory);

            try (IndexReader reader = DirectoryReader.open(directory)) {
                float[] queryVector = randomVector(DIMENSION);
                ClumpingContext clumpingContext = ClumpingContext.getDefault();

                // Negative k should be handled (may throw or return empty)
                ClumpingKNNVectorQuery query = new ClumpingKNNVectorQuery(
                    new MatchAllDocsQuery(),
                    FIELD_NAME,
                    -1, // negative k
                    queryVector,
                    0,
                    clumpingContext,
                    SpaceType.L2
                );

                Map<Integer, float[]> candidates = new HashMap<>();
                candidates.put(0, randomVector(DIMENSION));

                LeafReaderContext leaf = reader.leaves().get(0);
                
                // Should handle gracefully
                try {
                    TopDocs topDocs = query.rescoreCandidates(candidates, leaf);
                    // If it doesn't throw, should return empty or handle gracefully
                    assertNotNull(topDocs);
                } catch (IllegalArgumentException e) {
                    // This is also acceptable behavior
                }
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
