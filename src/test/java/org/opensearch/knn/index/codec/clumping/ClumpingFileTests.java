/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clumping;

import lombok.SneakyThrows;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * Unit tests for {@link ClumpingFileWriter} and {@link ClumpingFileReader}.
 * 
 * Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5
 */
public class ClumpingFileTests extends KNNTestCase {

    private static final String SEGMENT_NAME = "test_segment";
    private static final String FIELD_NAME = "test_field";
    private static final int CLUMPING_FACTOR = 8;
    private static final int DIMENSION = 4;

    /**
     * Test round-trip: write hidden vectors and read them back.
     * Validates: Requirement 4.5 (round-trip property)
     */
    @SneakyThrows
    public void testRoundTrip() {
        try (Directory directory = newDirectory()) {
            // Create test data
            float[] vector1 = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            float[] vector2 = new float[] { 5.0f, 6.0f, 7.0f, 8.0f };
            float[] vector3 = new float[] { 9.0f, 10.0f, 11.0f, 12.0f };

            // Write clumping file
            String fileName = ClumpingFileReader.buildClumpingFileName(SEGMENT_NAME, FIELD_NAME);
            try (IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT)) {
                ClumpingFileWriter writer = new ClumpingFileWriter(output, CLUMPING_FACTOR, DIMENSION, VectorDataType.FLOAT);
                writer.addHiddenVector(10, vector1, 1);  // docId=10, markerDocId=1
                writer.addHiddenVector(20, vector2, 1);  // docId=20, markerDocId=1
                writer.addHiddenVector(30, vector3, 2);  // docId=30, markerDocId=2
                writer.finish();
            }

            // Read clumping file and verify
            try (ClumpingFileReader reader = ClumpingFileReader.open(directory, SEGMENT_NAME, FIELD_NAME)) {
                ClumpingFileHeader header = reader.getHeader();
                
                // Verify header
                assertEquals(CLUMPING_FACTOR, header.getClumpingFactor());
                assertEquals(DIMENSION, header.getDimension());
                assertEquals(VectorDataType.FLOAT, header.getVectorDataType());
                assertEquals(3, header.getHiddenVectorCount());
                assertEquals(2, header.getMarkerCount());

                // Verify hidden vectors for marker 1
                List<HiddenVectorEntry> marker1Vectors = reader.getHiddenVectors(1);
                assertEquals(2, marker1Vectors.size());
                
                HiddenVectorEntry entry1 = marker1Vectors.get(0);
                assertEquals(10, entry1.getDocId());
                assertEquals(1, entry1.getMarkerDocId());
                assertArrayEquals(vector1, entry1.getVector(), 0.0001f);
                
                HiddenVectorEntry entry2 = marker1Vectors.get(1);
                assertEquals(20, entry2.getDocId());
                assertEquals(1, entry2.getMarkerDocId());
                assertArrayEquals(vector2, entry2.getVector(), 0.0001f);

                // Verify hidden vectors for marker 2
                List<HiddenVectorEntry> marker2Vectors = reader.getHiddenVectors(2);
                assertEquals(1, marker2Vectors.size());
                
                HiddenVectorEntry entry3 = marker2Vectors.get(0);
                assertEquals(30, entry3.getDocId());
                assertEquals(2, entry3.getMarkerDocId());
                assertArrayEquals(vector3, entry3.getVector(), 0.0001f);

                // Verify non-existent marker returns empty list
                List<HiddenVectorEntry> nonExistent = reader.getHiddenVectors(999);
                assertTrue(nonExistent.isEmpty());
            }
        }
    }

    /**
     * Test reading hidden vectors for multiple markers at once.
     */
    @SneakyThrows
    public void testGetHiddenVectorsForMarkers() {
        try (Directory directory = newDirectory()) {
            float[] vector1 = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            float[] vector2 = new float[] { 5.0f, 6.0f, 7.0f, 8.0f };
            float[] vector3 = new float[] { 9.0f, 10.0f, 11.0f, 12.0f };

            String fileName = ClumpingFileReader.buildClumpingFileName(SEGMENT_NAME, FIELD_NAME);
            try (IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT)) {
                ClumpingFileWriter writer = new ClumpingFileWriter(output, CLUMPING_FACTOR, DIMENSION, VectorDataType.FLOAT);
                writer.addHiddenVector(10, vector1, 1);
                writer.addHiddenVector(20, vector2, 2);
                writer.addHiddenVector(30, vector3, 3);
                writer.finish();
            }

            try (ClumpingFileReader reader = ClumpingFileReader.open(directory, SEGMENT_NAME, FIELD_NAME)) {
                Collection<Integer> markerIds = Arrays.asList(1, 2, 999); // 999 doesn't exist
                Map<Integer, List<HiddenVectorEntry>> result = reader.getHiddenVectorsForMarkers(markerIds);
                
                assertEquals(2, result.size()); // Only 1 and 2 have vectors
                assertTrue(result.containsKey(1));
                assertTrue(result.containsKey(2));
                assertFalse(result.containsKey(999));
                
                assertEquals(1, result.get(1).size());
                assertEquals(1, result.get(2).size());
            }
        }
    }

    /**
     * Test getAllHiddenVectors returns all vectors.
     */
    @SneakyThrows
    public void testGetAllHiddenVectors() {
        try (Directory directory = newDirectory()) {
            float[] vector1 = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            float[] vector2 = new float[] { 5.0f, 6.0f, 7.0f, 8.0f };
            float[] vector3 = new float[] { 9.0f, 10.0f, 11.0f, 12.0f };

            String fileName = ClumpingFileReader.buildClumpingFileName(SEGMENT_NAME, FIELD_NAME);
            try (IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT)) {
                ClumpingFileWriter writer = new ClumpingFileWriter(output, CLUMPING_FACTOR, DIMENSION, VectorDataType.FLOAT);
                writer.addHiddenVector(10, vector1, 1);
                writer.addHiddenVector(20, vector2, 2);
                writer.addHiddenVector(30, vector3, 2);
                writer.finish();
            }

            try (ClumpingFileReader reader = ClumpingFileReader.open(directory, SEGMENT_NAME, FIELD_NAME)) {
                List<HiddenVectorEntry> allVectors = reader.getAllHiddenVectors();
                assertEquals(3, allVectors.size());
            }
        }
    }

    /**
     * Test getAllMarkerDocIds returns all marker IDs.
     */
    @SneakyThrows
    public void testGetAllMarkerDocIds() {
        try (Directory directory = newDirectory()) {
            float[] vector = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };

            String fileName = ClumpingFileReader.buildClumpingFileName(SEGMENT_NAME, FIELD_NAME);
            try (IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT)) {
                ClumpingFileWriter writer = new ClumpingFileWriter(output, CLUMPING_FACTOR, DIMENSION, VectorDataType.FLOAT);
                writer.addHiddenVector(10, vector, 1);
                writer.addHiddenVector(20, vector, 5);
                writer.addHiddenVector(30, vector, 10);
                writer.finish();
            }

            try (ClumpingFileReader reader = ClumpingFileReader.open(directory, SEGMENT_NAME, FIELD_NAME)) {
                Collection<Integer> markerIds = reader.getAllMarkerDocIds();
                assertEquals(3, markerIds.size());
                assertTrue(markerIds.contains(1));
                assertTrue(markerIds.contains(5));
                assertTrue(markerIds.contains(10));
            }
        }
    }

    /**
     * Test empty clumping file (no hidden vectors).
     */
    @SneakyThrows
    public void testEmptyClumpingFile() {
        try (Directory directory = newDirectory()) {
            String fileName = ClumpingFileReader.buildClumpingFileName(SEGMENT_NAME, FIELD_NAME);
            try (IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT)) {
                ClumpingFileWriter writer = new ClumpingFileWriter(output, CLUMPING_FACTOR, DIMENSION, VectorDataType.FLOAT);
                writer.finish();
            }

            try (ClumpingFileReader reader = ClumpingFileReader.open(directory, SEGMENT_NAME, FIELD_NAME)) {
                ClumpingFileHeader header = reader.getHeader();
                assertEquals(0, header.getHiddenVectorCount());
                assertEquals(0, header.getMarkerCount());
                
                List<HiddenVectorEntry> vectors = reader.getHiddenVectors(1);
                assertTrue(vectors.isEmpty());
            }
        }
    }

    /**
     * Test file existence check.
     */
    @SneakyThrows
    public void testExists() {
        try (Directory directory = newDirectory()) {
            // File doesn't exist yet
            assertFalse(ClumpingFileReader.exists(directory, SEGMENT_NAME, FIELD_NAME));

            // Create file
            String fileName = ClumpingFileReader.buildClumpingFileName(SEGMENT_NAME, FIELD_NAME);
            try (IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT)) {
                ClumpingFileWriter writer = new ClumpingFileWriter(output, CLUMPING_FACTOR, DIMENSION, VectorDataType.FLOAT);
                writer.finish();
            }

            // File exists now
            assertTrue(ClumpingFileReader.exists(directory, SEGMENT_NAME, FIELD_NAME));
        }
    }

    /**
     * Test invalid magic bytes throws exception.
     * Validates: Requirement 4.4 (format validation)
     */
    @SneakyThrows
    public void testInvalidMagicBytesThrowsException() {
        try (Directory directory = newDirectory()) {
            String fileName = ClumpingFileReader.buildClumpingFileName(SEGMENT_NAME, FIELD_NAME);
            
            // Write file with invalid magic bytes
            try (IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT)) {
                output.writeBytes(new byte[] { 'B', 'A', 'D', '!' }, 4); // Invalid magic
                output.writeInt(1); // version
                // ... rest of header doesn't matter, should fail on magic bytes
            }

            IOException exception = expectThrows(
                IOException.class,
                () -> ClumpingFileReader.open(directory, SEGMENT_NAME, FIELD_NAME)
            );
            assertTrue(exception.getMessage().contains("Invalid clumping file"));
        }
    }

    /**
     * Test unsupported version throws exception.
     * Validates: Requirement 4.4 (format validation), 12.2 (version mismatch)
     */
    @SneakyThrows
    public void testUnsupportedVersionThrowsException() {
        try (Directory directory = newDirectory()) {
            String fileName = ClumpingFileReader.buildClumpingFileName(SEGMENT_NAME, FIELD_NAME);
            
            // Write file with unsupported version
            try (IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT)) {
                output.writeBytes(ClumpingFileHeader.MAGIC_BYTES, 4);
                output.writeInt(999); // Unsupported version
                // ... rest of header doesn't matter, should fail on version
            }

            IOException exception = expectThrows(
                IOException.class,
                () -> ClumpingFileReader.open(directory, SEGMENT_NAME, FIELD_NAME)
            );
            assertTrue(exception.getMessage().contains("Unsupported clumping file format version"));
        }
    }

    /**
     * Test writer rejects null vector.
     */
    public void testWriterRejectsNullVector() {
        try (Directory directory = newDirectory()) {
            String fileName = ClumpingFileReader.buildClumpingFileName(SEGMENT_NAME, FIELD_NAME);
            try (IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT)) {
                ClumpingFileWriter writer = new ClumpingFileWriter(output, CLUMPING_FACTOR, DIMENSION, VectorDataType.FLOAT);
                
                IllegalArgumentException exception = expectThrows(
                    IllegalArgumentException.class,
                    () -> writer.addHiddenVector(1, null, 1)
                );
                assertTrue(exception.getMessage().contains("Vector cannot be null"));
            }
        } catch (IOException e) {
            fail("Unexpected IOException: " + e.getMessage());
        }
    }

    /**
     * Test writer rejects vector with wrong dimension.
     */
    public void testWriterRejectsWrongDimension() {
        try (Directory directory = newDirectory()) {
            String fileName = ClumpingFileReader.buildClumpingFileName(SEGMENT_NAME, FIELD_NAME);
            try (IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT)) {
                ClumpingFileWriter writer = new ClumpingFileWriter(output, CLUMPING_FACTOR, DIMENSION, VectorDataType.FLOAT);
                
                float[] wrongDimVector = new float[] { 1.0f, 2.0f }; // Wrong dimension
                
                IllegalArgumentException exception = expectThrows(
                    IllegalArgumentException.class,
                    () -> writer.addHiddenVector(1, wrongDimVector, 1)
                );
                assertTrue(exception.getMessage().contains("dimension mismatch"));
            }
        } catch (IOException e) {
            fail("Unexpected IOException: " + e.getMessage());
        }
    }

    /**
     * Test writer rejects adding vectors after finish.
     */
    @SneakyThrows
    public void testWriterRejectsAddAfterFinish() {
        try (Directory directory = newDirectory()) {
            String fileName = ClumpingFileReader.buildClumpingFileName(SEGMENT_NAME, FIELD_NAME);
            try (IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT)) {
                ClumpingFileWriter writer = new ClumpingFileWriter(output, CLUMPING_FACTOR, DIMENSION, VectorDataType.FLOAT);
                writer.finish();
                
                float[] vector = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
                
                IllegalStateException exception = expectThrows(
                    IllegalStateException.class,
                    () -> writer.addHiddenVector(1, vector, 1)
                );
                assertTrue(exception.getMessage().contains("finish()"));
            }
        }
    }

    /**
     * Test writer rejects calling finish twice.
     */
    @SneakyThrows
    public void testWriterRejectsDoubleFinish() {
        try (Directory directory = newDirectory()) {
            String fileName = ClumpingFileReader.buildClumpingFileName(SEGMENT_NAME, FIELD_NAME);
            try (IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT)) {
                ClumpingFileWriter writer = new ClumpingFileWriter(output, CLUMPING_FACTOR, DIMENSION, VectorDataType.FLOAT);
                writer.finish();
                
                IllegalStateException exception = expectThrows(
                    IllegalStateException.class,
                    () -> writer.finish()
                );
                assertTrue(exception.getMessage().contains("finish()"));
            }
        }
    }

    /**
     * Test file name building.
     */
    public void testBuildClumpingFileName() {
        String fileName = ClumpingFileReader.buildClumpingFileName("segment_0", "my_vector_field");
        assertEquals("segment_0_my_vector_field.clump", fileName);
    }

    /**
     * Test with larger dataset to verify scalability.
     */
    @SneakyThrows
    public void testLargerDataset() {
        try (Directory directory = newDirectory()) {
            int numMarkers = 100;
            int hiddenPerMarker = 10;
            int dimension = 128;

            String fileName = ClumpingFileReader.buildClumpingFileName(SEGMENT_NAME, FIELD_NAME);
            try (IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT)) {
                ClumpingFileWriter writer = new ClumpingFileWriter(output, CLUMPING_FACTOR, dimension, VectorDataType.FLOAT);
                
                int docId = 0;
                for (int marker = 0; marker < numMarkers; marker++) {
                    for (int hidden = 0; hidden < hiddenPerMarker; hidden++) {
                        float[] vector = new float[dimension];
                        for (int d = 0; d < dimension; d++) {
                            vector[d] = (float) (marker * 1000 + hidden * 10 + d);
                        }
                        writer.addHiddenVector(docId++, vector, marker);
                    }
                }
                writer.finish();
                
                assertEquals(numMarkers * hiddenPerMarker, writer.getHiddenVectorCount());
                assertEquals(numMarkers, writer.getMarkerCount());
            }

            try (ClumpingFileReader reader = ClumpingFileReader.open(directory, SEGMENT_NAME, FIELD_NAME)) {
                ClumpingFileHeader header = reader.getHeader();
                assertEquals(numMarkers * hiddenPerMarker, header.getHiddenVectorCount());
                assertEquals(numMarkers, header.getMarkerCount());
                assertEquals(dimension, header.getDimension());

                // Verify a few random markers
                for (int marker : new int[] { 0, 50, 99 }) {
                    List<HiddenVectorEntry> vectors = reader.getHiddenVectors(marker);
                    assertEquals(hiddenPerMarker, vectors.size());
                    
                    for (int i = 0; i < vectors.size(); i++) {
                        HiddenVectorEntry entry = vectors.get(i);
                        assertEquals(marker, entry.getMarkerDocId());
                        assertEquals(dimension, entry.getVector().length);
                    }
                }
            }
        }
    }
}
