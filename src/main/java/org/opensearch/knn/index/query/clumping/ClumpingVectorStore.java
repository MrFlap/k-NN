/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clumping;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.index.VectorDataType;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Stores and retrieves hidden vectors for clumping-based search.
 * 
 * Hidden vectors are stored in a separate file from the main index.
 * The file format is:
 * - Header: magic number, version, vector data type, dimension, hidden vector count
 * - For each hidden vector: markerDocId (int), hiddenDocId (int), vector data
 * 
 * This allows efficient retrieval of hidden vectors associated with marker vectors
 * found during the initial k-NN search.
 */
@Log4j2
public class ClumpingVectorStore implements Closeable {

    /**
     * File extension for clumping hidden vector files.
     */
    public static final String HIDDEN_VECTORS_EXTENSION = "clump";

    /**
     * Magic number for file format validation.
     */
    private static final int MAGIC_NUMBER = 0x434C554D; // "CLUM"

    /**
     * Current file format version.
     */
    private static final int VERSION = 1;

    private final Directory directory;
    private final String segmentName;
    private final String fieldName;

    public ClumpingVectorStore(Directory directory, String segmentName, String fieldName) {
        this.directory = directory;
        this.segmentName = segmentName;
        this.fieldName = fieldName;
    }

    /**
     * Gets the filename for hidden vectors.
     */
    public String getHiddenVectorsFileName() {
        return segmentName + "_" + fieldName + "." + HIDDEN_VECTORS_EXTENSION;
    }

    /**
     * Writes hidden vectors to disk.
     * 
     * @param hiddenVectors map from hidden doc ID to its vector
     * @param markerAssignments map from hidden doc ID to its assigned marker doc ID
     * @param vectorDataType the vector data type
     * @param dimension the vector dimension
     * @throws IOException if writing fails
     */
    public void writeHiddenVectors(
        Map<Integer, float[]> hiddenVectors,
        Map<Integer, Integer> markerAssignments,
        VectorDataType vectorDataType,
        int dimension
    ) throws IOException {
        String fileName = getHiddenVectorsFileName();
        
        try (IndexOutput output = directory.createOutput(fileName, IOContext.DEFAULT)) {
            // Write header
            output.writeInt(MAGIC_NUMBER);
            output.writeInt(VERSION);
            output.writeInt(vectorDataType.ordinal());
            output.writeInt(dimension);
            output.writeInt(hiddenVectors.size());

            // Write each hidden vector with its marker assignment
            for (Map.Entry<Integer, float[]> entry : hiddenVectors.entrySet()) {
                int hiddenDocId = entry.getKey();
                float[] vector = entry.getValue();
                int markerDocId = markerAssignments.get(hiddenDocId);

                output.writeInt(markerDocId);
                output.writeInt(hiddenDocId);
                writeVector(output, vector, vectorDataType);
            }

            log.debug("Wrote {} hidden vectors to {}", hiddenVectors.size(), fileName);
        }
    }

    /**
     * Reads the hidden vector mapping from disk.
     * This loads the marker-to-hidden mapping without loading the actual vectors.
     * 
     * @return the hidden vector mapping
     * @throws IOException if reading fails
     */
    public HiddenVectorMapping readHiddenVectorMapping() throws IOException {
        String fileName = getHiddenVectorsFileName();
        
        if (!fileExists(fileName)) {
            return HiddenVectorMapping.EMPTY;
        }

        Map<Integer, List<Integer>> markerToHiddenDocs = new HashMap<>();
        Set<Integer> markerDocIds = new HashSet<>();

        try (IndexInput input = directory.openInput(fileName, IOContext.READONCE)) {
            // Read and validate header
            int magic = input.readInt();
            if (magic != MAGIC_NUMBER) {
                throw new IOException("Invalid clumping file format: bad magic number");
            }

            int version = input.readInt();
            if (version != VERSION) {
                throw new IOException("Unsupported clumping file version: " + version);
            }

            int dataTypeOrdinal = input.readInt();
            int dimension = input.readInt();
            int hiddenCount = input.readInt();

            VectorDataType vectorDataType = VectorDataType.values()[dataTypeOrdinal];
            int bytesPerVector = getBytesPerVector(vectorDataType, dimension);

            // Read mappings (skip vector data)
            for (int i = 0; i < hiddenCount; i++) {
                int markerDocId = input.readInt();
                int hiddenDocId = input.readInt();
                
                markerDocIds.add(markerDocId);
                markerToHiddenDocs.computeIfAbsent(markerDocId, k -> new ArrayList<>()).add(hiddenDocId);
                
                // Skip vector data
                input.seek(input.getFilePointer() + bytesPerVector);
            }
        }

        return new HiddenVectorMapping(markerToHiddenDocs, markerDocIds);
    }

    /**
     * Reads hidden vectors for specific marker documents.
     * 
     * @param markerDocIds the marker document IDs to get hidden vectors for
     * @return map from hidden doc ID to its vector
     * @throws IOException if reading fails
     */
    public Map<Integer, float[]> readHiddenVectorsForMarkers(Set<Integer> markerDocIds) throws IOException {
        String fileName = getHiddenVectorsFileName();
        
        if (!fileExists(fileName)) {
            return new HashMap<>();
        }

        Map<Integer, float[]> result = new HashMap<>();

        try (IndexInput input = directory.openInput(fileName, IOContext.READONCE)) {
            // Read header
            int magic = input.readInt();
            if (magic != MAGIC_NUMBER) {
                throw new IOException("Invalid clumping file format: bad magic number");
            }

            int version = input.readInt();
            if (version != VERSION) {
                throw new IOException("Unsupported clumping file version: " + version);
            }

            int dataTypeOrdinal = input.readInt();
            int dimension = input.readInt();
            int hiddenCount = input.readInt();

            VectorDataType vectorDataType = VectorDataType.values()[dataTypeOrdinal];
            int bytesPerVector = getBytesPerVector(vectorDataType, dimension);

            // Read vectors for matching markers
            for (int i = 0; i < hiddenCount; i++) {
                int markerDocId = input.readInt();
                int hiddenDocId = input.readInt();

                if (markerDocIds.contains(markerDocId)) {
                    float[] vector = readVector(input, vectorDataType, dimension);
                    result.put(hiddenDocId, vector);
                } else {
                    // Skip vector data
                    input.seek(input.getFilePointer() + bytesPerVector);
                }
            }
        }

        return result;
    }

    /**
     * Reads a single hidden vector by its document ID.
     * 
     * @param hiddenDocId the hidden document ID
     * @return the vector, or null if not found
     * @throws IOException if reading fails
     */
    public float[] readHiddenVector(int hiddenDocId) throws IOException {
        String fileName = getHiddenVectorsFileName();
        
        if (!fileExists(fileName)) {
            return null;
        }

        try (IndexInput input = directory.openInput(fileName, IOContext.READONCE)) {
            // Read header
            int magic = input.readInt();
            if (magic != MAGIC_NUMBER) {
                throw new IOException("Invalid clumping file format: bad magic number");
            }

            int version = input.readInt();
            if (version != VERSION) {
                throw new IOException("Unsupported clumping file version: " + version);
            }

            int dataTypeOrdinal = input.readInt();
            int dimension = input.readInt();
            int hiddenCount = input.readInt();

            VectorDataType vectorDataType = VectorDataType.values()[dataTypeOrdinal];
            int bytesPerVector = getBytesPerVector(vectorDataType, dimension);

            // Search for the hidden doc
            for (int i = 0; i < hiddenCount; i++) {
                input.readInt(); // marker doc ID (not needed for lookup)
                int currentHiddenDocId = input.readInt();

                if (currentHiddenDocId == hiddenDocId) {
                    return readVector(input, vectorDataType, dimension);
                } else {
                    // Skip vector data
                    input.seek(input.getFilePointer() + bytesPerVector);
                }
            }
        }

        return null;
    }

    private boolean fileExists(String fileName) throws IOException {
        String[] files = directory.listAll();
        for (String file : files) {
            if (file.equals(fileName)) {
                return true;
            }
        }
        return false;
    }

    private void writeVector(IndexOutput output, float[] vector, VectorDataType vectorDataType) throws IOException {
        switch (vectorDataType) {
            case FLOAT:
                for (float v : vector) {
                    output.writeInt(Float.floatToIntBits(v));
                }
                break;
            case BYTE:
                for (float v : vector) {
                    output.writeByte((byte) v);
                }
                break;
            case BINARY:
                for (float v : vector) {
                    output.writeByte((byte) v);
                }
                break;
            default:
                throw new IllegalArgumentException("Unsupported vector data type: " + vectorDataType);
        }
    }

    private float[] readVector(IndexInput input, VectorDataType vectorDataType, int dimension) throws IOException {
        float[] vector = new float[dimension];
        
        switch (vectorDataType) {
            case FLOAT:
                for (int i = 0; i < dimension; i++) {
                    vector[i] = Float.intBitsToFloat(input.readInt());
                }
                break;
            case BYTE:
            case BINARY:
                for (int i = 0; i < dimension; i++) {
                    vector[i] = input.readByte();
                }
                break;
            default:
                throw new IllegalArgumentException("Unsupported vector data type: " + vectorDataType);
        }
        
        return vector;
    }

    private int getBytesPerVector(VectorDataType vectorDataType, int dimension) {
        switch (vectorDataType) {
            case FLOAT:
                return dimension * Float.BYTES;
            case BYTE:
            case BINARY:
                return dimension;
            default:
                throw new IllegalArgumentException("Unsupported vector data type: " + vectorDataType);
        }
    }

    @Override
    public void close() throws IOException {
        // Directory is managed externally, nothing to close here
    }
}
