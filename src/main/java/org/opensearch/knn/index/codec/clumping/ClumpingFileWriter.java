/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clumping;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.index.VectorDataType;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Writes hidden vectors to a clumping file during indexing.
 * 
 * The clumping file stores hidden vectors that are not indexed in the main k-NN index
 * (FAISS/HNSW) but are associated with marker vectors. This writer buffers hidden vectors
 * grouped by their associated marker document ID and writes them to disk in a format
 * optimized for efficient retrieval during search.
 * 
 * <h2>File Format</h2>
 * <pre>
 * +------------------+
 * | Header (fixed)   |
 * +------------------+
 * | Magic: 4 bytes   | "KNNC"
 * | Version: 4 bytes | Format version (1)
 * | Factor: 4 bytes  | Clumping factor
 * | Dim: 4 bytes     | Vector dimension
 * | Type: 1 byte     | Vector data type (0=float, 1=byte, 2=binary)
 * | Hidden#: 4 bytes | Total hidden vector count
 * | Marker#: 4 bytes | Total marker count
 * | DataOff: 8 bytes | Offset to data section
 * | IdxOff: 8 bytes  | Reserved (actual index offset stored at end)
 * +------------------+
 * | Data Section     |
 * +------------------+
 * | MarkerID: 4 bytes| Marker doc ID
 * | Count: 4 bytes   | Number of hidden vectors for this marker
 * | [HiddenEntry]... | Repeated for each hidden vector:
 * |   DocID: 4 bytes |   Hidden doc ID
 * |   Vector: D*4 B  |   Vector data (D = dimension)
 * | ... repeat for each marker group
 * +------------------+
 * | Index Section    |
 * +------------------+
 * | Entry#: 4 bytes  | Number of index entries
 * | [IndexEntry]...  | Repeated for each marker:
 * |   MarkerID: 4 B  |   Marker doc ID
 * |   Offset: 8 B    |   File offset to data
 * +------------------+
 * | Index Pointer    |
 * +------------------+
 * | IdxOff: 8 bytes  | Offset to index section (for reader)
 * +------------------+
 * | Footer           |
 * +------------------+
 * | Checksum         | CRC32 checksum (via CodecUtil)
 * +------------------+
 * </pre>
 * 
 * <h2>Usage</h2>
 * <pre>
 * try (ClumpingFileWriter writer = new ClumpingFileWriter(output, clumpingFactor, dimension, vectorDataType)) {
 *     writer.addHiddenVector(docId1, vector1, markerDocId1);
 *     writer.addHiddenVector(docId2, vector2, markerDocId2);
 *     // ... add more hidden vectors
 *     writer.finish();
 * }
 * </pre>
 * 
 * @see ClumpingFileReader for reading clumping files
 * @see ClumpingFileHeader for header structure
 * @see HiddenVectorEntry for hidden vector representation
 */
@Log4j2
public class ClumpingFileWriter implements Closeable {

    /**
     * File extension for clumping files.
     */
    public static final String CLUMPING_EXTENSION = "clump";

    /**
     * Magic bytes identifying a clumping file.
     * The value "KNNC" stands for "KNN Clumping".
     */
    public static final byte[] MAGIC_BYTES = new byte[] { 'K', 'N', 'N', 'C' };

    /**
     * Current format version of the clumping file.
     */
    public static final int FORMAT_VERSION = 1;

    /**
     * Header size in bytes (fixed portion before data section).
     * Magic(4) + Version(4) + Factor(4) + Dim(4) + Type(1) + Hidden#(4) + Marker#(4) + DataOff(8) + IdxOff(8) = 41 bytes
     */
    public static final int HEADER_SIZE = 41;

    private final IndexOutput output;
    private final int clumpingFactor;
    private final int dimension;
    private final VectorDataType vectorDataType;

    /**
     * Temporary storage during indexing: maps marker doc ID to list of hidden vectors.
     * Using TreeMap to ensure deterministic ordering when writing.
     */
    private final Map<Integer, List<HiddenVectorEntry>> markerToHiddenVectors;

    /**
     * Total count of hidden vectors added.
     */
    private int hiddenVectorCount;

    /**
     * Flag to track if finish() has been called.
     */
    private boolean finished;

    /**
     * Creates a new ClumpingFileWriter.
     *
     * @param output         The IndexOutput to write to
     * @param clumpingFactor The clumping factor (ratio of total vectors to marker vectors)
     * @param dimension      The dimension of vectors
     * @param vectorDataType The data type of vectors
     */
    public ClumpingFileWriter(IndexOutput output, int clumpingFactor, int dimension, VectorDataType vectorDataType) {
        this.output = output;
        this.clumpingFactor = clumpingFactor;
        this.dimension = dimension;
        this.vectorDataType = vectorDataType;
        this.markerToHiddenVectors = new TreeMap<>();
        this.hiddenVectorCount = 0;
        this.finished = false;
    }

    /**
     * Adds a hidden vector to be written to the clumping file.
     * 
     * The hidden vector is buffered in memory and will be written to disk when
     * {@link #finish()} is called. Hidden vectors are grouped by their associated
     * marker document ID for efficient retrieval during search.
     *
     * @param docId       The document ID of the hidden vector
     * @param vector      The vector data
     * @param markerDocId The document ID of the associated marker vector
     * @throws IllegalStateException if finish() has already been called
     */
    public void addHiddenVector(int docId, float[] vector, int markerDocId) {
        if (finished) {
            throw new IllegalStateException("Cannot add hidden vectors after finish() has been called");
        }

        if (vector == null) {
            throw new IllegalArgumentException("Vector cannot be null");
        }

        if (vector.length != dimension) {
            throw new IllegalArgumentException(
                String.format("Vector dimension mismatch: expected %d, got %d", dimension, vector.length)
            );
        }

        HiddenVectorEntry entry = HiddenVectorEntry.builder()
            .docId(docId)
            .vector(vector)
            .markerDocId(markerDocId)
            .build();

        markerToHiddenVectors.computeIfAbsent(markerDocId, k -> new ArrayList<>()).add(entry);
        hiddenVectorCount++;

        if (log.isTraceEnabled()) {
            log.trace("Added hidden vector: docId={}, markerDocId={}", docId, markerDocId);
        }
    }

    /**
     * Finishes writing the clumping file.
     * 
     * This method writes the complete file structure:
     * 1. Header with metadata (data offset known, index offset placeholder)
     * 2. Data section with hidden vectors grouped by marker
     * 3. Index section with marker-to-offset mappings
     * 4. Index offset pointer (8 bytes) - allows reader to find index section
     * 5. Footer with checksum (via CodecUtil)
     *
     * @throws IOException if an I/O error occurs
     * @throws IllegalStateException if finish() has already been called
     */
    public void finish() throws IOException {
        if (finished) {
            throw new IllegalStateException("finish() has already been called");
        }
        finished = true;

        log.debug("Finishing clumping file: hiddenVectors={}, markers={}", hiddenVectorCount, markerToHiddenVectors.size());

        // Calculate data offset - header is written first, then data section
        long dataOffset = HEADER_SIZE;

        // Write header (index offset will be 0 as placeholder - actual offset stored at end)
        writeHeader(dataOffset, 0);

        // Write hidden vectors grouped by marker and collect offsets
        Map<Integer, Long> markerToOffset = writeHiddenVectorData();

        // Record where index section starts
        long indexOffset = output.getFilePointer();

        // Write marker-to-offset index
        writeMarkerIndex(markerToOffset);

        // Write the index offset at the end of the file (before footer)
        // This allows the reader to find the index section without seeking
        // The reader reads this value from (fileLength - footerLength - 8)
        output.writeLong(indexOffset);

        // Write footer with checksum
        CodecUtil.writeFooter(output);

        log.debug("Clumping file finished: dataOffset={}, indexOffset={}", dataOffset, indexOffset);
    }

    /**
     * Writes the file header.
     *
     * @param dataOffset  Offset to the data section
     * @param indexOffset Offset to the index section (0 if not yet known)
     * @throws IOException if an I/O error occurs
     */
    private void writeHeader(long dataOffset, long indexOffset) throws IOException {
        // Magic bytes
        output.writeBytes(MAGIC_BYTES, MAGIC_BYTES.length);

        // Format version
        output.writeInt(FORMAT_VERSION);

        // Clumping factor
        output.writeInt(clumpingFactor);

        // Vector dimension
        output.writeInt(dimension);

        // Vector data type (0=float, 1=byte, 2=binary)
        output.writeByte(getVectorDataTypeByte(vectorDataType));

        // Hidden vector count
        output.writeInt(hiddenVectorCount);

        // Marker count
        output.writeInt(markerToHiddenVectors.size());

        // Data section offset
        output.writeLong(dataOffset);

        // Index section offset - this will be written at the end of the file
        // since we don't know it until after writing the data section
        // We write a placeholder here and the actual offset at the end
        output.writeLong(indexOffset);
    }

    /**
     * Writes the hidden vector data section.
     * 
     * Hidden vectors are written grouped by their associated marker document ID.
     * For each marker group:
     * - Marker document ID (4 bytes)
     * - Count of hidden vectors (4 bytes)
     * - For each hidden vector:
     *   - Document ID (4 bytes)
     *   - Vector data (dimension * 4 bytes for float)
     *
     * @return Map of marker document ID to file offset
     * @throws IOException if an I/O error occurs
     */
    private Map<Integer, Long> writeHiddenVectorData() throws IOException {
        Map<Integer, Long> markerToOffset = new HashMap<>();

        for (Map.Entry<Integer, List<HiddenVectorEntry>> entry : markerToHiddenVectors.entrySet()) {
            int markerDocId = entry.getKey();
            List<HiddenVectorEntry> hiddenVectors = entry.getValue();

            // Record offset for this marker group
            long offset = output.getFilePointer();
            markerToOffset.put(markerDocId, offset);

            // Write marker document ID
            output.writeInt(markerDocId);

            // Write count of hidden vectors for this marker
            output.writeInt(hiddenVectors.size());

            // Write each hidden vector
            for (HiddenVectorEntry hiddenVector : hiddenVectors) {
                // Write hidden document ID
                output.writeInt(hiddenVector.getDocId());

                // Write vector data
                writeVector(hiddenVector.getVector());
            }
        }

        return markerToOffset;
    }

    /**
     * Writes a vector to the output.
     *
     * @param vector The vector to write
     * @throws IOException if an I/O error occurs
     */
    private void writeVector(float[] vector) throws IOException {
        for (float value : vector) {
            output.writeInt(Float.floatToIntBits(value));
        }
    }

    /**
     * Writes the marker-to-offset index section.
     * 
     * The index section contains:
     * - Number of index entries (4 bytes)
     * - For each marker:
     *   - Marker document ID (4 bytes)
     *   - File offset to data (8 bytes)
     *
     * @param markerToOffset Map of marker document ID to file offset
     * @throws IOException if an I/O error occurs
     */
    private void writeMarkerIndex(Map<Integer, Long> markerToOffset) throws IOException {
        // Write number of index entries
        output.writeInt(markerToOffset.size());

        // Write each index entry (sorted by marker ID for deterministic output)
        for (Map.Entry<Integer, Long> entry : new TreeMap<>(markerToOffset).entrySet()) {
            // Write marker document ID
            output.writeInt(entry.getKey());

            // Write file offset
            output.writeLong(entry.getValue());
        }
    }

    /**
     * Converts VectorDataType to byte representation for file storage.
     *
     * @param vectorDataType The vector data type
     * @return Byte representation (0=float, 1=byte, 2=binary)
     */
    private byte getVectorDataTypeByte(VectorDataType vectorDataType) {
        switch (vectorDataType) {
            case FLOAT:
                return 0;
            case BYTE:
                return 1;
            case BINARY:
                return 2;
            default:
                throw new IllegalArgumentException("Unknown vector data type: " + vectorDataType);
        }
    }

    /**
     * Returns the total number of hidden vectors added.
     *
     * @return The hidden vector count
     */
    public int getHiddenVectorCount() {
        return hiddenVectorCount;
    }

    /**
     * Returns the number of markers with associated hidden vectors.
     *
     * @return The marker count
     */
    public int getMarkerCount() {
        return markerToHiddenVectors.size();
    }

    /**
     * Closes this writer and releases resources.
     * 
     * Note: This does NOT close the underlying IndexOutput, as that is typically
     * managed by the caller. Call {@link #finish()} before closing to ensure
     * all data is written.
     *
     * @throws IOException if an I/O error occurs
     */
    @Override
    public void close() throws IOException {
        if (!finished && hiddenVectorCount > 0) {
            log.warn("ClumpingFileWriter closed without calling finish(). Data may be incomplete.");
        }
        // Clear the buffer to release memory
        markerToHiddenVectors.clear();
    }
}
