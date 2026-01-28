/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clumping;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.index.VectorDataType;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Reads hidden vectors from a clumping file during search.
 * 
 * The clumping file stores hidden vectors that are not indexed in the main k-NN index
 * (FAISS/HNSW) but are associated with marker vectors. This reader provides efficient
 * access to hidden vectors by marker document ID using the marker-to-offset index.
 * 
 * <h2>File Format</h2>
 * <pre>
 * +------------------+
 * | Header (41 bytes)|
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
 * try (ClumpingFileReader reader = ClumpingFileReader.open(directory, segmentName, fieldName)) {
 *     ClumpingFileHeader header = reader.getHeader();
 *     List&lt;HiddenVectorEntry&gt; hiddenVectors = reader.getHiddenVectors(markerDocId);
 *     // ... process hidden vectors
 * }
 * </pre>
 * 
 * @see ClumpingFileWriter for writing clumping files
 * @see ClumpingFileHeader for header structure
 * @see HiddenVectorEntry for hidden vector representation
 */
@Log4j2
public class ClumpingFileReader implements Closeable {

    private final IndexInput input;
    private final ClumpingFileHeader header;
    private final Map<Integer, Long> markerToOffset;

    /**
     * Private constructor. Use {@link #open(Directory, String, String)} to create instances.
     *
     * @param input          The IndexInput to read from
     * @param header         The parsed file header
     * @param markerToOffset Map of marker document IDs to file offsets
     */
    private ClumpingFileReader(IndexInput input, ClumpingFileHeader header, Map<Integer, Long> markerToOffset) {
        this.input = input;
        this.header = header;
        this.markerToOffset = markerToOffset;
    }

    /**
     * Opens a clumping file for reading.
     * 
     * This method validates the file format and loads the marker-to-offset index
     * into memory for efficient lookups.
     *
     * @param directory   The directory containing the clumping file
     * @param segmentName The segment name
     * @param fieldName   The field name
     * @return A new ClumpingFileReader instance
     * @throws IOException if an I/O error occurs or the file format is invalid
     */
    public static ClumpingFileReader open(Directory directory, String segmentName, String fieldName) throws IOException {
        String fileName = buildClumpingFileName(segmentName, fieldName);
        IndexInput input = directory.openInput(fileName, IOContext.DEFAULT);

        try {
            // Read and validate header
            ClumpingFileHeader header = readAndValidateHeader(input);

            // Read the index offset from the end of the file (before footer)
            // The index offset is stored at (fileLength - footerLength - 8)
            long fileLength = input.length();
            long indexOffsetPosition = fileLength - CodecUtil.footerLength() - 8;
            input.seek(indexOffsetPosition);
            long indexOffset = input.readLong();

            // Read the marker-to-offset index
            Map<Integer, Long> markerToOffset = readMarkerIndex(input, indexOffset);

            log.debug(
                "Opened clumping file: segment={}, field={}, hiddenVectors={}, markers={}",
                segmentName,
                fieldName,
                header.getHiddenVectorCount(),
                header.getMarkerCount()
            );

            return new ClumpingFileReader(input, header, markerToOffset);
        } catch (Exception e) {
            // Close input on error
            input.close();
            throw e;
        }
    }

    /**
     * Returns the file header containing metadata about the clumping file.
     *
     * @return The clumping file header
     */
    public ClumpingFileHeader getHeader() {
        return header;
    }

    /**
     * Retrieves hidden vectors associated with a specific marker document ID.
     * 
     * This method uses the marker-to-offset index for O(1) lookup of the file
     * position, then reads the hidden vector group from that position.
     *
     * @param markerDocId The marker document ID
     * @return List of hidden vectors associated with the marker, or empty list if none
     * @throws IOException if an I/O error occurs
     */
    public List<HiddenVectorEntry> getHiddenVectors(int markerDocId) throws IOException {
        Long offset = markerToOffset.get(markerDocId);
        if (offset == null) {
            return Collections.emptyList();
        }

        input.seek(offset);
        return readHiddenVectorGroup(markerDocId);
    }

    /**
     * Retrieves hidden vectors for multiple marker document IDs.
     * 
     * This method is more efficient than calling {@link #getHiddenVectors(int)}
     * multiple times when retrieving hidden vectors for many markers.
     *
     * @param markerDocIds Collection of marker document IDs
     * @return Map of marker document ID to list of hidden vectors
     * @throws IOException if an I/O error occurs
     */
    public Map<Integer, List<HiddenVectorEntry>> getHiddenVectorsForMarkers(Collection<Integer> markerDocIds) throws IOException {
        Map<Integer, List<HiddenVectorEntry>> result = new HashMap<>();

        for (Integer markerDocId : markerDocIds) {
            List<HiddenVectorEntry> hiddenVectors = getHiddenVectors(markerDocId);
            if (!hiddenVectors.isEmpty()) {
                result.put(markerDocId, hiddenVectors);
            }
        }

        return result;
    }

    /**
     * Returns all marker document IDs that have associated hidden vectors.
     * 
     * This is useful during merge operations to iterate through all markers
     * and retrieve their hidden vectors.
     *
     * @return Set of marker document IDs
     */
    public Collection<Integer> getAllMarkerDocIds() {
        return Collections.unmodifiableCollection(markerToOffset.keySet());
    }

    /**
     * Retrieves all hidden vectors from the clumping file.
     * 
     * This method reads all hidden vectors grouped by their marker document IDs.
     * It is useful during merge operations when all hidden vectors need to be
     * re-partitioned.
     *
     * @return List of all hidden vector entries
     * @throws IOException if an I/O error occurs
     */
    public List<HiddenVectorEntry> getAllHiddenVectors() throws IOException {
        List<HiddenVectorEntry> allEntries = new ArrayList<>();
        
        for (Integer markerDocId : markerToOffset.keySet()) {
            List<HiddenVectorEntry> entries = getHiddenVectors(markerDocId);
            allEntries.addAll(entries);
        }
        
        return allEntries;
    }

    /**
     * Builds the clumping file name from segment and field names.
     *
     * @param segmentName The segment name
     * @param fieldName   The field name
     * @return The clumping file name
     */
    public static String buildClumpingFileName(String segmentName, String fieldName) {
        return String.format("%s_%s.%s", segmentName, fieldName, ClumpingFileWriter.CLUMPING_EXTENSION);
    }

    /**
     * Checks if a clumping file exists for the given segment and field.
     *
     * @param directory   The directory to check
     * @param segmentName The segment name
     * @param fieldName   The field name
     * @return true if the clumping file exists, false otherwise
     * @throws IOException if an I/O error occurs while checking
     */
    public static boolean exists(Directory directory, String segmentName, String fieldName) throws IOException {
        String fileName = buildClumpingFileName(segmentName, fieldName);
        try {
            // Check if file exists by trying to get its length
            directory.fileLength(fileName);
            return true;
        } catch (java.io.FileNotFoundException e) {
            return false;
        }
    }

    /**
     * Reads and validates the file header.
     *
     * @param input The IndexInput to read from
     * @return The parsed header
     * @throws IOException if an I/O error occurs or the header is invalid
     */
    private static ClumpingFileHeader readAndValidateHeader(IndexInput input) throws IOException {
        // Read magic bytes
        byte[] magicBytes = new byte[4];
        input.readBytes(magicBytes, 0, 4);

        // Validate magic bytes
        if (!Arrays.equals(magicBytes, ClumpingFileHeader.MAGIC_BYTES)) {
            throw new IOException(
                String.format(
                    "Invalid clumping file: expected magic bytes %s, got %s",
                    Arrays.toString(ClumpingFileHeader.MAGIC_BYTES),
                    Arrays.toString(magicBytes)
                )
            );
        }

        // Read format version
        int formatVersion = input.readInt();

        // Validate format version
        if (formatVersion != ClumpingFileHeader.FORMAT_VERSION) {
            throw new IOException(
                String.format(
                    "Unsupported clumping file format version: expected %d, got %d",
                    ClumpingFileHeader.FORMAT_VERSION,
                    formatVersion
                )
            );
        }

        // Read remaining header fields
        int clumpingFactor = input.readInt();
        int dimension = input.readInt();
        byte vectorDataTypeByte = input.readByte();
        VectorDataType vectorDataType = byteToVectorDataType(vectorDataTypeByte);
        int hiddenVectorCount = input.readInt();
        int markerCount = input.readInt();
        long dataOffset = input.readLong();
        long indexOffset = input.readLong(); // This is a placeholder in the header; actual value is at end of file

        return ClumpingFileHeader.builder()
            .magicBytes(magicBytes)
            .formatVersion(formatVersion)
            .clumpingFactor(clumpingFactor)
            .dimension(dimension)
            .vectorDataType(vectorDataType)
            .hiddenVectorCount(hiddenVectorCount)
            .markerCount(markerCount)
            .dataOffset(dataOffset)
            .indexOffset(indexOffset)
            .build();
    }

    /**
     * Reads the marker-to-offset index from the file.
     *
     * @param input       The IndexInput to read from
     * @param indexOffset The file offset to the index section
     * @return Map of marker document ID to file offset
     * @throws IOException if an I/O error occurs
     */
    private static Map<Integer, Long> readMarkerIndex(IndexInput input, long indexOffset) throws IOException {
        input.seek(indexOffset);

        // Read number of index entries
        int entryCount = input.readInt();

        Map<Integer, Long> markerToOffset = new HashMap<>(entryCount);

        // Read each index entry
        for (int i = 0; i < entryCount; i++) {
            int markerDocId = input.readInt();
            long offset = input.readLong();
            markerToOffset.put(markerDocId, offset);
        }

        return markerToOffset;
    }

    /**
     * Reads a group of hidden vectors for a marker from the current file position.
     *
     * @param expectedMarkerDocId The expected marker document ID (for validation)
     * @return List of hidden vectors
     * @throws IOException if an I/O error occurs
     */
    private List<HiddenVectorEntry> readHiddenVectorGroup(int expectedMarkerDocId) throws IOException {
        // Read marker document ID
        int markerDocId = input.readInt();

        // Validate marker document ID matches expected
        if (markerDocId != expectedMarkerDocId) {
            throw new IOException(
                String.format(
                    "Marker document ID mismatch: expected %d, got %d",
                    expectedMarkerDocId,
                    markerDocId
                )
            );
        }

        // Read count of hidden vectors
        int count = input.readInt();

        List<HiddenVectorEntry> hiddenVectors = new ArrayList<>(count);

        // Read each hidden vector
        for (int i = 0; i < count; i++) {
            int docId = input.readInt();
            float[] vector = readVector();

            HiddenVectorEntry entry = HiddenVectorEntry.builder()
                .docId(docId)
                .vector(vector)
                .markerDocId(markerDocId)
                .build();

            hiddenVectors.add(entry);
        }

        return hiddenVectors;
    }

    /**
     * Reads a vector from the current file position.
     *
     * @return The vector data
     * @throws IOException if an I/O error occurs
     */
    private float[] readVector() throws IOException {
        int dimension = header.getDimension();
        float[] vector = new float[dimension];

        for (int i = 0; i < dimension; i++) {
            vector[i] = Float.intBitsToFloat(input.readInt());
        }

        return vector;
    }

    /**
     * Converts a byte value to VectorDataType.
     *
     * @param value The byte value (0=float, 1=byte, 2=binary)
     * @return The corresponding VectorDataType
     * @throws IOException if the value is invalid
     */
    private static VectorDataType byteToVectorDataType(byte value) throws IOException {
        switch (value) {
            case 0:
                return VectorDataType.FLOAT;
            case 1:
                return VectorDataType.BYTE;
            case 2:
                return VectorDataType.BINARY;
            default:
                throw new IOException("Invalid vector data type byte: " + value);
        }
    }

    /**
     * Closes this reader and releases resources.
     *
     * @throws IOException if an I/O error occurs
     */
    @Override
    public void close() throws IOException {
        input.close();
    }
}
