/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.clumping;

/**
 * Defines the binary format of a .clump file (v2 — vectors inline).
 * <p>
 * This format stores all vector data directly in the clump file so that expansion
 * at query time can read vectors sequentially without random access into Lucene's
 * vector storage.
 * <p>
 * Layout:
 * <pre>
 *   [Header]
 *     4 bytes: numMarkers (int)
 *     4 bytes: dimension (int)
 *     1 byte:  vectorDataType (0 = float, 1 = byte)
 *
 *   [Marker Table] — numMarkers entries, each (4 + 4 + 8) = 16 bytes
 *     For each marker i:
 *       4 bytes: markerDocId (int)
 *       4 bytes: numHidden (int) — count of hidden vectors for this marker
 *       8 bytes: clumpDataOffset (long) — byte offset into the file where this marker's
 *                clump data begins (absolute file position)
 *
 *   [Clump Data] — sequential per marker
 *     For each marker i (in marker table order):
 *       // Marker's own vector (so the marker vector is also available for sequential read)
 *       D * elementSize bytes: marker vector data
 *       // Hidden vectors
 *       For each hidden vector j in marker i's clump:
 *         4 bytes: hiddenDocId (int)
 *         D * elementSize bytes: hidden vector data
 * </pre>
 *
 * To expand marker i at query time:
 * 1. Read the marker table entry at index i to get (markerDocId, numHidden, clumpDataOffset).
 * 2. Seek to clumpDataOffset.
 * 3. Skip the marker vector (D * elementSize bytes) — or read it if needed.
 * 4. Sequentially read numHidden entries of (hiddenDocId, hiddenVector).
 * 5. Score each hidden vector against the query vector inline.
 *
 * Binary search on the marker table (sorted by markerDocId) maps a Lucene doc ID
 * to a marker index.
 */
public final class ClumpFileFormat {

    public static final String CLUMP_FILE_EXTENSION = ".clump";

    /** Vector data type codes stored in the header. */
    public static final byte VECTOR_TYPE_FLOAT = 0;
    public static final byte VECTOR_TYPE_BYTE = 1;

    /** Header size: numMarkers (4) + dimension (4) + vectorDataType (1) = 9 bytes. */
    public static final int HEADER_BYTES = Integer.BYTES + Integer.BYTES + Byte.BYTES;

    /** Each marker table entry: markerDocId (4) + numHidden (4) + clumpDataOffset (8) = 16 bytes. */
    public static final int MARKER_TABLE_ENTRY_BYTES = Integer.BYTES + Integer.BYTES + Long.BYTES;

    private ClumpFileFormat() {}

    /**
     * Byte offset of the marker table (immediately after header).
     */
    public static long markerTableStart() {
        return HEADER_BYTES;
    }

    /**
     * Byte offset of the marker table entry for marker index i.
     */
    public static long markerTableEntryOffset(int markerIndex) {
        return HEADER_BYTES + (long) markerIndex * MARKER_TABLE_ENTRY_BYTES;
    }

    /**
     * Byte offset where the clump data section starts (after header + marker table).
     */
    public static long clumpDataStart(int numMarkers) {
        return HEADER_BYTES + (long) numMarkers * MARKER_TABLE_ENTRY_BYTES;
    }

    /**
     * Returns the number of bytes per vector element for the given data type code.
     */
    public static int bytesPerElement(byte vectorDataType) {
        return vectorDataType == VECTOR_TYPE_FLOAT ? Float.BYTES : Byte.BYTES;
    }

    /**
     * Returns the total bytes for one vector of the given dimension and data type.
     */
    public static int vectorBytes(int dimension, byte vectorDataType) {
        return dimension * bytesPerElement(vectorDataType);
    }

    /**
     * Returns the byte size of one hidden entry (docId + vector).
     */
    public static int hiddenEntryBytes(int dimension, byte vectorDataType) {
        return Integer.BYTES + vectorBytes(dimension, vectorDataType);
    }
}
