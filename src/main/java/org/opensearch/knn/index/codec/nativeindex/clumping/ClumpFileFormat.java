/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.clumping;

/**
 * Defines the binary format of a .clump file (v3 — separated doc IDs and vectors).
 * <p>
 * This format stores vector data contiguously per marker (separated from doc IDs)
 * so that the vector block can be passed directly to SIMD bulk scoring via
 * {@code SimdVectorComputeService} when the file is memory-mapped.
 * <p>
 * Layout:
 * <pre>
 *   [Header]
 *     4 bytes: numMarkers (int)
 *     4 bytes: dimension (int)
 *     1 byte:  vectorDataType (0 = float, 1 = byte, 2 = fp16)
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
 *       D * elementSize bytes: marker vector data
 *       numHidden * 4 bytes:   hidden doc IDs (int[]) — contiguous block
 *       numHidden * D * elementSize bytes: hidden vectors — contiguous block
 * </pre>
 *
 * To expand marker i at query time:
 * 1. Read the marker table entry at index i to get (markerDocId, numHidden, clumpDataOffset).
 * 2. docIdBlockOffset = clumpDataOffset + D * elementSize
 * 3. vectorBlockOffset = docIdBlockOffset + numHidden * 4
 * 4. Read numHidden doc IDs from docIdBlockOffset.
 * 5. Score the contiguous vector block at vectorBlockOffset via SIMD bulk scoring
 *    (or read + decode individually for non-mmap directories).
 *
 * Binary search on the marker table (sorted by markerDocId) maps a Lucene doc ID
 * to a marker index.
 */
public final class ClumpFileFormat {

    public static final String CLUMP_FILE_EXTENSION = ".clump";

    /** Vector data type codes stored in the header. */
    public static final byte VECTOR_TYPE_FLOAT = 0;
    public static final byte VECTOR_TYPE_BYTE = 1;
    /** FP16 (half-precision float): 2 bytes per dimension. Halves I/O for hidden vectors. */
    public static final byte VECTOR_TYPE_FP16 = 2;

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
        if (vectorDataType == VECTOR_TYPE_FLOAT) {
            return Float.BYTES;
        } else if (vectorDataType == VECTOR_TYPE_FP16) {
            return Short.BYTES;
        }
        return Byte.BYTES;
    }

    /**
     * Returns the total bytes for one vector of the given dimension and data type.
     */
    public static int vectorBytes(int dimension, byte vectorDataType) {
        return dimension * bytesPerElement(vectorDataType);
    }

    /**
     * Returns the byte size of one hidden entry (docId + vector) — total bytes per hidden
     * vector across both the doc ID block and the vector block.
     */
    public static int hiddenEntryBytes(int dimension, byte vectorDataType) {
        return Integer.BYTES + vectorBytes(dimension, vectorDataType);
    }

    /**
     * Returns the byte size of the doc ID block for a marker with the given number of hidden vectors.
     */
    public static int docIdBlockBytes(int numHidden) {
        return numHidden * Integer.BYTES;
    }

    /**
     * Returns the byte size of the contiguous vector block for a marker.
     */
    public static int vectorBlockBytes(int numHidden, int dimension, byte vectorDataType) {
        return numHidden * vectorBytes(dimension, vectorDataType);
    }

    /**
     * Returns the total clump data size for a marker (marker vector + doc ID block + vector block).
     */
    public static long markerClumpDataSize(int numHidden, int dimension, byte vectorDataType) {
        return vectorBytes(dimension, vectorDataType)
            + docIdBlockBytes(numHidden)
            + vectorBlockBytes(numHidden, dimension, vectorDataType);
    }
}
