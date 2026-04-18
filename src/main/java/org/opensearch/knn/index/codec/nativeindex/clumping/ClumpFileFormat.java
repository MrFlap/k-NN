/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.clumping;

/**
 * Defines the binary format of a .clump file.
 * <p>
 * Two layouts coexist, distinguished by the {@code vectorDataType} byte in the header:
 * <ul>
 *   <li><b>v3 (FLOAT, BYTE, FP16):</b> fixed per-vector element size = {@code D * bytesPerElement}.
 *       No per-vector metadata beyond the raw element bytes.</li>
 *   <li><b>v4 (SQ_1BIT):</b> each vector is stored as a quantized binary code followed by four
 *       correction factors ({@code lowerInterval}, {@code upperInterval},
 *       {@code additionalCorrection}, {@code quantizedComponentSum}), matching the exact layout
 *       consumed by the native SIMD SQ scoring context. Per-vector size is
 *       {@code quantizedVecBytes + 16} bytes.</li>
 * </ul>
 * In both cases the marker's own vector appears first, followed by a contiguous doc-ID block,
 * followed by a contiguous hidden-vector block.
 * <p>
 * Common layout:
 * <pre>
 *   [Header]
 *     4 bytes: numMarkers (int)
 *     4 bytes: dimension (int)
 *     1 byte:  vectorDataType (0 = float, 1 = byte, 2 = fp16, 3 = sq1bit)
 *     — SQ_1BIT only (additional header fields follow immediately) —
 *       4 bytes: quantizedVecBytes (int)   — length of one binary code
 *       4 bytes: centroidDp (float)
 *       D * 4 bytes: centroid vector (float[])
 *
 *   [Marker Table] — numMarkers entries, each (4 + 4 + 8) = 16 bytes
 *     For each marker i:
 *       4 bytes: markerDocId (int)
 *       4 bytes: numHidden (int)
 *       8 bytes: clumpDataOffset (long)
 *
 *   [Clump Data] — sequential per marker, in marker table order
 *     For each marker i:
 *       [one marker vector entry]
 *       numHidden * 4 bytes:  hidden doc IDs (int[])
 *       [numHidden contiguous hidden vector entries]
 *
 *   Where "one vector entry":
 *     FLOAT:   D * 4 bytes                                             (little-endian IEEE754)
 *     FP16:    D * 2 bytes                                             (little-endian half-float)
 *     BYTE:    D bytes
 *     SQ_1BIT: quantizedVecBytes bytes  +  4 * 4 bytes correction factors
 *              (lowerInterval, upperInterval, additionalCorrection, quantizedComponentSum — 16 B)
 * </pre>
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
    /**
     * SQ 1-bit binary code + 16 B correction factors per vector.
     * Entry size = {@code quantizedVecBytes + 16}. {@code quantizedVecBytes} is carried in the
     * header since it depends on dimension padding and isn't derivable from {@code dimension} alone.
     */
    public static final byte VECTOR_TYPE_SQ_1BIT = 3;

    /** Size of correction factors appended after each SQ binary code: 4 floats / 4 ints = 16 B. */
    public static final int SQ_CORRECTION_BYTES = Integer.BYTES * 4;

    /** Header size for fixed-element types (FLOAT / BYTE / FP16): numMarkers + dim + type. */
    public static final int BASE_HEADER_BYTES = Integer.BYTES + Integer.BYTES + Byte.BYTES;

    /**
     * Additional header bytes for SQ_1BIT: quantizedVecBytes (4) + centroidDp (4) + centroid (D*4).
     * Use {@link #sqHeaderExtraBytes(int)} to compute for a given dimension.
     */
    public static int sqHeaderExtraBytes(int dimension) {
        return Integer.BYTES + Float.BYTES + dimension * Float.BYTES;
    }

    /** Each marker table entry: markerDocId (4) + numHidden (4) + clumpDataOffset (8) = 16 bytes. */
    public static final int MARKER_TABLE_ENTRY_BYTES = Integer.BYTES + Integer.BYTES + Long.BYTES;

    private ClumpFileFormat() {}

    /**
     * Returns the total header size for the given vector data type and dimension.
     * For SQ_1BIT the header carries the quantizer's centroid as well.
     */
    public static int headerBytes(byte vectorDataType, int dimension) {
        if (vectorDataType == VECTOR_TYPE_SQ_1BIT) {
            return BASE_HEADER_BYTES + sqHeaderExtraBytes(dimension);
        }
        return BASE_HEADER_BYTES;
    }

    /**
     * Byte offset of the marker table entry for marker index i.
     */
    public static long markerTableEntryOffset(int markerIndex, byte vectorDataType, int dimension) {
        return headerBytes(vectorDataType, dimension) + (long) markerIndex * MARKER_TABLE_ENTRY_BYTES;
    }

    /**
     * Byte offset where the clump data section starts (after header + marker table).
     */
    public static long clumpDataStart(int numMarkers, byte vectorDataType, int dimension) {
        return headerBytes(vectorDataType, dimension) + (long) numMarkers * MARKER_TABLE_ENTRY_BYTES;
    }

    /**
     * Returns the number of bytes per vector element for the given data type code.
     * Not valid for SQ_1BIT (entries are variable-shaped; use {@link #vectorBytes}).
     */
    public static int bytesPerElement(byte vectorDataType) {
        if (vectorDataType == VECTOR_TYPE_FLOAT) {
            return Float.BYTES;
        } else if (vectorDataType == VECTOR_TYPE_FP16) {
            return Short.BYTES;
        } else if (vectorDataType == VECTOR_TYPE_BYTE) {
            return Byte.BYTES;
        }
        throw new IllegalArgumentException("bytesPerElement not defined for type " + vectorDataType);
    }

    /**
     * Returns the total bytes for one vector entry. For SQ_1BIT this is
     * {@code quantizedVecBytes + SQ_CORRECTION_BYTES}; for other types it's
     * {@code dimension * bytesPerElement(type)}. {@code quantizedVecBytes} is ignored for
     * non-SQ types.
     */
    public static int vectorBytes(int dimension, byte vectorDataType, int quantizedVecBytes) {
        if (vectorDataType == VECTOR_TYPE_SQ_1BIT) {
            return quantizedVecBytes + SQ_CORRECTION_BYTES;
        }
        return dimension * bytesPerElement(vectorDataType);
    }

    /**
     * Convenience overload for non-SQ callers.
     */
    public static int vectorBytes(int dimension, byte vectorDataType) {
        return vectorBytes(dimension, vectorDataType, 0);
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
    public static int vectorBlockBytes(int numHidden, int dimension, byte vectorDataType, int quantizedVecBytes) {
        return numHidden * vectorBytes(dimension, vectorDataType, quantizedVecBytes);
    }

    /**
     * Returns the total clump data size for a marker (marker vector + doc ID block + vector block).
     */
    public static long markerClumpDataSize(int numHidden, int dimension, byte vectorDataType, int quantizedVecBytes) {
        int vecSize = vectorBytes(dimension, vectorDataType, quantizedVecBytes);
        return vecSize
            + (long) docIdBlockBytes(numHidden)
            + (long) numHidden * vecSize;
    }
}
