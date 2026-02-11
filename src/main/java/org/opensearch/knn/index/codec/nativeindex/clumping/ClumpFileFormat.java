/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.clumping;

/**
 * Defines the binary format of a .clump file.
 * <p>
 * Layout:
 * <pre>
 *   [Header]
 *     4 bytes: numMarkers (int)
 *
 *   [Offset table] — (numMarkers + 1) entries, each 4 bytes (int)
 *     offset[0] .. offset[numMarkers]
 *     The adjacency list for marker i spans from offset[i] to offset[i+1] (exclusive).
 *     Offsets are relative to the start of the adjacency data section.
 *
 *   [Adjacency data]
 *     Each entry is a 4-byte int (docId of a hidden vector).
 *     For marker i, hidden doc IDs are at positions:
 *       adjDataStart + offset[i] .. adjDataStart + offset[i+1]
 *     where adjDataStart = 4 + (numMarkers + 1) * 4
 *
 *   [Marker doc ID table] — numMarkers entries, each 4 bytes (int)
 *     markerDocId[0] .. markerDocId[numMarkers - 1]
 *     Maps marker index to actual Lucene doc ID.
 * </pre>
 *
 * To look up hidden vectors for a marker with Lucene docId D:
 * 1. Binary search the marker doc ID table to find marker index i.
 * 2. Read offset[i] and offset[i+1] from the offset table.
 * 3. Read hidden doc IDs from the adjacency data section.
 */
public final class ClumpFileFormat {

    public static final String CLUMP_FILE_EXTENSION = ".clump";

    /** Size of the header: just numMarkers (4 bytes). */
    public static final int HEADER_BYTES = Integer.BYTES;

    private ClumpFileFormat() {}

    /**
     * Byte offset of the offset table entry for marker index i.
     */
    public static int offsetTablePosition(int markerIndex) {
        return HEADER_BYTES + markerIndex * Integer.BYTES;
    }

    /**
     * Byte offset where the adjacency data section starts.
     */
    public static int adjDataStart(int numMarkers) {
        return HEADER_BYTES + (numMarkers + 1) * Integer.BYTES;
    }

    /**
     * Byte offset of the marker doc ID table.
     */
    public static int markerDocIdTableStart(int numMarkers, int totalHiddenVectors) {
        return adjDataStart(numMarkers) + totalHiddenVectors * Integer.BYTES;
    }
}
