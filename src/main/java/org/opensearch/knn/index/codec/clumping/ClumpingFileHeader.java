/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clumping;

import lombok.Builder;
import lombok.Value;
import org.opensearch.knn.index.VectorDataType;

/**
 * Header structure for the clumping file.
 * 
 * The clumping file stores hidden vectors that are not indexed in the main k-NN index
 * (FAISS/HNSW) but are associated with marker vectors. This header contains metadata
 * about the file format and the vectors stored within.
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
 * | IdxOff: 8 bytes  | Offset to index section
 * +------------------+
 * </pre>
 * 
 * @see ClumpingFileWriter for writing clumping files
 * @see ClumpingFileReader for reading clumping files
 */
@Value
@Builder
public class ClumpingFileHeader {

    /**
     * Magic bytes identifying a clumping file.
     * The value "KNNC" stands for "KNN Clumping".
     */
    public static final byte[] MAGIC_BYTES = new byte[] { 'K', 'N', 'N', 'C' };

    /**
     * Current format version of the clumping file.
     * This version number is incremented when backward-incompatible changes
     * are made to the file format.
     */
    public static final int FORMAT_VERSION = 1;

    /**
     * The magic bytes read from the file header.
     * Must match {@link #MAGIC_BYTES} for a valid clumping file.
     */
    byte[] magicBytes;

    /**
     * The format version of the clumping file.
     * Used to ensure compatibility when reading files created by different versions.
     */
    int formatVersion;

    /**
     * The clumping factor used when creating this file.
     * This determines the ratio of total vectors to marker vectors (approximately 1/clumpingFactor
     * vectors become markers).
     */
    int clumpingFactor;

    /**
     * The dimension of vectors stored in this file.
     * All vectors in the file must have the same dimension.
     */
    int dimension;

    /**
     * The data type of vectors stored in this file.
     * Determines how vector data is serialized (float, byte, or binary).
     */
    VectorDataType vectorDataType;

    /**
     * The total number of hidden vectors stored in this file.
     * Hidden vectors are vectors that are not indexed in the main k-NN index.
     */
    int hiddenVectorCount;

    /**
     * The total number of marker vectors that have associated hidden vectors.
     * This is the number of entries in the marker-to-offset index section.
     */
    int markerCount;

    /**
     * The file offset to the beginning of the hidden vector data section.
     * Used for seeking directly to the data when reading.
     */
    long dataOffset;

    /**
     * The file offset to the beginning of the marker-to-offset index section.
     * The index section provides O(1) lookup of hidden vectors by marker document ID.
     */
    long indexOffset;
}
