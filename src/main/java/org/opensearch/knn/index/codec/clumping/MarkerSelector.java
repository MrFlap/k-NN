/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clumping;

/**
 * Utility class for deterministic marker vector selection during clumping.
 * 
 * When clumping is enabled, vectors are partitioned into two groups:
 * <ul>
 *   <li><b>Marker vectors</b>: Indexed in the main k-NN index (FAISS/HNSW)</li>
 *   <li><b>Hidden vectors</b>: Stored in a separate clumping file on disk</li>
 * </ul>
 * 
 * This class provides deterministic selection of marker vectors based on document ID.
 * The selection is reproducible - the same document ID and clumping factor will always
 * produce the same result, ensuring consistent behavior across indexing operations.
 * 
 * <h2>Selection Algorithm</h2>
 * The selection uses a hash-based approach:
 * <ol>
 *   <li>Apply a mixing function to the document ID for better distribution</li>
 *   <li>Take the hash modulo the clumping factor</li>
 *   <li>If the result is 0, the document is a marker; otherwise, it's hidden</li>
 * </ol>
 * 
 * This ensures approximately 1/clumpingFactor vectors become markers.
 * 
 * @see ClumpingFileWriter for writing hidden vectors
 * @see ClumpingFileReader for reading hidden vectors
 */
public final class MarkerSelector {

    /**
     * Private constructor to prevent instantiation.
     * This is a utility class with only static methods.
     */
    private MarkerSelector() {
        // Utility class - do not instantiate
    }

    /**
     * Determines if a document should be a marker vector.
     * 
     * Uses deterministic selection based on document ID hash to ensure reproducibility.
     * The same document ID and clumping factor will always produce the same result.
     * 
     * @param docId The document ID (Lucene segment-level document ID)
     * @param clumpingFactor The clumping factor (ratio of total to markers).
     *                       Must be at least 2. A clumping factor of N means
     *                       approximately 1/N vectors become markers.
     * @return true if the document should be a marker (indexed in main index),
     *         false if the document should be hidden (stored in clumping file)
     * @throws IllegalArgumentException if clumpingFactor is less than 2
     */
    public static boolean isMarker(int docId, int clumpingFactor) {
        if (clumpingFactor < 2) {
            throw new IllegalArgumentException("Clumping factor must be at least 2, got: " + clumpingFactor);
        }
        int hash = hash(docId);
        return (hash % clumpingFactor) == 0;
    }

    /**
     * Applies a mixing function to the document ID for better distribution.
     * 
     * This uses the finalizer from MurmurHash3, which provides excellent
     * avalanche properties - small changes in input produce large changes
     * in output, ensuring uniform distribution across the hash space.
     * 
     * @param docId The document ID to hash
     * @return A non-negative hash value with good distribution properties
     */
    private static int hash(int docId) {
        // MurmurHash3 finalizer - provides excellent avalanche properties
        int h = docId;
        h ^= h >>> 16;
        h *= 0x85ebca6b;
        h ^= h >>> 13;
        h *= 0xc2b2ae35;
        h ^= h >>> 16;
        // Ensure non-negative result by masking off the sign bit
        return h & 0x7FFFFFFF;
    }
}
