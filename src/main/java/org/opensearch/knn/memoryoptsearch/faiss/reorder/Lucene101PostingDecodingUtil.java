/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.store.IndexInput;

import java.io.IOException;

public class Lucene101PostingDecodingUtil {

    /** The wrapper {@link IndexInput}. */
    public final IndexInput in;

    /** Sole constructor, called by sub-classes. */
    public Lucene101PostingDecodingUtil(IndexInput in) {
        this.in = in;
    }

    /**
     * Core methods for decoding blocks of docs / freqs / positions / offsets.
     *
     * <ul>
     *   <li>Read {@code count} ints.
     *   <li>For all {@code i} &gt;= 0 so that {@code bShift - i * dec} &gt; 0, apply shift {@code
     *       bShift - i * dec} and store the result in {@code b} at offset {@code count * i}.
     *   <li>Apply mask {@code cMask} and store the result in {@code c} starting at offset {@code
     *       cIndex}.
     * </ul>
     */
    public void splitInts(int count, int[] b, int bShift, int dec, int bMask, int[] c, int cIndex, int cMask) throws IOException {
        in.readInts(c, cIndex, count);
        final int maxIter = (bShift - 1) / dec;

        // Process each shift level across all elements (better for vectorization)
        for (int j = 0; j <= maxIter; ++j) {
            final int shift = bShift - j * dec;
            final int bOffset = count * j;
            // Vectorizable loop: contiguous memory access with simple operations
            for (int i = 0; i < count; ++i) {
                b[bOffset + i] = (c[cIndex + i] >>> shift) & bMask;
            }
        }

        // Apply mask to c array (vectorizable)
        for (int i = 0; i < count; ++i) {
            c[cIndex + i] &= cMask;
        }
    }
}
