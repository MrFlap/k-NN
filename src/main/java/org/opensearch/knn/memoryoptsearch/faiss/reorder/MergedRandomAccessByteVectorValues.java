/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.search.VectorScorer;

import java.io.IOException;
import java.util.Arrays;

/**
 * A {@link ByteVectorValues} that provides random-access {@code vectorValue(int mergedOrd)}
 * over multiple source segments' quantized vectors. Each segment contributes a contiguous
 * range of merged ordinals.
 * <p>
 * Used during merge to read quantized vectors from source segments' .faiss files for
 * BP permutation computation. The working set is much smaller than full-precision vectors
 * (e.g., 96 bytes vs 3072 bytes per vector for 768-dim 32x BQ).
 */
public class MergedRandomAccessByteVectorValues extends ByteVectorValues {

    private final ByteVectorValues[] segmentValues;
    private final int[] segmentStarts;
    private final int[][] liveLocalOrds;

    /**
     * @param segmentValues  per-segment ByteVectorValues from .faiss flat storage
     * @param segmentStarts  length = numSegments + 1; segmentStarts[i] is the first merged ordinal for segment i
     * @param liveLocalOrds  per-segment mapping from live index to raw local ordinal; null entry means no deletions
     */
    public MergedRandomAccessByteVectorValues(ByteVectorValues[] segmentValues, int[] segmentStarts, int[][] liveLocalOrds) {
        this.segmentValues = segmentValues;
        this.segmentStarts = segmentStarts;
        this.liveLocalOrds = liveLocalOrds;
    }

    @Override
    public byte[] vectorValue(int mergedOrd) throws IOException {
        int seg = findSegment(mergedOrd);
        int liveIdx = mergedOrd - segmentStarts[seg];
        int rawLocalOrd = liveLocalOrds[seg] != null ? liveLocalOrds[seg][liveIdx] : liveIdx;
        return segmentValues[seg].vectorValue(rawLocalOrd);
    }

    @Override
    public MergedRandomAccessByteVectorValues copy() throws IOException {
        ByteVectorValues[] copies = new ByteVectorValues[segmentValues.length];
        for (int i = 0; i < segmentValues.length; i++) {
            copies[i] = segmentValues[i] != null ? segmentValues[i].copy() : null;
        }
        return new MergedRandomAccessByteVectorValues(copies, segmentStarts, liveLocalOrds);
    }

    @Override
    public int size() {
        return segmentStarts[segmentStarts.length - 1];
    }

    @Override
    public int dimension() {
        for (ByteVectorValues sv : segmentValues) {
            if (sv != null) return sv.dimension();
        }
        throw new IllegalStateException("No non-null segment values");
    }

    @Override
    public DocIndexIterator iterator() {
        throw new UnsupportedOperationException("Random access only");
    }

    @Override
    public int ordToDoc(int ord) {
        throw new UnsupportedOperationException("Random access only");
    }

    @Override
    public VectorScorer scorer(byte[] query) {
        throw new UnsupportedOperationException("Random access only");
    }

    private int findSegment(int mergedOrd) {
        int idx = Arrays.binarySearch(segmentStarts, mergedOrd);
        if (idx < 0) {
            idx = -idx - 2;
        }
        while (segmentValues[idx] == null) {
            idx++;
        }
        return idx;
    }
}
