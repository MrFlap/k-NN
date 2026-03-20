/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.VectorScorer;

import java.io.IOException;
import java.util.Arrays;

/**
 * A {@link FloatVectorValues} that provides random-access {@code vectorValue(int mergedOrd)}
 * over multiple source segments. Each segment contributes a contiguous range of merged ordinals.
 * An optional {@code liveLocalOrds} mapping handles deleted documents by translating live indices
 * to raw local ordinals within each segment.
 */
public class MergedRandomAccessFloatVectorValues extends FloatVectorValues {

    private final FloatVectorValues[] segmentValues;
    private final int[] segmentStarts;
    private final int[][] liveLocalOrds;

    /**
     * @param segmentValues  per-segment FloatVectorValues sources
     * @param segmentStarts  length = numSegments + 1; segmentStarts[i] is the first merged ordinal for segment i
     * @param liveLocalOrds  per-segment mapping from live index to raw local ordinal; null entry means no deletions
     */
    public MergedRandomAccessFloatVectorValues(FloatVectorValues[] segmentValues, int[] segmentStarts, int[][] liveLocalOrds) {
        this.segmentValues = segmentValues;
        this.segmentStarts = segmentStarts;
        this.liveLocalOrds = liveLocalOrds;
    }

    @Override
    public float[] vectorValue(int mergedOrd) throws IOException {
        int seg = findSegment(mergedOrd);
        int liveIdx = mergedOrd - segmentStarts[seg];
        int rawLocalOrd = liveLocalOrds[seg] != null ? liveLocalOrds[seg][liveIdx] : liveIdx;
        return segmentValues[seg].vectorValue(rawLocalOrd);
    }

    @Override
    public MergedRandomAccessFloatVectorValues copy() throws IOException {
        FloatVectorValues[] copies = new FloatVectorValues[segmentValues.length];
        for (int i = 0; i < segmentValues.length; i++) {
            copies[i] = segmentValues[i].copy();
        }
        return new MergedRandomAccessFloatVectorValues(copies, segmentStarts, liveLocalOrds);
    }

    @Override
    public int size() {
        return segmentStarts[segmentStarts.length - 1];
    }

    @Override
    public int dimension() {
        return segmentValues[0].dimension();
    }

    @Override
    public DocIndexIterator iterator() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int ordToDoc(int ord) {
        throw new UnsupportedOperationException();
    }

    @Override
    public VectorScorer scorer(float[] query) {
        throw new UnsupportedOperationException();
    }

    private int findSegment(int mergedOrd) {
        int idx = Arrays.binarySearch(segmentStarts, mergedOrd);
        // If exact match, idx is the segment index.
        // If not found, binarySearch returns -(insertion point) - 1.
        // The segment is (insertion point - 1).
        if (idx < 0) {
            idx = -idx - 2;
        }
        return idx;
    }
}
