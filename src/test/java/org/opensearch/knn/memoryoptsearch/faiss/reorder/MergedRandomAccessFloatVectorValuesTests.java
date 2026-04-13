/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.index.FloatVectorValues;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MergedRandomAccessFloatVectorValuesTests extends OpenSearchTestCase {

    private static FloatVectorValues makeSegment(float[][] vectors) {
        int dim = vectors[0].length;
        List<float[]> list = new ArrayList<>(vectors.length);
        for (float[] v : vectors) {
            list.add(v);
        }
        return FloatVectorValues.fromFloats(list, dim);
    }

    /**
     * 3 segments (3, 4, 2 vectors), no deletions. Verify all boundaries.
     */
    public void testNoDeletions() throws IOException {
        float[][] seg0 = { { 1, 2 }, { 3, 4 }, { 5, 6 } };
        float[][] seg1 = { { 10, 20 }, { 30, 40 }, { 50, 60 }, { 70, 80 } };
        float[][] seg2 = { { 100, 200 }, { 300, 400 } };

        FloatVectorValues[] segments = { makeSegment(seg0), makeSegment(seg1), makeSegment(seg2) };
        int[] segmentStarts = { 0, 3, 7, 9 };
        int[][] liveLocalOrds = { null, null, null };

        MergedRandomAccessFloatVectorValues merged = new MergedRandomAccessFloatVectorValues(segments, segmentStarts, liveLocalOrds);

        assertEquals(9, merged.size());
        assertEquals(2, merged.dimension());

        // Segment 0: merged ords 0,1,2
        assertArrayEquals(new float[] { 1, 2 }, merged.vectorValue(0), 0f);
        assertArrayEquals(new float[] { 3, 4 }, merged.vectorValue(1), 0f);
        assertArrayEquals(new float[] { 5, 6 }, merged.vectorValue(2), 0f);

        // Segment 1: merged ords 3,4,5,6
        assertArrayEquals(new float[] { 10, 20 }, merged.vectorValue(3), 0f);
        assertArrayEquals(new float[] { 30, 40 }, merged.vectorValue(4), 0f);
        assertArrayEquals(new float[] { 50, 60 }, merged.vectorValue(5), 0f);
        assertArrayEquals(new float[] { 70, 80 }, merged.vectorValue(6), 0f);

        // Segment 2: merged ords 7,8
        assertArrayEquals(new float[] { 100, 200 }, merged.vectorValue(7), 0f);
        assertArrayEquals(new float[] { 300, 400 }, merged.vectorValue(8), 0f);
    }

    /**
     * With liveLocalOrds mapping (simulating deletions).
     */
    public void testWithLiveLocalOrds() throws IOException {
        // Segment has 4 raw vectors, but only ords 1 and 3 are live
        float[][] seg0Raw = { { 0, 0 }, { 10, 10 }, { 20, 20 }, { 30, 30 } };
        FloatVectorValues[] segments = { makeSegment(seg0Raw) };
        int[] segmentStarts = { 0, 2 };
        // liveLocalOrds[0] maps live index 0 -> raw 1, live index 1 -> raw 3
        int[][] liveLocalOrds = { { 1, 3 } };

        MergedRandomAccessFloatVectorValues merged = new MergedRandomAccessFloatVectorValues(segments, segmentStarts, liveLocalOrds);

        assertEquals(2, merged.size());
        assertArrayEquals(new float[] { 10, 10 }, merged.vectorValue(0), 0f);
        assertArrayEquals(new float[] { 30, 30 }, merged.vectorValue(1), 0f);
    }

    /**
     * copy() returns an independent instance that reads the same data.
     */
    public void testCopy() throws IOException {
        float[][] seg0 = { { 1, 2 }, { 3, 4 } };
        FloatVectorValues[] segments = { makeSegment(seg0) };
        int[] segmentStarts = { 0, 2 };
        int[][] liveLocalOrds = { null };

        MergedRandomAccessFloatVectorValues merged = new MergedRandomAccessFloatVectorValues(segments, segmentStarts, liveLocalOrds);
        MergedRandomAccessFloatVectorValues copy = merged.copy();

        assertNotSame(merged, copy);
        assertEquals(merged.size(), copy.size());
        assertEquals(merged.dimension(), copy.dimension());
        assertArrayEquals(new float[] { 1, 2 }, copy.vectorValue(0), 0f);
        assertArrayEquals(new float[] { 3, 4 }, copy.vectorValue(1), 0f);
    }

    /**
     * size() and dimension() return correct values.
     */
    public void testSizeAndDimension() throws IOException {
        float[][] seg0 = { { 1, 2, 3 } };
        float[][] seg1 = { { 4, 5, 6 }, { 7, 8, 9 } };
        FloatVectorValues[] segments = { makeSegment(seg0), makeSegment(seg1) };
        int[] segmentStarts = { 0, 1, 3 };
        int[][] liveLocalOrds = { null, null };

        MergedRandomAccessFloatVectorValues merged = new MergedRandomAccessFloatVectorValues(segments, segmentStarts, liveLocalOrds);

        assertEquals(3, merged.size());
        assertEquals(3, merged.dimension());
    }

    public void testIteratorThrows() throws IOException {
        float[][] seg0 = { { 1, 2 } };
        FloatVectorValues[] segments = { makeSegment(seg0) };
        int[] segmentStarts = { 0, 1 };
        int[][] liveLocalOrds = { null };

        MergedRandomAccessFloatVectorValues merged = new MergedRandomAccessFloatVectorValues(segments, segmentStarts, liveLocalOrds);

        expectThrows(UnsupportedOperationException.class, merged::iterator);
        expectThrows(UnsupportedOperationException.class, () -> merged.ordToDoc(0));
        expectThrows(UnsupportedOperationException.class, () -> merged.scorer(new float[] { 1, 2 }));
    }
}
