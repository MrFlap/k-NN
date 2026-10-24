/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.FilterIndexInput;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.mockito.MockedStatic;
import org.opensearch.knn.common.featureflags.KNNFeatureFlags;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.mockito.Mockito.mockStatic;

public class DocIdsIteratorValuesPrefetchTests extends OpenSearchTestCase {
    // Vectors exist only for odd docs, so ord = (doc - 1) / 2.
    private static final int NUM_VECTORS = 50;

    public void testPrefetchByDocIds_whenCandidatesFarApart_thenPrefetchesEachCandidateOrd() throws IOException {
        // 16KB vectors: two ords are never coalesced into one 32KB prefetch unless adjacent.
        final int dimension = 4096;
        final int vectorBytes = dimension * Float.BYTES;
        try (
            ByteBuffersDirectory dir = new ByteBuffersDirectory();
            MockedStatic<KNNFeatureFlags> flags = mockStatic(KNNFeatureFlags.class)
        ) {
            flags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);
            final RecordingIndexInput slice = createSlice(dir, vectorBytes);

            // doc 4 has no vector and must be skipped.
            new KNNVectorValuesIterator.DocIdsIteratorValues(new SparseSliceValues(slice, dimension)).prefetchByDocIds(
                new int[] { 3, 4, 9, 41 }
            );

            assertEquals(List.of(range(1, 1, vectorBytes), range(4, 1, vectorBytes), range(20, 1, vectorBytes)), slice.prefetches);
        }
    }

    public void testPrefetchByDocIds_whenCandidatesWithin32KB_thenCoalesced() throws IOException {
        final int dimension = 4;
        final int vectorBytes = dimension * Float.BYTES;
        try (
            ByteBuffersDirectory dir = new ByteBuffersDirectory();
            MockedStatic<KNNFeatureFlags> flags = mockStatic(KNNFeatureFlags.class)
        ) {
            flags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);
            final RecordingIndexInput slice = createSlice(dir, vectorBytes);

            new KNNVectorValuesIterator.DocIdsIteratorValues(new SparseSliceValues(slice, dimension)).prefetchByDocIds(
                new int[] { 3, 9, 41 }
            );

            assertEquals(List.of(range(1, 20, vectorBytes)), slice.prefetches);
        }
    }

    public void testPrefetchByDocIds_whenPrefetchDisabled_thenNoOp() throws IOException {
        final int dimension = 4;
        try (
            ByteBuffersDirectory dir = new ByteBuffersDirectory();
            MockedStatic<KNNFeatureFlags> flags = mockStatic(KNNFeatureFlags.class)
        ) {
            flags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(false);
            final RecordingIndexInput slice = createSlice(dir, dimension * Float.BYTES);

            new KNNVectorValuesIterator.DocIdsIteratorValues(new SparseSliceValues(slice, dimension)).prefetchByDocIds(
                new int[] { 3, 9, 41 }
            );

            assertTrue(slice.prefetches.isEmpty());
        }
    }

    public void testPrefetchByDocIds_whenNoSlice_thenNoOp() throws IOException {
        try (MockedStatic<KNNFeatureFlags> flags = mockStatic(KNNFeatureFlags.class)) {
            flags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);

            // Not a HasIndexSlice, and a HasIndexSlice whose slice is null: both must silently skip.
            new KNNVectorValuesIterator.DocIdsIteratorValues(FloatVectorValues.fromFloats(List.of(new float[4], new float[4]), 4))
                .prefetchByDocIds(new int[] { 0, 1 });
            new KNNVectorValuesIterator.DocIdsIteratorValues(new SparseSliceValues(null, 4)).prefetchByDocIds(new int[] { 3, 9 });
        }
    }

    private static List<Long> range(final int startOrd, final int numVectors, final int vectorBytes) {
        return List.of((long) startOrd * vectorBytes, (long) numVectors * vectorBytes);
    }

    private static RecordingIndexInput createSlice(final ByteBuffersDirectory dir, final int vectorBytes) throws IOException {
        try (IndexOutput out = dir.createOutput("vec", IOContext.DEFAULT)) {
            out.writeBytes(new byte[NUM_VECTORS * vectorBytes], NUM_VECTORS * vectorBytes);
        }
        return new RecordingIndexInput(dir.openInput("vec", IOContext.DEFAULT));
    }

    private static class RecordingIndexInput extends FilterIndexInput {
        final List<List<Long>> prefetches = new ArrayList<>();

        RecordingIndexInput(final IndexInput in) {
            super("recording", in);
        }

        @Override
        public void prefetch(final long offset, final long length) {
            prefetches.add(List.of(offset, length));
        }
    }

    private static class SparseSliceValues extends FloatVectorValues implements HasIndexSlice {
        private final IndexInput slice;
        private final int dimension;

        SparseSliceValues(final IndexInput slice, final int dimension) {
            this.slice = slice;
            this.dimension = dimension;
        }

        @Override
        public IndexInput getSlice() {
            return slice;
        }

        @Override
        public int ordToDoc(final int ord) {
            return 2 * ord + 1;
        }

        @Override
        public DocIndexIterator iterator() {
            return createSparseIterator();
        }

        @Override
        public float[] vectorValue(final int ord) {
            throw new UnsupportedOperationException();
        }

        @Override
        public FloatVectorValues copy() {
            return new SparseSliceValues(slice, dimension);
        }

        @Override
        public int dimension() {
            return dimension;
        }

        @Override
        public int size() {
            return NUM_VECTORS;
        }
    }
}
