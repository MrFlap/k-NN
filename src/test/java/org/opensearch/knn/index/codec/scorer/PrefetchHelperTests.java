/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.Constants;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.common.featureflags.KNNFeatureFlags;

import java.io.IOException;

import static org.mockito.Mockito.mockStatic;

public class PrefetchHelperTests extends KNNTestCase {

    public void testPrefetch_whenNullOrds_thenNoOp() throws IOException {
        TrackingIndexInput trackingInput = createTrackingInput(1024);
        PrefetchHelper.prefetch(trackingInput, 0L, 16L, null, 0);
        assertTrue(trackingInput.prefetchCalls.isEmpty());
    }

    public void testPrefetch_whenZeroOrds_thenNoOp() throws IOException {
        TrackingIndexInput trackingInput = createTrackingInput(1024);
        int[] ords = { 0, 1, 2 };
        PrefetchHelper.prefetch(trackingInput, 0L, 16L, ords, 0);
        assertTrue(trackingInput.prefetchCalls.isEmpty());
    }

    public void testPrefetch_whenSingleOrd_thenNoOp() throws IOException {
        TrackingIndexInput trackingInput = createTrackingInput(200 * 1024);
        int[] ords = { 5 };
        PrefetchHelper.prefetch(trackingInput, 100L, 16L, ords, 1);
        assertTrue(trackingInput.prefetchCalls.isEmpty());
    }

    public void testPrefetch_whenPrefetchDisabled_thenNoPrefetch() throws IOException {
        try (MockedStatic<KNNFeatureFlags> mockedFlags = mockStatic(KNNFeatureFlags.class)) {
            mockedFlags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(false);
            TrackingIndexInput trackingInput = createTrackingInput(200 * 1024);
            int[] ords = { 0, 1, 2 };
            long vectorSize = 1024L;
            PrefetchHelper.prefetch(trackingInput, 0L, vectorSize, ords, 3);
            assertEquals(0, trackingInput.prefetchCalls.size());
            assertTrue(trackingInput.prefetchCalls.isEmpty());
        }
    }

    public void testPrefetch_whenVectorsWithinSameGroup_thenSingleExactPrefetch() throws IOException {
        try (MockedStatic<KNNFeatureFlags> mockedFlags = mockStatic(KNNFeatureFlags.class)) {
            mockedFlags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);
            TrackingIndexInput trackingInput = createTrackingInput(200 * 1024);
            int[] ords = { 0, 1, 2 };
            long vectorSize = 1024L;
            PrefetchHelper.prefetch(trackingInput, 0L, vectorSize, ords, 3);
            assertEquals(1, trackingInput.prefetchCalls.size());
            assertEquals(0L, trackingInput.prefetchCalls.get(0).offset());
            assertEquals(3 * vectorSize, trackingInput.prefetchCalls.get(0).length());
        }
    }

    public void testPrefetch_whenVectorsSpanMultipleGroups_thenMultipleExactPrefetches() throws IOException {
        try (MockedStatic<KNNFeatureFlags> mockedFlags = mockStatic(KNNFeatureFlags.class)) {
            mockedFlags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);
            TrackingIndexInput trackingInput = createTrackingInput(500 * 1024);
            int[] ords = { 0, 10, 100 };
            long vectorSize = 1024L;
            PrefetchHelper.prefetch(trackingInput, 0L, vectorSize, ords, 3);
            assertEquals(2, trackingInput.prefetchCalls.size());
            assertEquals(0L, trackingInput.prefetchCalls.get(0).offset());
            assertEquals(11 * vectorSize, trackingInput.prefetchCalls.get(0).length());
            assertEquals(100 * vectorSize, trackingInput.prefetchCalls.get(1).offset());
            assertEquals(vectorSize, trackingInput.prefetchCalls.get(1).length());
        }
    }

    public void testPrefetch_whenNumOrdsLessThanArrayLength_thenOnlyPrefetchesNumOrds() throws IOException {
        try (MockedStatic<KNNFeatureFlags> mockedFlags = mockStatic(KNNFeatureFlags.class)) {
            mockedFlags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);

            TrackingIndexInput trackingInput = createTrackingInput(200 * 1024);
            int[] ords = { 0, 1, 2, 3, 4 };
            PrefetchHelper.prefetch(trackingInput, 0L, 8L, ords, 2);
            assertEquals(1, trackingInput.prefetchCalls.size());
            assertEquals(0L, trackingInput.prefetchCalls.get(0).offset());
            assertEquals(16L, trackingInput.prefetchCalls.get(0).length());
        }
    }

    public void testPrefetch_whenUnorderedOrds_thenSortsAndGroups() throws IOException {
        try (MockedStatic<KNNFeatureFlags> mockedFlags = mockStatic(KNNFeatureFlags.class)) {
            mockedFlags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);
            TrackingIndexInput trackingInput = createTrackingInput(500 * 1024);
            int[] ords = { 100, 0, 10 };
            long vectorSize = 1024L;
            PrefetchHelper.prefetch(trackingInput, 0L, vectorSize, ords, 3);
            assertEquals(2, trackingInput.prefetchCalls.size());
            assertEquals(0L, trackingInput.prefetchCalls.get(0).offset());
            assertEquals(11 * vectorSize, trackingInput.prefetchCalls.get(0).length());
            assertEquals(100 * vectorSize, trackingInput.prefetchCalls.get(1).offset());
            assertEquals(vectorSize, trackingInput.prefetchCalls.get(1).length());
        }
    }

    public void testPrefetch_whenBaseOffsetNonZero_thenOffsetsCalculatedCorrectly() throws IOException {
        try (MockedStatic<KNNFeatureFlags> mockedFlags = mockStatic(KNNFeatureFlags.class)) {
            mockedFlags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);

            TrackingIndexInput trackingInput = createTrackingInput(200 * 1024);
            long baseOffset = 10000L;
            int[] ords = { 0, 1 };
            long vectorSize = 1024L;
            PrefetchHelper.prefetch(trackingInput, baseOffset, vectorSize, ords, 2);
            assertEquals(1, trackingInput.prefetchCalls.size());
            assertEquals(baseOffset, trackingInput.prefetchCalls.get(0).offset());
            assertEquals(2 * vectorSize, trackingInput.prefetchCalls.get(0).length());
        }
    }

    public void testPrefetch_whenVectorsExactly32KBApart_thenCreatesNewGroup() throws IOException {
        try (MockedStatic<KNNFeatureFlags> mockedFlags = mockStatic(KNNFeatureFlags.class)) {
            mockedFlags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);
            TrackingIndexInput trackingInput = createTrackingInput(300 * 1024);
            int[] ords = { 0, 32 };
            long vectorSize = 1024L;
            PrefetchHelper.prefetch(trackingInput, 0L, vectorSize, ords, 2);
            assertEquals(2, trackingInput.prefetchCalls.size());
            assertEquals(0L, trackingInput.prefetchCalls.get(0).offset());
            assertEquals(vectorSize, trackingInput.prefetchCalls.get(0).length());
            assertEquals(32 * vectorSize, trackingInput.prefetchCalls.get(1).offset());
            assertEquals(vectorSize, trackingInput.prefetchCalls.get(1).length());
        }
    }

    public void testPrefetch_whenVectorsJustUnder32KBApart_thenSameGroup() throws IOException {
        try (MockedStatic<KNNFeatureFlags> mockedFlags = mockStatic(KNNFeatureFlags.class)) {
            mockedFlags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);
            TrackingIndexInput trackingInput = createTrackingInput(300 * 1024);
            int[] ords = { 0, 31 };
            long vectorSize = 1024L;
            PrefetchHelper.prefetch(trackingInput, 0L, vectorSize, ords, 2);
            assertEquals(1, trackingInput.prefetchCalls.size());
            assertEquals(0L, trackingInput.prefetchCalls.get(0).offset());
            assertEquals(32 * vectorSize, trackingInput.prefetchCalls.get(0).length());
        }
    }

    public void testPrefetch_whenMultipleGroupsAndFileBoundary_thenRespectsFileLength() throws IOException {
        try (MockedStatic<KNNFeatureFlags> mockedFlags = mockStatic(KNNFeatureFlags.class)) {
            mockedFlags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);
            TrackingIndexInput trackingInput = createTrackingInput(350 * 1024);
            int[] ords = { 0, 10, 340 };
            long vectorSize = 1024L;
            PrefetchHelper.prefetch(trackingInput, 0L, vectorSize, ords, 3);
            assertEquals(2, trackingInput.prefetchCalls.size());
            assertEquals(0L, trackingInput.prefetchCalls.get(0).offset());
            assertEquals(11 * vectorSize, trackingInput.prefetchCalls.get(0).length());
            assertEquals(340 * vectorSize, trackingInput.prefetchCalls.get(1).offset());
            assertEquals(vectorSize, trackingInput.prefetchCalls.get(1).length());
        }
    }

    public void testPrefetch_whenFinalVectorEndsExactlyAtFileLength_thenPrefetchesExactly() throws IOException {
        try (MockedStatic<KNNFeatureFlags> mockedFlags = mockStatic(KNNFeatureFlags.class)) {
            mockedFlags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);
            TrackingIndexInput trackingInput = createTrackingInput(32 * 1024);
            int[] ords = { 0, 31 };
            long vectorSize = 1024L;
            PrefetchHelper.prefetch(trackingInput, 0L, vectorSize, ords, 2);
            assertEquals(1, trackingInput.prefetchCalls.size());
            assertEquals(0L, trackingInput.prefetchCalls.get(0).offset());
            assertEquals(32 * vectorSize, trackingInput.prefetchCalls.get(0).length());
        }
    }

    public void testPrefetch_whenGroupLengthExactly32KB_thenPrefetches32KB() throws IOException {
        try (MockedStatic<KNNFeatureFlags> mockedFlags = mockStatic(KNNFeatureFlags.class)) {
            mockedFlags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);
            TrackingIndexInput trackingInput = createTrackingInput(300 * 1024);
            int[] ords = { 0, 31 };
            long vectorSize = 1024L;
            PrefetchHelper.prefetch(trackingInput, 0L, vectorSize, ords, 2);
            assertEquals(1, trackingInput.prefetchCalls.size());
            assertEquals(0L, trackingInput.prefetchCalls.get(0).offset());
            assertEquals(32 * 1024L, trackingInput.prefetchCalls.get(0).length());
        }
    }

    public void testPrefetch_whenMultipleVectorsAt32KBIntervals_thenMultipleGroups() throws IOException {
        try (MockedStatic<KNNFeatureFlags> mockedFlags = mockStatic(KNNFeatureFlags.class)) {
            mockedFlags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);
            TrackingIndexInput trackingInput = createTrackingInput(600 * 1024);
            int[] ords = { 0, 32, 64 };
            long vectorSize = 1024L;
            PrefetchHelper.prefetch(trackingInput, 0L, vectorSize, ords, 3);
            assertEquals(3, trackingInput.prefetchCalls.size());
            assertEquals(0L, trackingInput.prefetchCalls.get(0).offset());
            assertEquals(vectorSize, trackingInput.prefetchCalls.get(0).length());
            assertEquals(32 * vectorSize, trackingInput.prefetchCalls.get(1).offset());
            assertEquals(vectorSize, trackingInput.prefetchCalls.get(1).length());
            assertEquals(64 * vectorSize, trackingInput.prefetchCalls.get(2).offset());
            assertEquals(vectorSize, trackingInput.prefetchCalls.get(2).length());
        }
    }

    public void testPrefetch_whenDuplicateOrds_thenSortsAndDeduplicatesInPrefetch() throws IOException {
        try (MockedStatic<KNNFeatureFlags> mockedFlags = mockStatic(KNNFeatureFlags.class)) {
            mockedFlags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);

            TrackingIndexInput trackingInput = createTrackingInput(200 * 1024);
            int[] ords = { 5, 2, 5, 2, 10 };
            long vectorSize = 1024L;
            PrefetchHelper.prefetch(trackingInput, 0L, vectorSize, ords, 5);
            assertEquals(1, trackingInput.prefetchCalls.size());
            assertEquals(2 * vectorSize, trackingInput.prefetchCalls.get(0).offset());
            assertEquals(9 * vectorSize, trackingInput.prefetchCalls.get(0).length());
        }
    }

    public void testPrefetch_whenThreeOrMoreGroups_thenCreatesAllGroups() throws IOException {
        try (MockedStatic<KNNFeatureFlags> mockedFlags = mockStatic(KNNFeatureFlags.class)) {
            mockedFlags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);
            TrackingIndexInput trackingInput = createTrackingInput(800 * 1024);
            int[] ords = { 0, 20, 50, 100, 150 };
            long vectorSize = 1024L;
            PrefetchHelper.prefetch(trackingInput, 0L, vectorSize, ords, 5);
            assertEquals(4, trackingInput.prefetchCalls.size());
            assertEquals(0L, trackingInput.prefetchCalls.get(0).offset());
            assertEquals(21 * vectorSize, trackingInput.prefetchCalls.get(0).length());
            assertEquals(50 * vectorSize, trackingInput.prefetchCalls.get(1).offset());
            assertEquals(vectorSize, trackingInput.prefetchCalls.get(1).length());
            assertEquals(100 * vectorSize, trackingInput.prefetchCalls.get(2).offset());
            assertEquals(vectorSize, trackingInput.prefetchCalls.get(2).length());
            assertEquals(150 * vectorSize, trackingInput.prefetchCalls.get(3).offset());
            assertEquals(vectorSize, trackingInput.prefetchCalls.get(3).length());
        }
    }

    public void testPrefetch_whenVectorEnd32KBPlusBytes_thenCreatesNewGroup() throws IOException {
        try (MockedStatic<KNNFeatureFlags> mockedFlags = mockStatic(KNNFeatureFlags.class)) {
            mockedFlags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);
            TrackingIndexInput trackingInput = createTrackingInput(300 * 1024);
            int[] ords = { 0, 31 };
            long vectorSize = 1025L;
            PrefetchHelper.prefetch(trackingInput, 0L, vectorSize, ords, 2);
            assertEquals(2, trackingInput.prefetchCalls.size());
            assertEquals(0L, trackingInput.prefetchCalls.get(0).offset());
            assertEquals(vectorSize, trackingInput.prefetchCalls.get(0).length());
            assertEquals(31 * vectorSize, trackingInput.prefetchCalls.get(1).offset());
            assertEquals(vectorSize, trackingInput.prefetchCalls.get(1).length());
        }
    }

    private TrackingIndexInput createTrackingInput(int sizeInBytes) throws IOException {
        ByteBuffersDirectory dir = new ByteBuffersDirectory();
        try (IndexOutput out = dir.createOutput("test", IOContext.DEFAULT)) {
            out.writeBytes(new byte[sizeInBytes], sizeInBytes);
        }
        IndexInput delegate = dir.openInput("test", IOContext.DEFAULT);
        return new TrackingIndexInput(delegate);
    }

    private static final int PAGE = 4096;

    public void testPrefetchRange_whenMMapInput_thenDirectMadvise() throws IOException {
        try (MMapDirectory dir = new MMapDirectory(createTempDir()); IndexInput in = openPattern(dir, 8 * PAGE)) {
            assertEquals(Constants.LINUX, PrefetchHelper.tryMadvise(in, 0, PAGE));
            try (IndexInput slice = in.slice("slice", PAGE, 4 * PAGE)) {
                assertEquals(Constants.LINUX, PrefetchHelper.tryMadvise(slice, 0, PAGE));
            }
            assertFalse(PrefetchHelper.tryMadvise(in, 0, 0));
        }
    }

    public void testPrefetchRange_whenDirectMadviseDisabled_thenUsesLucenePrefetch() throws IOException {
        try (MMapDirectory dir = new MMapDirectory(createTempDir()); IndexInput in = openPattern(dir, 8 * PAGE)) {
            assertEquals(Constants.LINUX, PrefetchHelper.directMadvise(in, 0, PAGE));
            try (MockedStatic<KNNSettings> settings = mockStatic(KNNSettings.class, Mockito.CALLS_REAL_METHODS)) {
                settings.when(KNNSettings::isDirectMadviseEnabled).thenReturn(false);
                assertFalse(PrefetchHelper.directMadvise(in, 0, PAGE));
                PrefetchHelper.prefetchRange(in, 0, 2 * PAGE); // falls back to IndexInput#prefetch
                assertPattern(in, 8 * PAGE);
            }
        }
    }

    public void testPrefetchRange_whenByteBuffersInput_thenFallsBackToLucenePrefetch() throws IOException {
        try (ByteBuffersDirectory dir = new ByteBuffersDirectory(); IndexInput in = openPattern(dir, 4 * PAGE)) {
            assertFalse(PrefetchHelper.tryMadvise(in, 0, PAGE));
            PrefetchHelper.prefetchRange(in, 10, 2 * PAGE); // Lucene's default prefetch: no-op
            assertPattern(in, 4 * PAGE);
        }
    }

    public void testPrefetchRange_whenUnalignedRangesCrossPages_thenAdvisesAndDataIntact() throws IOException {
        try (MMapDirectory dir = new MMapDirectory(createTempDir()); IndexInput in = openPattern(dir, 8 * PAGE + 123)) {
            long[][] ranges = { { PAGE - 1, 2 }, { 1, 3L * PAGE + 5 }, { 7L * PAGE + 100, PAGE + 23 }, { 8L * PAGE + 122, 1 } };
            for (long[] r : ranges) {
                assertEquals(Constants.LINUX, PrefetchHelper.tryMadvise(in, r[0], r[1]));
                PrefetchHelper.prefetchRange(in, r[0], r[1]);
            }
            assertPattern(in, 8 * PAGE + 123);
        }
    }

    public void testPrefetchRange_whenRangeSpansMMapChunks_thenFallsBack() throws IOException {
        // 4KB chunks: a range crossing a chunk boundary has no single segment, so Lucene's prefetch is used
        try (MMapDirectory dir = new MMapDirectory(createTempDir(), PAGE); IndexInput in = openPattern(dir, 4 * PAGE)) {
            assertFalse(PrefetchHelper.tryMadvise(in, PAGE - 10, 20));
            assertEquals(Constants.LINUX, PrefetchHelper.tryMadvise(in, PAGE + 10, 20));
            PrefetchHelper.prefetchRange(in, PAGE - 10, 20);
            assertPattern(in, 4 * PAGE);
        }
    }

    private static IndexInput openPattern(org.apache.lucene.store.Directory dir, int size) throws IOException {
        try (IndexOutput out = dir.createOutput("pattern", IOContext.DEFAULT)) {
            for (int i = 0; i < size; i++) {
                out.writeByte((byte) i);
            }
        }
        return dir.openInput("pattern", IOContext.DEFAULT);
    }

    private static void assertPattern(IndexInput in, int size) throws IOException {
        in.seek(0);
        for (int i = 0; i < size; i++) {
            assertEquals((byte) i, in.readByte());
        }
    }
}
