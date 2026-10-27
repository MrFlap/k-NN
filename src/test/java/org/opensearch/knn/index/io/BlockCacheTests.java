/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.io;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import static org.opensearch.knn.index.io.NativeIo.PAGE;

@ThreadLeakFilters(defaultFilters = true, filters = { IoUringReaperFilter.class })
public class BlockCacheTests extends KNNTestCase {

    private static IoUring ring;

    private static synchronized IoUring ring() {
        if (ring == null) {
            try {
                ring = IoUring.create(64);
            } catch (IOException e) {
                assumeTrue("io_uring unavailable here: " + e, false);
            }
        }
        return ring;
    }

    private static byte at(final long position) {
        return (byte) (position * 31 + (position >>> 12));
    }

    private FSDirectory directoryWithFile(final String name, final int length) throws IOException {
        final FSDirectory dir = FSDirectory.open(createTempDir());
        try (IndexOutput out = dir.createOutput(name, IOContext.DEFAULT)) {
            for (int i = 0; i < length; i++) {
                out.writeByte(at(i));
            }
        }
        return dir;
    }

    private static void assertBytes(final byte[] actual, final long position, final int length) {
        for (int i = 0; i < length; i++) {
            if (actual[i] != at(position + i)) {
                fail("byte " + (position + i) + " differs");
            }
        }
    }

    public void testRead_whenMissThenHit_thenBytesMatch() throws IOException {
        try (FSDirectory dir = directoryWithFile("f", 10 * PAGE + 123); DirectFile file = DirectFile.open(dir, "f")) {
            assertNotNull(file);
            final BlockCache cache = new BlockCache(64L * PAGE, ring());
            final byte[] buffer = new byte[3 * PAGE];
            // unaligned, crossing pages, and the partial last page
            final long[][] ranges = { { 0, 1 }, { PAGE - 3, 7 }, { 5, 2 * PAGE + 9 }, { 10L * PAGE, 123 }, { 10L * PAGE + 120, 3 } };
            for (final long[] r : ranges) {
                cache.read(file, r[0], buffer, 0, (int) r[1]);
                assertBytes(buffer, r[0], (int) r[1]);
            }
            final long misses = cache.misses();
            assertTrue(misses > 0);
            cache.read(file, PAGE - 3, buffer, 0, 7);
            assertBytes(buffer, PAGE - 3, 7);
            assertEquals(misses, cache.misses());
            assertTrue(cache.hits() > 0);
            assertEquals(4L * PAGE, cache.usedBytes()); // pages 0, 1, 2 and 10
        }
    }

    public void testPrefetch_thenReadWaitsForInFlightPagesWithoutReadingAgain() throws IOException {
        try (FSDirectory dir = directoryWithFile("f", 200 * PAGE); DirectFile file = DirectFile.open(dir, "f")) {
            final BlockCache cache = new BlockCache(256L * PAGE, ring());
            cache.prefetch(file, 7, 100L * PAGE); // unaligned: pages 0..100, two IOs
            final byte[] buffer = new byte[100 * PAGE];
            cache.read(file, 7, buffer, 0, buffer.length);
            assertBytes(buffer, 7, buffer.length);
            assertEquals(0, cache.misses());
            cache.prefetch(file, 0, 10L * PAGE); // all cached: nothing to read
            assertEquals(101L * PAGE, cache.usedBytes());
        }
    }

    public void testPrefetch_whenRangePartlyCached_thenOneIoForTheWholeRangeAndGapPagesInserted() throws IOException {
        try (FSDirectory dir = directoryWithFile("f", 100 * PAGE); DirectFile file = DirectFile.open(dir, "f")) {
            final BlockCache cache = new BlockCache(256L * PAGE, ring());
            final byte[] buffer = new byte[16 * PAGE];
            cache.read(file, 1L * PAGE, buffer, 0, 1);
            cache.read(file, 3L * PAGE, buffer, 0, 1);
            final long ioBytes = cache.ioBytes();
            cache.prefetch(file, 0, 16L * PAGE); // pages 1 and 3 cached: still one IO of all 16 pages
            assertEquals(1, cache.prefetchIos());
            assertEquals(16L * PAGE, cache.ioBytes() - ioBytes);
            final long misses = cache.misses();
            cache.read(file, 0, buffer, 0, buffer.length);
            assertBytes(buffer, 0, buffer.length);
            assertEquals("every page of the range was inserted", misses, cache.misses());
            assertEquals(16L * PAGE, cache.usedBytes());
            cache.prefetch(file, 0, 16L * PAGE); // all cached: no IO
            assertEquals(1, cache.prefetchIos());
        }
    }

    public void testRead_whenFileLargerThanCache_thenEvictsAndStaysCorrect() throws IOException {
        try (FSDirectory dir = directoryWithFile("f", 300 * PAGE); DirectFile file = DirectFile.open(dir, "f")) {
            final BlockCache cache = new BlockCache(64L * PAGE, ring());
            final byte[] buffer = new byte[PAGE];
            for (int round = 0; round < 2; round++) {
                for (long page = 0; page < 300; page++) {
                    cache.read(file, page * PAGE, buffer, 0, PAGE);
                    assertBytes(buffer, page * PAGE, PAGE);
                }
            }
            assertTrue(cache.evictions() > 0);
            assertEquals(64L * PAGE, cache.usedBytes());
        }
    }

    public void testRead_whenConcurrentWithSmallCache_thenBytesMatch() throws Exception {
        final int length = 500 * PAGE;
        try (FSDirectory dir = directoryWithFile("f", length); DirectFile file = DirectFile.open(dir, "f")) {
            final BlockCache cache = new BlockCache(64L * PAGE, ring());
            final ExecutorService pool = Executors.newFixedThreadPool(8);
            try {
                final List<Future<?>> futures = new ArrayList<>();
                for (int t = 0; t < 8; t++) {
                    final Random random = new Random(t);
                    futures.add(pool.submit(() -> {
                        final byte[] buffer = new byte[3 * PAGE];
                        for (int i = 0; i < 1000; i++) {
                            final long position = random.nextInt(length - buffer.length);
                            final int n = 1 + random.nextInt(buffer.length);
                            if (random.nextInt(4) == 0) {
                                cache.prefetch(file, position, 16L * PAGE);
                            }
                            cache.read(file, position, buffer, 0, n);
                            assertBytes(buffer, position, n);
                        }
                        return null;
                    }));
                }
                for (Future<?> future : futures) {
                    future.get();
                }
            } finally {
                pool.shutdownNow();
            }
        }
    }

    public void testCachedIndexInput_thenReadsMatchMMapInput() throws IOException {
        ring();
        final int length = 20 * PAGE + 5;
        try (
            MockedStatic<KNNSettings> settings = Mockito.mockStatic(KNNSettings.class, Mockito.CALLS_REAL_METHODS);
            FSDirectory dir = directoryWithFile("f", length);
            IndexInput mmap = dir.openInput("f", IOContext.DEFAULT);
            IndexInput cached = KnnIo.openCached(dir, "f")
        ) {
            settings.when(KNNSettings::getBlockCachePercent).thenReturn(25.0);
            assertNotNull(cached);
            assertEquals(length, cached.length());
            final IndexInput slice = cached.slice("s", PAGE - 2, 10L * PAGE);
            final IndexInput expected = mmap.slice("s", PAGE - 2, 10L * PAGE);
            slice.seek(3);
            expected.seek(3);
            assertEquals(expected.readInt(), slice.readInt());
            assertEquals(expected.readLong(), slice.readLong());
            assertEquals(expected.readShort(), slice.readShort());
            assertEquals(expected.readByte(), slice.readByte());
            final float[] floats = new float[1000];
            final float[] expectedFloats = new float[1000];
            slice.readFloats(floats, 0, 1000);
            expected.readFloats(expectedFloats, 0, 1000);
            assertArrayEquals(expectedFloats, floats, 0f);
            final IndexInput clone = slice.clone();
            assertEquals(slice.getFilePointer(), clone.getFilePointer());
            final int[] ints = new int[100];
            final int[] expectedInts = new int[100];
            clone.readInts(ints, 0, 100);
            expected.readInts(expectedInts, 0, 100);
            assertArrayEquals(expectedInts, ints);
            slice.prefetch(0, 20L * PAGE); // clipped to the slice
            expectThrows(IOException.class, () -> slice.seek(10L * PAGE + 1));
            cached.seek(length - 1);
            assertEquals(at(length - 1), cached.readByte());
            expectThrows(IOException.class, cached::readByte);
        }
    }

    public void testCapacity_thenPercentOfVectorDataClampedToHalfOfRamOutsideHeap() {
        final long gb = 1L << 30;
        assertEquals((long) (2.5 * gb), KnnIo.capacity(25.0, 10 * gb, 64 * gb, 16 * gb));
        assertEquals(24 * gb, KnnIo.capacity(25.0, 100 * gb, 64 * gb, 16 * gb)); // 25 GB asked, (64 - 16) / 2 allowed
        assertEquals(16L << 20, KnnIo.capacity(25.0, 1000, 64 * gb, 16 * gb)); // floor
        assertEquals(10 * gb, KnnIo.capacity(100.0, 10 * gb, 64 * gb, 16 * gb));
    }

    public void testBlockCache_whenPercentChanges_thenReallocated() {
        ring();
        final long vectorData = 4L << 30;
        KnnIo.addVectorData(vectorData);
        try (MockedStatic<KNNSettings> settings = Mockito.mockStatic(KNNSettings.class, Mockito.CALLS_REAL_METHODS)) {
            settings.when(KNNSettings::getBlockCachePercent).thenReturn(1.0);
            final BlockCache first = KnnIo.blockCache();
            assertNotNull(first);
            assertSame(first, KnnIo.blockCache());
            settings.when(KNNSettings::getBlockCachePercent).thenReturn(2.0);
            final BlockCache second = KnnIo.blockCache();
            assertNotSame(first, second);
            assertTrue(second.capacityBytes > first.capacityBytes);
            assertEquals(second.capacityBytes, KnnIo.stats().get("capacity_bytes"));
        } finally {
            KnnIo.addVectorData(-vectorData);
        }
    }

    public void testUseIoUring_whenUnavailable_thenFallsBackToMMap() throws IOException {
        try (MockedStatic<KNNSettings> settings = Mockito.mockStatic(KNNSettings.class, Mockito.CALLS_REAL_METHODS)) {
            settings.when(KNNSettings::getIoBackend).thenReturn(KnnIo.IO_URING);
            KnnIo.setAvailableForTesting(false);
            assertFalse(KnnIo.useIoUring());
            try (FSDirectory dir = directoryWithFile("f", PAGE); IndexInput mmap = dir.openInput("f", IOContext.DEFAULT)) {
                final KnnIo.LazyCachedInput lazy = new KnnIo.LazyCachedInput(dir, "f");
                final IndexInput selected = lazy.select(mmap, 0, PAGE);
                assertFalse(selected instanceof CachedIndexInput);
                lazy.close();
            }
            settings.when(KNNSettings::getIoBackend).thenReturn(KnnIo.MMAP);
            KnnIo.setAvailableForTesting(null);
            assertFalse(KnnIo.useIoUring());
        } finally {
            KnnIo.setAvailableForTesting(null);
        }
    }
}
