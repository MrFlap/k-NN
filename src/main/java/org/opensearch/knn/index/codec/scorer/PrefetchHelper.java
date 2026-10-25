/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MemorySegmentAccessInput;
import org.apache.lucene.util.Constants;
import org.opensearch.knn.common.featureflags.KNNFeatureFlags;

import java.io.IOException;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Optional;

/**
 * Helper class to prefetch vector data from disk to improve query performance.
 * <p>
 * Prefetches vector data using exact byte range strategy. Vectors are grouped together when
 * the end of a vector (start offset + vector size) falls within 128KB from the group's start offset.
 * This minimizes I/O operations while fetching only the exact bytes needed.
 * <p>
 * Prefetching is controlled by {@link KNNFeatureFlags#isPrefetchEnabled()}. When disabled,
 * no prefetch operations are performed.
 * <p>
 * <b>Example:</b><br>
 * Given vectors at offsets [100KB, 120KB, 300KB] with 10KB vector size:
 * <ul>
 *   <li>Group 1: Prefetch 40KB (100KB to 130KB, covers vectors at 100KB and 120KB)</li>
 *   <li>Group 2: Prefetch 10KB (300KB to 310KB, covers vector at 300KB)</li>
 * </ul>
 * Result: 2 prefetch operations, 50KB total prefetched (vs 6 random seeks without prefetch)
 */
@Log4j2
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public class PrefetchHelper {

    // TODO: If needed we can get this value via Cluster Settings
    private static final long BYTES_32 = 32 * 1024;

    /**
     * Prefetches vector data from disk using exact byte range strategy.
     * <p>
     * Vectors are sorted by offset and grouped when the end of a vector falls within 128KB
     * from the group start. Each group is prefetched with exact byte range needed.
     * Returns early if prefetch is disabled, ordinals are null, or fewer than 2 vectors.
     *
     * @param indexInput the index input to prefetch from
     * @param baseOffset the base offset in the file where vectors start
     * @param oneVectorByteSize the size of one vector in bytes
     * @param ordsToPrefetch array of vector ordinals to prefetch
     * @param numOrds number of valid ordinals in the array
     * @throws IOException if an I/O error occurs during prefetch
     */
    public static void prefetch(
        final IndexInput indexInput,
        final long baseOffset,
        final long oneVectorByteSize,
        final int[] ordsToPrefetch,
        final int numOrds
    ) throws IOException {
        if (ordsToPrefetch == null || numOrds <= 1) {
            return;
        }
        // Instrumentation: account the distinct physical pages this batch touches, for the per-query
        // distinct-page metric. No-op unless -Dknn.pageTouch.enabled=true; recorded independently of the
        // prefetch feature flag so the metric reflects reads whether or not prefetch I/O is issued.
        PageTouchTracker.get().recordBatch(baseOffset, oneVectorByteSize, ordsToPrefetch, numOrds);
        if (KNNFeatureFlags.isPrefetchEnabled()) {
            prefetchExactVectorSize(indexInput, baseOffset, oneVectorByteSize, ordsToPrefetch, numOrds);
        } else {
            log.debug("KNNVectors Prefetch is disabled");
        }
    }

    /**
     * Prefetches records for the locality-reordered store, coalescing ordinals that fall on the same
     * absolute 32KB block into a single prefetch.
     * <p>
     * Unlike {@link #prefetch} (whose grouping window floats — it is anchored at the first ordinal of
     * the batch), the blocks here are fixed to multiples of {@link #BYTES_32} measured from the store
     * start (vector 0). Two records that share a physical 32KB page therefore always coalesce into one
     * {@code prefetch}, which matches exactly how {@link PageTouchTracker} counts touched pages
     * ({@code offset / 32KB}). This suits the reordered layout, where reordering co-locates a node's
     * neighbors onto a few pages, so most of a batch collapses to a handful of block-aligned prefetches.
     * <p>
     * Returns early if prefetch is disabled, ordinals are null, or fewer than 2 vectors.
     *
     * @param indexInput the index input to prefetch from
     * @param baseOffset the base offset in the file where records start (0 for a record slice)
     * @param oneVectorByteSize the size of one record in bytes
     * @param ordsToPrefetch physical ordinals to prefetch (already translated for the reordered store)
     * @param numOrds number of valid ordinals in the array
     * @throws IOException if an I/O error occurs during prefetch
     */
    public static void prefetchBasedOnOrdinals(
        final IndexInput indexInput,
        final long baseOffset,
        final long oneVectorByteSize,
        final int[] ordsToPrefetch,
        final int numOrds
    ) throws IOException {
        if (ordsToPrefetch == null || numOrds <= 1) {
            return;
        }
        Arrays.sort(ordsToPrefetch, 0, numOrds);

        // Instrumentation: account the distinct physical pages this batch touches, for the per-query
        // distinct-page metric. No-op unless -Dknn.pageTouch.enabled=true; recorded independently of the
        // prefetch feature flag so the metric reflects reads whether or not prefetch I/O is issued.
        PageTouchTracker.get().recordBatch(baseOffset, oneVectorByteSize, ordsToPrefetch, numOrds);
        if (!KNNFeatureFlags.isPrefetchEnabled()) {
            log.debug("KNNVectors Prefetch is disabled");
            return;
        }

        int groupCount = 1;
        long groupStartOffset = baseOffset + (long) ordsToPrefetch[0] * oneVectorByteSize;
        // Block boundaries are anchored at offset 0 (vector 0), so a group flushes whenever the next
        // record starts in a different absolute 32KB block.
        long groupBlock = groupStartOffset / BYTES_32;
        for (int i = 1; i < numOrds; i++) {
            long currentOffset = baseOffset + (long) ordsToPrefetch[i] * oneVectorByteSize;
            long currentBlock = currentOffset / BYTES_32;
            if (currentBlock != groupBlock) {
                long prevEnd = baseOffset + (long) ordsToPrefetch[i - 1] * oneVectorByteSize + oneVectorByteSize;
                prefetchRange(indexInput, groupStartOffset, prevEnd - groupStartOffset);
                groupCount++;
                groupStartOffset = currentOffset;
                groupBlock = currentBlock;
            }
        }
        // Prefetch final group
        long lastEnd = baseOffset + (long) ordsToPrefetch[numOrds - 1] * oneVectorByteSize + oneVectorByteSize;
        prefetchRange(indexInput, groupStartOffset, lastEnd - groupStartOffset);

        log.trace("Prefetching [{}] block-aligned groups from [{}] ordinals, record size: {}", groupCount, numOrds, oneVectorByteSize);
    }

    /**
     * Prefetches vectors using exact byte ranges.
     * <p>
     * Groups vectors within 128KB ranges and prefetches only the exact bytes needed for each group.
     *
     * @param indexInput the index input to prefetch from
     * @param baseOffset the base offset in the file where vectors start
     * @param oneVectorByteSize the size of one vector in bytes
     * @param ordsToPrefetch array of vector ordinals to prefetch
     * @param numOrds number of valid ordinals in the array
     * @throws IOException if an I/O error occurs during prefetch
     */
    private static void prefetchExactVectorSize(
        final IndexInput indexInput,
        final long baseOffset,
        final long oneVectorByteSize,
        final int[] ordsToPrefetch,
        final int numOrds
    ) throws IOException {
        Arrays.sort(ordsToPrefetch, 0, numOrds);

        int groupCount = 1;
        long groupStartOffset = baseOffset + (long) ordsToPrefetch[0] * oneVectorByteSize;

        for (int i = 1; i < numOrds; i++) {
            long currentOffset = baseOffset + (long) ordsToPrefetch[i] * oneVectorByteSize;
            if ((currentOffset + oneVectorByteSize) - groupStartOffset > BYTES_32) {
                long prevOffset = baseOffset + (long) ordsToPrefetch[i - 1] * oneVectorByteSize;
                prefetchRange(indexInput, groupStartOffset, (prevOffset + oneVectorByteSize) - groupStartOffset);
                groupCount++;
                groupStartOffset = currentOffset;
            }
        }
        // Prefetch final group
        long lastOffset = baseOffset + (long) ordsToPrefetch[numOrds - 1] * oneVectorByteSize;
        prefetchRange(indexInput, groupStartOffset, (lastOffset + oneVectorByteSize) - groupStartOffset);

        log.trace(
            "Prefetching grouped [{}] vectors where num of ords was [{}] using exact prefetch size, vector size: {}",
            groupCount,
            numOrds,
            oneVectorByteSize
        );
    }

    /**
     * Asks the OS to start reading {@code [offset, offset + length)} of {@code indexInput}. Callers gate this on
     * {@link KNNFeatureFlags#isPrefetchEnabled()}.
     * <p>
     * For mmap'd inputs this calls {@code madvise(MADV_WILLNEED)} directly. Lucene's
     * {@link IndexInput#prefetch} skips the call unless a per-file counter (shared across clones) is zero or a power
     * of two, and resets it only when a range is not resident, so on a partly cached file most prefetches are
     * silently dropped and the reads that follow go serial.
     */
    public static void prefetchRange(final IndexInput indexInput, final long offset, final long length) throws IOException {
        if (tryMadvise(indexInput, offset, length) == false) {
            indexInput.prefetch(offset, length);
        }
    }

    /** Returns true if madvise was issued, false if the caller should fall back to {@link IndexInput#prefetch}. */
    static boolean tryMadvise(final IndexInput indexInput, final long offset, final long length) throws IOException {
        if (DirectMadvise.MADVISE == null || length <= 0 || (indexInput instanceof MemorySegmentAccessInput) == false) {
            return false;
        }
        try {
            // null when the range spans two mmap chunks
            final Object segment = (Object) DirectMadvise.SLICE.invokeExact((MemorySegmentAccessInput) indexInput, offset, length);
            if (segment == null) {
                return false;
            }
            final long address = (long) DirectMadvise.ADDRESS.invokeExact(segment);
            // mmap chunks are page aligned, so rounding the start down stays inside the mapping; the kernel rounds len up.
            final long alignedAddress = address & DirectMadvise.PAGE_MASK;
            // Non-zero return ignored: prefetch is only a hint.
            final int ignored = (int) DirectMadvise.MADVISE.invokeExact(alignedAddress, address + length - alignedAddress, MADV_WILLNEED);
            return true;
        } catch (IOException | RuntimeException | Error e) {
            throw e;
        } catch (Throwable t) {
            throw new IOException(t);
        }
    }

    private static final int MADV_WILLNEED = 3; // Linux value

    /**
     * Binds {@code madvise} through the FFM {@code Linker} once. Uses reflection and method handles because the
     * plugin compiles for Java 21, where {@code java.lang.foreign} is still a preview API. The pointer argument is
     * passed as a 64-bit integer, which matches the LP64 calling convention on Linux x86_64 and aarch64.
     */
    private static final class DirectMadvise {
        static final MethodHandle SLICE; // (MemorySegmentAccessInput, long, long) -> MemorySegment as Object
        static final MethodHandle ADDRESS; // (MemorySegment as Object) -> long
        static final MethodHandle MADVISE; // (long addr, long len, int advice) -> int
        static final long PAGE_MASK;

        static {
            MethodHandle slice = null, address = null, madvise = null;
            long pageMask = 0;
            if (Constants.LINUX && Constants.JRE_IS_64BIT) {
                try {
                    final Class<?> segmentClass = Class.forName("java.lang.foreign.MemorySegment");
                    final Class<?> layoutClass = Class.forName("java.lang.foreign.MemoryLayout");
                    final Class<?> descriptorClass = Class.forName("java.lang.foreign.FunctionDescriptor");
                    final Class<?> linkerClass = Class.forName("java.lang.foreign.Linker");
                    final Class<?> optionClass = Class.forName("java.lang.foreign.Linker$Option");
                    final Class<?> valueLayoutClass = Class.forName("java.lang.foreign.ValueLayout");
                    final Object javaInt = valueLayoutClass.getField("JAVA_INT").get(null);
                    final Object javaLong = valueLayoutClass.getField("JAVA_LONG").get(null);
                    final Object linker = linkerClass.getMethod("nativeLinker").invoke(null);
                    final Object libc = linkerClass.getMethod("defaultLookup").invoke(linker);
                    final java.lang.reflect.Method find = Class.forName("java.lang.foreign.SymbolLookup").getMethod("find", String.class);
                    final java.lang.reflect.Method descriptorOf = descriptorClass.getMethod("of", layoutClass, layoutClass.arrayType());
                    final java.lang.reflect.Method downcall = linkerClass.getMethod(
                        "downcallHandle",
                        segmentClass,
                        descriptorClass,
                        optionClass.arrayType()
                    );
                    final Object noOptions = Array.newInstance(optionClass, 0);

                    final Object madviseArgs = Array.newInstance(layoutClass, 3);
                    Array.set(madviseArgs, 0, javaLong);
                    Array.set(madviseArgs, 1, javaLong);
                    Array.set(madviseArgs, 2, javaInt);
                    madvise = (MethodHandle) downcall.invoke(
                        linker,
                        ((Optional<?>) find.invoke(libc, "madvise")).orElseThrow(),
                        descriptorOf.invoke(null, javaInt, madviseArgs),
                        noOptions
                    );
                    final MethodHandle getPageSize = (MethodHandle) downcall.invoke(
                        linker,
                        ((Optional<?>) find.invoke(libc, "getpagesize")).orElseThrow(),
                        descriptorOf.invoke(null, javaInt, Array.newInstance(layoutClass, 0)),
                        noOptions
                    );
                    pageMask = -(long) (int) getPageSize.invokeExact();

                    final MethodHandles.Lookup lookup = MethodHandles.publicLookup();
                    slice = lookup.findVirtual(
                        MemorySegmentAccessInput.class,
                        "segmentSliceOrNull",
                        MethodType.methodType(segmentClass, long.class, long.class)
                    ).asType(MethodType.methodType(Object.class, MemorySegmentAccessInput.class, long.class, long.class));
                    address = lookup.findVirtual(segmentClass, "address", MethodType.methodType(long.class))
                        .asType(MethodType.methodType(long.class, Object.class));
                    log.info("k-NN prefetch uses direct madvise(MADV_WILLNEED)");
                } catch (Throwable t) {
                    // e.g. no FFM, native access denied (--illegal-native-access=deny): keep Lucene's prefetch.
                    log.warn("Direct madvise unavailable, falling back to IndexInput#prefetch: {}", t.toString());
                    slice = address = madvise = null;
                }
            }
            SLICE = slice;
            ADDRESS = address;
            MADVISE = madvise;
            PAGE_MASK = pageMask;
        }
    }
}
