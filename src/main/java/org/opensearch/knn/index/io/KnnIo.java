/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.io;

import com.google.common.annotations.VisibleForTesting;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.monitor.os.OsProbe;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

/**
 * The {@code knn.io.backend} switch. With {@code io_uring}, search reads of k-NN vector files go through an off-heap
 * {@link BlockCache} filled by O_DIRECT reads, so a merged prefetch range is one device IO whatever the page cache
 * holds. With {@code mmap} (the default), or when io_uring is unavailable, reads use the mmap'd inputs unchanged.
 * <p>
 * The cache holds {@code knn.io.block_cache.percent} of the k-NN vector data of this node's open segments, at most
 * half of the RAM outside the JVM heap. It is allocated on first io_uring use.
 */
@Log4j2
public final class KnnIo {
    public static final String MMAP = "mmap";
    public static final String IO_URING = "io_uring";

    /** Extensions of the per-segment k-NN files counted as vector data, besides each field's engine file (.faiss). */
    public static final List<String> VECTOR_DATA_EXTENSIONS = List.of("vec", "vemf", "veq", "vemq", "veqlo", "vemlo");

    private static final int RING_ENTRIES = 256;
    private static final long MIN_CAPACITY_BYTES = 16L << 20;
    private static final AtomicLong VECTOR_DATA_BYTES = new AtomicLong();
    // Open k-NN vector files by path: {readers holding it, its length}.
    private static final Map<String, long[]> VECTOR_DATA_FILES = new HashMap<>();
    private static final LongAdder PREFETCH_BATCHES = new LongAdder();
    private static final long PHYSICAL_BYTES = OsProbe.getInstance().getTotalPhysicalMemorySize();
    private static final long MAX_HEAP_BYTES = Runtime.getRuntime().maxMemory();

    private static volatile IoUring ring;
    private static volatile String unavailable;
    private static volatile BlockCache cache;
    private static volatile Boolean availableForTesting;

    private KnnIo() {}

    /** Whether a search should read through the block cache: {@code knn.io.backend} is io_uring and io_uring works. */
    public static boolean useIoUring() {
        return IO_URING.equals(KNNSettings.getIoBackend()) && blockCache() != null;
    }

    /** Adjusts the vector data total directly, for tests. */
    static void addVectorData(final long bytes) {
        VECTOR_DATA_BYTES.addAndGet(bytes);
    }

    /**
     * Counts a k-NN reader's vector files into the node's vector data: the {@link #VECTOR_DATA_EXTENSIONS} files of its
     * segment suffix plus the engine files of the segment's vector fields. Each file counts once while any reader
     * holding it is open, so several readers of one segment, or two formats sharing an engine file, add it once.
     *
     * @return releases this reader's hold; closing it again does nothing
     */
    // shortcut: files inside a compound file count but are read with mmap.
    public static Closeable registerVectorData(final SegmentReadState state) {
        final Map<String, Long> files = vectorDataFiles(state);
        synchronized (VECTOR_DATA_FILES) {
            for (final Map.Entry<String, Long> file : files.entrySet()) {
                final long[] entry = VECTOR_DATA_FILES.computeIfAbsent(file.getKey(), k -> new long[] { 0, file.getValue() });
                if (entry[0]++ == 0) {
                    VECTOR_DATA_BYTES.addAndGet(entry[1]);
                }
            }
        }
        final AtomicBoolean released = new AtomicBoolean();
        return () -> {
            if (released.compareAndSet(false, true) == false) {
                return;
            }
            synchronized (VECTOR_DATA_FILES) {
                for (final String path : files.keySet()) {
                    final long[] entry = VECTOR_DATA_FILES.get(path);
                    if (entry != null && --entry[0] == 0) {
                        VECTOR_DATA_FILES.remove(path);
                        VECTOR_DATA_BYTES.addAndGet(-entry[1]);
                    }
                }
            }
        };
    }

    /** The reader's k-NN vector files with their lengths, keyed by absolute path (or directory and name off a file system). */
    private static Map<String, Long> vectorDataFiles(final SegmentReadState state) {
        final Map<String, Long> result = new HashMap<>();
        try {
            final Set<String> present = Set.of(state.directory.listAll());
            final List<String> names = new ArrayList<>();
            for (final String extension : VECTOR_DATA_EXTENSIONS) {
                names.add(IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, extension));
            }
            for (final FieldInfo field : state.fieldInfos) {
                if (field.hasVectorValues()) {
                    names.add(KNNCodecUtil.getNativeEngineFileFromFieldInfo(field, state.segmentInfo));
                }
            }
            final Directory unwrapped = FilterDirectory.unwrap(state.directory);
            for (final String name : names) {
                if (name != null && present.contains(name)) {
                    final String key = unwrapped instanceof FSDirectory
                        ? ((FSDirectory) unwrapped).getDirectory().resolve(name).toAbsolutePath().toString()
                        : System.identityHashCode(unwrapped) + "/" + name;
                    result.put(key, state.directory.fileLength(name));
                }
            }
        } catch (IOException | RuntimeException e) {
            // e.g. segment files not set yet
            log.debug("Could not size k-NN vector data of segment [{}]", state.segmentInfo.name, e);
        }
        return result;
    }

    /** Node-wide k-NN vector data bytes counted by {@link #registerVectorData}. */
    static long vectorDataBytes() {
        return VECTOR_DATA_BYTES.get();
    }

    /**
     * Cache capacity: {@code percent} of {@code vectorDataBytes}, at most half of {@code physicalBytes - maxHeapBytes},
     * and at least {@value #MIN_CAPACITY_BYTES} bytes.
     */
    static long capacity(final double percent, final long vectorDataBytes, final long physicalBytes, final long maxHeapBytes) {
        return Math.max(MIN_CAPACITY_BYTES, Math.min(requested(percent, vectorDataBytes), ramLimit(physicalBytes, maxHeapBytes)));
    }

    private static long requested(final double percent, final long vectorDataBytes) {
        return (long) (percent / 100.0 * vectorDataBytes);
    }

    private static long ramLimit(final long physicalBytes, final long maxHeapBytes) {
        return Math.max(0, physicalBytes - maxHeapBytes) / 2;
    }

    /**
     * The block cache, allocated on first use and reallocated when its target capacity moves by more than 10%; null
     * when io_uring is unavailable.
     */
    static BlockCache blockCache() {
        final IoUring r = ring();
        if (r == null) {
            return null;
        }
        final double percent = KNNSettings.getBlockCachePercent();
        final long vectorData = VECTOR_DATA_BYTES.get();
        final long target = capacity(percent, vectorData, PHYSICAL_BYTES, MAX_HEAP_BYTES);
        final BlockCache current = cache;
        if (current != null && Math.abs(current.capacityBytes - target) <= current.capacityBytes / 10) {
            return current;
        }
        synchronized (KnnIo.class) {
            if (cache == null || Math.abs(cache.capacityBytes - target) > cache.capacityBytes / 10) {
                // shortcut: a resize drops the cache and starts a new one; inputs opened before keep the old one until
                // their search ends, and the GC frees it. Fine for benchmark sweeps; warm a resized cache before measuring.
                if (requested(percent, vectorData) > ramLimit(PHYSICAL_BYTES, MAX_HEAP_BYTES)) {
                    log.warn(
                        "knn.io.block_cache.percent asks for [{}] bytes; clamped to [{}], half of the RAM outside the heap",
                        requested(percent, vectorData),
                        ramLimit(PHYSICAL_BYTES, MAX_HEAP_BYTES)
                    );
                }
                log.info("Allocating a [{}] byte k-NN block cache for [{}] bytes of vector data", target, vectorData);
                cache = new BlockCache(target, r);
            }
            return cache;
        }
    }

    private static IoUring ring() {
        if (availableForTesting != null) {
            return availableForTesting ? ringOrNull() : null;
        }
        return ringOrNull();
    }

    private static IoUring ringOrNull() {
        if (ring != null || unavailable != null) {
            return ring;
        }
        synchronized (KnnIo.class) {
            if (ring == null && unavailable == null) {
                try {
                    ring = IoUring.create(RING_ENTRIES);
                    log.info("k-NN io_uring backend ready");
                } catch (IOException | RuntimeException e) {
                    unavailable = e.toString();
                    log.warn("k-NN io_uring backend unavailable, reads use mmap: {}", unavailable);
                }
            }
            return ring;
        }
    }

    /** Forces io_uring unavailable ({@code false}), or back to probing ({@code null}). */
    @VisibleForTesting
    public static void setAvailableForTesting(final Boolean available) {
        availableForTesting = available;
    }

    /** Opens {@code fileName} through the block cache, or returns null if it cannot be opened with O_DIRECT. */
    static IndexInput openCached(final Directory directory, final String fileName) throws IOException {
        final DirectFile file = DirectFile.open(directory, fileName);
        return file == null ? null : new CachedIndexInput(file, KnnIo::blockCache);
    }

    /**
     * A directory that opens files with {@code extension} through the block cache, so a Lucene reader built on it reads
     * them with O_DIRECT. Files that cannot be opened that way are opened by {@code directory}.
     */
    public static Directory cachedDirectory(final Directory directory, final String extension) {
        return new FilterDirectory(directory) {
            @Override
            public IndexInput openInput(final String name, final IOContext context) throws IOException {
                if (name.endsWith("." + extension)) {
                    final IndexInput cached = openCached(in, name);
                    if (cached != null) {
                        return cached;
                    }
                }
                return super.openInput(name, context);
            }
        };
    }

    /** Counts a prefetch batch (one PrefetchHelper grouping pass) on an input that reads through the block cache. */
    public static void recordPrefetchBatch(final IndexInput input) {
        if (input instanceof CachedIndexInput) {
            PREFETCH_BATCHES.increment();
        }
    }

    /**
     * k-NN stats: backend, availability, the cache's capacity, use, hits, misses and evictions, and its IOs: prefetch
     * IOs per prefetch batch (one beam or neighbor batch) and the average IO size.
     */
    public static Map<String, Object> stats() {
        final Map<String, Object> stats = new LinkedHashMap<>();
        final BlockCache c = cache;
        stats.put("backend", KNNSettings.getIoBackend());
        stats.put("io_uring_unavailable_reason", unavailable == null ? "" : unavailable);
        stats.put("vector_data_bytes", VECTOR_DATA_BYTES.get());
        stats.put("capacity_bytes", c == null ? 0 : c.capacityBytes);
        stats.put("used_bytes", c == null ? 0 : c.usedBytes());
        stats.put("hits", c == null ? 0 : c.hits());
        stats.put("misses", c == null ? 0 : c.misses());
        stats.put("evictions", c == null ? 0 : c.evictions());
        final long prefetchIos = c == null ? 0 : c.prefetchIos();
        final long ios = prefetchIos + (c == null ? 0 : c.syncIos());
        final long batches = PREFETCH_BATCHES.sum();
        stats.put("prefetch_batches", batches);
        stats.put("prefetch_ios", prefetchIos);
        stats.put("read_miss_ios", c == null ? 0 : c.syncIos());
        stats.put("prefetch_ios_per_batch", batches == 0 ? 0.0 : (double) prefetchIos / batches);
        stats.put("avg_io_bytes", ios == 0 ? 0L : (c.ioBytes() / ios));
        return stats;
    }

    /**
     * A file a reader can read either way: the mmap'd input it already has, or, opened on first io_uring use, the same
     * file through the block cache.
     */
    public static final class LazyCachedInput implements Closeable {
        private final Directory directory;
        private final String fileName;
        private volatile IndexInput cached;
        private volatile boolean opened;

        public LazyCachedInput(final Directory directory, final String fileName) {
            this.directory = directory;
            this.fileName = fileName;
        }

        /**
         * For one search: a clone of {@code mmapView} on the mmap backend, otherwise the same bytes, {@code [offset,
         * offset + length)} of the file, through the block cache.
         */
        public IndexInput select(final IndexInput mmapView, final long offset, final long length) throws IOException {
            if (useIoUring() == false) {
                return mmapView.clone();
            }
            final IndexInput file = open();
            return file == null ? mmapView.clone() : file.slice("cached " + mmapView, offset, length);
        }

        private IndexInput open() throws IOException {
            if (opened == false) {
                synchronized (this) {
                    if (opened == false) {
                        cached = openCached(directory, fileName);
                        opened = true;
                    }
                }
            }
            return cached;
        }

        @Override
        public void close() throws IOException {
            if (cached != null) {
                cached.close();
            }
        }
    }
}
