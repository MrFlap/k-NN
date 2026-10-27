/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.io;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.util.ThreadInterruptedException;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicLongArray;
import java.util.concurrent.atomic.LongAdder;

import static org.opensearch.knn.index.io.NativeIo.PAGE;

/**
 * An off-heap cache of 4 KB file pages, filled by O_DIRECT reads through io_uring, so a merged prefetch range is one
 * device IO whatever the page cache holds.
 * <p>
 * Pages are keyed by (file id, page index) and evicted with CLOCK. A page being read is in flight: readers of it wait
 * for the completion rather than reading it again. Each read lands in a 256 KiB staging buffer and is copied into the
 * page slots on completion. A slot's version changes when it is evicted, and readers copy out of a slot and then check
 * the version, so a page evicted mid-copy is read again.
 */
@Log4j2
final class BlockCache {
    static final int MAX_IO_PAGES = IoUring.STAGING_BYTES / PAGE; // 256 KiB, one gp3 IO
    private static final int FREE = 0, IN_FLIGHT = 1, READY = 2;
    private static final int CHUNK_PAGES = 1 << 18; // 1 GiB per ByteBuffer view
    private static final int MAX_ATTEMPTS = 10_000;
    private static final int NO_SLOT = -2;

    final long capacityBytes;
    private final int numSlots;
    private final NativeIo.Memory memory;
    private final ByteBuffer[] chunks;
    // shortcut: boxed Long keys allocate per lookup; use a primitive open-addressing table if it shows in profiles.
    private final ConcurrentHashMap<Long, Integer> index = new ConcurrentHashMap<>();
    private final AtomicLongArray keys;
    private final AtomicIntegerArray states;
    private final AtomicIntegerArray versions;
    private final byte[] referenced;
    private final Object clockLock = new Object();
    // shortcut: one monitor wakes every waiter on every completion; stripe it if waiters contend.
    private final Object waitLock = new Object();
    private int hand;

    private final IoUring ring;

    private final LongAdder hits = new LongAdder();
    private final LongAdder misses = new LongAdder();
    private final LongAdder evictions = new LongAdder();
    private final LongAdder prefetchIos = new LongAdder();
    private final LongAdder syncIos = new LongAdder();
    private final LongAdder ioBytes = new LongAdder();
    private final AtomicLong usedPages = new AtomicLong();

    BlockCache(final long capacityBytes, final IoUring ring) {
        this.numSlots = Math.toIntExact(Math.max(MAX_IO_PAGES, capacityBytes / PAGE));
        this.capacityBytes = (long) numSlots * PAGE;
        this.memory = NativeIo.allocate(this.capacityBytes);
        this.chunks = new ByteBuffer[(numSlots + CHUNK_PAGES - 1) / CHUNK_PAGES];
        for (int c = 0; c < chunks.length; c++) {
            final int pages = Math.min(CHUNK_PAGES, numSlots - c * CHUNK_PAGES);
            chunks[c] = memory.buffer((long) c * CHUNK_PAGES * PAGE, pages * PAGE);
        }
        this.keys = new AtomicLongArray(numSlots);
        for (int s = 0; s < numSlots; s++) {
            keys.set(s, -1L);
        }
        this.states = new AtomicIntegerArray(numSlots);
        this.versions = new AtomicIntegerArray(numSlots);
        this.referenced = new byte[numSlots];
        this.ring = ring;
    }

    private static long key(final DirectFile file, final long page) {
        return ((long) file.id << 40) | page;
    }

    /** Copies {@code [position, position + length)} of {@code file} into {@code dst}, reading missing pages. */
    void read(final DirectFile file, long position, final byte[] dst, int offset, int length) throws IOException {
        final long lastPage = (position + length - 1) / PAGE;
        while (length > 0) {
            final long page = position / PAGE;
            final int inPage = (int) (position % PAGE);
            final int n = Math.min(length, PAGE - inPage);
            copyPage(file, page, lastPage, inPage, dst, offset, n);
            position += n;
            offset += n;
            length -= n;
        }
    }

    private void copyPage(
        final DirectFile file,
        final long page,
        final long lastPage,
        final int inPage,
        final byte[] dst,
        final int offset,
        final int length
    ) throws IOException {
        final long key = key(file, page);
        boolean submitted = false;
        for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
            final Integer slot = index.get(key);
            if (slot == null) {
                if (submitted) {
                    pause(); // every slot was in flight, so nothing was read
                }
                // Miss: read this page and the rest of the caller's range in one IO, then wait for it.
                submitted = true;
                submit(file, page, Math.min(lastPage, page + MAX_IO_PAGES - 1), true);
                continue;
            }
            final int s = slot;
            final int version = versions.get(s);
            if (keys.get(s) != key) {
                continue;
            }
            final int state = states.get(s);
            if (state == IN_FLIGHT) {
                awaitCompletion(s, key);
                continue;
            }
            if (state != READY) {
                continue;
            }
            chunks[s / CHUNK_PAGES].get((s % CHUNK_PAGES) * PAGE + inPage, dst, offset, length);
            if (versions.get(s) != version) {
                continue; // evicted while copying
            }
            referenced[s] = 1;
            (submitted ? misses : hits).increment();
            return;
        }
        throw new IOException("Could not read page " + page + " of " + file.name + " through the block cache");
    }

    private void pause() {
        synchronized (waitLock) {
            try {
                waitLock.wait(1);
            } catch (InterruptedException e) {
                throw new ThreadInterruptedException(e);
            }
        }
    }

    private void awaitCompletion(final int slot, final long key) {
        synchronized (waitLock) {
            while (keys.get(slot) == key && states.get(slot) == IN_FLIGHT) {
                try {
                    waitLock.wait();
                } catch (InterruptedException e) {
                    throw new ThreadInterruptedException(e);
                }
            }
        }
    }

    /**
     * Unless every page of {@code [position, position + length)} is cached or in flight, reads the whole range as one IO
     * (per {@value #MAX_IO_PAGES} pages) and returns without waiting. Every page it reads that was missing, gap pages
     * included, is inserted; cached and in-flight pages are read again and their bytes dropped. If every request is in
     * use, the prefetch is skipped.
     */
    void prefetch(final DirectFile file, final long position, final long length) throws IOException {
        if (length <= 0) {
            return;
        }
        final long lastPage = Math.min(file.lastPage(), (position + length - 1) / PAGE);
        // PrefetchHelper keeps a merged range within 64 pages; only a single range larger than that takes several IOs.
        for (long first = position / PAGE; first <= lastPage; first += MAX_IO_PAGES) {
            if (submit(file, first, Math.min(lastPage, first + MAX_IO_PAGES - 1), false) == false) {
                return;
            }
        }
    }

    /**
     * Claims slots for the missing pages of {@code [firstPage, lastPage]} and submits one read from the first missing
     * page to the last. Returns false if no request was free and {@code block} is false.
     */
    private boolean submit(final DirectFile file, final long firstPage, long lastPage, final boolean block) throws IOException {
        lastPage = Math.min(lastPage, file.lastPage());
        final Integer requestId = ring.acquire(block);
        if (requestId == null) {
            return false;
        }
        final int[] slots = new int[(int) (lastPage - firstPage + 1)];
        java.util.Arrays.fill(slots, -1);
        boolean missing = false;
        for (int i = 0; i < slots.length; i++) {
            slots[i] = claim(key(file, firstPage + i));
            if (slots[i] == NO_SLOT) {
                slots[i] = -1;
                break; // every slot is in flight: the rest of the range is read but not kept
            }
            missing |= slots[i] >= 0;
        }
        if (missing == false) {
            ring.release(requestId);
            return true;
        }
        // The whole range is one IO: cached and in-flight pages inside it are read again and their bytes dropped.
        (block ? syncIos : prefetchIos).increment();
        ioBytes.add((long) slots.length * PAGE);
        ring.submitRead(
            requestId,
            file.fd,
            firstPage * PAGE,
            slots.length * PAGE,
            (result, staging) -> complete(file, firstPage, slots, result, staging)
        );
        return true;
    }

    /**
     * Takes a slot for {@code key} in flight. Returns -1 if the page is already cached or in flight, and
     * {@link #NO_SLOT} if every slot is in flight.
     */
    private int claim(final long key) {
        if (index.containsKey(key)) {
            return -1;
        }
        synchronized (clockLock) {
            for (int scanned = 0; scanned < 2 * numSlots; scanned++) {
                final int s = hand;
                hand = hand + 1 == numSlots ? 0 : hand + 1;
                final int state = states.get(s);
                if (state == IN_FLIGHT) {
                    continue;
                }
                if (state == READY && referenced[s] != 0) {
                    referenced[s] = 0;
                    continue;
                }
                final long oldKey = keys.get(s);
                if (state == READY) {
                    index.remove(oldKey, s);
                    evictions.increment();
                } else {
                    usedPages.incrementAndGet();
                }
                versions.incrementAndGet(s);
                keys.set(s, key);
                states.set(s, IN_FLIGHT);
                if (index.putIfAbsent(key, s) != null) {
                    keys.set(s, -1L);
                    states.set(s, FREE);
                    usedPages.decrementAndGet();
                    return -1;
                }
                return s;
            }
            return NO_SLOT;
        }
    }

    /** Runs on the ring's reaper thread: copies the read pages into their slots and wakes waiters. */
    private void complete(final DirectFile file, final long firstPage, final int[] slots, final int result, final ByteBuffer buffer) {
        final int[] failed = new int[slots.length];
        int numFailed = 0;
        for (int i = 0; i < slots.length; i++) {
            final int s = slots[i];
            if (s < 0) {
                continue;
            }
            // A short read only ends at EOF, and every claimed page holds file bytes, so it must have reached this page.
            if (result <= i * PAGE) {
                failed[numFailed++] = s;
                continue;
            }
            chunks[s / CHUNK_PAGES].put((s % CHUNK_PAGES) * PAGE, buffer, i * PAGE, PAGE);
            states.set(s, READY);
        }
        if (numFailed > 0) {
            log.warn("O_DIRECT read of {} pages at page {} of {} returned {}", slots.length, firstPage, file.name, result);
            release(java.util.Arrays.copyOf(failed, numFailed));
        }
        synchronized (waitLock) {
            waitLock.notifyAll();
        }
    }

    private void release(final int[] slots) {
        for (final int s : slots) {
            if (s < 0) {
                continue;
            }
            index.remove(keys.get(s), s);
            versions.incrementAndGet(s);
            keys.set(s, -1L);
            states.set(s, FREE);
            usedPages.decrementAndGet();
        }
        synchronized (waitLock) {
            waitLock.notifyAll();
        }
    }

    long usedBytes() {
        return usedPages.get() * PAGE;
    }

    long hits() {
        return hits.sum();
    }

    long misses() {
        return misses.sum();
    }

    long evictions() {
        return evictions.sum();
    }

    /** IOs started by {@link #prefetch}. */
    long prefetchIos() {
        return prefetchIos.sum();
    }

    /** IOs started by reads that missed. */
    long syncIos() {
        return syncIos.sum();
    }

    /** Bytes requested by all IOs. */
    long ioBytes() {
        return ioBytes.sum();
    }
}
