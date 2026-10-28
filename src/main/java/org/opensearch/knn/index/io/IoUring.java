/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.io;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.util.ThreadInterruptedException;

import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.ArrayBlockingQueue;

/**
 * A minimal io_uring instance over raw syscalls: one submission queue fed under a lock, and a daemon reaper thread that
 * waits for completions and runs each request's callback. Only {@code IORING_OP_READ} (kernel 5.6+) is used. Each
 * request reads into its own {@value #STAGING_BYTES}-byte staging buffer; a caller takes a request id with
 * {@link #acquire} and the id is freed after its callback runs, so at most {@link #entries()} requests are in flight and
 * neither queue can overflow (the completion queue is twice the submission queue).
 */
@Log4j2
final class IoUring {
    static final int STAGING_BYTES = 256 * 1024;

    interface Completion {
        /** @param result bytes read, or a negative errno; {@code staging} holds the bytes until this returns */
        void complete(int result, ByteBuffer staging);
    }

    private static final long SYS_IO_URING_SETUP = 425; // same number on x86_64 and aarch64
    private static final long SYS_IO_URING_ENTER = 426;
    private static final int IORING_ENTER_GETEVENTS = 1;
    private static final int IORING_FEAT_SINGLE_MMAP = 1;
    private static final byte IORING_OP_READ = 22;
    private static final long IORING_OFF_SQ_RING = 0L;
    private static final long IORING_OFF_CQ_RING = 0x8000000L;
    private static final long IORING_OFF_SQES = 0x10000000L;
    private static final int PROT_READ_WRITE = 0x3;
    private static final int MAP_SHARED_POPULATE = 0x1 | 0x8000;
    private static final int SQE_BYTES = 64;
    private static final int CQE_BYTES = 16;
    private static final VarHandle INT = MethodHandles.byteBufferViewVarHandle(int[].class, ByteOrder.nativeOrder());

    private final int ringFd;
    private final int entries;
    private final ByteBuffer sq;
    private final ByteBuffer cq;
    private final ByteBuffer sqes;
    private final int sqHead, sqTail, sqMask;
    private final int cqHead, cqTail, cqMask, cqes;
    private final NativeIo.Memory staging;
    private final ByteBuffer[] stagingBuffers;
    private final Completion[] completions;
    private final ArrayBlockingQueue<Integer> freeIds;
    private int tail;

    private IoUring(final int requestedEntries) throws IOException {
        // struct io_uring_params is 120 bytes; the kernel fills sq_off at 40 and cq_off at 80.
        final NativeIo.Memory paramsMemory = NativeIo.allocate(120);
        final ByteBuffer p = paramsMemory.buffer(0, 120).order(ByteOrder.nativeOrder());
        final long fd = NativeIo.syscall(SYS_IO_URING_SETUP, requestedEntries, paramsMemory.address, 0, 0, 0, 0);
        if (fd < 0) {
            throw new IOException("io_uring_setup failed (kernel.io_uring_disabled, seccomp, or no io_uring)");
        }
        this.ringFd = (int) fd;
        this.entries = p.getInt(0);
        final int cqEntries = p.getInt(4);
        final int features = p.getInt(20);
        final int sqArray = p.getInt(64);
        this.cqes = p.getInt(100);
        final long sqBytes = sqArray + (long) entries * Integer.BYTES;
        final long cqBytes = cqes + (long) cqEntries * CQE_BYTES;
        final boolean single = (features & IORING_FEAT_SINGLE_MMAP) != 0;
        final long sqAddress = map(single ? Math.max(sqBytes, cqBytes) : sqBytes, IORING_OFF_SQ_RING);
        this.sq = NativeIo.view(sqAddress, single ? Math.max(sqBytes, cqBytes) : sqBytes);
        this.cq = single ? sq : NativeIo.view(map(cqBytes, IORING_OFF_CQ_RING), cqBytes);
        this.sqes = NativeIo.view(map((long) entries * SQE_BYTES, IORING_OFF_SQES), (long) entries * SQE_BYTES);
        this.sqHead = p.getInt(40);
        this.sqTail = p.getInt(44);
        this.sqMask = sq.getInt(p.getInt(48));
        this.cqHead = p.getInt(80);
        this.cqTail = p.getInt(84);
        this.cqMask = cq.getInt(p.getInt(88));
        // The SQ array indirection is the identity: SQE i is always submitted from array slot i.
        for (int i = 0; i < entries; i++) {
            sq.putInt(sqArray + i * Integer.BYTES, i);
        }
        this.tail = (int) INT.getAcquire(sq, sqTail);
        this.staging = NativeIo.mapAnonymous((long) entries * STAGING_BYTES);
        this.stagingBuffers = new ByteBuffer[entries];
        this.completions = new Completion[entries];
        this.freeIds = new ArrayBlockingQueue<>(entries);
        for (int i = 0; i < entries; i++) {
            stagingBuffers[i] = staging.buffer((long) i * STAGING_BYTES, STAGING_BYTES);
            freeIds.add(i);
        }
        // shortcut: the ring and its reaper live as long as the JVM; there is one per node, so nothing closes them.
        final Thread reaper = new Thread(this::reap, "knn-io-uring-reaper");
        reaper.setDaemon(true);
        reaper.start();
    }

    /** @throws IOException if io_uring is unavailable */
    static IoUring create(final int entries) throws IOException {
        if (NativeIo.UNAVAILABLE != null) {
            throw new IOException("native bindings unavailable: " + NativeIo.UNAVAILABLE);
        }
        return new IoUring(entries);
    }

    private long map(final long length, final long offset) throws IOException {
        final long address = NativeIo.mmap(length, PROT_READ_WRITE, MAP_SHARED_POPULATE, ringFd, offset);
        if (address == -1L) {
            throw new IOException("mmap of io_uring ring failed");
        }
        return address;
    }

    /** The most requests that may be in flight at once. */
    int entries() {
        return entries;
    }

    /** A free request id, waiting for one if {@code block}; null if none is free and not blocking. */
    Integer acquire(final boolean block) {
        if (block == false) {
            return freeIds.poll();
        }
        try {
            return freeIds.take();
        } catch (InterruptedException e) {
            throw new ThreadInterruptedException(e);
        }
    }

    /** Returns an acquired id that was not submitted. */
    void release(final int requestId) {
        freeIds.add(requestId);
    }

    /**
     * Submits a read of {@code length} (at most {@value #STAGING_BYTES}) bytes at {@code offset} of {@code fd} into the
     * request's staging buffer, and runs {@code completion} on the reaper thread when it finishes, after which the id
     * is free again.
     */
    synchronized void submitRead(final int requestId, final int fd, final long offset, final int length, final Completion completion) {
        completions[requestId] = completion;
        final long address = staging.address + (long) requestId * STAGING_BYTES;
        final int sqe = (tail & sqMask) * SQE_BYTES;
        for (int i = 0; i < SQE_BYTES; i += Long.BYTES) {
            sqes.putLong(sqe + i, 0L);
        }
        sqes.put(sqe, IORING_OP_READ);
        sqes.putInt(sqe + 4, fd);
        sqes.putLong(sqe + 8, offset);
        sqes.putLong(sqe + 16, address);
        sqes.putInt(sqe + 24, length);
        sqes.putLong(sqe + 32, requestId);
        tail++;
        INT.setRelease(sq, sqTail, tail);
        // Submit everything the kernel has not consumed yet, so an earlier failed enter cannot strand an entry.
        final int pending = tail - (int) INT.getAcquire(sq, sqHead);
        if (pending > 0 && NativeIo.syscall(SYS_IO_URING_ENTER, ringFd, pending, 0, 0, 0, 0) < 0) {
            // shortcut: the entry stays queued and goes out with the next submit; nothing retries it sooner.
            log.warn("io_uring_enter submit of {} entries failed", pending);
        }
    }

    private void reap() {
        while (true) {
            try {
                NativeIo.syscall(SYS_IO_URING_ENTER, ringFd, 0, 1, IORING_ENTER_GETEVENTS, 0, 0);
                int head = (int) INT.getAcquire(cq, cqHead);
                final int last = (int) INT.getAcquire(cq, cqTail);
                while (head != last) {
                    final int cqe = cqes + (head & cqMask) * CQE_BYTES;
                    final int requestId = (int) cq.getLong(cqe);
                    final int result = cq.getInt(cqe + 8);
                    head++;
                    INT.setRelease(cq, cqHead, head);
                    final Completion completion = completions[requestId];
                    completions[requestId] = null;
                    try {
                        completion.complete(result, stagingBuffers[requestId]);
                    } finally {
                        freeIds.add(requestId);
                    }
                }
            } catch (Throwable t) {
                log.error("io_uring reaper failed", t);
            }
        }
    }
}
