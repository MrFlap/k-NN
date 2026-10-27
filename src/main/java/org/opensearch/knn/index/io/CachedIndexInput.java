/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.io;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.BitUtil;

import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.function.Supplier;

/**
 * An {@link IndexInput} over a {@link DirectFile} that reads every byte through the {@link BlockCache}, and whose
 * {@link #prefetch} starts O_DIRECT reads of the missing pages. Slices and clones share the file; only the input that
 * opened it closes it. Each slice or clone takes the cache current at its creation, so a reallocated cache is used by
 * the next search.
 */
final class CachedIndexInput extends IndexInput {
    private final DirectFile file;
    private final Supplier<BlockCache> caches;
    private final BlockCache cache;
    private final long base;
    private final long length;
    private final boolean owner;
    private long position;
    private byte[] scratch = new byte[Long.BYTES];

    CachedIndexInput(final DirectFile file, final Supplier<BlockCache> caches) {
        this("CachedIndexInput(" + file.name + ")", file, caches, 0, file.length, true);
    }

    private CachedIndexInput(
        final String description,
        final DirectFile file,
        final Supplier<BlockCache> caches,
        final long base,
        final long length,
        final boolean owner
    ) {
        super(description);
        this.file = file;
        this.caches = caches;
        this.cache = caches.get();
        this.base = base;
        this.length = length;
        this.owner = owner;
    }

    @Override
    public void readBytes(final byte[] b, final int offset, final int len) throws IOException {
        if (len > length - position) {
            throw new EOFException("read past EOF: " + this + " position=" + position + " len=" + len + " length=" + length);
        }
        cache.read(file, base + position, b, offset, len);
        position += len;
    }

    @Override
    public byte readByte() throws IOException {
        readBytes(scratch, 0, 1);
        return scratch[0];
    }

    @Override
    public short readShort() throws IOException {
        readBytes(scratch, 0, Short.BYTES);
        return (short) BitUtil.VH_LE_SHORT.get(scratch, 0);
    }

    @Override
    public int readInt() throws IOException {
        readBytes(scratch, 0, Integer.BYTES);
        return (int) BitUtil.VH_LE_INT.get(scratch, 0);
    }

    @Override
    public long readLong() throws IOException {
        readBytes(scratch, 0, Long.BYTES);
        return (long) BitUtil.VH_LE_LONG.get(scratch, 0);
    }

    @Override
    public void readInts(final int[] dst, final int offset, final int len) throws IOException {
        littleEndian(len * Integer.BYTES).asIntBuffer().get(dst, offset, len);
    }

    @Override
    public void readFloats(final float[] dst, final int offset, final int len) throws IOException {
        littleEndian(len * Float.BYTES).asFloatBuffer().get(dst, offset, len);
    }

    @Override
    public void readLongs(final long[] dst, final int offset, final int len) throws IOException {
        littleEndian(len * Long.BYTES).asLongBuffer().get(dst, offset, len);
    }

    private ByteBuffer littleEndian(final int bytes) throws IOException {
        if (scratch.length < bytes) {
            scratch = new byte[Math.max(bytes, 2 * scratch.length)];
        }
        readBytes(scratch, 0, bytes);
        return ByteBuffer.wrap(scratch, 0, bytes).order(ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    public void prefetch(final long offset, final long len) throws IOException {
        cache.prefetch(file, base + offset, Math.min(len, length - offset));
    }

    @Override
    public long getFilePointer() {
        return position;
    }

    @Override
    public void seek(final long pos) throws IOException {
        if (pos < 0 || pos > length) {
            throw new EOFException("seek past EOF: " + this + " pos=" + pos + " length=" + length);
        }
        position = pos;
    }

    @Override
    public long length() {
        return length;
    }

    @Override
    public IndexInput slice(final String sliceDescription, final long offset, final long len) {
        if (offset < 0 || len < 0 || offset + len > length) {
            throw new IllegalArgumentException("slice out of bounds: " + sliceDescription + " offset=" + offset + " length=" + len);
        }
        return new CachedIndexInput(getFullSliceDescription(sliceDescription), file, caches, base + offset, len, false);
    }

    @Override
    public CachedIndexInput clone() {
        final CachedIndexInput clone = new CachedIndexInput(toString(), file, caches, base, length, false);
        clone.position = position;
        return clone;
    }

    @Override
    public void close() {
        if (owner) {
            file.close();
        }
    }
}
