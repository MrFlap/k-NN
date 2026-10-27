/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.io;

import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;

import java.io.Closeable;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/** A file opened with O_DIRECT for the block cache. Its id keys its cached pages and is never reused. */
final class DirectFile implements Closeable {
    private static final AtomicInteger NEXT_ID = new AtomicInteger();

    final int id;
    final int fd;
    final long length;
    final String name;
    private final AtomicBoolean closed = new AtomicBoolean();

    private DirectFile(final int fd, final long length, final String name) {
        this.id = NEXT_ID.getAndIncrement();
        this.fd = fd;
        this.length = length;
        this.name = name;
    }

    /**
     * Opens {@code fileName} of {@code directory} with O_DIRECT, or returns null when it has no path on a local file
     * system (for example inside a compound file) or the open fails (for example tmpfs before 6.6).
     */
    static DirectFile open(final Directory directory, final String fileName) throws IOException {
        if (NativeIo.UNAVAILABLE != null || (FilterDirectory.unwrap(directory) instanceof FSDirectory) == false) {
            return null;
        }
        final FSDirectory fsDirectory = (FSDirectory) FilterDirectory.unwrap(directory);
        final java.nio.file.Path path = fsDirectory.getDirectory().resolve(fileName).toAbsolutePath();
        final int fd = NativeIo.openDirect(path.toString());
        return fd < 0 ? null : new DirectFile(fd, directory.fileLength(fileName), path.toString());
    }

    /** Index of the last page holding file bytes. */
    long lastPage() {
        return (length - 1) / NativeIo.PAGE;
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            NativeIo.close(fd);
        }
    }
}
