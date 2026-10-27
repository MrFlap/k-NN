/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.io;

import com.carrotsearch.randomizedtesting.ThreadFilter;

/** The io_uring reaper is a daemon thread that lives as long as the JVM. */
public final class IoUringReaperFilter implements ThreadFilter {
    @Override
    public boolean reject(final Thread t) {
        return "knn-io-uring-reaper".equals(t.getName());
    }
}
