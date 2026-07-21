/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

/**
 * ThreadLocal holder for the per-segment QuantizationErrorFile.Reader.
 * Must be set before any search that uses this scorer and cleared after.
 */
public final class QefContext {
    public static final ThreadLocal<QuantizationErrorFile.Reader> CURRENT = new ThreadLocal<>();

    private QefContext() {}
}
