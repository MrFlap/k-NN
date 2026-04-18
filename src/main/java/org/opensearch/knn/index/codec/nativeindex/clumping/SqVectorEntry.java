/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.clumping;

/**
 * A fully-materialized scalar-quantized (1-bit) vector: the packed binary code
 * plus the four correction factors produced by the {@code OptimizedScalarQuantizer}.
 *
 * <p>This mirrors the exact byte layout that
 * {@link org.opensearch.knn.index.codec.nativeindex.MemOptimizedScalarQuantizedIndexBuildStrategy}
 * ships to native code and that the SIMD SQ search context consumes. Using a single record keeps
 * the clumping build/write/read paths symmetric with the native expectations and avoids redundant
 * unpacking.
 */
public final class SqVectorEntry {

    /** The 1-bit quantized binary code. Length equals {@code quantizedVecBytes}. */
    public final byte[] code;
    public final float lowerInterval;
    public final float upperInterval;
    public final float additionalCorrection;
    public final int quantizedComponentSum;

    public SqVectorEntry(
        byte[] code,
        float lowerInterval,
        float upperInterval,
        float additionalCorrection,
        int quantizedComponentSum
    ) {
        this.code = code;
        this.lowerInterval = lowerInterval;
        this.upperInterval = upperInterval;
        this.additionalCorrection = additionalCorrection;
        this.quantizedComponentSum = quantizedComponentSum;
    }
}
