/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.padrotate;

import org.apache.lucene.util.VectorUtil;

import java.util.Random;

/**
 * Deterministic random orthogonal matrix. Rows are sampled from a standard Gaussian then
 * Gram-Schmidt orthonormalized, so the resulting matrix preserves L2 norms and inner products.
 *
 * <p>The matrix is deterministic given a 64-bit seed, which lets us regenerate it from metadata
 * rather than persisting the full {@code dim x dim} matrix per segment.
 */
public final class RandomRotation {

    private final int dimensions;
    private final float[][] matrix;

    public RandomRotation(int dimensions, long seed) {
        if (dimensions <= 0) {
            throw new IllegalArgumentException("dimensions must be positive, got " + dimensions);
        }
        this.dimensions = dimensions;
        this.matrix = generate(dimensions, seed);
    }

    public int dimensions() {
        return dimensions;
    }

    /**
     * Apply the rotation to {@code input}, writing the result into {@code output}. Both arrays
     * must have length {@link #dimensions()}. {@code input} and {@code output} must not alias.
     */
    public void apply(float[] input, float[] output) {
        if (input.length != dimensions || output.length != dimensions) {
            throw new IllegalArgumentException(
                "expected length " + dimensions + ", got input=" + input.length + " output=" + output.length
            );
        }
        for (int i = 0; i < dimensions; i++) {
            output[i] = VectorUtil.dotProduct(matrix[i], input);
        }
    }

    /**
     * Apply the transpose (= inverse, since the matrix is orthogonal) to {@code input}, writing
     * the result into {@code output}. Both arrays must have length {@link #dimensions()}.
     */
    public void applyTranspose(float[] input, float[] output) {
        if (input.length != dimensions || output.length != dimensions) {
            throw new IllegalArgumentException(
                "expected length " + dimensions + ", got input=" + input.length + " output=" + output.length
            );
        }
        // output[j] = sum_i matrix[i][j] * input[i]
        for (int j = 0; j < dimensions; j++) {
            output[j] = 0f;
        }
        for (int i = 0; i < dimensions; i++) {
            final float xi = input[i];
            final float[] row = matrix[i];
            for (int j = 0; j < dimensions; j++) {
                output[j] += row[j] * xi;
            }
        }
    }

    private static float[][] generate(int d, long seed) {
        final Random random = new Random(seed);
        final float[][] m = new float[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                m[i][j] = (float) random.nextGaussian();
            }
        }
        // Modified Gram-Schmidt. Slightly more numerically stable than classical GS at no
        // meaningful cost for our dim ranges.
        for (int i = 0; i < d; i++) {
            for (int k = 0; k < i; k++) {
                float dot = 0f;
                for (int j = 0; j < d; j++) {
                    dot += m[i][j] * m[k][j];
                }
                for (int j = 0; j < d; j++) {
                    m[i][j] -= dot * m[k][j];
                }
            }
            float norm = 0f;
            for (int j = 0; j < d; j++) {
                norm += m[i][j] * m[i][j];
            }
            norm = (float) Math.sqrt(norm);
            if (norm == 0f) {
                throw new IllegalStateException("degenerate row at i=" + i + " during orthonormalization");
            }
            float inv = 1f / norm;
            for (int j = 0; j < d; j++) {
                m[i][j] *= inv;
            }
        }
        return m;
    }
}
