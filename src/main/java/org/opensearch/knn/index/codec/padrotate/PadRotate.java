/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.padrotate;

/**
 * Pad-and-rotate transform: pads a {@code D}-length float vector up to {@code PAD_FACTOR * D}
 * (zero-filling the tail) and applies a per-segment random orthogonal rotation. Used to spread
 * input variance across a higher-dimensional space before aggressive scalar quantization.
 *
 * <p>The rotation is deterministic given a seed so the writer and reader for a given segment
 * agree on the transform without persisting the full matrix.
 */
public final class PadRotate {

    /** Factor by which the input dimension is expanded before quantization. */
    public static final int PAD_FACTOR = 4;

    private final int originalDim;
    private final int paddedDim;
    private final RandomRotation rotation;

    public PadRotate(int originalDim, long seed) {
        this.originalDim = originalDim;
        this.paddedDim = paddedDim(originalDim);
        this.rotation = new RandomRotation(paddedDim, seed);
    }

    public static int paddedDim(int originalDim) {
        return originalDim * PAD_FACTOR;
    }

    public int originalDimensions() {
        return originalDim;
    }

    public int paddedDimensions() {
        return paddedDim;
    }

    /**
     * Pads {@code input} (length {@code D}) to the padded dimension and applies the rotation.
     * {@code scratch} and {@code output} must both have length {@code paddedDim}; {@code scratch}
     * is used to hold the zero-padded vector before rotation. Neither array is returned.
     */
    public void forward(float[] input, float[] scratch, float[] output) {
        if (input.length != originalDim) {
            throw new IllegalArgumentException("expected input length " + originalDim + ", got " + input.length);
        }
        System.arraycopy(input, 0, scratch, 0, originalDim);
        for (int i = originalDim; i < paddedDim; i++) {
            scratch[i] = 0f;
        }
        rotation.apply(scratch, output);
    }

    /**
     * Reverses the rotation on {@code paddedRotated} (length {@code paddedDim}) and writes the
     * first {@code originalDim} components of the result (i.e. the reconstructed input) into
     * {@code output} (length {@code originalDim}). {@code scratch} has length {@code paddedDim}
     * and is used to hold the rotation-reversed vector before truncation.
     */
    public void reverseAndTruncate(float[] paddedRotated, float[] scratch, float[] output) {
        if (paddedRotated.length != paddedDim) {
            throw new IllegalArgumentException("expected padded length " + paddedDim + ", got " + paddedRotated.length);
        }
        if (output.length != originalDim) {
            throw new IllegalArgumentException("expected output length " + originalDim + ", got " + output.length);
        }
        rotation.applyTranspose(paddedRotated, scratch);
        System.arraycopy(scratch, 0, output, 0, originalDim);
    }
}
