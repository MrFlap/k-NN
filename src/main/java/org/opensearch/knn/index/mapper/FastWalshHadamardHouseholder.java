/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.Getter;

import java.util.SplittableRandom;

/**
 * Structured random orthogonal transform of R^d built from random sign flips and a recursive fast Walsh-Hadamard transform (FWHT) that
 * handles any {@code d} without padding.
 * <p>
 * The transform {@code T(n)} on a block of size {@code n} applies {@code T(n / 2)} to the two halves of its first {@code 2 * (n / 2)}
 * coordinates and combines them with the normalized butterfly {@code (a, b) -> ((a + b) / sqrt(2), (a - b) / sqrt(2))}. When
 * {@code n} is odd, the last coordinate is left unpaired, so a Householder reflection then swaps it with a flat, random-sign unit
 * vector {@code f}, spreading it across the whole block. {@code f} has random signs because the reflection also maps {@code f} back
 * onto the last coordinate, and the butterflies produce exactly flat same-sign vectors.
 * <p>
 * Each round applies random signs followed by {@code T(d)}. Every step is orthogonal, so the transform preserves norms, inner products
 * and L2 distances without changing the dimension. Cost per round is at most about {@code 6 d log2(d)} flops. All coefficients are
 * derived from {@code (dimension, seed, rounds)}, so the same inputs always produce the same transform. Instances are immutable and
 * thread safe, and applying it allocates nothing.
 * <p>
 * This is not a rotation in the strict sense: the butterflies, reflections and sign flips can have determinant -1. Only
 * orthogonality matters for preserving distances.
 */
public final class FastWalshHadamardHouseholder {
    public static final int DEFAULT_ROUNDS = 3;

    private static final float INV_SQRT2 = (float) (1.0 / Math.sqrt(2.0));

    @Getter
    private final int dimension;
    // Per round: random signs applied before the transform.
    private final float[][] roundSigns;
    // Block size at each recursion depth: levelSizes[0] = dimension, levelSizes[j + 1] = levelSizes[j] / 2, down to 1.
    private final int[] levelSizes;
    // Per depth with an odd block size: signs of the reflection target f = signs / sqrt(n), otherwise null.
    private final float[][] targetSigns;
    // Per depth with an odd block size: 1 / sqrt(n).
    private final float[] invSqrtSizes;
    // Per depth with an odd block size: 2 / ||e_last - f||^2.
    private final float[] reflectionScales;

    public FastWalshHadamardHouseholder(final int dimension, final long seed) {
        this(dimension, seed, DEFAULT_ROUNDS);
    }

    public FastWalshHadamardHouseholder(final int dimension, final long seed, final int rounds) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("Dimension must be positive, got " + dimension);
        }
        if (rounds <= 0) {
            throw new IllegalArgumentException("Rounds must be positive, got " + rounds);
        }
        this.dimension = dimension;

        final SplittableRandom random = new SplittableRandom(seed);
        this.roundSigns = new float[rounds][];
        for (int round = 0; round < rounds; round++) {
            roundSigns[round] = randomSigns(random, dimension);
        }

        final int depth = Integer.SIZE - Integer.numberOfLeadingZeros(dimension);
        this.levelSizes = new int[depth];
        this.targetSigns = new float[depth][];
        this.invSqrtSizes = new float[depth];
        this.reflectionScales = new float[depth];
        for (int level = 0, n = dimension; level < depth; level++, n >>= 1) {
            levelSizes[level] = n;
            if (n > 1 && (n & 1) == 1) {
                final float[] signs = randomSigns(random, n);
                // Fixing f's last sign to -1 keeps ||e_last - f||^2 = 2 + 2 / sqrt(n) away from zero.
                signs[n - 1] = -1f;
                final double invSqrt = 1.0 / Math.sqrt(n);
                targetSigns[level] = signs;
                invSqrtSizes[level] = (float) invSqrt;
                reflectionScales[level] = (float) (2.0 / (2.0 + 2.0 * invSqrt));
            }
        }
    }

    /**
     * Transforms the given vector in place.
     *
     * @param vector vector of length {@link #getDimension()}
     */
    public void apply(final float[] vector) {
        validate(vector);
        for (final float[] signs : roundSigns) {
            for (int i = 0; i < dimension; i++) {
                vector[i] *= signs[i];
            }
            transform(vector, 0, 0);
        }
    }

    /**
     * Undoes {@link #apply(float[])} in place.
     *
     * @param vector vector of length {@link #getDimension()}
     */
    public void applyInverse(final float[] vector) {
        validate(vector);
        for (int round = roundSigns.length - 1; round >= 0; round--) {
            inverseTransform(vector, 0, 0);
            final float[] signs = roundSigns[round];
            for (int i = 0; i < dimension; i++) {
                vector[i] *= signs[i];
            }
        }
    }

    private void validate(final float[] vector) {
        if (vector == null || vector.length != dimension) {
            throw new IllegalArgumentException(
                "Vector must have dimension " + dimension + ", got " + (vector == null ? "null" : vector.length)
            );
        }
    }

    /**
     * Applies {@code T(levelSizes[level])} to the block of {@code vector} starting at {@code offset}.
     */
    private void transform(final float[] vector, final int offset, final int level) {
        final int n = levelSizes[level];
        if (n == 1) {
            return;
        }
        final int half = n >> 1;
        transform(vector, offset, level + 1);
        transform(vector, offset + half, level + 1);
        for (int i = offset, j = offset + half; i < offset + half; i++, j++) {
            final float a = vector[i];
            final float b = vector[j];
            vector[i] = (a + b) * INV_SQRT2;
            vector[j] = (a - b) * INV_SQRT2;
        }
        if ((n & 1) == 1) {
            reflect(vector, offset, n, level);
        }
    }

    /**
     * Undoes {@link #transform}. The reflection, the normalized butterfly and the child transforms' inverses are applied in
     * reverse order; the reflection and butterfly are their own inverses.
     */
    private void inverseTransform(final float[] vector, final int offset, final int level) {
        final int n = levelSizes[level];
        if (n == 1) {
            return;
        }
        if ((n & 1) == 1) {
            reflect(vector, offset, n, level);
        }
        final int half = n >> 1;
        for (int i = offset, j = offset + half; i < offset + half; i++, j++) {
            final float a = vector[i];
            final float b = vector[j];
            vector[i] = (a + b) * INV_SQRT2;
            vector[j] = (a - b) * INV_SQRT2;
        }
        inverseTransform(vector, offset, level + 1);
        inverseTransform(vector, offset + half, level + 1);
    }

    /**
     * Applies the Householder reflection {@code y -> y - 2 (u . y) u} with {@code u = (e_last - f) / ||e_last - f||} to the block
     * {@code vector[offset, offset + n)}, without materializing {@code u}. It swaps the block's last coordinate with {@code f}.
     */
    private void reflect(final float[] vector, final int offset, final int n, final int level) {
        final float[] signs = targetSigns[level];
        final float invSqrt = invSqrtSizes[level];
        final int last = offset + n - 1;
        float dot = 0;
        for (int i = 0; i < n; i++) {
            dot += signs[i] * vector[offset + i];
        }
        // y - 2 (u . y) u = y - alpha * (e_last - f), with alpha = 2 (y_last - f . y) / ||e_last - f||^2.
        final float alpha = (vector[last] - dot * invSqrt) * reflectionScales[level];
        final float beta = alpha * invSqrt;
        for (int i = 0; i < n; i++) {
            vector[offset + i] += beta * signs[i];
        }
        vector[last] -= alpha;
    }

    private static float[] randomSigns(final SplittableRandom random, final int n) {
        final float[] signs = new float[n];
        for (int i = 0; i < n; i++) {
            signs[i] = random.nextBoolean() ? 1f : -1f;
        }
        return signs;
    }
}
