/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.quantization.quantizer;

import lombok.experimental.UtilityClass;

import java.util.Random;

/**
 * Structured orthogonal rotation built from log2(d) layers of independent 2x2 Givens rotations.
 *
 * Each layer pairs indices at a fixed stride and rotates them by an independent uniform random
 * angle. Layers run largest-stride to smallest-stride: (d/2, d/4, ..., 2, 1). With that ordering
 * signal spreads across the full space after the first layer, which matters when the input is
 * zero-padded: round 0 mixes (i, i + d/2), immediately carrying signal into the zero region.
 *
 * Cost: (d/2) * log2(d) pairwise rotations per vector = O(d log d), vs O(d^2) for a dense
 * orthogonal matrix. Setup is O(d log d) angles, vs O(d^3) Gram-Schmidt.
 *
 * Orthogonality is exact (product of orthogonal 2x2 blocks), so ||Mx|| = ||x|| modulo float error.
 * It is not Haar-uniform on O(d) - only ~d log(d) degrees of freedom vs d(d-1)/2 - but for
 * variance equalization prior to scalar quantization that difference has not been observed to
 * matter (cf. Fastfood, Orthogonal Random Features).
 */
@UtilityClass
public class ButterflyRotation {

    /**
     * Precomputed cos/sin values for every Givens rotation across all layers. Layout is
     * flat row-major: index {@code round * (dimensions/2) + pairIndex}.
     */
    public static final class Plan {
        public final int dimensions;
        public final int numRounds;
        final float[] cos;
        final float[] sin;

        Plan(int dimensions, int numRounds, float[] cos, float[] sin) {
            this.dimensions = dimensions;
            this.numRounds = numRounds;
            this.cos = cos;
            this.sin = sin;
        }
    }

    /**
     * Build a rotation plan for a power-of-two dimension, deterministic from the given seed.
     */
    public Plan generatePlan(final int dimensions, final long seed) {
        if (dimensions <= 0 || (dimensions & (dimensions - 1)) != 0) {
            throw new IllegalArgumentException("dimensions must be a positive power of 2, got " + dimensions);
        }
        final int numRounds = Integer.numberOfTrailingZeros(dimensions);
        final int pairsPerRound = dimensions / 2;
        final int total = numRounds * pairsPerRound;
        final float[] cos = new float[total];
        final float[] sin = new float[total];
        final Random random = new Random(seed);
        final double twoPi = 2.0 * Math.PI;
        for (int i = 0; i < total; i++) {
            final double theta = random.nextDouble() * twoPi;
            cos[i] = (float) Math.cos(theta);
            sin[i] = (float) Math.sin(theta);
        }
        return new Plan(dimensions, numRounds, cos, sin);
    }

    /**
     * Apply the rotation in-place. No allocations.
     */
    public void applyInPlace(final float[] vec, final Plan plan) {
        if (vec.length != plan.dimensions) {
            throw new IllegalArgumentException("input length " + vec.length + " != plan dimensions " + plan.dimensions);
        }
        final int pairsPerRound = plan.dimensions / 2;
        for (int round = 0; round < plan.numRounds; round++) {
            final int stride = 1 << (plan.numRounds - 1 - round);
            int pairIdx = round * pairsPerRound;
            for (int blockStart = 0; blockStart < plan.dimensions; blockStart += stride << 1) {
                for (int j = 0; j < stride; j++) {
                    final int idxA = blockStart + j;
                    final int idxB = idxA + stride;
                    final float a = vec[idxA];
                    final float b = vec[idxB];
                    final float c = plan.cos[pairIdx];
                    final float s = plan.sin[pairIdx];
                    vec[idxA] = c * a - s * b;
                    vec[idxB] = s * a + c * b;
                    pairIdx++;
                }
            }
        }
    }

    /**
     * Apply the rotation to a copy of {@code input} and return the rotated vector. The input
     * array is not modified. Prefer {@link #applyInPlace} on hot paths to avoid the copy.
     */
    public float[] apply(final float[] input, final Plan plan) {
        final float[] vec = input.clone();
        applyInPlace(vec, plan);
        return vec;
    }
}
