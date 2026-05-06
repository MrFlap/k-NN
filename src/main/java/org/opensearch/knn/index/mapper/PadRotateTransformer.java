/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.index.mapper;

import org.opensearch.knn.quantization.quantizer.ButterflyRotation;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

import static org.opensearch.knn.common.KNNConstants.QUANTIZATION_RANDOM_ROTATION_DEFAULT_SEED;

/**
 * POC-only: pads a D-length float vector up to {@code PAD_FACTOR*D}, rounded up to the next
 * power of 2, then applies a fixed butterfly rotation (log2(paddedDim) layers of random 2x2
 * Givens rotations). Used at both index- and query-time so indexed and query vectors live in
 * the same rotated space before scalar quantization runs.
 *
 * PAD_FACTOR is a minimum multiplier. The butterfly requires a power-of-two dimension, so we
 * round up when PAD_FACTOR*D isn't already one (e.g. D=768, PAD_FACTOR=4 -> 3072 -> 4096).
 * Worst-case overhead is slightly under 2x vs the requested PAD_FACTOR.
 *
 * Rotation plan is cached per paddedDim and seeded from QUANTIZATION_RANDOM_ROTATION_DEFAULT_SEED,
 * so all shards/nodes produce identical rotations.
 */
public final class PadRotateTransformer {

    public static final int PAD_FACTOR = 4;

    private static final ConcurrentMap<Integer, ButterflyRotation.Plan> PLAN_CACHE = new ConcurrentHashMap<>();

    private PadRotateTransformer() {}

    public static int paddedDim(int originalDim) {
        final int minDim = originalDim * PAD_FACTOR;
        // Round up to next power of 2. For a power-of-2 input this is a no-op.
        return Integer.highestOneBit(minDim - 1) << 1;
    }

    public static float[] padAndRotate(float[] input) {
        final int paddedDim = paddedDim(input.length);
        // Single allocation: zero-fill region beyond input.length is implicit.
        final float[] padded = new float[paddedDim];
        System.arraycopy(input, 0, padded, 0, input.length);
        final ButterflyRotation.Plan plan = PLAN_CACHE.computeIfAbsent(
            paddedDim,
            dim -> ButterflyRotation.generatePlan(dim, QUANTIZATION_RANDOM_ROTATION_DEFAULT_SEED)
        );
        ButterflyRotation.applyInPlace(padded, plan);
        return padded;
    }
}
