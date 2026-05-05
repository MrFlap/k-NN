/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.knn.index.mapper;

import org.opensearch.knn.quantization.quantizer.RandomGaussianRotation;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * POC-only: pads a D-length float vector with 3*D zeros (to 4*D), then applies a
 * fixed random rotation. Used at both index- and query-time so indexed and query
 * vectors live in the same rotated 4*D space before Lucene scalar quantization runs.
 *
 * Rotation matrix is cached per padded-dim and deterministic (fixed seed via
 * RandomGaussianRotation), so all shards/nodes produce identical rotations.
 */
public final class PadRotateTransformer {

    public static final int PAD_FACTOR = 4;

    private static final ConcurrentMap<Integer, float[][]> ROTATION_CACHE = new ConcurrentHashMap<>();

    private PadRotateTransformer() {}

    public static int paddedDim(int originalDim) {
        return originalDim * PAD_FACTOR;
    }

    public static float[] padAndRotate(float[] input) {
        final int paddedDim = paddedDim(input.length);
        final float[] padded = new float[paddedDim];
        System.arraycopy(input, 0, padded, 0, input.length);
        final float[][] rotation = ROTATION_CACHE.computeIfAbsent(paddedDim, RandomGaussianRotation::generateRotationMatrix);
        return RandomGaussianRotation.applyRotation(padded, rotation);
    }
}
