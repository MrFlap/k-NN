/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.padrotate;

import org.opensearch.knn.KNNTestCase;

import java.util.Random;

public class PadRotateTests extends KNNTestCase {

    public void testRoundTripReconstructsInput() {
        int originalDim = 8;
        long seed = 12345L;
        PadRotate padRotate = new PadRotate(originalDim, seed);
        assertEquals(PadRotate.PAD_FACTOR * originalDim, padRotate.paddedDimensions());

        float[] input = new float[originalDim];
        Random r = new Random(42L);
        for (int i = 0; i < originalDim; i++) {
            input[i] = (float) r.nextGaussian();
        }

        float[] scratch = new float[padRotate.paddedDimensions()];
        float[] rotated = new float[padRotate.paddedDimensions()];
        padRotate.forward(input, scratch, rotated);

        float[] reconstructed = new float[originalDim];
        padRotate.reverseAndTruncate(rotated, scratch, reconstructed);

        for (int i = 0; i < originalDim; i++) {
            assertEquals("dim " + i, input[i], reconstructed[i], 1e-4f);
        }
    }

    public void testForwardPreservesNorm() {
        int originalDim = 16;
        PadRotate padRotate = new PadRotate(originalDim, 7L);

        float[] input = new float[originalDim];
        Random r = new Random(99L);
        for (int i = 0; i < originalDim; i++) {
            input[i] = (float) r.nextGaussian();
        }

        float[] scratch = new float[padRotate.paddedDimensions()];
        float[] rotated = new float[padRotate.paddedDimensions()];
        padRotate.forward(input, scratch, rotated);

        double inputNorm = 0;
        for (float v : input) inputNorm += v * v;
        double rotatedNorm = 0;
        for (float v : rotated) rotatedNorm += v * v;

        assertEquals(Math.sqrt(inputNorm), Math.sqrt(rotatedNorm), 1e-3);
    }

    public void testDifferentSeedsProduceDifferentRotations() {
        int originalDim = 8;
        float[] input = new float[originalDim];
        Random r = new Random(1L);
        for (int i = 0; i < originalDim; i++) input[i] = (float) r.nextGaussian();

        PadRotate a = new PadRotate(originalDim, 1L);
        PadRotate b = new PadRotate(originalDim, 2L);

        float[] scratch = new float[a.paddedDimensions()];
        float[] ra = new float[a.paddedDimensions()];
        float[] rb = new float[b.paddedDimensions()];
        a.forward(input, scratch, ra);
        b.forward(input, scratch, rb);

        boolean anyDifferent = false;
        for (int i = 0; i < ra.length; i++) {
            if (Math.abs(ra[i] - rb[i]) > 1e-4f) {
                anyDifferent = true;
                break;
            }
        }
        assertTrue("different seeds should produce different rotations", anyDifferent);
    }
}
