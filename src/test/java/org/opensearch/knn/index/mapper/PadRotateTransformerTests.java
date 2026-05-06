/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.opensearch.knn.KNNTestCase;

public class PadRotateTransformerTests extends KNNTestCase {

    public void testPaddedDim_powerOfTwoInputStaysAtPadFactor() {
        // 128 * 4 = 512 is already a power of 2
        assertEquals(128 * PadRotateTransformer.PAD_FACTOR, PadRotateTransformer.paddedDim(128));
        // 1024 * 4 = 4096 is already a power of 2
        assertEquals(1024 * PadRotateTransformer.PAD_FACTOR, PadRotateTransformer.paddedDim(1024));
    }

    public void testPaddedDim_nonPowerOfTwoRoundsUp() {
        // 768 * 4 = 3072 -> round up to 4096
        assertEquals(4096, PadRotateTransformer.paddedDim(768));
        // 384 * 4 = 1536 -> round up to 2048
        assertEquals(2048, PadRotateTransformer.paddedDim(384));
        // 1536 * 4 = 6144 -> round up to 8192
        assertEquals(8192, PadRotateTransformer.paddedDim(1536));
    }

    public void testPadAndRotate_outputLengthMatchesPaddedDim() {
        float[] input = new float[768];
        for (int i = 0; i < input.length; i++) {
            input[i] = i + 1.0f;
        }

        float[] rotated = PadRotateTransformer.padAndRotate(input);

        assertEquals(PadRotateTransformer.paddedDim(768), rotated.length);
    }

    public void testPadAndRotate_preservesNorm() {
        float[] input = new float[768];
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) Math.sin(i);
        }
        float inputNorm = norm(input);

        float[] rotated = PadRotateTransformer.padAndRotate(input);

        assertEquals("pad+rotate must preserve norm", inputNorm, norm(rotated), 0.001f * inputNorm);
    }

    private static float norm(float[] v) {
        float sumSq = 0f;
        for (float value : v) {
            sumSq += value * value;
        }
        return (float) Math.sqrt(sumSq);
    }
}
