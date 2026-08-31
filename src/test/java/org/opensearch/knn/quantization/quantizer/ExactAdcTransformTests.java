/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizer;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.plugin.script.KNNScoringUtil;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;

import java.util.Random;

/**
 * Verifies that the exact ADC transform, combined with the offset it returns, reproduces the true distance from the
 * full-precision query to the reconstruction implied by a document's bits.
 * <p>
 * This is the property that makes scores comparable across segments: two segments trained on different data produce
 * different thresholds, but both report a distance on the same absolute scale.
 */
public class ExactAdcTransformTests extends KNNTestCase {

    private static final int DIMENSION = 64;
    private static final float DELTA = 1e-3f;

    public void testExactAdcMatchesReconstructionDistanceForL2() {
        assertExactAdcMatchesReconstruction(SpaceType.L2, 42L);
    }

    public void testExactAdcMatchesReconstructionDistanceForInnerProduct() {
        assertExactAdcMatchesReconstruction(SpaceType.INNER_PRODUCT, 7L);
    }

    /**
     * Two segments with deliberately different thresholds must agree on the distance to the same underlying vector.
     * The legacy transform fails this because it drops the per-dimension (above - below) weighting.
     */
    public void testTwoSegmentsAgreeOnTheSameReconstruction() {
        final Random random = new Random(1234L);
        final float[] query = randomVector(random);

        // A vector both segments happen to quantize identically, so the reconstructions differ only via the state.
        final OneBitScalarQuantizationState wideState = state(0.0f, 1.0f);
        final OneBitScalarQuantizationState narrowState = state(0.4f, 0.6f);

        final byte[] bits = new byte[DIMENSION / 8];
        for (int i = 0; i < bits.length; i++) {
            bits[i] = (byte) random.nextInt(256);
        }

        final double wide = adcDistance(query, bits, wideState, SpaceType.L2);
        final double wideTruth = reconstructionL2(query, bits, wideState);
        final double narrow = adcDistance(query, bits, narrowState, SpaceType.L2);
        final double narrowTruth = reconstructionL2(query, bits, narrowState);

        assertEquals(wideTruth, wide, DELTA);
        assertEquals(narrowTruth, narrow, DELTA);
        // Sanity: the two segments really do disagree, so the test above is not vacuous.
        assertTrue("expected the two states to imply different distances", Math.abs(wideTruth - narrowTruth) > DELTA);
    }

    private void assertExactAdcMatchesReconstruction(final SpaceType spaceType, final long seed) {
        final Random random = new Random(seed);
        final OneBitScalarQuantizationState state = randomState(random);

        for (int trial = 0; trial < 20; trial++) {
            final float[] query = randomVector(random);
            final byte[] bits = new byte[DIMENSION / 8];
            for (int i = 0; i < bits.length; i++) {
                bits[i] = (byte) random.nextInt(256);
            }

            final double actual = adcDistance(query, bits, state, spaceType);
            final double expected = spaceType == SpaceType.L2
                ? reconstructionL2(query, bits, state)
                : reconstructionInnerProduct(query, bits, state);

            assertEquals("trial " + trial, expected, actual, DELTA);
        }
    }

    /** Runs the transform and the matching JVM ADC kernel, then applies the returned offset. */
    private double adcDistance(
        final float[] query,
        final byte[] bits,
        final OneBitScalarQuantizationState state,
        final SpaceType spaceType
    ) {
        final float[] transformed = query.clone();
        final float offset = new OneBitScalarQuantizer().transformWithADC(transformed, state, spaceType, true);
        final float raw = spaceType == SpaceType.L2
            ? KNNScoringUtil.l2SquaredADC(transformed, bits)
            : KNNScoringUtil.innerProductADC(transformed, bits);
        return raw + offset;
    }

    private double reconstructionL2(final float[] query, final byte[] bits, final OneBitScalarQuantizationState state) {
        double sum = 0.0;
        for (int i = 0; i < DIMENSION; i++) {
            final double diff = query[i] - reconstruct(bits, state, i);
            sum += diff * diff;
        }
        return sum;
    }

    private double reconstructionInnerProduct(final float[] query, final byte[] bits, final OneBitScalarQuantizationState state) {
        double sum = 0.0;
        for (int i = 0; i < DIMENSION; i++) {
            sum += (double) query[i] * reconstruct(bits, state, i);
        }
        return sum;
    }

    /** The value the ADC kernel implicitly scores against: below-mean when the bit is 0, above-mean when it is 1. */
    private float reconstruct(final byte[] bits, final OneBitScalarQuantizationState state, final int dimension) {
        final int bit = (bits[dimension / 8] >> (7 - (dimension % 8))) & 1;
        return bit == 1 ? state.getAboveThresholdMeans()[dimension] : state.getBelowThresholdMeans()[dimension];
    }

    private float[] randomVector(final Random random) {
        final float[] vector = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            vector[i] = (float) random.nextGaussian();
        }
        return vector;
    }

    private OneBitScalarQuantizationState randomState(final Random random) {
        final float[] thresholds = new float[DIMENSION];
        final float[] below = new float[DIMENSION];
        final float[] above = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            thresholds[i] = (float) random.nextGaussian();
            // Keep spreads uneven across dimensions - that is exactly what the legacy transform ignores.
            below[i] = thresholds[i] - (0.1f + random.nextFloat());
            above[i] = thresholds[i] + (0.1f + random.nextFloat());
        }
        return buildState(thresholds, below, above);
    }

    private OneBitScalarQuantizationState state(final float below, final float above) {
        final float[] thresholds = new float[DIMENSION];
        final float[] belows = new float[DIMENSION];
        final float[] aboves = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            thresholds[i] = (below + above) / 2f;
            belows[i] = below;
            aboves[i] = above;
        }
        return buildState(thresholds, belows, aboves);
    }

    private OneBitScalarQuantizationState buildState(final float[] thresholds, final float[] below, final float[] above) {
        return OneBitScalarQuantizationState.builder()
            .quantizationParams(ScalarQuantizationParams.builder().sqType(ScalarQuantizationType.ONE_BIT).build())
            .meanThresholds(thresholds)
            .belowThresholdMeans(below)
            .aboveThresholdMeans(above)
            .build();
    }
}
