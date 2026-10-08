/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer.QuantizationResult;
import org.opensearch.knn.KNNTestCase;

import java.util.Arrays;

public class Int8ScalarQuantizerTests extends KNNTestCase {

    private static final float[] ZERO_CENTROID = new float[0];

    // Lucene's UNSIGNED_BYTE decoding: lower + code * (upper - lower) / 255.
    private static float decode(final byte code, final QuantizationResult result) {
        final float step = (result.upperInterval() - result.lowerInterval()) / 255f;
        return result.lowerInterval() + (code & 0xFF) * step;
    }

    private float[] randomVector(final int dimension, final float magnitude) {
        final float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            vector[i] = (random().nextFloat() * 2 - 1) * magnitude;
        }
        return vector;
    }

    private static QuantizationResult quantize(final VectorSimilarityFunction similarity, final float[] vector, final byte[] codes) {
        return new Int8ScalarQuantizer(similarity).scalarQuantize(vector, codes, Int8ScalarQuantizer.BITS, ZERO_CENTROID);
    }

    public void testQuantize_whenRandomVector_thenDecodedValueIsWithinHalfStep() {
        for (int iteration = 0; iteration < 50; iteration++) {
            final int dimension = 1 + random().nextInt(300);
            final float[] vector = randomVector(dimension, 0.01f + random().nextFloat() * 100);
            final byte[] codes = new byte[dimension];
            final QuantizationResult result = quantize(VectorSimilarityFunction.EUCLIDEAN, vector, codes);

            final float step = (result.upperInterval() - result.lowerInterval()) / 255f;
            for (int i = 0; i < dimension; i++) {
                // Half a step is the rounding error; the slack covers the float arithmetic on top of it.
                assertEquals(vector[i], decode(codes[i], result), step * 0.5f + Math.ulp(vector[i]) * 4);
            }
        }
    }

    public void testQuantize_whenNegatedVector_thenCodesAreMirroredAroundTheOffset() {
        final float[] vector = randomVector(64, 3f);
        final float[] negated = new float[vector.length];
        for (int i = 0; i < vector.length; i++) {
            negated[i] = -vector[i];
        }
        final byte[] codes = new byte[vector.length];
        final byte[] negatedCodes = new byte[vector.length];
        final QuantizationResult result = quantize(VectorSimilarityFunction.DOT_PRODUCT, vector, codes);
        final QuantizationResult negatedResult = quantize(VectorSimilarityFunction.DOT_PRODUCT, negated, negatedCodes);

        // Symmetric quantization: a code and the code of its negation are equidistant from the 128 offset.
        for (int i = 0; i < vector.length; i++) {
            assertEquals(256, (codes[i] & 0xFF) + (negatedCodes[i] & 0xFF));
        }
        assertEquals(result.lowerInterval(), negatedResult.lowerInterval(), 0f);
        assertEquals(result.upperInterval(), negatedResult.upperInterval(), 0f);
    }

    public void testQuantize_whenSignsDiffer_thenSignsAreKept() {
        final float[] vector = { -5f, 5f, -0.5f, 0.5f, 0f };
        final byte[] codes = new byte[vector.length];
        quantize(VectorSimilarityFunction.EUCLIDEAN, vector, codes);

        assertTrue((codes[0] & 0xFF) < 128);
        assertTrue((codes[1] & 0xFF) > 128);
        assertTrue((codes[2] & 0xFF) < 128);
        assertTrue((codes[3] & 0xFF) > 128);
        assertEquals(128, codes[4] & 0xFF);
    }

    public void testQuantize_whenLargestMagnitude_thenMapsToTheExtremeCodes() {
        final float[] vector = { 2f, -8f, 1f, 4f };
        final byte[] codes = new byte[vector.length];
        quantize(VectorSimilarityFunction.EUCLIDEAN, vector, codes);

        // -128 is deliberately unused so the range stays symmetric: the extremes are 1 and 255.
        assertEquals(1, codes[1] & 0xFF);
        final float[] positive = { 2f, 8f, 1f, 4f };
        quantize(VectorSimilarityFunction.EUCLIDEAN, positive, codes);
        assertEquals(255, codes[1] & 0xFF);
    }

    public void testQuantize_correctiveTermsMatchLuceneUnsignedByteConvention() {
        final float[] vector = randomVector(100, 7f);
        final byte[] codes = new byte[vector.length];
        final QuantizationResult result = quantize(VectorSimilarityFunction.EUCLIDEAN, vector, codes);

        float maxAbs = 0;
        float norm2 = 0;
        for (final float v : vector) {
            maxAbs = Math.max(maxAbs, Math.abs(v));
            norm2 += v * v;
        }
        final float scale = maxAbs / 127f;
        assertEquals(-128 * scale, result.lowerInterval(), 1e-6f);
        assertEquals(127 * scale, result.upperInterval(), 1e-6f);
        // Lucene derives the per-code step as (upper - lower) / 255, which has to be the scale.
        assertEquals(scale, (result.upperInterval() - result.lowerInterval()) / 255f, 1e-6f);
        assertEquals(norm2, result.additionalCorrection(), 1e-3f);
        int expectedSum = 0;
        for (final byte code : codes) {
            expectedSum += code & 0xFF;
        }
        assertEquals(expectedSum, result.quantizedComponentSum());
    }

    public void testQuantize_whenNotEuclidean_thenAdditionalCorrectionIsZero() {
        // With a zero centroid the dot product of the vector with the centroid is zero.
        for (final VectorSimilarityFunction similarity : new VectorSimilarityFunction[] {
            VectorSimilarityFunction.DOT_PRODUCT,
            VectorSimilarityFunction.COSINE,
            VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT }) {
            final float[] vector = randomVector(32, 1f);
            final QuantizationResult result = quantize(similarity, vector, new byte[vector.length]);
            assertEquals(similarity.name(), 0f, result.additionalCorrection(), 0f);
        }
    }

    public void testQuantize_whenZeroVector_thenAllCodesAreTheOffset() {
        final float[] vector = new float[17];
        final byte[] codes = new byte[vector.length];
        final QuantizationResult result = quantize(VectorSimilarityFunction.EUCLIDEAN, vector, codes);

        for (final byte code : codes) {
            assertEquals(128, code & 0xFF);
        }
        assertEquals(128 * vector.length, result.quantizedComponentSum());
        assertEquals(0f, result.additionalCorrection(), 0f);
        for (final byte code : codes) {
            assertEquals(0f, decode(code, result), 0f);
        }
    }

    public void testQuantize_ignoresCentroid() {
        final float[] vector = randomVector(40, 2f);
        final byte[] withoutCentroid = new byte[vector.length];
        final byte[] withCentroid = new byte[vector.length];
        final Int8ScalarQuantizer quantizer = new Int8ScalarQuantizer(VectorSimilarityFunction.EUCLIDEAN);
        final QuantizationResult first = quantizer.scalarQuantize(vector, withoutCentroid, Int8ScalarQuantizer.BITS, null);
        final float[] centroid = randomVector(vector.length, 5f);
        final QuantizationResult second = quantizer.scalarQuantize(vector, withCentroid, Int8ScalarQuantizer.BITS, centroid);

        assertArrayEquals(withoutCentroid, withCentroid);
        assertEquals(first, second);
    }

    public void testQuantize_doesNotModifyTheInput() {
        final float[] vector = randomVector(40, 2f);
        final float[] copy = Arrays.copyOf(vector, vector.length);
        quantize(VectorSimilarityFunction.EUCLIDEAN, vector, new byte[vector.length]);
        assertArrayEquals(copy, vector, 0f);
    }

    public void testQuantize_whenBitsIsNotEight_thenThrows() {
        final Int8ScalarQuantizer quantizer = new Int8ScalarQuantizer(VectorSimilarityFunction.EUCLIDEAN);
        for (final byte bits : new byte[] { 1, 4, 7 }) {
            expectThrows(IllegalArgumentException.class, () -> quantizer.scalarQuantize(new float[4], new byte[4], bits, ZERO_CENTROID));
        }
    }

    public void testQuantize_whenDestinationTooShort_thenThrows() {
        final Int8ScalarQuantizer quantizer = new Int8ScalarQuantizer(VectorSimilarityFunction.EUCLIDEAN);
        expectThrows(
            IllegalArgumentException.class,
            () -> quantizer.scalarQuantize(new float[8], new byte[7], Int8ScalarQuantizer.BITS, ZERO_CENTROID)
        );
    }

    // The scale comes from the largest magnitude, so a single outlier coarsens everything else. That is the
    // trade-off of the simple scheme and the reason the random orthogonal transform helps it: it spreads an outlier
    // across the components. Documented here so a change in that behavior is a conscious one.
    public void testQuantize_whenOutlier_thenOtherComponentsLoseResolution() {
        final float[] flat = new float[64];
        Arrays.fill(flat, 1f);
        final float[] withOutlier = Arrays.copyOf(flat, flat.length);
        withOutlier[0] = 1000f;

        final QuantizationResult flatResult = quantize(VectorSimilarityFunction.EUCLIDEAN, flat, new byte[64]);
        final QuantizationResult outlierResult = quantize(VectorSimilarityFunction.EUCLIDEAN, withOutlier, new byte[64]);

        final float flatStep = (flatResult.upperInterval() - flatResult.lowerInterval()) / 255f;
        final float outlierStep = (outlierResult.upperInterval() - outlierResult.lowerInterval()) / 255f;
        assertTrue(outlierStep > 100 * flatStep);
    }
}
