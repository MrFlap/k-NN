/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.opensearch.knn.KNNTestCase;

import java.util.Arrays;

public class FastWalshHadamardHouseholderTests extends KNNTestCase {
    private static final int[] DIMENSIONS = { 1, 2, 3, 5, 7, 8, 100, 128, 384, 513, 768, 999, 1000, 1023, 1024, 1025, 1536 };

    public void testRotate_isOrthogonalForSmallDimensions() {
        for (int d = 1; d <= 70; d++) {
            final FastWalshHadamardHouseholder rotation = new FastWalshHadamardHouseholder(d, d, 1);
            final float[][] columns = new float[d][];
            for (int i = 0; i < d; i++) {
                columns[i] = new float[d];
                columns[i][i] = 1f;
                rotation.apply(columns[i]);
            }
            for (int i = 0; i < d; i++) {
                for (int j = i; j < d; j++) {
                    assertEquals("d=" + d + " (" + i + ", " + j + ")", i == j ? 1.0 : 0.0, dot(columns[i], columns[j]), 1e-5);
                }
            }
        }
    }

    public void testRotate_preservesNormsAndInnerProducts() {
        for (int d : DIMENSIONS) {
            final FastWalshHadamardHouseholder rotation = new FastWalshHadamardHouseholder(d, randomLong());
            final float[] a = randomVector(d);
            final float[] b = randomVector(d);
            final double dot = dot(a, b);
            final double normA = dot(a, a);
            rotation.apply(a);
            rotation.apply(b);

            final double tolerance = 1e-4 * d;
            assertEquals("d=" + d, normA, dot(a, a), tolerance);
            assertEquals("d=" + d, dot, dot(a, b), tolerance);
        }
    }

    public void testRotate_spreadsSpikesForAnyDimension() {
        for (int d : new int[] { 100, 513, 768, 999, 1000, 1023, 1024, 1025 }) {
            final FastWalshHadamardHouseholder rotation = new FastWalshHadamardHouseholder(d, 17L);
            for (int spike : new int[] { 0, d / 2, d - 1 }) {
                final float[] vector = new float[d];
                vector[spike] = 1f;
                rotation.apply(vector);
                float max = 0;
                for (float v : vector) {
                    max = Math.max(max, Math.abs(v));
                }
                // A random rotation gives max |x| * sqrt(d) around sqrt(2 ln d) ~ 3.7; a poorly mixed one leaves ~10 or more.
                assertTrue("d=" + d + " spike=" + spike + " max*sqrt(d)=" + max * Math.sqrt(d), max * Math.sqrt(d) < 7);
            }
        }
    }

    public void testApplyInverse_restoresOriginal() {
        for (int d : DIMENSIONS) {
            final FastWalshHadamardHouseholder transform = new FastWalshHadamardHouseholder(d, randomLong());
            final float[] original = randomVector(d);
            final float[] vector = Arrays.copyOf(original, d);
            transform.apply(vector);
            transform.applyInverse(vector);
            assertArrayEquals("d=" + d, original, vector, 1e-5f);
        }
        expectThrows(IllegalArgumentException.class, () -> new FastWalshHadamardHouseholder(8, 0L).applyInverse(new float[7]));
    }

    public void testRotate_isDeterministicPerSeed() {
        final int d = 1000;
        final float[] input = randomVector(d);
        final float[] first = Arrays.copyOf(input, d);
        final float[] second = Arrays.copyOf(input, d);
        final float[] otherSeed = Arrays.copyOf(input, d);
        new FastWalshHadamardHouseholder(d, 11L).apply(first);
        new FastWalshHadamardHouseholder(d, 11L).apply(second);
        new FastWalshHadamardHouseholder(d, 12L).apply(otherSeed);

        assertArrayEquals(first, second, 0f);
        assertFalse(Arrays.equals(first, otherSeed));
    }

    public void testRotate_withWrongDimension_thenThrowsException() {
        final FastWalshHadamardHouseholder rotation = new FastWalshHadamardHouseholder(8, 0L);
        expectThrows(IllegalArgumentException.class, () -> rotation.apply(new float[7]));
        expectThrows(IllegalArgumentException.class, () -> rotation.apply(null));
    }

    public void testConstructor_withInvalidArguments_thenThrowsException() {
        expectThrows(IllegalArgumentException.class, () -> new FastWalshHadamardHouseholder(0, 0L));
        expectThrows(IllegalArgumentException.class, () -> new FastWalshHadamardHouseholder(8, 0L, 0));
    }

    public void testTransformer_transformsInPlaceOrCopy() {
        final int d = 100;
        final RandomOrthogonalVectorTransformer transformer = new RandomOrthogonalVectorTransformer(d, 5L);
        final float[] input = randomVector(d);
        final float[] original = Arrays.copyOf(input, d);

        final float[] copy = transformer.transform(input, false);
        assertNotSame(input, copy);
        assertArrayEquals(original, input, 0f);

        final float[] inPlace = transformer.transform(input, true);
        assertSame(input, inPlace);
        assertArrayEquals(copy, inPlace, 0f);

        expectThrows(IllegalArgumentException.class, () -> transformer.transform(null, true));
        expectThrows(UnsupportedOperationException.class, () -> transformer.transform(new byte[d]));
    }

    private static float[] randomVector(int d) {
        final float[] vector = new float[d];
        for (int i = 0; i < d; i++) {
            vector[i] = randomFloat() * 2 - 1;
        }
        return vector;
    }

    private static double dot(float[] a, float[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += (double) a[i] * b[i];
        }
        return sum;
    }
}
