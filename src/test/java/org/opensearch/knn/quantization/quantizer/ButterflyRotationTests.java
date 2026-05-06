/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizer;

import org.opensearch.knn.KNNTestCase;

import java.util.Random;

public class ButterflyRotationTests extends KNNTestCase {

    private static final long SEED = 42L;

    public void testGeneratePlan_rejectsNonPowerOfTwo() {
        expectThrows(IllegalArgumentException.class, () -> ButterflyRotation.generatePlan(6, SEED));
        expectThrows(IllegalArgumentException.class, () -> ButterflyRotation.generatePlan(0, SEED));
        expectThrows(IllegalArgumentException.class, () -> ButterflyRotation.generatePlan(-8, SEED));
    }

    public void testGeneratePlan_shape() {
        int dimensions = 64;
        ButterflyRotation.Plan plan = ButterflyRotation.generatePlan(dimensions, SEED);
        assertEquals(dimensions, plan.dimensions);
        assertEquals(6, plan.numRounds); // log2(64)
    }

    public void testApply_preservesNorm() {
        int[] dims = { 2, 8, 64, 512 };
        float delta = 0.0001f;
        for (int dim : dims) {
            ButterflyRotation.Plan plan = ButterflyRotation.generatePlan(dim, SEED);
            float[] vector = randomVector(dim, 7L);
            float originalNorm = norm(vector);

            float[] rotated = ButterflyRotation.apply(vector, plan);

            assertEquals(dim, rotated.length);
            assertEquals("norm not preserved for dim=" + dim, originalNorm, norm(rotated), delta * originalNorm);
        }
    }

    public void testApply_deterministicFromSeed() {
        int dimensions = 128;
        float[] vector = randomVector(dimensions, 13L);

        ButterflyRotation.Plan planA = ButterflyRotation.generatePlan(dimensions, SEED);
        ButterflyRotation.Plan planB = ButterflyRotation.generatePlan(dimensions, SEED);

        float[] rotatedA = ButterflyRotation.apply(vector, planA);
        float[] rotatedB = ButterflyRotation.apply(vector, planB);

        assertArrayEquals(rotatedA, rotatedB, 0.0f);
    }

    public void testApply_differentSeedsProduceDifferentRotations() {
        int dimensions = 128;
        float[] vector = randomVector(dimensions, 13L);

        float[] rotatedA = ButterflyRotation.apply(vector, ButterflyRotation.generatePlan(dimensions, SEED));
        float[] rotatedB = ButterflyRotation.apply(vector, ButterflyRotation.generatePlan(dimensions, SEED + 1));

        boolean anyDifferent = false;
        for (int i = 0; i < dimensions; i++) {
            if (Math.abs(rotatedA[i] - rotatedB[i]) > 0.0001f) {
                anyDifferent = true;
                break;
            }
        }
        assertTrue("different seeds should produce different rotations", anyDifferent);
    }

    public void testApply_doesNotMutateInput() {
        int dimensions = 32;
        float[] vector = randomVector(dimensions, 21L);
        float[] snapshot = vector.clone();

        ButterflyRotation.apply(vector, ButterflyRotation.generatePlan(dimensions, SEED));

        assertArrayEquals(snapshot, vector, 0.0f);
    }

    public void testApply_preservesPairwiseDistance() {
        // Orthogonal transforms preserve all pairwise L2 distances, not just norms.
        int dimensions = 256;
        ButterflyRotation.Plan plan = ButterflyRotation.generatePlan(dimensions, SEED);

        float[] a = randomVector(dimensions, 101L);
        float[] b = randomVector(dimensions, 202L);
        float originalDistance = l2Distance(a, b);

        float[] rotatedA = ButterflyRotation.apply(a, plan);
        float[] rotatedB = ButterflyRotation.apply(b, plan);

        assertEquals("pairwise L2 distance not preserved", originalDistance, l2Distance(rotatedA, rotatedB), 0.0001f * originalDistance);
    }

    public void testApply_spreadsSignalFromZeroPaddedInput() {
        // With the largest-stride-first ordering, a vector padded with zeros in the back half
        // should have signal in every coordinate after one full rotation.
        int dimensions = 64;
        float[] padded = new float[dimensions];
        for (int i = 0; i < dimensions / 4; i++) {
            padded[i] = i + 1.0f;
        }

        float[] rotated = ButterflyRotation.apply(padded, ButterflyRotation.generatePlan(dimensions, SEED));

        int nonZero = 0;
        for (float v : rotated) {
            if (Math.abs(v) > 1e-6f) {
                nonZero++;
            }
        }
        // All coordinates reachable after log2(d) layers.
        assertEquals("every dimension should receive signal after full butterfly", dimensions, nonZero);
    }

    public void testApply_rejectsMismatchedInputLength() {
        ButterflyRotation.Plan plan = ButterflyRotation.generatePlan(8, SEED);
        expectThrows(IllegalArgumentException.class, () -> ButterflyRotation.apply(new float[7], plan));
        expectThrows(IllegalArgumentException.class, () -> ButterflyRotation.apply(new float[16], plan));
    }

    private static float[] randomVector(int dimensions, long seed) {
        Random random = new Random(seed);
        float[] vector = new float[dimensions];
        for (int i = 0; i < dimensions; i++) {
            vector[i] = (float) random.nextGaussian();
        }
        return vector;
    }

    private static float norm(float[] v) {
        float sumSq = 0f;
        for (float value : v) {
            sumSq += value * value;
        }
        return (float) Math.sqrt(sumSq);
    }

    private static float l2Distance(float[] a, float[] b) {
        float sumSq = 0f;
        for (int i = 0; i < a.length; i++) {
            float d = a[i] - b[i];
            sumSq += d * d;
        }
        return (float) Math.sqrt(sumSq);
    }
}
