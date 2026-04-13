/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder.bpreorder;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.test.OpenSearchTestCase;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;

public class BpReordererTests extends OpenSearchTestCase {

    /**
     * Verify that the permutation is valid: contains every ord exactly once.
     */
    private void assertValidPermutation(int[] permutation, int expectedSize) {
        assertEquals(expectedSize, permutation.length);
        Set<Integer> seen = new HashSet<>();
        for (int ord : permutation) {
            assertTrue("ord out of range: " + ord, ord >= 0 && ord < expectedSize);
            assertTrue("duplicate ord: " + ord, seen.add(ord));
        }
    }

    public void testComputePermutationProducesValidPermutation() {
        float[][] vectors = generateClusteredVectors(500, 32, 2);
        int[] permutation = BpReorderer.computePermutation(vectors);
        assertValidPermutation(permutation, vectors.length);
    }

    public void testComputePermutationWithEuclidean() {
        float[][] vectors = generateClusteredVectors(200, 16, 3);
        int[] permutation = BpReorderer.computePermutation(vectors, VectorSimilarityFunction.EUCLIDEAN);
        assertValidPermutation(permutation, vectors.length);
    }

    public void testSmallInput() {
        // Minimum viable input: 2 vectors
        float[][] vectors = { { 1.0f, 0.0f }, { 0.0f, 1.0f } };
        int[] permutation = BpReorderer.computePermutation(vectors);
        assertValidPermutation(permutation, 2);
    }

    public void testIdenticalVectors() {
        // All vectors the same — permutation should still be valid
        float[][] vectors = new float[100][8];
        for (float[] v : vectors) {
            java.util.Arrays.fill(v, 1.0f);
        }
        int[] permutation = BpReorderer.computePermutation(vectors);
        assertValidPermutation(permutation, 100);
    }

    public void testClusteredVectorsGroupedAfterReorder() {
        // Create two well-separated clusters and verify that after reordering,
        // vectors from the same cluster tend to be adjacent.
        int perCluster = 250;
        int dim = 16;
        float[][] vectors = new float[perCluster * 2][dim];
        Random rand = new Random(42);

        // Cluster A centered at origin, cluster B centered at (100, 100, ...)
        // Interleave them: even indices = cluster A, odd = cluster B
        for (int i = 0; i < perCluster; i++) {
            for (int d = 0; d < dim; d++) {
                vectors[2 * i][d] = rand.nextFloat();           // cluster A
                vectors[2 * i + 1][d] = 100.0f + rand.nextFloat(); // cluster B
            }
        }

        int[] permutation = BpReorderer.computePermutation(vectors);
        assertValidPermutation(permutation, vectors.length);

        // After reordering, count how many adjacent pairs are from the same cluster.
        // With interleaved input, random ordering would give ~50% same-cluster adjacency.
        // BP should achieve significantly higher.
        int sameClusterAdjacent = 0;
        for (int i = 0; i < permutation.length - 1; i++) {
            boolean curIsA = permutation[i] % 2 == 0;
            boolean nextIsA = permutation[i + 1] % 2 == 0;
            if (curIsA == nextIsA) {
                sameClusterAdjacent++;
            }
        }
        float adjacencyRate = (float) sameClusterAdjacent / (permutation.length - 1);
        // BP should group clusters together — expect >90% same-cluster adjacency
        assertTrue(
            "Expected high same-cluster adjacency after BP reorder, got " + adjacencyRate,
            adjacencyRate > 0.9f
        );
    }

    /**
     * Generate vectors in `numClusters` clusters, interleaved.
     */
    private float[][] generateClusteredVectors(int totalVectors, int dim, int numClusters) {
        float[][] vectors = new float[totalVectors][dim];
        Random rand = new Random(42);
        for (int i = 0; i < totalVectors; i++) {
            int cluster = i % numClusters;
            for (int d = 0; d < dim; d++) {
                vectors[i][d] = cluster * 100.0f + rand.nextFloat();
            }
        }
        return vectors;
    }
}
