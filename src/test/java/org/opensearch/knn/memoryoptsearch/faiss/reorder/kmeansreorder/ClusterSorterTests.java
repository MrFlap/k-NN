/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder.kmeansreorder;

import org.opensearch.test.OpenSearchTestCase;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;

public class ClusterSorterTests extends OpenSearchTestCase {

    private void assertValidPermutation(int[] permutation, int expectedSize) {
        assertEquals(expectedSize, permutation.length);
        Set<Integer> seen = new HashSet<>();
        for (int ord : permutation) {
            assertTrue("ord out of range: " + ord, ord >= 0 && ord < expectedSize);
            assertTrue("duplicate ord: " + ord, seen.add(ord));
        }
    }

    public void testSortByClusterProducesValidPermutation() {
        int[] assignments = { 1, 0, 2, 0, 1 };
        float[] distances = { 0.5f, 0.1f, 0.3f, 0.2f, 0.4f };
        int[] result = ClusterSorter.sortByCluster(assignments, distances, KMeansClusterer.METRIC_L2);
        assertValidPermutation(result, 5);
    }

    public void testSortByClusterGroupsByCluster() {
        int[] assignments = { 2, 0, 1, 0, 2 };
        float[] distances = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f };
        int[] result = ClusterSorter.sortByCluster(assignments, distances, KMeansClusterer.METRIC_L2);

        // Verify output is sorted by cluster assignment
        for (int i = 0; i < result.length - 1; i++) {
            assertTrue(
                "cluster order violated at position " + i,
                assignments[result[i]] <= assignments[result[i + 1]]
            );
        }
    }

    public void testSortByClusterSortsByDistanceWithinCluster() {
        // All same cluster, different distances
        int[] assignments = { 0, 0, 0, 0 };
        float[] distances = { 3.0f, 1.0f, 4.0f, 2.0f };
        int[] result = ClusterSorter.sortByCluster(assignments, distances, KMeansClusterer.METRIC_L2);

        // Within cluster 0, should be sorted by ascending distance for L2
        for (int i = 0; i < result.length - 1; i++) {
            assertTrue(
                "distance order violated at position " + i,
                distances[result[i]] <= distances[result[i + 1]]
            );
        }
    }

    public void testKMeansClustererProducesValidResult() {
        float[][] vectors = generateClusteredVectors(200, 8, 3);
        KMeansResult result = KMeansClusterer.cluster(vectors, 3, 10, KMeansClusterer.METRIC_L2);

        assertEquals(vectors.length, result.assignments().length);
        assertEquals(vectors.length, result.distances().length);

        // All assignments should be in [0, k)
        for (int a : result.assignments()) {
            assertTrue("assignment out of range: " + a, a >= 0 && a < 3);
        }
        // All distances should be non-negative for L2
        for (float d : result.distances()) {
            assertTrue("negative distance: " + d, d >= 0);
        }
    }

    public void testClusterAndSortProducesValidPermutation() {
        // Use pure Java path (no JNI)
        System.setProperty("kmeans.useJni", "false");
        try {
            float[][] vectors = generateClusteredVectors(300, 8, 4);
            int[] result = ClusterSorter.clusterAndSort(vectors, 4);
            assertValidPermutation(result, vectors.length);
        } finally {
            System.clearProperty("kmeans.useJni");
        }
    }

    public void testClusterAndSortGroupsClusteredVectors() {
        System.setProperty("kmeans.useJni", "false");
        try {
            int perCluster = 100;
            int dim = 8;
            int k = 3;
            float[][] vectors = generateClusteredVectors(perCluster * k, dim, k);

            int[] permutation = ClusterSorter.clusterAndSort(vectors, k);
            assertValidPermutation(permutation, vectors.length);

            // After sorting, adjacent vectors should mostly be from the same original cluster.
            // Original cluster = index % k (from how we generated them).
            int sameClusterAdjacent = 0;
            for (int i = 0; i < permutation.length - 1; i++) {
                int curCluster = permutation[i] % k;
                int nextCluster = permutation[i + 1] % k;
                if (curCluster == nextCluster) {
                    sameClusterAdjacent++;
                }
            }
            float adjacencyRate = (float) sameClusterAdjacent / (permutation.length - 1);
            assertTrue(
                "Expected high same-cluster adjacency after kmeans reorder, got " + adjacencyRate,
                adjacencyRate > 0.8f
            );
        } finally {
            System.clearProperty("kmeans.useJni");
        }
    }

    public void testKMeansClustererWithSingleCluster() {
        float[][] vectors = generateClusteredVectors(50, 4, 1);
        KMeansResult result = KMeansClusterer.cluster(vectors, 1, 5, KMeansClusterer.METRIC_L2);

        // All should be assigned to cluster 0
        for (int a : result.assignments()) {
            assertEquals(0, a);
        }
    }

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
