/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder.kmeansreorder;

import java.util.Arrays;

/**
 * Utility class for clustering vectors and sorting by cluster assignment.
 */
public class ClusterSorter {

    private static final boolean USE_JNI = Boolean.parseBoolean(System.getProperty("kmeans.useJni", "true"));

    /**
     * Sort indices by (cluster_id, distance_to_centroid).
     * @return newOrder where newOrder[newIdx] = oldIdx
     */
    public static int[] sortByCluster(int[] assignments, float[] distances, int metricType) {
        int n = assignments.length;
        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) indices[i] = i;

        final boolean reverseDistance = (metricType == KMeansClusterer.METRIC_INNER_PRODUCT);

        Arrays.sort(indices, (a, b) -> {
            int cmp = Integer.compare(assignments[a], assignments[b]);
            if (cmp != 0) return cmp;
            return reverseDistance
                ? Float.compare(distances[b], distances[a])
                : Float.compare(distances[a], distances[b]);
        });

        int[] newOrder = new int[n];
        for (int i = 0; i < n; i++) newOrder[i] = indices[i];
        return newOrder;
    }

    /**
     * Cluster vectors and return sorted order.
     * @return newOrder where newOrder[newIdx] = oldIdx
     */
    public static int[] clusterAndSort(float[][] vectors, int k, int niter, int metricType) {
        KMeansResult result;
        if (USE_JNI) {
            result = clusterWithJni(vectors, k, niter, metricType);
        } else {
            result = KMeansClusterer.cluster(vectors, k, niter, metricType);
        }
        return sortByCluster(result.assignments(), result.distances(), metricType);
    }

    private static KMeansResult clusterWithJni(float[][] vectors, int k, int niter, int metricType) {
        int n = vectors.length;
        int dim = vectors[0].length;
        long addr = FaissKMeansService.storeVectors(vectors);
        try {
            return FaissKMeansService.kmeansWithDistances(addr, n, dim, k, niter, metricType);
        } finally {
            FaissKMeansService.freeVectors(addr);
        }
    }

    /**
     * Cluster vectors with L2 metric and return sorted order.
     */
    public static int[] clusterAndSort(float[][] vectors, int k) {
        return clusterAndSort(vectors, k, 25, KMeansClusterer.METRIC_L2);
    }
}
