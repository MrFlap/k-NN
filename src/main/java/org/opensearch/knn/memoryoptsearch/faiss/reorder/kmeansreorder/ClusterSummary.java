/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder.kmeansreorder;

/**
 * Cluster summary for one field in one segment. Contains sufficient statistics
 * to merge clusters across segments without re-reading vectors.
 */
public class ClusterSummary {
    public final int k;
    public final int dimension;
    public final int metricType;
    public final int numVectors;
    public final float[][] centroids;    // [k][dimension]
    public final float[][] linearSums;   // [k][dimension]
    public final int[] counts;           // [k]
    public final float[] sumOfSquares;   // [k]

    public ClusterSummary(
        int k,
        int dimension,
        int metricType,
        int numVectors,
        float[][] centroids,
        float[][] linearSums,
        int[] counts,
        float[] sumOfSquares
    ) {
        this.k = k;
        this.dimension = dimension;
        this.metricType = metricType;
        this.numVectors = numVectors;
        this.centroids = centroids;
        this.linearSums = linearSums;
        this.counts = counts;
        this.sumOfSquares = sumOfSquares;
    }

    /**
     * Build a ClusterSummary from raw k-means results and the original vectors.
     */
    public static ClusterSummary fromAssignments(
        float[][] vectors,
        int[] assignments,
        float[] distances,
        float[][] centroids,
        int k,
        int dimension,
        int metricType
    ) {
        int n = vectors.length;
        float[][] linearSums = new float[k][dimension];
        int[] counts = new int[k];
        float[] sumOfSquares = new float[k];

        for (int i = 0; i < n; i++) {
            int c = assignments[i];
            counts[c]++;
            sumOfSquares[c] += distances[i];
            for (int d = 0; d < dimension; d++) {
                linearSums[c][d] += vectors[i][d];
            }
        }
        return new ClusterSummary(k, dimension, metricType, n, centroids, linearSums, counts, sumOfSquares);
    }
}
