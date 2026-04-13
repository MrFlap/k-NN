/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder.kmeansreorder;

import java.util.Random;

/**
 * Pure-Java weighted k-means for merging pooled centroids. Only called when
 * pool_size > k_max (typically a few thousand points at most).
 */
public class WeightedKMeans {

    /**
     * Run weighted k-means on a small set of weighted points.
     *
     * @param points     [n][d] — the pooled centroids from source segments
     * @param weights    [n] — weight of each point (cluster count from source)
     * @param k          target number of output clusters
     * @param niter      number of iterations
     * @param metricType METRIC_L2 (0) or METRIC_INNER_PRODUCT (1)
     * @return [k][d] — the new centroids
     */
    public static float[][] cluster(float[][] points, float[] weights, int k, int niter, int metricType) {
        int n = points.length;
        int d = points[0].length;
        if (k >= n) {
            // Nothing to reduce — return a copy
            float[][] copy = new float[n][];
            for (int i = 0; i < n; i++) {
                copy[i] = points[i].clone();
            }
            return copy;
        }

        float[][] centroids = initWeightedKMeansPlusPlus(points, weights, k, metricType);

        for (int iter = 0; iter < niter; iter++) {
            // Assign
            int[] assignments = new int[n];
            for (int i = 0; i < n; i++) {
                assignments[i] = nearest(points[i], centroids, metricType);
            }

            // Update as weighted mean
            float[][] newCentroids = new float[k][d];
            float[] totalWeight = new float[k];
            for (int i = 0; i < n; i++) {
                int c = assignments[i];
                totalWeight[c] += weights[i];
                for (int j = 0; j < d; j++) {
                    newCentroids[c][j] += weights[i] * points[i][j];
                }
            }
            for (int c = 0; c < k; c++) {
                if (totalWeight[c] > 0) {
                    for (int j = 0; j < d; j++) {
                        newCentroids[c][j] /= totalWeight[c];
                    }
                } else {
                    // Dead centroid — keep previous
                    System.arraycopy(centroids[c], 0, newCentroids[c], 0, d);
                }
            }
            centroids = newCentroids;
        }
        return centroids;
    }

    static int nearest(float[] point, float[][] centroids, int metricType) {
        float bestDist = Float.MAX_VALUE;
        int best = 0;
        for (int c = 0; c < centroids.length; c++) {
            float dist = distance(point, centroids[c], metricType);
            if (dist < bestDist) {
                bestDist = dist;
                best = c;
            }
        }
        return best;
    }

    static float distance(float[] a, float[] b, int metricType) {
        if (metricType == KMeansClusterer.METRIC_INNER_PRODUCT) {
            float dot = 0;
            for (int i = 0; i < a.length; i++) dot += a[i] * b[i];
            return -dot;
        } else {
            float sum = 0;
            for (int i = 0; i < a.length; i++) {
                float diff = a[i] - b[i];
                sum += diff * diff;
            }
            return sum;
        }
    }

    private static float[][] initWeightedKMeansPlusPlus(float[][] points, float[] weights, int k, int metricType) {
        int n = points.length;
        int d = points[0].length;
        float[][] centroids = new float[k][d];
        Random rand = new Random(42);

        // First centroid: weighted random
        int first = weightedRandomIndex(weights, rand);
        System.arraycopy(points[first], 0, centroids[0], 0, d);

        float[] minDist = new float[n];
        for (int i = 0; i < n; i++) minDist[i] = Float.MAX_VALUE;

        for (int c = 1; c < k; c++) {
            for (int i = 0; i < n; i++) {
                float dist = distance(points[i], centroids[c - 1], metricType);
                if (dist < minDist[i]) minDist[i] = dist;
            }
            // Weighted by distance * point weight
            float sum = 0;
            for (int i = 0; i < n; i++) sum += minDist[i] * weights[i];
            float r = rand.nextFloat() * sum;
            float cumSum = 0;
            int next = 0;
            for (int i = 0; i < n; i++) {
                cumSum += minDist[i] * weights[i];
                if (cumSum >= r) {
                    next = i;
                    break;
                }
            }
            System.arraycopy(points[next], 0, centroids[c], 0, d);
        }
        return centroids;
    }

    private static int weightedRandomIndex(float[] weights, Random rand) {
        float sum = 0;
        for (float w : weights) sum += w;
        float r = rand.nextFloat() * sum;
        float cumSum = 0;
        for (int i = 0; i < weights.length; i++) {
            cumSum += weights[i];
            if (cumSum >= r) return i;
        }
        return weights.length - 1;
    }
}
