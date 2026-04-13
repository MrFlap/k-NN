/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder.kmeansreorder;

import java.util.Random;

/**
 * Pure Java k-means clustering implementation.
 */
public class KMeansClusterer {

    public static final int METRIC_L2 = 0;
    public static final int METRIC_INNER_PRODUCT = 1;

    /**
     * Run k-means clustering.
     *
     * @param vectors 2D array [numVectors][dimension]
     * @param numClusters number of clusters (k)
     * @param numIterations number of iterations
     * @param metricType METRIC_L2 or METRIC_INNER_PRODUCT
     * @return KMeansResult with assignments and distances
     */
    public static KMeansResult cluster(float[][] vectors, int numClusters, int numIterations, int metricType) {
        int n = vectors.length;
        int d = vectors[0].length;

        // Initialize centroids using k-means++
        float[][] centroids = initCentroidsKMeansPlusPlus(vectors, numClusters, metricType);

        int[] assignments = new int[n];
        float[] distances = new float[n];

        for (int iter = 0; iter < numIterations; iter++) {
            // Assign vectors to nearest centroid
            for (int i = 0; i < n; i++) {
                float bestDist = Float.MAX_VALUE;
                int bestCluster = 0;
                for (int c = 0; c < numClusters; c++) {
                    float dist = distance(vectors[i], centroids[c], metricType);
                    if (dist < bestDist) {
                        bestDist = dist;
                        bestCluster = c;
                    }
                }
                assignments[i] = bestCluster;
                distances[i] = bestDist;
            }

            // Update centroids
            float[][] newCentroids = new float[numClusters][d];
            int[] counts = new int[numClusters];

            for (int i = 0; i < n; i++) {
                int c = assignments[i];
                counts[c]++;
                for (int j = 0; j < d; j++) {
                    newCentroids[c][j] += vectors[i][j];
                }
            }

            for (int c = 0; c < numClusters; c++) {
                if (counts[c] > 0) {
                    for (int j = 0; j < d; j++) {
                        centroids[c][j] = newCentroids[c][j] / counts[c];
                    }
                }
            }
        }

        return new KMeansResult(assignments, distances);
    }

    private static float[][] initCentroidsKMeansPlusPlus(float[][] vectors, int k, int metricType) {
        int n = vectors.length;
        int d = vectors[0].length;
        float[][] centroids = new float[k][d];
        Random rand = new Random(42);

        // First centroid: random
        int first = rand.nextInt(n);
        System.arraycopy(vectors[first], 0, centroids[0], 0, d);

        float[] minDist = new float[n];
        for (int i = 0; i < n; i++) {
            minDist[i] = Float.MAX_VALUE;
        }

        for (int c = 1; c < k; c++) {
            // Update min distances
            for (int i = 0; i < n; i++) {
                float dist = distance(vectors[i], centroids[c - 1], metricType);
                if (dist < minDist[i]) {
                    minDist[i] = dist;
                }
            }

            // Choose next centroid with probability proportional to distance squared
            float sum = 0;
            for (int i = 0; i < n; i++) {
                sum += minDist[i];
            }

            float r = rand.nextFloat() * sum;
            float cumSum = 0;
            int next = 0;
            for (int i = 0; i < n; i++) {
                cumSum += minDist[i];
                if (cumSum >= r) {
                    next = i;
                    break;
                }
            }
            System.arraycopy(vectors[next], 0, centroids[c], 0, d);
        }

        return centroids;
    }

    private static float distance(float[] a, float[] b, int metricType) {
        if (metricType == METRIC_INNER_PRODUCT) {
            // For inner product, we want to maximize, so return negative
            float dot = 0;
            for (int i = 0; i < a.length; i++) {
                dot += a[i] * b[i];
            }
            return -dot;
        } else {
            // L2 squared distance
            float sum = 0;
            for (int i = 0; i < a.length; i++) {
                float diff = a[i] - b[i];
                sum += diff * diff;
            }
            return sum;
        }
    }
}
