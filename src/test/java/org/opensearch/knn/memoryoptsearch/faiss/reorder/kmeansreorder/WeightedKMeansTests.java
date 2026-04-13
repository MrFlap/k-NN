/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder.kmeansreorder;

import org.opensearch.test.OpenSearchTestCase;

public class WeightedKMeansTests extends OpenSearchTestCase {

    public void testBasicClustering() {
        // 6 points in 2D, 2 natural clusters around (0,0) and (100,100)
        float[][] points = {
            {1, 1}, {2, 2}, {0, 0},
            {100, 100}, {101, 101}, {99, 99}
        };
        float[] weights = {10, 10, 10, 10, 10, 10};

        float[][] centroids = WeightedKMeans.cluster(points, weights, 2, 25, KMeansClusterer.METRIC_L2);
        assertEquals(2, centroids.length);

        // One centroid should be near (1,1), the other near (100,100)
        float[] c0 = centroids[0];
        float[] c1 = centroids[1];
        // Sort so c0 is the smaller one
        if (c0[0] > c1[0]) { float[] tmp = c0; c0 = c1; c1 = tmp; }
        assertTrue("Expected centroid near (1,1) but got (" + c0[0] + "," + c0[1] + ")", c0[0] < 10);
        assertTrue("Expected centroid near (100,100) but got (" + c1[0] + "," + c1[1] + ")", c1[0] > 90);
    }

    public void testWeightsAffectCentroids() {
        // Two points, one heavily weighted
        float[][] points = {{0, 0}, {10, 10}};
        float[] weights = {100, 1};

        float[][] centroids = WeightedKMeans.cluster(points, weights, 1, 10, KMeansClusterer.METRIC_L2);
        assertEquals(1, centroids.length);
        // Centroid should be very close to (0,0) due to heavy weight
        assertTrue("Centroid should be near (0,0)", centroids[0][0] < 1.0f);
    }

    public void testKGreaterThanN() {
        float[][] points = {{1, 2}, {3, 4}};
        float[] weights = {1, 1};

        float[][] centroids = WeightedKMeans.cluster(points, weights, 5, 10, KMeansClusterer.METRIC_L2);
        // Should return all points as centroids (no reduction)
        assertEquals(2, centroids.length);
    }

    public void testSinglePoint() {
        float[][] points = {{5, 5}};
        float[] weights = {1};

        float[][] centroids = WeightedKMeans.cluster(points, weights, 1, 10, KMeansClusterer.METRIC_L2);
        assertEquals(1, centroids.length);
        assertEquals(5.0f, centroids[0][0], 0.001f);
        assertEquals(5.0f, centroids[0][1], 0.001f);
    }

    public void testInnerProductMetric() {
        float[][] points = {
            {1, 0}, {0.9f, 0.1f},
            {0, 1}, {0.1f, 0.9f}
        };
        float[] weights = {1, 1, 1, 1};

        float[][] centroids = WeightedKMeans.cluster(points, weights, 2, 25, KMeansClusterer.METRIC_INNER_PRODUCT);
        assertEquals(2, centroids.length);
    }
}
