/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder.kmeansreorder;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

public class KMeansReorderStrategyTests extends OpenSearchTestCase {

    @Override
    public void setUp() throws Exception {
        super.setUp();
        System.setProperty("kmeans.useJni", "false");
    }

    @Override
    public void tearDown() throws Exception {
        System.clearProperty("kmeans.useJni");
        super.tearDown();
    }

    private void assertValidPermutation(int[] permutation, int expectedSize) {
        assertEquals(expectedSize, permutation.length);
        Set<Integer> seen = new HashSet<>();
        for (int ord : permutation) {
            assertTrue("ord out of range: " + ord, ord >= 0 && ord < expectedSize);
            assertTrue("duplicate ord: " + ord, seen.add(ord));
        }
    }

    private FloatVectorValues toFloatVectorValues(float[][] vectors) {
        int dim = vectors[0].length;
        List<float[]> list = new ArrayList<>(vectors.length);
        for (float[] v : vectors) {
            list.add(v);
        }
        return FloatVectorValues.fromFloats(list, dim);
    }

    public void testProducesValidPermutation() throws IOException {
        float[][] vectors = generateClusteredVectors(300, 8, 3);
        KMeansReorderStrategy strategy = new KMeansReorderStrategy(3, 10);
        int[] permutation = strategy.computePermutation(toFloatVectorValues(vectors), 1, VectorSimilarityFunction.EUCLIDEAN);
        assertValidPermutation(permutation, 300);
    }

    public void testDefaultParameters() throws IOException {
        float[][] vectors = generateClusteredVectors(500, 8, 4);
        KMeansReorderStrategy strategy = new KMeansReorderStrategy();
        // default k=256, but only 500 vectors so effective k = min(256, 500) = 256
        int[] permutation = strategy.computePermutation(toFloatVectorValues(vectors), 1, VectorSimilarityFunction.EUCLIDEAN);
        assertValidPermutation(permutation, 500);
    }

    public void testKLargerThanNumVectors() throws IOException {
        // k=256 but only 10 vectors — should cap at 10
        float[][] vectors = generateClusteredVectors(10, 4, 2);
        KMeansReorderStrategy strategy = new KMeansReorderStrategy(256, 5);
        int[] permutation = strategy.computePermutation(toFloatVectorValues(vectors), 1, VectorSimilarityFunction.EUCLIDEAN);
        assertValidPermutation(permutation, 10);
    }

    public void testClusteringQuality() throws IOException {
        int perCluster = 100;
        int k = 3;
        int dim = 8;
        float[][] vectors = generateClusteredVectors(perCluster * k, dim, k);

        KMeansReorderStrategy strategy = new KMeansReorderStrategy(k, 25);
        int[] permutation = strategy.computePermutation(toFloatVectorValues(vectors), 1, VectorSimilarityFunction.EUCLIDEAN);
        assertValidPermutation(permutation, vectors.length);

        int sameClusterAdjacent = 0;
        for (int i = 0; i < permutation.length - 1; i++) {
            if ((permutation[i] % k) == (permutation[i + 1] % k)) {
                sameClusterAdjacent++;
            }
        }
        float rate = (float) sameClusterAdjacent / (permutation.length - 1);
        assertTrue("Expected >80% same-cluster adjacency, got " + rate, rate > 0.8f);
    }

    public void testCustomKAndNiter() throws IOException {
        float[][] vectors = generateClusteredVectors(200, 8, 5);
        KMeansReorderStrategy strategy = new KMeansReorderStrategy(5, 3);
        int[] permutation = strategy.computePermutation(toFloatVectorValues(vectors), 1, VectorSimilarityFunction.EUCLIDEAN);
        assertValidPermutation(permutation, 200);
    }

    private float[][] generateClusteredVectors(int total, int dim, int clusters) {
        float[][] vectors = new float[total][dim];
        Random rand = new Random(42);
        for (int i = 0; i < total; i++) {
            int c = i % clusters;
            for (int d = 0; d < dim; d++) {
                vectors[i][d] = c * 100.0f + rand.nextFloat();
            }
        }
        return vectors;
    }
}
