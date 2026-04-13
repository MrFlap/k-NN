/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder.bpreorder;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

public class BipartiteReorderStrategyTests extends OpenSearchTestCase {

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
        float[][] vectors = generateClusteredVectors(500, 16, 2);
        BipartiteReorderStrategy strategy = new BipartiteReorderStrategy();
        int[] permutation = strategy.computePermutation(toFloatVectorValues(vectors), 2, VectorSimilarityFunction.EUCLIDEAN);
        assertValidPermutation(permutation, 500);
    }

    public void testSingleThread() throws IOException {
        float[][] vectors = generateClusteredVectors(200, 8, 2);
        BipartiteReorderStrategy strategy = new BipartiteReorderStrategy();
        int[] permutation = strategy.computePermutation(toFloatVectorValues(vectors), 1, VectorSimilarityFunction.EUCLIDEAN);
        assertValidPermutation(permutation, 200);
    }

    public void testMultiThread() throws IOException {
        float[][] vectors = generateClusteredVectors(200, 8, 2);
        BipartiteReorderStrategy strategy = new BipartiteReorderStrategy();
        int[] permutation = strategy.computePermutation(toFloatVectorValues(vectors), 4, VectorSimilarityFunction.EUCLIDEAN);
        assertValidPermutation(permutation, 200);
    }

    public void testMinimumInput() throws IOException {
        float[][] vectors = { { 1.0f, 0.0f }, { 0.0f, 1.0f } };
        BipartiteReorderStrategy strategy = new BipartiteReorderStrategy();
        int[] permutation = strategy.computePermutation(toFloatVectorValues(vectors), 1, VectorSimilarityFunction.EUCLIDEAN);
        assertValidPermutation(permutation, 2);
    }

    public void testClusteringQuality() throws IOException {
        int perCluster = 250;
        int dim = 16;
        float[][] vectors = new float[perCluster * 2][dim];
        Random rand = new Random(42);
        for (int i = 0; i < perCluster; i++) {
            for (int d = 0; d < dim; d++) {
                vectors[2 * i][d] = rand.nextFloat();
                vectors[2 * i + 1][d] = 100.0f + rand.nextFloat();
            }
        }

        BipartiteReorderStrategy strategy = new BipartiteReorderStrategy();
        int[] permutation = strategy.computePermutation(toFloatVectorValues(vectors), 2, VectorSimilarityFunction.EUCLIDEAN);
        assertValidPermutation(permutation, vectors.length);

        int sameClusterAdjacent = 0;
        for (int i = 0; i < permutation.length - 1; i++) {
            if ((permutation[i] % 2 == 0) == (permutation[i + 1] % 2 == 0)) {
                sameClusterAdjacent++;
            }
        }
        float rate = (float) sameClusterAdjacent / (permutation.length - 1);
        assertTrue("Expected >90% same-cluster adjacency, got " + rate, rate > 0.9f);
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
