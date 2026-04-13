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

public class MergeAwareKMeansReorderStrategyTests extends OpenSearchTestCase {

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

    public void testFlushProducesValidPermutationAndSummary() throws IOException {
        float[][] vectors = generateClusteredVectors(300, 8, 3);
        MergeAwareKMeansReorderStrategy strategy = new MergeAwareKMeansReorderStrategy(100);

        ClusterResult result = strategy.computePermutationWithSummary(
            toFloatVectorValues(vectors), 1, VectorSimilarityFunction.EUCLIDEAN
        );

        assertValidPermutation(result.permutation(), 300);
        assertNotNull(result.summary());
        assertTrue("k should be > 0", result.summary().k > 0);
        assertTrue("k should be <= 100", result.summary().k <= 100);
        assertEquals(300, result.summary().numVectors);
        assertEquals(8, result.summary().dimension);
    }

    public void testCapKAdaptsToSmallSegments() {
        MergeAwareKMeansReorderStrategy strategy = new MergeAwareKMeansReorderStrategy(1500);

        // 5000 vectors: sqrt(5000)=70, 5000/39=128 → min(70,128)=70, min(70,1500)=70
        assertEquals(70, strategy.capK(5000));

        // 100 vectors: sqrt(100)=10, 100/39=2 → min(2,10)=2, min(2,1500)=2
        assertEquals(2, strategy.capK(100));

        // 2M vectors: sqrt(2M)=1414, 2M/39=51282 → min(1414,51282)=1414, min(1414,1500)=1414
        assertEquals(1414, strategy.capK(2_000_000));
    }

    public void testMergeConcatenatesWhenPoolBelowKMax() throws IOException {
        // Two source summaries with k=5 each, kMax=20 → pool=10 < 20 → concatenate
        int dim = 4;
        ClusterSummary s1 = makeSummary(5, dim, 50);
        ClusterSummary s2 = makeSummary(5, dim, 50);

        float[][] mergedVectors = generateClusteredVectors(100, dim, 3);
        MergeAwareKMeansReorderStrategy strategy = new MergeAwareKMeansReorderStrategy(20);

        ClusterResult result = strategy.computePermutationFromMergedSummaries(
            List.of(s1, s2), toFloatVectorValues(mergedVectors), 1, VectorSimilarityFunction.EUCLIDEAN
        );

        assertValidPermutation(result.permutation(), 100);
        // Pool was 10, kMax=20, so output k should be 10 (concatenation)
        assertEquals(10, result.summary().k);
    }

    public void testMergeReducesWhenPoolExceedsKMax() throws IOException {
        // Two source summaries with k=10 each, kMax=15 → pool=20 > 15 → weighted k-means
        int dim = 4;
        ClusterSummary s1 = makeSummary(10, dim, 100);
        ClusterSummary s2 = makeSummary(10, dim, 100);

        float[][] mergedVectors = generateClusteredVectors(200, dim, 5);
        MergeAwareKMeansReorderStrategy strategy = new MergeAwareKMeansReorderStrategy(15);

        ClusterResult result = strategy.computePermutationFromMergedSummaries(
            List.of(s1, s2), toFloatVectorValues(mergedVectors), 1, VectorSimilarityFunction.EUCLIDEAN
        );

        assertValidPermutation(result.permutation(), 200);
        assertEquals(15, result.summary().k);
    }

    public void testMergeProducesSummaryForFutureMerges() throws IOException {
        int dim = 4;
        // Simulate a 3-level cascade: flush → merge → merge
        MergeAwareKMeansReorderStrategy strategy = new MergeAwareKMeansReorderStrategy(50);

        // Level 0: two flush segments
        float[][] v1 = generateClusteredVectors(100, dim, 3);
        float[][] v2 = generateClusteredVectors(100, dim, 3);
        ClusterResult r1 = strategy.computePermutationWithSummary(
            toFloatVectorValues(v1), 1, VectorSimilarityFunction.EUCLIDEAN);
        ClusterResult r2 = strategy.computePermutationWithSummary(
            toFloatVectorValues(v2), 1, VectorSimilarityFunction.EUCLIDEAN);

        // Level 1: merge r1 + r2
        float[][] merged12 = concat(v1, v2);
        ClusterResult r12 = strategy.computePermutationFromMergedSummaries(
            List.of(r1.summary(), r2.summary()),
            toFloatVectorValues(merged12), 1, VectorSimilarityFunction.EUCLIDEAN
        );
        assertValidPermutation(r12.permutation(), 200);
        assertNotNull(r12.summary());

        // Level 2: merge r12 + another flush
        float[][] v3 = generateClusteredVectors(100, dim, 3);
        ClusterResult r3 = strategy.computePermutationWithSummary(
            toFloatVectorValues(v3), 1, VectorSimilarityFunction.EUCLIDEAN);

        float[][] merged123 = concat(merged12, v3);
        ClusterResult r123 = strategy.computePermutationFromMergedSummaries(
            List.of(r12.summary(), r3.summary()),
            toFloatVectorValues(merged123), 1, VectorSimilarityFunction.EUCLIDEAN
        );
        assertValidPermutation(r123.permutation(), 300);
        assertEquals(300, r123.summary().numVectors);
    }

    public void testComputePermutationFallback() throws IOException {
        float[][] vectors = generateClusteredVectors(200, 8, 3);
        MergeAwareKMeansReorderStrategy strategy = new MergeAwareKMeansReorderStrategy(100);

        int[] perm = strategy.computePermutation(
            toFloatVectorValues(vectors), 1, VectorSimilarityFunction.EUCLIDEAN
        );
        assertValidPermutation(perm, 200);
    }

    // --- helpers ---

    private ClusterSummary makeSummary(int k, int dim, int numVectors) {
        Random rand = new Random(42);
        float[][] centroids = new float[k][dim];
        float[][] linearSums = new float[k][dim];
        int[] counts = new int[k];
        float[] sumOfSquares = new float[k];
        int perCluster = numVectors / k;
        for (int i = 0; i < k; i++) {
            counts[i] = perCluster;
            sumOfSquares[i] = rand.nextFloat() * 10;
            for (int d = 0; d < dim; d++) {
                centroids[i][d] = i * 50.0f + rand.nextFloat();
                linearSums[i][d] = centroids[i][d] * counts[i];
            }
        }
        return new ClusterSummary(k, dim, KMeansClusterer.METRIC_L2, numVectors,
            centroids, linearSums, counts, sumOfSquares);
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
        for (float[] v : vectors) list.add(v);
        return FloatVectorValues.fromFloats(list, dim);
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

    private float[][] concat(float[][] a, float[][] b) {
        float[][] result = new float[a.length + b.length][];
        System.arraycopy(a, 0, result, 0, a.length);
        System.arraycopy(b, 0, result, a.length, b.length);
        return result;
    }
}
