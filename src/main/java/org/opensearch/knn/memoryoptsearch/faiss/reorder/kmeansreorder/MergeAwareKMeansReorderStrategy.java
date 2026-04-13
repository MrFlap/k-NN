/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder.kmeansreorder;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.MergeAwareReorderStrategy;

import java.io.IOException;
import java.util.List;

/**
 * Merge-aware k-means reorder strategy. Runs full k-means at flush (with adaptive k),
 * and merges centroids at merge time (pool-or-reduce) followed by a single assignment pass.
 */
@Log4j2
public class MergeAwareKMeansReorderStrategy implements MergeAwareReorderStrategy {

    private static final int DEFAULT_NITER = 25;
    private static final int MIN_POINTS_PER_CENTROID = 39;

    private final int kMax;
    private final int niter;

    public MergeAwareKMeansReorderStrategy(int kMax) {
        this(kMax, DEFAULT_NITER);
    }

    public MergeAwareKMeansReorderStrategy(int kMax, int niter) {
        this.kMax = kMax;
        this.niter = niter;
    }

    @Override
    public ClusterResult computePermutationWithSummary(
        FloatVectorValues vectors,
        int numThreads,
        VectorSimilarityFunction similarityFunction
    ) throws IOException {
        int n = vectors.size();
        int d = vectors.dimension();
        int metricType = toFaissMetric(similarityFunction);
        int effectiveK = capK(n);

        if (effectiveK < 2) {
            return trivialResult(n, d, metricType);
        }

        // Materialize vectors for clustering
        float[][] heapVectors = new float[n][];
        for (int i = 0; i < n; i++) {
            float[] src = vectors.vectorValue(i);
            heapVectors[i] = new float[src.length];
            System.arraycopy(src, 0, heapVectors[i], 0, src.length);
        }

        log.info("Flush k-means: n={}, dim={}, effectiveK={}, kMax={}", n, d, effectiveK, kMax);

        // Run k-means with adaptive k
        KMeansResult result = KMeansClusterer.cluster(heapVectors, effectiveK, niter, metricType);

        // Build centroids from assignments (KMeansClusterer already computed them internally,
        // but we need them explicitly for the summary)
        float[][] centroids = computeCentroids(heapVectors, result.assignments(), effectiveK, d);

        // Build summary
        ClusterSummary summary = ClusterSummary.fromAssignments(
            heapVectors, result.assignments(), result.distances(), centroids, effectiveK, d, metricType
        );

        // Sort by (cluster, distance) → permutation
        int[] permutation = ClusterSorter.sortByCluster(result.assignments(), result.distances(), metricType);
        return new ClusterResult(permutation, summary);
    }

    @Override
    public ClusterResult computePermutationFromMergedSummaries(
        List<ClusterSummary> sourceSummaries,
        FloatVectorValues mergedVectors,
        int numThreads,
        VectorSimilarityFunction similarityFunction
    ) throws IOException {
        int metricType = toFaissMetric(similarityFunction);
        int d = mergedVectors.dimension();

        // 1. Pool centroids and weights
        int poolSize = sourceSummaries.stream().mapToInt(s -> s.k).sum();
        float[][] pooledCentroids = new float[poolSize][];
        float[] weights = new float[poolSize];
        int offset = 0;
        for (ClusterSummary src : sourceSummaries) {
            for (int i = 0; i < src.k; i++) {
                pooledCentroids[offset] = src.centroids[i];
                weights[offset] = src.counts[i];
                offset++;
            }
        }

        // 2. Pool-or-reduce
        float[][] finalCentroids;
        int outputK;
        if (poolSize <= kMax) {
            log.info("Merge centroid concatenation: poolSize={} <= kMax={}", poolSize, kMax);
            finalCentroids = pooledCentroids;
            outputK = poolSize;
        } else {
            log.info("Merge weighted k-means: poolSize={} -> kMax={}", poolSize, kMax);
            finalCentroids = WeightedKMeans.cluster(pooledCentroids, weights, kMax, niter, metricType);
            outputK = kMax;
        }

        // 3. Brute-force assignment pass
        int n = mergedVectors.size();
        int[] assignments = new int[n];
        float[] distances = new float[n];
        for (int i = 0; i < n; i++) {
            float[] vec = mergedVectors.vectorValue(i);
            int best = 0;
            float bestDist = Float.MAX_VALUE;
            for (int c = 0; c < outputK; c++) {
                float dist = WeightedKMeans.distance(vec, finalCentroids[c], metricType);
                if (dist < bestDist) {
                    bestDist = dist;
                    best = c;
                }
            }
            assignments[i] = best;
            distances[i] = bestDist;
        }

        // 4. Build summary for merged segment
        // We need linearSums — accumulate from merged vectors
        float[][] linearSums = new float[outputK][d];
        int[] counts = new int[outputK];
        float[] sumOfSquares = new float[outputK];
        for (int i = 0; i < n; i++) {
            int c = assignments[i];
            counts[c]++;
            sumOfSquares[c] += distances[i];
            float[] vec = mergedVectors.vectorValue(i);
            for (int j = 0; j < d; j++) {
                linearSums[c][j] += vec[j];
            }
        }
        ClusterSummary mergedSummary = new ClusterSummary(
            outputK, d, metricType, n, finalCentroids, linearSums, counts, sumOfSquares
        );

        // 5. Sort → permutation
        int[] permutation = ClusterSorter.sortByCluster(assignments, distances, metricType);
        return new ClusterResult(permutation, mergedSummary);
    }

    @Override
    public int[] computePermutation(
        FloatVectorValues vectors,
        int numThreads,
        VectorSimilarityFunction similarityFunction
    ) throws IOException {
        return computePermutationWithSummary(vectors, numThreads, similarityFunction).permutation();
    }

    int capK(int numVectors) {
        int effectiveK = Math.min((int) Math.sqrt(numVectors), kMax);
        effectiveK = Math.min(effectiveK, Math.max(1, numVectors / MIN_POINTS_PER_CENTROID));
        return effectiveK;
    }

    private static float[][] computeCentroids(float[][] vectors, int[] assignments, int k, int d) {
        float[][] centroids = new float[k][d];
        int[] counts = new int[k];
        for (int i = 0; i < vectors.length; i++) {
            int c = assignments[i];
            counts[c]++;
            for (int j = 0; j < d; j++) {
                centroids[c][j] += vectors[i][j];
            }
        }
        for (int c = 0; c < k; c++) {
            if (counts[c] > 0) {
                for (int j = 0; j < d; j++) {
                    centroids[c][j] /= counts[c];
                }
            }
        }
        return centroids;
    }

    private static ClusterResult trivialResult(int n, int d, int metricType) {
        int[] perm = new int[n];
        for (int i = 0; i < n; i++) perm[i] = i;
        ClusterSummary summary = new ClusterSummary(
            1, d, metricType, n,
            new float[][] { new float[d] },
            new float[][] { new float[d] },
            new int[] { n },
            new float[] { 0f }
        );
        return new ClusterResult(perm, summary);
    }

    private static int toFaissMetric(VectorSimilarityFunction similarityFunction) {
        return (similarityFunction == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT
            || similarityFunction == VectorSimilarityFunction.COSINE)
            ? KMeansClusterer.METRIC_INNER_PRODUCT
            : KMeansClusterer.METRIC_L2;
    }
}
