/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder.kmeansreorder;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.memoryoptsearch.MemorySegmentAddressExtractorUtil;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.VectorReorderStrategy;

import java.io.IOException;

/**
 * K-means clustering reorder strategy.
 * Prefers mmap passthrough to FAISS (zero heap) when the .vec IndexInput is backed by a single
 * contiguous mmap segment. Falls back to heap materialization otherwise.
 */
@Log4j2
public class KMeansReorderStrategy implements VectorReorderStrategy {

    private static final int DEFAULT_K = 256;
    private static final int DEFAULT_NITER = 25;
    /** FAISS default min_points_per_centroid */
    private static final int MIN_POINTS_PER_CENTROID = 39;

    private final int k;
    private final int niter;

    public KMeansReorderStrategy() {
        this(DEFAULT_K, DEFAULT_NITER);
    }

    public KMeansReorderStrategy(int k, int niter) {
        this.k = k;
        this.niter = niter;
    }

    /**
     * Compute permutation via mmap passthrough — zero Java heap for vectors.
     * The vecData IndexInput must be opened from an MMapDirectory (position 0, raw file).
     *
     * @param vecData             raw .vec file IndexInput (not past codec header)
     * @param vectorDataOffset    absolute byte offset in the file where vector data begins
     *                            (past codec header + alignment padding)
     * @param numVectors          number of vectors
     * @param dimension           vector dimension
     * @param similarityFunction  similarity function for metric type
     * @return permutation or null if mmap extraction fails (caller should fall back)
     */
    public int[] computePermutationMMap(
        IndexInput vecData,
        long vectorDataOffset,
        int numVectors,
        int dimension,
        VectorSimilarityFunction similarityFunction
    ) {
        int effectiveK = capK(numVectors);
        if (effectiveK < 2) {
            return identityPermutation(numVectors);
        }

        long vectorDataLength = (long) numVectors * dimension * Float.BYTES;

        long[] addressAndSize = MemorySegmentAddressExtractorUtil.tryExtractAddressAndSize(
            vecData, vectorDataOffset, vectorDataLength
        );

        if (addressAndSize == null || addressAndSize.length != 2) {
            // Multi-segment mmap (file > ~2GB) or extraction failed
            log.warn("mmap address extraction returned {} segments, need exactly 1. Falling back to heap path.",
                addressAndSize == null ? 0 : addressAndSize.length / 2);
            return null;
        }

        long mmapAddress = addressAndSize[0];
        int metricType = toFaissMetric(similarityFunction);

        log.info("Running mmap KMeans: n={}, dim={}, k={}, niter={}", numVectors, dimension, effectiveK, niter);
        KMeansResult result = FaissKMeansService.kmeansWithDistancesMMap(
            mmapAddress, numVectors, dimension, effectiveK, niter, metricType
        );
        return ClusterSorter.sortByCluster(result.assignments(), result.distances(), metricType);
    }

    @Override
    public int[] computePermutation(FloatVectorValues vectors, int numThreads, VectorSimilarityFunction similarityFunction)
        throws IOException {
        int n = vectors.size();
        int effectiveK = capK(n);
        if (effectiveK < 2) {
            return identityPermutation(n);
        }

        int metricType = toFaissMetric(similarityFunction);

        // Materialize vectors into heap — fallback path
        float[][] heapVectors = new float[n][];
        for (int i = 0; i < n; i++) {
            float[] src = vectors.vectorValue(i);
            heapVectors[i] = new float[src.length];
            System.arraycopy(src, 0, heapVectors[i], 0, src.length);
        }

        int[] perm = ClusterSorter.clusterAndSort(heapVectors, effectiveK, niter, metricType);
        return perm;
    }

    int capK(int numVectors) {
        int effectiveK = Math.min((int) Math.sqrt(numVectors), k);
        // Respect FAISS min_points_per_centroid to avoid warnings
        effectiveK = Math.min(effectiveK, Math.max(1, numVectors / MIN_POINTS_PER_CENTROID));
        return effectiveK;
    }

    private static int[] identityPermutation(int n) {
        int[] perm = new int[n];
        for (int i = 0; i < n; i++) perm[i] = i;
        return perm;
    }

    private static int toFaissMetric(VectorSimilarityFunction similarityFunction) {
        return (similarityFunction == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT
            || similarityFunction == VectorSimilarityFunction.COSINE)
            ? FaissKMeansService.METRIC_INNER_PRODUCT
            : FaissKMeansService.METRIC_L2;
    }
}
