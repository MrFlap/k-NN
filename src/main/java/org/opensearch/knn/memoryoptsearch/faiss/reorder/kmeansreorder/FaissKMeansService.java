/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder.kmeansreorder;

import org.opensearch.knn.jni.JNICommons;
import org.opensearch.knn.jni.KNNLibraryLoader;

/**
 * JNI service for FAISS k-means clustering.
 * Uses the existing k-NN native library.
 */
public class FaissKMeansService {

    static {
        KNNLibraryLoader.loadFaissLibraryDirect();
    }

    public static final int METRIC_L2 = 0;
    public static final int METRIC_INNER_PRODUCT = 1;

    /**
     * Run k-means clustering and return both assignments and distances to centroids.
     *
     * @param vectorsAddress pointer to native memory where vectors are stored (std::vector&lt;float&gt;*)
     * @param numVectors number of vectors
     * @param dimension dimension of each vector
     * @param numClusters number of clusters (k)
     * @param numIterations number of k-means iterations
     * @param metricType METRIC_L2 or METRIC_INNER_PRODUCT
     * @return KMeansResult containing assignments and distances
     */
    public static native KMeansResult kmeansWithDistances(
        long vectorsAddress, int numVectors, int dimension,
        int numClusters, int numIterations, int metricType
    );

    /**
     * Run k-means on mmap'd vector data. The address points directly to contiguous float data
     * (not a std::vector wrapper). FAISS reads from this pointer; it is NOT freed by this method.
     *
     * @param mmapAddress raw pointer to contiguous float vector data
     * @param numVectors number of vectors
     * @param dimension dimension of each vector
     * @param numClusters number of clusters (k)
     * @param numIterations number of k-means iterations
     * @param metricType METRIC_L2 or METRIC_INNER_PRODUCT
     * @return KMeansResult containing assignments and distances
     */
    public static native KMeansResult kmeansWithDistancesMMap(
        long mmapAddress, int numVectors, int dimension,
        int numClusters, int numIterations, int metricType
    );

    /**
     * Allocate native memory and copy vectors into it.
     *
     * @param vectors 2D array of vectors [numVectors][dimension]
     * @return pointer to native memory
     */
    public static long storeVectors(float[][] vectors) {
        if (vectors == null || vectors.length == 0) return 0;
        // Use existing JNICommons to store vectors
        return JNICommons.storeVectorData(0, vectors, (long) vectors.length * vectors[0].length);
    }

    /**
     * Free native memory.
     *
     * @param address pointer to native memory
     */
    public static void freeVectors(long address) {
        if (address != 0) {
            JNICommons.freeVectorData(address);
        }
    }
}
