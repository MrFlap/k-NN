/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.kmeansreorder.ClusterResult;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.kmeansreorder.ClusterSummary;

import java.io.IOException;
import java.util.List;

/**
 * Extension of {@link VectorReorderStrategy} that supports merge-aware clustering:
 * full k-means at flush, centroid merging at merge.
 */
public interface MergeAwareReorderStrategy extends VectorReorderStrategy {

    /**
     * Run full k-means with adaptive k and return both the permutation and cluster summary.
     * Called during flush.
     */
    ClusterResult computePermutationWithSummary(
        FloatVectorValues vectors,
        int numThreads,
        VectorSimilarityFunction similarityFunction
    ) throws IOException;

    /**
     * Merge cluster summaries from source segments, then assign all merged vectors
     * to the new centroids. Adaptive k: if pooled centroid count <= k_max, centroids
     * are concatenated; otherwise weighted k-means reduces to k_max.
     * Called during merge.
     */
    ClusterResult computePermutationFromMergedSummaries(
        List<ClusterSummary> sourceSummaries,
        FloatVectorValues mergedVectors,
        int numThreads,
        VectorSimilarityFunction similarityFunction
    ) throws IOException;
}
