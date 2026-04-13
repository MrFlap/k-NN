/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;

import java.io.IOException;

/**
 * Strategy interface for computing vector reorder permutations.
 * Implementations can be mmap-friendly (BP) or heap-based (KMeans).
 */
public interface VectorReorderStrategy {

    /**
     * Compute a permutation array mapping new ord to old ord.
     *
     * @param vectors    FloatVectorValues — may be mmap-backed (from Lucene99FlatVectorsReader)
     *                   or heap-backed (from FloatVectorValues.fromFloats()).
     *                   Accessed via random-access vectorValue(int ord).
     *                   Each thread should call vectors.copy() for its own view.
     * @param numThreads number of CPU threads to use for the computation
     * @param similarityFunction the similarity function used by the field
     * @return permutation array where permutation[newOrd] = oldOrd
     * @throws IOException if reading vectors fails
     */
    int[] computePermutation(FloatVectorValues vectors, int numThreads, VectorSimilarityFunction similarityFunction) throws IOException;
}
