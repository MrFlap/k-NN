/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder.bpreorder;

import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.VectorReorderStrategy;

import java.io.IOException;

/**
 * BP reorder strategy that computes the permutation from quantized (binary) vectors
 * instead of full-precision float vectors.
 * <p>
 * For 32x binary quantization on 768-dim vectors, the working set drops from ~6GB
 * to ~192MB, making the permutation computation fit entirely in page cache.
 * <p>
 * Uses {@link BinaryToFloatVectorValues} to decode packed bits into {+1, -1} floats,
 * then delegates to the standard {@link BipartiteReorderStrategy} which uses Lucene's
 * SIMD-optimized {@code BpVectorReorderer}. L2 distance on {+1, -1} vectors is
 * monotonically related to Hamming distance, so the permutation quality is equivalent.
 */
public class QuantizedBipartiteReorderStrategy implements VectorReorderStrategy {

    private final BipartiteReorderStrategy delegate = new BipartiteReorderStrategy();

    /**
     * Fallback: compute permutation from full-precision float vectors.
     * Used when quantized vectors are not available (e.g., uncompressed indices).
     */
    @Override
    public int[] computePermutation(FloatVectorValues vectors, int numThreads,
                                    VectorSimilarityFunction similarityFunction) throws IOException {
        return delegate.computePermutation(vectors, numThreads, similarityFunction);
    }

    /**
     * Compute permutation from quantized byte vectors by decoding to {+1, -1} floats
     * and using standard BP with Euclidean distance.
     *
     * @param quantizedVectors ByteVectorValues from source segments' .faiss files
     * @param numThreads       number of threads for the ForkJoinPool
     * @return permutation array where permutation[newOrd] = oldOrd
     */
    public int[] computePermutationFromQuantized(ByteVectorValues quantizedVectors, int numThreads)
        throws IOException {
        FloatVectorValues decoded = new BinaryToFloatVectorValues(quantizedVectors);
        // L2 on {+1, -1} is monotonic with Hamming: L2² = 4 * hamming_distance
        return delegate.computePermutation(decoded, numThreads, VectorSimilarityFunction.EUCLIDEAN);
    }
}
