/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder.bpreorder;

import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.misc.index.BpByteVectorReorderer;
import org.apache.lucene.search.TaskExecutor;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.VectorReorderStrategy;

import java.io.IOException;
import java.util.concurrent.ForkJoinPool;

/**
 * BP reorder strategy that computes the permutation from quantized (byte) vectors
 * instead of full-precision float vectors.
 * <p>
 * For 32x binary quantization on 768-dim vectors, the working set drops from ~6GB
 * (768 * 4 bytes * 2M vectors) to ~192MB (96 bytes * 2M vectors), making the
 * permutation computation fit entirely in page cache and turning an I/O-bound
 * operation into a compute-bound one.
 * <p>
 * The permutation quality is comparable to full-precision BP because binary
 * quantization preserves the coarse spatial relationships that BP needs for
 * partitioning (which vector is closer to which centroid).
 * <p>
 * Usage: the caller provides quantized ByteVectorValues (from source segments'
 * .faiss files) via {@link #computePermutationFromQuantized}. The standard
 * {@link #computePermutation} method falls back to the float-based BP.
 */
public class QuantizedBipartiteReorderStrategy implements VectorReorderStrategy {

    /**
     * Fallback: compute permutation from full-precision float vectors.
     * Used when quantized vectors are not available (e.g., uncompressed indices).
     */
    @Override
    public int[] computePermutation(FloatVectorValues vectors, int numThreads,
                                    VectorSimilarityFunction similarityFunction) throws IOException {
        // Delegate to the standard float-based BP strategy
        return new BipartiteReorderStrategy().computePermutation(vectors, numThreads, similarityFunction);
    }

    /**
     * Compute permutation from quantized byte vectors using Hamming/L2 distance.
     * This is the fast path for compressed indices (8x, 16x, 32x).
     * <p>
     * The byte vectors are typically read from source segments' .faiss flat storage
     * during merge, before the new .vec or .faiss files are written.
     *
     * @param quantizedVectors ByteVectorValues from source segments' .faiss files
     * @param numThreads       number of threads for the ForkJoinPool
     * @return permutation array where permutation[newOrd] = oldOrd
     */
    public int[] computePermutationFromQuantized(ByteVectorValues quantizedVectors, int numThreads)
        throws IOException {
        BpByteVectorReorderer reorderer = new BpByteVectorReorderer();

        ForkJoinPool pool = new ForkJoinPool(numThreads);
        try {
            TaskExecutor executor = new TaskExecutor(pool);
            Sorter.DocMap map = reorderer.computeValueMap(quantizedVectors, executor);

            int n = quantizedVectors.size();
            int[] permutation = new int[n];
            for (int i = 0; i < n; i++) {
                permutation[i] = map.newToOld(i);
            }
            return permutation;
        } finally {
            pool.shutdown();
        }
    }
}
