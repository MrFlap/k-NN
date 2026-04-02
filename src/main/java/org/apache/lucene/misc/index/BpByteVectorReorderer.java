/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Derived from Lucene's BpVectorReorderer, adapted for ByteVectorValues
 * to enable BP reordering from binary quantized vectors using Hamming distance.
 */
package org.apache.lucene.misc.index;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Arrays;
import java.util.concurrent.Executor;
import java.util.concurrent.RecursiveAction;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.CloseableThreadLocal;
import org.apache.lucene.util.IntroSelector;
import org.apache.lucene.util.IntsRef;

/**
 * BP reorderer for binary quantized vectors using Hamming distance.
 * <p>
 * Binary quantized vectors pack bits into bytes — each byte contains 8 binary decisions.
 * A 768-dim vector becomes 96 bytes. The natural distance metric is Hamming distance
 * (popcount of XOR), not L2 on raw byte values.
 * <p>
 * Centroids are represented as per-bit probabilities: for each of the codeSize*8 bit positions,
 * the centroid stores the fraction of partition vectors that have bit=1 at that position.
 * The bias (attraction to left vs right centroid) is computed as the expected Hamming distance
 * difference: for each bit in the vector, we accumulate (prob_left - prob_right) if bit=1,
 * or (prob_right - prob_left) if bit=0. This is equivalent to computing
 * E[hamming(vec, left)] - E[hamming(vec, right)] using the probabilistic centroids.
 */
public class BpByteVectorReorderer extends AbstractBPReorderer {

    private static final int FORK_THRESHOLD = 8192;

    public BpByteVectorReorderer() {
        setMinPartitionSize(DEFAULT_MIN_PARTITION_SIZE);
        setMaxIters(DEFAULT_MAX_ITERS);
        setRAMBudgetMB(Runtime.getRuntime().totalMemory() / 1024d / 1024d / 10d);
    }

    /**
     * Per-thread state. Centroids are float arrays of size totalBits = codeSize * 8,
     * where centroid[bitPos] = fraction of vectors in the partition with bit=1 at that position.
     */
    private static class PerThreadState {
        final ByteVectorValues vectors;
        final int codeSize;
        final int totalBits;
        final float[] leftCentroid;
        final float[] rightCentroid;

        PerThreadState(ByteVectorValues vectors) {
            try {
                this.vectors = vectors.copy();
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
            int detectedCodeSize;
            try {
                detectedCodeSize = vectors.size() > 0 ? vectors.vectorValue(0).length : vectors.dimension();
            } catch (IOException e) {
                detectedCodeSize = vectors.dimension();
            }
            this.codeSize = detectedCodeSize;
            this.totalBits = codeSize * 8;
            this.leftCentroid = new float[totalBits];
            this.rightCentroid = new float[totalBits];
        }
    }

    private static class DocMap extends Sorter.DocMap {
        private final int[] newToOld;
        private final int[] oldToNew;

        public DocMap(int[] newToOld) {
            this.newToOld = newToOld;
            oldToNew = new int[newToOld.length];
            for (int i = 0; i < newToOld.length; ++i) {
                oldToNew[newToOld[i]] = i;
            }
        }

        @Override public int size() { return newToOld.length; }
        @Override public int oldToNew(int docID) { return oldToNew[docID]; }
        @Override public int newToOld(int docID) { return newToOld[docID]; }
    }

    private abstract class BaseRecursiveAction extends RecursiveAction {
        protected final TaskExecutor executor;
        protected final int depth;

        BaseRecursiveAction(TaskExecutor executor, int depth) {
            this.executor = executor;
            this.depth = depth;
        }

        protected final boolean shouldFork(int problemSize, int totalProblemSize) {
            if (executor == null) return false;
            if (getSurplusQueuedTaskCount() > 3) return false;
            if (problemSize == totalProblemSize) return true;
            return problemSize > FORK_THRESHOLD;
        }
    }

    // ---- Hamming-based centroid and distance ----

    /**
     * Compute a probabilistic centroid over packed binary vectors.
     * centroid[bitPos] = fraction of vectors in the partition with bit=1 at that position.
     * centroid has length codeSize * 8.
     */
    static void computeBitCentroid(IntsRef ids, ByteVectorValues vectors, float[] centroid) throws IOException {
        Arrays.fill(centroid, 0);
        int codeSize = centroid.length / 8;
        for (int i = ids.offset; i < ids.offset + ids.length; i++) {
            byte[] vec = vectors.vectorValue(ids.ints[i]);
            for (int byteIdx = 0; byteIdx < codeSize; byteIdx++) {
                int b = vec[byteIdx] & 0xFF;
                int basePos = byteIdx * 8;
                // Unpack each bit
                for (int bit = 0; bit < 8; bit++) {
                    if ((b & (1 << bit)) != 0) {
                        centroid[basePos + bit] += 1.0f;
                    }
                }
            }
        }
        float scale = 1.0f / ids.length;
        for (int j = 0; j < centroid.length; j++) {
            centroid[j] *= scale;
        }
    }

    /**
     * Compute bias: expected Hamming distance to left centroid minus expected Hamming distance
     * to right centroid. Negative = closer to left, positive = closer to right.
     *
     * For each bit position:
     *   if vec bit = 1: contribution = (1 - leftProb) - (1 - rightProb) = rightProb - leftProb
     *   if vec bit = 0: contribution = leftProb - rightProb
     *
     * This simplifies to: for each bit, bias += (vec_bit == 1) ? (right - left) : (left - right)
     * Which is: bias += (2 * vec_bit - 1) * (rightProb - leftProb)
     */
    static float computeHammingBias(byte[] vec, float[] leftCentroid, float[] rightCentroid) {
        float bias = 0;
        int codeSize = leftCentroid.length / 8;
        for (int byteIdx = 0; byteIdx < codeSize; byteIdx++) {
            int b = vec[byteIdx] & 0xFF;
            int basePos = byteIdx * 8;
            for (int bit = 0; bit < 8; bit++) {
                float diff = rightCentroid[basePos + bit] - leftCentroid[basePos + bit];
                if ((b & (1 << bit)) != 0) {
                    bias += diff;  // bit=1: closer to right if rightProb > leftProb
                } else {
                    bias -= diff;  // bit=0: closer to left if leftProb < rightProb
                }
            }
        }
        return bias;
    }

    /**
     * Compute a scale factor for convergence check, analogous to the float BP's
     * centroid distance. We use the L2 norm of the centroid difference vector.
     */
    static float centroidDiffScale(float[] leftCentroid, float[] rightCentroid) {
        float sum = 0;
        for (int i = 0; i < leftCentroid.length; i++) {
            float d = leftCentroid[i] - rightCentroid[i];
            sum += d * d;
        }
        return (float) Math.sqrt(sum);
    }

    // ---- ReorderTask ----

    private class ReorderTask extends BaseRecursiveAction {
        private final IntsRef ids;
        private final float[] biases;
        private final CloseableThreadLocal<PerThreadState> threadLocal;

        ReorderTask(IntsRef ids, float[] biases, CloseableThreadLocal<PerThreadState> threadLocal,
                    TaskExecutor executor, int depth) {
            super(executor, depth);
            this.ids = ids;
            this.biases = biases;
            this.threadLocal = threadLocal;
        }

        @Override
        protected void compute() {
            if (depth > 0) {
                Arrays.sort(ids.ints, ids.offset, ids.offset + ids.length);
            }

            int halfLength = ids.length >>> 1;
            if (halfLength < minPartitionSize) return;

            IntsRef left = new IntsRef(ids.ints, ids.offset, halfLength);
            IntsRef right = new IntsRef(ids.ints, ids.offset + halfLength, ids.length - halfLength);

            PerThreadState state = threadLocal.get();
            float[] leftCentroid = state.leftCentroid;
            float[] rightCentroid = state.rightCentroid;

            try {
                computeBitCentroid(left, state.vectors, leftCentroid);
                computeBitCentroid(right, state.vectors, rightCentroid);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }

            for (int iter = 0; iter < maxIters; ++iter) {
                int moved;
                try {
                    moved = shuffle(ids, right.offset, leftCentroid, rightCentroid, biases);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
                if (moved == 0) break;
                // Always recompute centroids (incremental updates disabled for binary)
                try {
                    computeBitCentroid(left, state.vectors, leftCentroid);
                    computeBitCentroid(right, state.vectors, rightCentroid);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            }

            ReorderTask leftTask = new ReorderTask(left, biases, threadLocal, executor, depth + 1);
            ReorderTask rightTask = new ReorderTask(right, biases, threadLocal, executor, depth + 1);

            if (shouldFork(ids.length, ids.ints.length)) {
                invokeAll(leftTask, rightTask);
            } else {
                leftTask.compute();
                rightTask.compute();
            }
        }

        private int shuffle(IntsRef ids, int midPoint,
                            float[] leftCentroid, float[] rightCentroid,
                            float[] biases) throws IOException {
            new ComputeBiasTask(ids.ints, biases, ids.offset, ids.offset + ids.length,
                leftCentroid, rightCentroid, threadLocal, executor, depth).compute();

            float scale = centroidDiffScale(leftCentroid, rightCentroid);
            float maxLeftBias = Float.NEGATIVE_INFINITY;
            for (int i = ids.offset; i < midPoint; ++i) {
                maxLeftBias = Math.max(maxLeftBias, biases[i]);
            }
            float minRightBias = Float.POSITIVE_INFINITY;
            for (int i = midPoint, end = ids.offset + ids.length; i < end; ++i) {
                minRightBias = Math.min(minRightBias, biases[i]);
            }
            if (500 * (maxLeftBias - minRightBias) <= scale) return 0;

            class Selector extends IntroSelector {
                int count = 0;
                int pivotDoc;
                float pivotBias;

                @Override
                public void setPivot(int i) {
                    pivotDoc = ids.ints[i];
                    pivotBias = biases[i];
                }

                @Override
                public int comparePivot(int j) {
                    int cmp = Float.compare(pivotBias, biases[j]);
                    return cmp == 0 ? pivotDoc - ids.ints[j] : cmp;
                }

                @Override
                public void swap(int i, int j) {
                    float tmpBias = biases[i];
                    biases[i] = biases[j];
                    biases[j] = tmpBias;

                    int tmpDoc = ids.ints[i];
                    ids.ints[i] = ids.ints[j];
                    ids.ints[j] = tmpDoc;

                    if (!(i < midPoint == j < midPoint)) {
                        count++;
                    }
                }
            }

            Selector selector = new Selector();
            selector.select(ids.offset, ids.offset + ids.length, midPoint);
            return selector.count;
        }
    }

    // ---- ComputeBiasTask ----

    private class ComputeBiasTask extends BaseRecursiveAction {
        private final int[] ids;
        private final float[] biases;
        private final int start;
        private final int end;
        private final float[] leftCentroid;
        private final float[] rightCentroid;
        private final CloseableThreadLocal<PerThreadState> threadLocal;

        ComputeBiasTask(int[] ids, float[] biases, int start, int end,
                        float[] leftCentroid, float[] rightCentroid,
                        CloseableThreadLocal<PerThreadState> threadLocal,
                        TaskExecutor executor, int depth) {
            super(executor, depth);
            this.ids = ids;
            this.biases = biases;
            this.start = start;
            this.end = end;
            this.leftCentroid = leftCentroid;
            this.rightCentroid = rightCentroid;
            this.threadLocal = threadLocal;
        }

        @Override
        protected void compute() {
            final int problemSize = end - start;
            if (problemSize > 1 && shouldFork(problemSize, ids.length)) {
                final int mid = (start + end) >>> 1;
                invokeAll(
                    new ComputeBiasTask(ids, biases, start, mid, leftCentroid, rightCentroid,
                        threadLocal, executor, depth),
                    new ComputeBiasTask(ids, biases, mid, end, leftCentroid, rightCentroid,
                        threadLocal, executor, depth));
            } else {
                ByteVectorValues vectors = threadLocal.get().vectors;
                try {
                    for (int i = start; i < end; ++i) {
                        byte[] vec = vectors.vectorValue(ids[i]);
                        biases[i] = computeHammingBias(vec, leftCentroid, rightCentroid);
                    }
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            }
        }
    }

    // ---- Public API ----

    /**
     * Compute a permutation from binary quantized byte vectors using Hamming distance.
     *
     * @param vectors  ByteVectorValues — packed binary vectors from .faiss
     * @param executor task executor for parallelism, or null for single-threaded
     * @return DocMap mapping old ords to new ords
     */
    public Sorter.DocMap computeValueMap(ByteVectorValues vectors, TaskExecutor executor) {
        if (docRAMRequirements(vectors.size()) >= ramBudgetMB * 1024 * 1024) {
            throw new NotEnoughRAMException(
                "At least " + Math.ceil(docRAMRequirements(vectors.size()) / 1024. / 1024.)
                + "MB of RAM required, but budget is " + ramBudgetMB + "MB");
        }
        return new DocMap(computePermutation(vectors, executor));
    }

    private int[] computePermutation(ByteVectorValues vectors, TaskExecutor executor) {
        final int size = vectors.size();
        int[] sortedIds = new int[size];
        for (int i = 0; i < size; ++i) {
            sortedIds[i] = i;
        }
        try (CloseableThreadLocal<PerThreadState> threadLocal =
                 new CloseableThreadLocal<>() {
                     @Override
                     protected PerThreadState initialValue() {
                         return new PerThreadState(vectors);
                     }
                 }) {
            IntsRef ids = new IntsRef(sortedIds, 0, sortedIds.length);
            new ReorderTask(ids, new float[size], threadLocal, executor, 0).compute();
        }
        return sortedIds;
    }

    private static long docRAMRequirements(int maxDoc) {
        // sortedIds (int[]) + biases (float[]) + centroid overhead
        return 2L * Integer.BYTES * maxDoc;
    }

    @Override
    public Sorter.DocMap computeDocMap(CodecReader reader, Directory tempDir, Executor executor) throws IOException {
        throw new UnsupportedOperationException("Use computeValueMap(ByteVectorValues, TaskExecutor) directly");
    }
}
