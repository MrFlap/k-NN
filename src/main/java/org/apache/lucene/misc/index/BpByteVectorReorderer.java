/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Derived from Lucene's BpVectorReorderer, adapted for ByteVectorValues
 * to enable BP reordering from quantized (binary) vectors without decoding to float.
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
 * BP reorderer that operates directly on {@link ByteVectorValues} using Hamming distance.
 * <p>
 * For binary quantized vectors (e.g., 32x compression), this avoids decoding to float entirely.
 * Centroids are maintained as float arrays (accumulated byte values), and bias computation uses
 * a Hamming-like distance between byte vectors and float centroids.
 * <p>
 * The working set is 32x smaller than the float-based BP, making it fit in page cache
 * and turning an I/O-bound operation into a compute-bound one.
 */
public class BpByteVectorReorderer extends AbstractBPReorderer {

    private static final int FORK_THRESHOLD = 8192;
    private static final int MAX_CENTROID_UPDATES = 0;

    public BpByteVectorReorderer() {
        setMinPartitionSize(DEFAULT_MIN_PARTITION_SIZE);
        setMaxIters(DEFAULT_MAX_ITERS);
        setRAMBudgetMB(Runtime.getRuntime().totalMemory() / 1024d / 1024d / 10d);
    }

    /**
     * Per-thread state holding a copy of the byte vector values and scratch arrays.
     * Each thread gets its own mmap view via copy().
     */
    private static class PerThreadState {
        final ByteVectorValues vectors;
        final float[] leftCentroid;
        final float[] rightCentroid;
        final float[] scratch;
        final int codeSize;

        PerThreadState(ByteVectorValues vectors) {
            try {
                this.vectors = vectors.copy();
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
            // codeSize = number of bytes per vector (e.g., 96 for 768-dim 32x BQ)
            this.codeSize = vectors.dimension();
            // Centroids are float arrays over the byte dimensions for precision during accumulation
            this.leftCentroid = new float[codeSize];
            this.rightCentroid = new float[codeSize];
            this.scratch = new float[codeSize];
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

        @Override
        public int size() { return newToOld.length; }

        @Override
        public int oldToNew(int docID) { return oldToNew[docID]; }

        @Override
        public int newToOld(int docID) { return newToOld[docID]; }
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

    // ---- Distance functions for byte vectors ----

    /**
     * Compute squared L2 distance between a byte vector and a float centroid.
     * Each byte is treated as an unsigned value [0, 255].
     */
    static float squareDistanceByteToCentroid(byte[] vec, float[] centroid) {
        float sum = 0;
        for (int i = 0; i < centroid.length; i++) {
            float diff = (vec[i] & 0xFF) - centroid[i];
            sum += diff * diff;
        }
        return sum;
    }

    /**
     * Compute dot product between a byte vector (unsigned) and a float centroid.
     */
    static float dotProductByteToCentroid(byte[] vec, float[] centroid) {
        float sum = 0;
        for (int i = 0; i < centroid.length; i++) {
            sum += (vec[i] & 0xFF) * centroid[i];
        }
        return sum;
    }

    /**
     * Compute centroid (mean) of a set of byte vectors, storing result as float.
     */
    static void computeCentroid(IntsRef ids, ByteVectorValues vectors, float[] centroid) throws IOException {
        Arrays.fill(centroid, 0);
        for (int i = ids.offset; i < ids.offset + ids.length; i++) {
            byte[] vec = vectors.vectorValue(ids.ints[i]);
            for (int d = 0; d < centroid.length; d++) {
                centroid[d] += (vec[d] & 0xFF);
            }
        }
        float scale = 1.0f / ids.length;
        for (int d = 0; d < centroid.length; d++) {
            centroid[d] *= scale;
        }
    }

    static void vectorSubtract(float[] u, float[] v, float[] result) {
        for (int i = 0; i < u.length; i++) {
            result[i] = u[i] - v[i];
        }
    }

    static float dotProduct(float[] u, float[] v) {
        float sum = 0;
        for (int i = 0; i < u.length; i++) {
            sum += u[i] * v[i];
        }
        return sum;
    }

    static void vectorScalarMul(float x, float[] v) {
        for (int i = 0; i < v.length; i++) {
            v[i] *= x;
        }
    }

    // ---- ReorderTask: recursive bisection on byte vectors ----

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
            ByteVectorValues vectors = state.vectors;
            float[] leftCentroid = state.leftCentroid;
            float[] rightCentroid = state.rightCentroid;
            float[] scratch = state.scratch;

            try {
                BpByteVectorReorderer.computeCentroid(left, vectors, leftCentroid);
                BpByteVectorReorderer.computeCentroid(right, vectors, rightCentroid);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }

            for (int iter = 0; iter < maxIters; ++iter) {
                int moved;
                try {
                    moved = shuffle(vectors, ids, right.offset, leftCentroid, rightCentroid, scratch, biases);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
                if (moved == 0) break;
                if (moved > MAX_CENTROID_UPDATES) {
                    try {
                        BpByteVectorReorderer.computeCentroid(left, vectors, leftCentroid);
                        BpByteVectorReorderer.computeCentroid(right, vectors, rightCentroid);
                    } catch (IOException e) {
                        throw new UncheckedIOException(e);
                    }
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

        private int shuffle(ByteVectorValues vectors, IntsRef ids, int midPoint,
                            float[] leftCentroid, float[] rightCentroid, float[] scratch,
                            float[] biases) throws IOException {
            new ComputeBiasTask(ids.ints, biases, ids.offset, ids.offset + ids.length,
                leftCentroid, rightCentroid, threadLocal, executor, depth).compute();

            vectorSubtract(leftCentroid, rightCentroid, scratch);
            float scale = (float) Math.sqrt(dotProduct(scratch, scratch));
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

                    if (i < midPoint == j < midPoint) {
                        int tmpDoc = ids.ints[i];
                        ids.ints[i] = ids.ints[j];
                        ids.ints[j] = tmpDoc;
                    } else {
                        count++;
                        int tmpDoc = ids.ints[i];
                        ids.ints[i] = ids.ints[j];
                        ids.ints[j] = tmpDoc;
                    }
                }
            }

            Selector selector = new Selector();
            selector.select(ids.offset, ids.offset + ids.length, midPoint);
            return selector.count;
        }
    }

    // ---- ComputeBiasTask: parallel bias computation on byte vectors ----

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
                        // Bias = distance_to_left - distance_to_right
                        // Negative bias → vector is closer to left centroid
                        biases[i] = squareDistanceByteToCentroid(vec, leftCentroid)
                                  - squareDistanceByteToCentroid(vec, rightCentroid);
                    }
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            }
        }
    }

    // ---- Public API ----

    /**
     * Compute a permutation from byte vector values using Hamming/L2 distance.
     * This is the main entry point for quantized-vector BP reordering.
     *
     * @param vectors  ByteVectorValues — typically mmap-backed quantized vectors from .faiss
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
        return 2L * Integer.BYTES * maxDoc;
    }

    @Override
    public Sorter.DocMap computeDocMap(CodecReader reader, Directory tempDir, Executor executor) throws IOException {
        throw new UnsupportedOperationException("Use computeValueMap(ByteVectorValues, TaskExecutor) directly");
    }
}
