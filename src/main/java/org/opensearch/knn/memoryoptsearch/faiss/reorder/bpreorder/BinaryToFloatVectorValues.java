/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder.bpreorder;

import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.VectorScorer;

import java.io.IOException;

/**
 * Adapts packed binary {@link ByteVectorValues} into {@link FloatVectorValues} by decoding
 * each bit into +1.0f (bit=1) or -1.0f (bit=0).
 * <p>
 * This allows Lucene's {@code BpVectorReorderer} to operate on binary quantized vectors
 * using its existing SIMD-optimized L2 distance functions. L2 distance on {+1, -1} vectors
 * is monotonically related to Hamming distance: {@code L2² = 4 * hamming_distance}.
 * <p>
 * The decoded dimension is {@code codeSize * 8} (e.g., 96 bytes → 768 floats for 32x BQ
 * on 768-dim vectors). Each {@code vectorValue()} call decodes on the fly into a reusable
 * buffer — no heap allocation per call.
 */
public class BinaryToFloatVectorValues extends FloatVectorValues {

    private final ByteVectorValues byteValues;
    private final int codeSize;
    private final int decodedDim;
    private final float[] buffer;

    public BinaryToFloatVectorValues(ByteVectorValues byteValues) throws IOException {
        this.byteValues = byteValues;
        // Detect actual byte vector length from the first vector, since dimension()
        // may return the original float dimension for binary FAISS indices.
        this.codeSize = byteValues.size() > 0 ? byteValues.vectorValue(0).length : byteValues.dimension();
        this.decodedDim = codeSize * 8;
        this.buffer = new float[decodedDim];
    }

    private BinaryToFloatVectorValues(ByteVectorValues byteValues, int codeSize) {
        this.byteValues = byteValues;
        this.codeSize = codeSize;
        this.decodedDim = codeSize * 8;
        this.buffer = new float[decodedDim];
    }

    @Override
    public float[] vectorValue(int ord) throws IOException {
        byte[] packed = byteValues.vectorValue(ord);
        for (int byteIdx = 0; byteIdx < codeSize; byteIdx++) {
            int b = packed[byteIdx] & 0xFF;
            int base = byteIdx * 8;
            for (int bit = 0; bit < 8; bit++) {
                buffer[base + bit] = ((b & (1 << bit)) != 0) ? 1.0f : -1.0f;
            }
        }
        return buffer;
    }

    @Override
    public int dimension() {
        return decodedDim;
    }

    @Override
    public int size() {
        return byteValues.size();
    }

    @Override
    public BinaryToFloatVectorValues copy() throws IOException {
        return new BinaryToFloatVectorValues(byteValues.copy(), codeSize);
    }

    @Override
    public DocIndexIterator iterator() {
        throw new UnsupportedOperationException("Random access only");
    }

    @Override
    public int ordToDoc(int ord) {
        throw new UnsupportedOperationException("Random access only");
    }

    @Override
    public VectorScorer scorer(float[] query) {
        throw new UnsupportedOperationException("Random access only");
    }
}
