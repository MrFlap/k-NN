/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;

import java.io.IOException;

/**
 * Off-heap data-blind int8 codes, one fixed-size record per vector ordinal:
 * <pre>
 *   [code: dim unsigned bytes][scale: f32][Σq: i32][‖v‖²: f32][unused: 4 bytes]
 * </pre>
 * All little-endian. The record is byte-for-byte what native {@code FaissSQFlat} stores per vector (code followed by
 * four 4-byte slots), so the graph build and native search read it without conversion.
 *
 * <p>Extends {@link QuantizedByteVectorValues} only so it travels the existing Faiss SQ write and search plumbing.
 * Ordinals are shared with the paired full-precision values, which own the ord-to-doc mapping. The corrections are
 * not Lucene's OSQ corrective terms, so {@link #getCorrectiveTerms} throws instead of letting an OSQ scorer read them.
 */
public class DataBlindInt8VectorValues extends QuantizedByteVectorValues {

    /** Bytes after the code: scale, Σq, ‖v‖² and four unused bytes. */
    public static final int CORRECTION_BYTES = 16;

    private final int dimension;
    private final int size;
    private final int recordBytes;
    private final IndexInput slice;
    private final byte[] code;
    private int lastOrd = -1;
    private float scale;
    private int sum;
    private float sqNorm;

    public DataBlindInt8VectorValues(final int dimension, final int size, final IndexInput slice) {
        this.dimension = dimension;
        this.size = size;
        this.recordBytes = dimension + CORRECTION_BYTES;
        this.slice = slice;
        this.code = new byte[dimension];
    }

    @Override
    public byte[] vectorValue(final int ord) throws IOException {
        load(ord);
        return code;
    }

    public float scale(final int ord) throws IOException {
        load(ord);
        return scale;
    }

    public int sum(final int ord) throws IOException {
        load(ord);
        return sum;
    }

    public float sqNorm(final int ord) throws IOException {
        load(ord);
        return sqNorm;
    }

    private void load(final int ord) throws IOException {
        if (ord == lastOrd) {
            return;
        }
        slice.seek((long) ord * recordBytes);
        slice.readBytes(code, 0, dimension);
        scale = Float.intBitsToFloat(slice.readInt());
        sum = slice.readInt();
        sqNorm = Float.intBitsToFloat(slice.readInt());
        lastOrd = ord;
    }

    @Override
    public OptimizedScalarQuantizer.QuantizationResult getCorrectiveTerms(final int vectorOrd) {
        throw new UnsupportedOperationException("Data-blind int8 stores scale, Σq and ‖v‖², not OSQ corrective terms");
    }

    @Override
    public OptimizedScalarQuantizer getQuantizer() {
        throw new UnsupportedOperationException("Data-blind int8 is not quantized with OptimizedScalarQuantizer");
    }

    @Override
    public ScalarEncoding getScalarEncoding() {
        return ScalarEncoding.UNSIGNED_BYTE;
    }

    @Override
    public float[] getCentroid() {
        // Data-blind: vectors are quantized absolutely, i.e. against a zero centroid.
        return new float[dimension];
    }

    @Override
    public float getCentroidDP() {
        return 0f;
    }

    /** One record, code and corrections, so prefetching strides over whole records. */
    @Override
    public int getVectorByteLength() {
        return recordBytes;
    }

    @Override
    public IndexInput getSlice() {
        return slice;
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public DataBlindInt8VectorValues copy() {
        return new DataBlindInt8VectorValues(dimension, size, slice.clone());
    }
}
