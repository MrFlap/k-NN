/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene104.QuantizedByteVectorValues;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.packed.DirectMonotonicReader;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

import java.io.IOException;

/**
 * A {@link QuantizedByteVectorValues} wrapper that remaps HNSW ordinals to full-segment ordinals
 * via a {@link DirectMonotonicReader} id-map.
 *
 * <p>When a Faiss index is built over a sparse subset of vectors (e.g., clumping markers),
 * the HNSW graph uses compact ordinals 0..N-1 while the underlying quantized byte storage
 * uses full-segment ordinals. This wrapper translates {@code vectorValue(i)} and
 * {@code getCorrectiveTerms(i)} calls from compact ordinal {@code i} to the corresponding
 * full-segment ordinal {@code idMappingReader.get(i)}, ensuring the scorer reads the correct
 * quantized data for each marker.
 */
class SparseQuantizedByteVectorValues extends QuantizedByteVectorValues {

    private final QuantizedByteVectorValues delegate;
    private final DirectMonotonicReader idMappingReader;

    SparseQuantizedByteVectorValues(QuantizedByteVectorValues delegate, DirectMonotonicReader idMappingReader) {
        this.delegate = delegate;
        this.idMappingReader = idMappingReader;
    }

    private int remap(int compactOrd) {
        return (int) idMappingReader.get(compactOrd);
    }

    @Override
    public byte[] vectorValue(int compactOrd) throws IOException {
        return delegate.vectorValue(remap(compactOrd));
    }

    @Override
    public OptimizedScalarQuantizer.QuantizationResult getCorrectiveTerms(int compactOrd) throws IOException {
        return delegate.getCorrectiveTerms(remap(compactOrd));
    }

    @Override
    public float getCentroidDP() throws IOException {
        return delegate.getCentroidDP();
    }

    @Override
    public float[] getCentroid() throws IOException {
        return delegate.getCentroid();
    }

    @Override
    public OptimizedScalarQuantizer getQuantizer() {
        return delegate.getQuantizer();
    }

    @Override
    public Lucene104ScalarQuantizedVectorsFormat.ScalarEncoding getScalarEncoding() {
        return delegate.getScalarEncoding();
    }

    @Override
    public org.apache.lucene.search.VectorScorer scorer(float[] target) throws IOException {
        return delegate.scorer(target);
    }

    @Override
    public int dimension() {
        return delegate.dimension();
    }

    @Override
    public int size() {
        return delegate.size();
    }

    @Override
    public IndexInput getSlice() {
        return delegate.getSlice();
    }

    @Override
    public QuantizedByteVectorValues copy() throws IOException {
        return new SparseQuantizedByteVectorValues(delegate.copy(), idMappingReader);
    }
}
