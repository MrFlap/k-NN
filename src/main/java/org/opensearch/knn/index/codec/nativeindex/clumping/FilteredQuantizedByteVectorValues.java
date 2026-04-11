/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.clumping;

import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene104.QuantizedByteVectorValues;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

import java.io.IOException;
import java.util.List;

/**
 * A {@link QuantizedByteVectorValues} wrapper that exposes only the marker-ordinal subset of a
 * full-segment quantized values instance.
 *
 * <p>{@link org.opensearch.knn.index.codec.nativeindex.MemOptimizedScalarQuantizedIndexBuildStrategy}
 * accesses quantized data by ordinal (0, 1, 2, ...) and expects the ordinal space to match the
 * vectors being indexed. When {@link ClumpingIndexBuildStrategy} builds the marker-only sub-index,
 * it passes a {@link FilteredKNNVectorValues} that skips hidden vectors, so the ordinals it
 * presents are 0..numMarkers-1. This class remaps those compact ordinals back to the original
 * full-segment ordinals so that the correct quantized codes and correction factors are returned.
 *
 * <p>Example: if markers are at full-segment ordinals [0, 8, 16, ...], then
 * {@code vectorValue(0)} returns the quantized code for ordinal 0,
 * {@code vectorValue(1)} returns the quantized code for ordinal 8, etc.
 */
class FilteredQuantizedByteVectorValues extends QuantizedByteVectorValues {

    private final QuantizedByteVectorValues delegate;
    /** Maps compact marker ordinal → original full-segment ordinal. */
    private final int[] markerOrdinals;

    /**
     * @param delegate       the full-segment {@link QuantizedByteVectorValues}
     * @param markerOrdinals list of original ordinals for each marker, in insertion order
     */
    FilteredQuantizedByteVectorValues(QuantizedByteVectorValues delegate, List<Integer> markerOrdinals) {
        this.delegate = delegate;
        this.markerOrdinals = markerOrdinals.stream().mapToInt(Integer::intValue).toArray();
    }

    @Override
    public byte[] vectorValue(int markerOrd) throws IOException {
        return delegate.vectorValue(markerOrdinals[markerOrd]);
    }

    @Override
    public OptimizedScalarQuantizer.QuantizationResult getCorrectiveTerms(int markerOrd) throws IOException {
        return delegate.getCorrectiveTerms(markerOrdinals[markerOrd]);
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
    public int dimension() {
        return delegate.dimension();
    }

    @Override
    public int size() {
        return markerOrdinals.length;
    }

    @Override
    public IndexInput getSlice() {
        return delegate.getSlice();
    }

    @Override
    public org.apache.lucene.search.VectorScorer scorer(float[] target) throws IOException {
        return delegate.scorer(target);
    }

    @Override
    public QuantizedByteVectorValues copy() throws IOException {
        return new FilteredQuantizedByteVectorValues(delegate.copy(), toList(markerOrdinals));
    }

    private static List<Integer> toList(int[] arr) {
        List<Integer> list = new java.util.ArrayList<>(arr.length);
        for (int v : arr) list.add(v);
        return list;
    }
}
