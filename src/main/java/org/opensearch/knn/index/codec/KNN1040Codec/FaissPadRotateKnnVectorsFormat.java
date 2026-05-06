/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import com.google.common.annotations.VisibleForTesting;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.codec.padrotate.PadRotateScalarQuantizedVectorsFormat;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;

/**
 * FAISS codec format for 8x compression: FAISS HNSW graph (.faiss) alongside a pad-and-rotate
 * scalar-quantized flat file (.peq/.pemq) that expands each D-dim input vector to 4*D before
 * single-bit scalar quantization.
 *
 * <p>Same overall shape as {@link org.opensearch.knn.index.codec.KNN1040Codec.Faiss1040ScalarQuantizedKnnVectorsFormat}
 * — the FAISS native engine builds the graph, the flat layer handles storage and quantized
 * scoring. The only difference is the underlying flat format:
 * {@link PadRotateScalarQuantizedVectorsFormat} instead of Lucene's straight 1-bit SQ.
 *
 * <p>Selected when {@code compression_level: 8x} is configured for a FAISS field. See
 * {@code FaissCodecFormatResolver} for routing.
 */
@Log4j2
public class FaissPadRotateKnnVectorsFormat extends KnnVectorsFormat {

    private static final String FORMAT_NAME = "FaissPadRotateKnnVectorsFormat";

    // Shared across all format instances; PadRotateScalarQuantizedVectorsFormat is stateless.
    private static final PadRotateScalarQuantizedVectorsFormat FLAT_FORMAT = new PadRotateScalarQuantizedVectorsFormat();

    private final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory;

    @VisibleForTesting
    static PadRotateScalarQuantizedVectorsFormat getFlatFormat() {
        return FLAT_FORMAT;
    }

    public FaissPadRotateKnnVectorsFormat() {
        this(new NativeIndexBuildStrategyFactory());
    }

    public FaissPadRotateKnnVectorsFormat(final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory) {
        super(FORMAT_NAME);
        this.nativeIndexBuildStrategyFactory = nativeIndexBuildStrategyFactory;
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new Faiss1040ScalarQuantizedKnnVectorsWriter(
            state,
            FLAT_FORMAT.fieldsWriter(state),
            FLAT_FORMAT::fieldsReader,
            nativeIndexBuildStrategyFactory
        );
    }

    @Override
    public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new Faiss1040ScalarQuantizedKnnVectorsReader(state, new Faiss1040ScalarQuantizedFlatVectorsReader(FLAT_FORMAT.fieldsReader(state)));
    }

    @Override
    public int getMaxDimensions(String fieldName) {
        return KNNEngine.getMaxDimensionByEngine(KNNEngine.FAISS);
    }

    @Override
    public String toString() {
        return FORMAT_NAME + "(flatFormat=" + FLAT_FORMAT + ")";
    }
}
