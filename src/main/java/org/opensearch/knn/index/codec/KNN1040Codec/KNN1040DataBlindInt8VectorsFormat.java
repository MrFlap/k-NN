/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.memoryoptsearch.faiss.FlatVectorsScorerProvider;

import java.io.IOException;

/**
 * Flat format for data-blind int8 quantization ({@link DataBlindInt8Quantizer}). Full-precision floats go to the
 * Lucene99 flat format ({@code .vec}) for rescoring; the int8 records ({@link DataBlindInt8VectorValues}) go to
 * {@code .vdbi}, with per-field metadata in {@code .vdbim}.
 *
 * <p>Data-blind means there is no per-segment state: merges re-encode the merged floats and reproduce the same bytes.
 */
public class KNN1040DataBlindInt8VectorsFormat extends FlatVectorsFormat {

    static final String NAME = "KNN1040DataBlindInt8VectorsFormat";
    static final String META_CODEC_NAME = "KNN1040DataBlindInt8VectorsFormatMeta";
    static final String DATA_CODEC_NAME = "KNN1040DataBlindInt8VectorsFormatData";
    static final String META_EXTENSION = "vdbim";
    static final String DATA_EXTENSION = "vdbi";
    static final int VERSION_START = 0;
    static final int VERSION_CURRENT = VERSION_START;

    private static final Lucene99FlatVectorsFormat RAW_VECTOR_FORMAT = new Lucene99FlatVectorsFormat(
        FlatVectorsScorerProvider.getLucene99FlatVectorsScorer()
    );

    public KNN1040DataBlindInt8VectorsFormat() {
        super(NAME);
    }

    @Override
    public FlatVectorsWriter fieldsWriter(final SegmentWriteState state) throws IOException {
        return new KNN1040DataBlindInt8VectorsWriter(state, RAW_VECTOR_FORMAT.fieldsWriter(state));
    }

    @Override
    public FlatVectorsReader fieldsReader(final SegmentReadState state) throws IOException {
        return new KNN1040DataBlindInt8VectorsReader(state, RAW_VECTOR_FORMAT.fieldsReader(state));
    }

    @Override
    public int getMaxDimensions(final String fieldName) {
        return KNNEngine.getMaxDimensionByEngine(KNNEngine.FAISS);
    }

    @Override
    public String toString() {
        return NAME + "(rawVectorFormat=" + RAW_VECTOR_FORMAT + ")";
    }
}
