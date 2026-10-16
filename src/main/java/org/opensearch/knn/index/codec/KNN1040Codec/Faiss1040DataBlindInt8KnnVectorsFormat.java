/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.index.FieldInfo;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;

/**
 * Faiss SQ format for data-blind int8 fields: the native HNSW graph as in {@link Faiss1040ScalarQuantizedKnnVectorsFormat},
 * with flat vectors in {@link KNN1040DataBlindInt8VectorsFormat}. A separate format name lets the read path open the int8
 * files without inspecting the segment.
 */
public class Faiss1040DataBlindInt8KnnVectorsFormat extends Faiss1040ScalarQuantizedKnnVectorsFormat {

    private static final String FORMAT_NAME = "Faiss1040DataBlindInt8KnnVectorsFormat";
    private static final KNN1040DataBlindInt8VectorsFormat FLAT_FORMAT = new KNN1040DataBlindInt8VectorsFormat();

    public Faiss1040DataBlindInt8KnnVectorsFormat() {
        this(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE, new NativeIndexBuildStrategyFactory());
    }

    public Faiss1040DataBlindInt8KnnVectorsFormat(
        final int approximateThreshold,
        final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory
    ) {
        super(FORMAT_NAME, approximateThreshold, nativeIndexBuildStrategyFactory, VectorDataType.FLOAT);
    }

    @Override
    protected FlatVectorsFormat flatFormatFor(final FieldInfo fieldInfo) {
        return FLAT_FORMAT;
    }

    @Override
    protected FlatVectorsFormat flatFormatFor() {
        return FLAT_FORMAT;
    }
}
