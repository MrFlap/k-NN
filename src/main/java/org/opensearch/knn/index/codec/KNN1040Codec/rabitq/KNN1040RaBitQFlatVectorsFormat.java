/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec.rabitq;

import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;

/**
 * Raw float vectors (Lucene99 flat, used for graph build, merge and exact search) plus tiered Extended RaBitQ records
 * against the segment's centroid: a 1-bit coarse tier that scores HNSW traversal and a 7-bit refine tier that rescores
 * the traversal's results.
 *
 * <p>Files: {@code .rbqm} per-field metadata and centroid, {@code .rbqc} coarse rows, {@code .rbqr} refine rows; see
 * {@link TieredRaBitQRecords}.
 */
public class KNN1040RaBitQFlatVectorsFormat extends FlatVectorsFormat {

    static final String NAME = "KNN1040RaBitQFlatVectorsFormat";
    static final String META_CODEC_NAME = "KNN1040RaBitQFlatVectorsFormatMeta";
    static final String COARSE_CODEC_NAME = "KNN1040RaBitQFlatVectorsFormatCoarse";
    static final String REFINE_CODEC_NAME = "KNN1040RaBitQFlatVectorsFormatRefine";
    static final String META_EXTENSION = "rbqm";
    static final String COARSE_EXTENSION = "rbqc";
    static final String REFINE_EXTENSION = "rbqr";
    static final int VERSION_START = 0;
    /** Adds the Hadamard rotation seed per field; version 0 fields are unrotated. */
    static final int VERSION_ROTATION = 1;
    static final int VERSION_CURRENT = VERSION_ROTATION;

    private static final FlatVectorsScorer RAW_SCORER = FlatVectorScorerUtil.getLucene99FlatVectorsScorer();
    private static final FlatVectorsFormat RAW_FORMAT = new Lucene99FlatVectorsFormat(RAW_SCORER);

    public KNN1040RaBitQFlatVectorsFormat() {
        super(NAME);
    }

    @Override
    public KNN1040RaBitQFlatVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new KNN1040RaBitQFlatVectorsWriter(state, RAW_FORMAT.fieldsWriter(state));
    }

    @Override
    public KNN1040RaBitQFlatVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new KNN1040RaBitQFlatVectorsReader(state, RAW_FORMAT.fieldsReader(state));
    }

    @Override
    public int getMaxDimensions(String fieldName) {
        return KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE);
    }

    @Override
    public String toString() {
        return NAME + "(raw=" + RAW_FORMAT + ")";
    }
}
