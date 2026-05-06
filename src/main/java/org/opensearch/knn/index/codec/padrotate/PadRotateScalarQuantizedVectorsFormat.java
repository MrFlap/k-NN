/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.padrotate;

import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorScorer;
import org.apache.lucene.codecs.lucene104.PadRotateScalarQuantizedVectorsReader;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

import java.io.IOException;

/**
 * Flat vectors format that pads each {@code D}-dim input vector to {@code PAD_FACTOR * D}, applies
 * a per-segment random orthogonal rotation, and stores the result using Lucene-104-style 1-bit
 * scalar quantization ({@code SINGLE_BIT_QUERY_NIBBLE}). No raw flat vectors are stored on disk;
 * reconstructed vectors are produced on read via dequantize + inverse-rotate + truncate.
 *
 * <p>Compared to plain 32x single-bit quantization at the original dimension, this format trades
 * 4x storage for better recall by spreading input variance across 4x more dimensions before
 * quantization. Net on-disk compression vs float32 input is approximately 8x.
 */
public class PadRotateScalarQuantizedVectorsFormat extends FlatVectorsFormat {

    public static final String NAME = "PadRotateScalarQuantizedVectorsFormat";

    public static final int VERSION_START = 0;
    public static final int VERSION_CURRENT = VERSION_START;

    public static final String META_CODEC_NAME = "PadRotateSQVectorsFormatMeta";
    public static final String DATA_CODEC_NAME = "PadRotateSQVectorsFormatData";
    public static final String META_EXTENSION = "pemq";
    public static final String DATA_EXTENSION = "peq";

    public static final int DIRECT_MONOTONIC_BLOCK_SHIFT = 16;

    private static final FlatVectorsScorer RAW_SCORER = FlatVectorScorerUtil.getLucene99FlatVectorsScorer();
    private static final Lucene104ScalarQuantizedVectorScorer QUANTIZED_SCORER = new Lucene104ScalarQuantizedVectorScorer(RAW_SCORER);

    public PadRotateScalarQuantizedVectorsFormat() {
        super(NAME);
    }

    @Override
    public FlatVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new PadRotateScalarQuantizedVectorsWriter(state, QUANTIZED_SCORER);
    }

    @Override
    public FlatVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new PadRotateScalarQuantizedVectorsReader(state, QUANTIZED_SCORER);
    }

    @Override
    public int getMaxDimensions(String fieldName) {
        // We internally operate at PAD_FACTOR * originalDim, so the user-facing cap is Lucene's
        // default / PAD_FACTOR.
        return 1024 / PadRotate.PAD_FACTOR;
    }

    @Override
    public String toString() {
        return NAME + "(padFactor=" + PadRotate.PAD_FACTOR + ")";
    }
}
