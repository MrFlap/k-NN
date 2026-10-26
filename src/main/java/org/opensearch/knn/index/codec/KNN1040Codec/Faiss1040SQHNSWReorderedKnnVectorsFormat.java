/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.opensearch.knn.index.KNNSettings;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.FilterIndexOutput;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;
import org.opensearch.knn.index.codec.locality.LocalityOrderedQuantizedVectorsReader;
import org.opensearch.knn.index.codec.locality.LocalityOrderedQuantizedVectorsWriter;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.memoryoptsearch.faiss.FlatVectorsScorerProvider;

import java.io.IOException;

/**
 * Dedicated format for Faiss SQ vector fields.
 *
 * <p>Uses {@link KNN1040LocalityAwareSQVectorsFormat} with 1-bit quantization
 * ({@link ScalarEncoding#SINGLE_BIT_QUERY_NIBBLE}) for flat vector storage (.vec/.veq files),
 * while HNSW graph construction is delegated to the native Faiss engine (.faiss files).
 *
 * <p>A field is routed to this format when its method parameters contain
 * {@code "encoder": {"name": "sq", "bits": 1}}. See {@code FaissCodecFormatResolver} for the
 * routing logic in {@code BasePerFieldKnnVectorsFormat.getKnnVectorsFormatForField}.
 *
 * @see LocalityOrderedQuantizedVectorsWriter for reordering
 * @see LocalityOrderedQuantizedVectorsReader for reordering
 */
@Log4j2
public class Faiss1040SQHNSWReorderedKnnVectorsFormat extends KnnVectorsFormat {

    private static final String FORMAT_NAME = "Faiss1040SQHNSWReorderedKnnVectorsFormat";

    // Lucene99FlatVectorsFormat's package-private VECTOR_DATA_EXTENSION and VECTOR_DATA_CODEC_NAME.
    private static final String RAW_VECTOR_DATA_EXTENSION = "vec";
    private static final String RAW_VECTOR_DATA_CODEC_NAME = "Lucene99FlatVectorsFormatData";
    static final int RAW_VECTOR_DATA_ALIGNMENT = 4096;

    // TODO : We have to make it scalable for other encoding types, not limit this on `ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE`.
    private static final KNN1040ScalarQuantizedVectorsFormat faissSqFlatFormat = new KNN1040ScalarQuantizedVectorsFormat(
        ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE
    );

    private static final Lucene99FlatVectorsFormat RAW_VECTOR_FORMAT = new Lucene99FlatVectorsFormat(
        FlatVectorsScorerProvider.getLucene99FlatVectorsScorer()
    );

    private static final KNN1040ScalarQuantizedVectorScorer KNN_1040_SCALAR_QUANTIZED_VECTOR_SCORER = FlatVectorsScorerProvider
        .getKNN1040ScalarQuantizedVectorScorer(FlatVectorsScorerProvider.getLucene99FlatVectorsScorer());

    private final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory;
    private final int approximateThreshold;

    /**
     * Constructor needed for SPI
     */
    public Faiss1040SQHNSWReorderedKnnVectorsFormat() {
        this(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE, new NativeIndexBuildStrategyFactory());
    }

    public Faiss1040SQHNSWReorderedKnnVectorsFormat(final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory) {
        this(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE, nativeIndexBuildStrategyFactory);
    }

    public Faiss1040SQHNSWReorderedKnnVectorsFormat(
        final int approximateThreshold,
        final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory
    ) {
        super(FORMAT_NAME);
        this.approximateThreshold = approximateThreshold;
        this.nativeIndexBuildStrategyFactory = nativeIndexBuildStrategyFactory;
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new Faiss1040SQHNSWReorderedWriter(
            state,
            faissSqFlatFormat.fieldsWriter(withPageAlignedRawVectorData(state)),
            faissSqFlatFormat::fieldsReader,
            new LocalityOrderedQuantizedVectorsWriter(state, ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE),
            nativeIndexBuildStrategyFactory,
            approximateThreshold
        );
    }

    @Override
    public KnnVectorsReader fieldsReader(final SegmentReadState state) throws IOException {
        return new Faiss1040SQHNSWReorderedReader(
            state,
            new LocalityOrderedQuantizedVectorsReader(state, RAW_VECTOR_FORMAT.fieldsReader(state), KNN_1040_SCALAR_QUANTIZED_VECTOR_SCORER)
        );
    }

    /**
     * Pads the raw {@code .vec} file with zeros from the end of its header to {@value #RAW_VECTOR_DATA_ALIGNMENT}, so
     * {@code Lucene99FlatVectorsWriter}'s 64-byte alignment leaves this format's single field starting on a page and
     * a vector whose size is a multiple of 4096 covers whole pages. The reader takes the data offset from the
     * {@code .vemf} metadata, so files written before or after this read the same way.
     */
    static SegmentWriteState withPageAlignedRawVectorData(final SegmentWriteState state) {
        final String vecFileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, RAW_VECTOR_DATA_EXTENSION);
        final long headerLength = CodecUtil.indexHeaderLength(RAW_VECTOR_DATA_CODEC_NAME, state.segmentSuffix);
        final FilterDirectory directory = new FilterDirectory(state.directory) {
            @Override
            public IndexOutput createOutput(final String name, final IOContext context) throws IOException {
                final IndexOutput output = super.createOutput(name, context);
                if (name.equals(vecFileName) == false) {
                    return output;
                }
                return new FilterIndexOutput("page-aligned " + output, name, output) {
                    @Override
                    public void writeByte(final byte b) throws IOException {
                        out.writeByte(b);
                        padAfterHeader();
                    }

                    @Override
                    public void writeBytes(final byte[] b, final int offset, final int length) throws IOException {
                        out.writeBytes(b, offset, length);
                        padAfterHeader();
                    }

                    private void padAfterHeader() throws IOException {
                        if (out.getFilePointer() == headerLength) {
                            out.alignFilePointer(RAW_VECTOR_DATA_ALIGNMENT);
                        }
                    }
                };
            }
        };
        return new SegmentWriteState(
            state.infoStream,
            directory,
            state.segmentInfo,
            state.fieldInfos,
            state.segUpdates,
            state.context,
            state.segmentSuffix
        );
    }

    /**
     * Uses Faiss max dimension since the HNSW graph is built by native Faiss.
     */
    @Override
    public int getMaxDimensions(String fieldName) {
        return KNNEngine.getMaxDimensionByEngine(KNNEngine.FAISS);
    }

    @Override
    public String toString() {
        return this.getClass().getSimpleName() + "(name=" + this.getClass().getSimpleName() + ")";
    }
}
