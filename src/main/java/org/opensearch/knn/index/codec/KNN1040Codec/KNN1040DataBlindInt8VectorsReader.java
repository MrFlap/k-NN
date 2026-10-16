/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Reads {@link KNN1040DataBlindInt8VectorsFormat}. {@link #getFloatVectorValues} returns the full-precision values paired
 * with their int8 codes in a {@link ScalarQuantizedFloatVectorValues}, the shape the Faiss SQ search path consumes.
 */
class KNN1040DataBlindInt8VectorsReader extends FlatVectorsReader {
    private static final long SHALLOW_SIZE = RamUsageEstimator.shallowSizeOfInstance(KNN1040DataBlindInt8VectorsReader.class);

    private record FieldEntry(FieldInfo fieldInfo, int dimension, int size, long offset, long length) {
    }

    private final FlatVectorsReader rawVectorsReader;
    private final Map<String, FieldEntry> fields = new HashMap<>();
    private final IndexInput data;

    KNN1040DataBlindInt8VectorsReader(final SegmentReadState state, final FlatVectorsReader rawVectorsReader) throws IOException {
        this.rawVectorsReader = rawVectorsReader;
        IndexInput dataIn = null;
        try {
            final int version = readMeta(state);
            dataIn = state.directory.openInput(fileName(state, KNN1040DataBlindInt8VectorsFormat.DATA_EXTENSION), state.context);
            final int dataVersion = CodecUtil.checkIndexHeader(
                dataIn,
                KNN1040DataBlindInt8VectorsFormat.DATA_CODEC_NAME,
                KNN1040DataBlindInt8VectorsFormat.VERSION_START,
                KNN1040DataBlindInt8VectorsFormat.VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix
            );
            if (version != dataVersion) {
                throw new CorruptIndexException("Format versions mismatch: meta=" + version + ", data=" + dataVersion, dataIn);
            }
            CodecUtil.retrieveChecksum(dataIn);
        } catch (Throwable t) {
            IOUtils.closeWhileSuppressingExceptions(t, dataIn, rawVectorsReader);
            throw t;
        }
        this.data = dataIn;
    }

    private int readMeta(final SegmentReadState state) throws IOException {
        try (
            ChecksumIndexInput meta = state.directory.openChecksumInput(fileName(state, KNN1040DataBlindInt8VectorsFormat.META_EXTENSION))
        ) {
            Throwable priorE = null;
            int version = -1;
            try {
                version = CodecUtil.checkIndexHeader(
                    meta,
                    KNN1040DataBlindInt8VectorsFormat.META_CODEC_NAME,
                    KNN1040DataBlindInt8VectorsFormat.VERSION_START,
                    KNN1040DataBlindInt8VectorsFormat.VERSION_CURRENT,
                    state.segmentInfo.getId(),
                    state.segmentSuffix
                );
                for (int number = meta.readInt(); number != -1; number = meta.readInt()) {
                    final FieldInfo info = state.fieldInfos.fieldInfo(number);
                    if (info == null) {
                        throw new CorruptIndexException("Invalid field number: " + number, meta);
                    }
                    fields.put(info.name, new FieldEntry(info, meta.readVInt(), meta.readVInt(), meta.readVLong(), meta.readVLong()));
                }
            } catch (Throwable t) {
                priorE = t;
            } finally {
                CodecUtil.checkFooter(meta, priorE);
            }
            return version;
        }
    }

    private static String fileName(final SegmentReadState state, final String extension) {
        return IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, extension);
    }

    @Override
    public FloatVectorValues getFloatVectorValues(final String field) throws IOException {
        final FloatVectorValues fullPrecision = rawVectorsReader.getFloatVectorValues(field);
        final FieldEntry entry = fields.get(field);
        if (fullPrecision == null || entry == null || entry.size() == 0) {
            return fullPrecision == null ? null : new ScalarQuantizedFloatVectorValues(fullPrecision, null);
        }
        final IndexInput slice = data.slice("data-blind-int8-codes", entry.offset(), entry.length());
        return new ScalarQuantizedFloatVectorValues(fullPrecision, new DataBlindInt8VectorValues(entry.dimension(), entry.size(), slice));
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(final String field, final float[] target) throws IOException {
        final ScalarQuantizedFloatVectorValues values = (ScalarQuantizedFloatVectorValues) getFloatVectorValues(field);
        if (values == null || values.getQuantizedVectorValues() == null) {
            return null;
        }
        return new DataBlindInt8RandomVectorScorer(
            values.getFloatVectorValues(),
            (DataBlindInt8VectorValues) values.getQuantizedVectorValues(),
            target,
            fields.get(field).fieldInfo().getVectorSimilarityFunction()
        );
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(final String field, final byte[] target) {
        throw new UnsupportedOperationException("Data-blind int8 supports float vectors only");
    }

    @Override
    public ByteVectorValues getByteVectorValues(final String field) {
        throw new UnsupportedOperationException("Data-blind int8 supports float vectors only");
    }

    @Override
    public FlatVectorsScorer getFlatVectorScorer(final String field) throws IOException {
        return rawVectorsReader.getFlatVectorScorer(field);
    }

    @Override
    public void checkIntegrity() throws IOException {
        rawVectorsReader.checkIntegrity();
        CodecUtil.checksumEntireFile(data);
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(data, rawVectorsReader);
    }

    @Override
    public long ramBytesUsed() {
        return SHALLOW_SIZE + rawVectorsReader.ramBytesUsed() + RamUsageEstimator.sizeOfMap(
            fields,
            RamUsageEstimator.shallowSizeOfInstance(FieldEntry.class)
        );
    }
}
