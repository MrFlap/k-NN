/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

/**
 * Writes {@link KNN1040DataBlindInt8VectorsFormat}: floats through the raw delegate, then one int8 record per ordinal in
 * the delegate's ordinal order, so the two files share ordinals.
 */
class KNN1040DataBlindInt8VectorsWriter extends FlatVectorsWriter {
    private static final long SHALLOW_SIZE = RamUsageEstimator.shallowSizeOfInstance(KNN1040DataBlindInt8VectorsWriter.class);

    private record Field(FieldInfo fieldInfo, FlatFieldVectorsWriter<float[]> vectors) {
    }

    private final FlatVectorsWriter rawVectorDelegate;
    private final List<Field> fields = new ArrayList<>();
    private final IndexOutput meta;
    private final IndexOutput data;
    private boolean finished;

    KNN1040DataBlindInt8VectorsWriter(final SegmentWriteState state, final FlatVectorsWriter rawVectorDelegate) throws IOException {
        super(rawVectorDelegate.getFlatVectorScorer());
        this.rawVectorDelegate = rawVectorDelegate;
        IndexOutput metaOut = null;
        IndexOutput dataOut = null;
        try {
            metaOut = state.directory.createOutput(fileName(state, KNN1040DataBlindInt8VectorsFormat.META_EXTENSION), state.context);
            dataOut = state.directory.createOutput(fileName(state, KNN1040DataBlindInt8VectorsFormat.DATA_EXTENSION), state.context);
            CodecUtil.writeIndexHeader(
                metaOut,
                KNN1040DataBlindInt8VectorsFormat.META_CODEC_NAME,
                KNN1040DataBlindInt8VectorsFormat.VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix
            );
            CodecUtil.writeIndexHeader(
                dataOut,
                KNN1040DataBlindInt8VectorsFormat.DATA_CODEC_NAME,
                KNN1040DataBlindInt8VectorsFormat.VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix
            );
        } catch (Throwable t) {
            IOUtils.closeWhileSuppressingExceptions(t, metaOut, dataOut, rawVectorDelegate);
            throw t;
        }
        this.meta = metaOut;
        this.data = dataOut;
    }

    private static String fileName(final SegmentWriteState state, final String extension) {
        return IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, extension);
    }

    @Override
    @SuppressWarnings("unchecked")
    public FlatFieldVectorsWriter<?> addField(final FieldInfo fieldInfo) throws IOException {
        if (fieldInfo.getVectorEncoding() != VectorEncoding.FLOAT32) {
            throw new IllegalArgumentException("Data-blind int8 supports float vectors only, field [" + fieldInfo.name + "] is not");
        }
        final FlatFieldVectorsWriter<float[]> vectors = (FlatFieldVectorsWriter<float[]>) rawVectorDelegate.addField(fieldInfo);
        fields.add(new Field(fieldInfo, vectors));
        return vectors;
    }

    @Override
    public void flush(final int maxDoc, final Sorter.DocMap sortMap) throws IOException {
        rawVectorDelegate.flush(maxDoc, sortMap);
        for (Field field : fields) {
            final List<float[]> vectors = field.vectors().getVectors();
            int[] newToOldOrd = null;
            if (sortMap != null) {
                newToOldOrd = new int[vectors.size()];
                mapOldOrdToNewOrd(field.vectors().getDocsWithFieldSet(), sortMap, null, newToOldOrd, null);
            }
            final long offset = data.getFilePointer();
            final Encoder encoder = new Encoder(field.fieldInfo().getVectorDimension());
            for (int ord = 0; ord < vectors.size(); ord++) {
                encoder.write(vectors.get(newToOldOrd == null ? ord : newToOldOrd[ord]));
            }
            writeMeta(field.fieldInfo(), vectors.size(), offset);
        }
    }

    @Override
    public void mergeOneFlatVectorField(final FieldInfo fieldInfo, final MergeState mergeState) throws IOException {
        rawVectorDelegate.mergeOneFlatVectorField(fieldInfo, mergeState);
        // Same iteration order as the raw delegate's merge, so ordinals line up. Data-blind, so re-encoding reproduces
        // each source segment's bytes.
        final FloatVectorValues merged = MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
        final long offset = data.getFilePointer();
        final Encoder encoder = new Encoder(fieldInfo.getVectorDimension());
        int count = 0;
        final KnnVectorValues.DocIndexIterator iterator = merged.iterator();
        for (int doc = iterator.nextDoc(); doc != NO_MORE_DOCS; doc = iterator.nextDoc()) {
            encoder.write(merged.vectorValue(iterator.index()));
            count++;
        }
        writeMeta(fieldInfo, count, offset);
    }

    private void writeMeta(final FieldInfo fieldInfo, final int count, final long offset) throws IOException {
        meta.writeInt(fieldInfo.number);
        meta.writeVInt(fieldInfo.getVectorDimension());
        meta.writeVInt(count);
        meta.writeVLong(offset);
        meta.writeVLong(data.getFilePointer() - offset);
    }

    /** Encodes vectors into {@link DataBlindInt8VectorValues} records appended to {@code data}. */
    private final class Encoder {
        private final DataBlindInt8Quantizer quantizer;
        private final byte[] code;

        Encoder(final int dimension) {
            this.quantizer = new DataBlindInt8Quantizer(dimension);
            this.code = new byte[dimension];
        }

        void write(final float[] vector) throws IOException {
            final DataBlindInt8Quantizer.Int8Code corrections = quantizer.encode(vector, code);
            data.writeBytes(code, code.length);
            data.writeInt(Float.floatToIntBits(corrections.scale()));
            data.writeInt(corrections.sum());
            data.writeInt(Float.floatToIntBits(corrections.sqNorm()));
            data.writeInt(0);
        }
    }

    @Override
    public void finish() throws IOException {
        if (finished) {
            throw new IllegalStateException("already finished");
        }
        finished = true;
        rawVectorDelegate.finish();
        meta.writeInt(-1);
        CodecUtil.writeFooter(meta);
        CodecUtil.writeFooter(data);
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(meta, data, rawVectorDelegate);
    }

    @Override
    public long ramBytesUsed() {
        return SHALLOW_SIZE + rawVectorDelegate.ramBytesUsed();
    }
}
