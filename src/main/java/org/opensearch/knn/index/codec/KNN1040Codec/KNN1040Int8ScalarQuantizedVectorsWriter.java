/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.apache.lucene.util.RamUsageEstimator.shallowSizeOfInstance;

/**
 * Flat vectors writer for int8 ({@link ScalarEncoding#UNSIGNED_BYTE}) fields that quantizes with {@link Int8ScalarQuantizer}
 * instead of Lucene's mean/stddev-aware {@code OptimizedScalarQuantizer}.
 *
 * <p>Derived from {@code Lucene104ScalarQuantizedVectorsWriter}, trimmed to the 8-bit case. The files it produces use the same
 * names, codec headers and layout as that writer, so they are read back by the stock
 * {@code Lucene104ScalarQuantizedVectorsReader}. The differences are in how the codes are chosen:
 * <ul>
 *   <li>vectors are quantized by {@link Int8ScalarQuantizer} (symmetric, one scale per vector);</li>
 *   <li>no centroid is computed or merged, the recorded centroid is the zero vector, so no centroid is subtracted at search time.</li>
 * </ul>
 */
public class KNN1040Int8ScalarQuantizedVectorsWriter extends FlatVectorsWriter {
    private static final long SHALLOW_RAM_BYTES_USED = shallowSizeOfInstance(KNN1040Int8ScalarQuantizedVectorsWriter.class);

    // Must match Lucene104ScalarQuantizedVectorsFormat, whose reader consumes these files. Those constants are package private.
    static final String META_CODEC_NAME = "Lucene104ScalarQuantizedVectorsFormatMeta";
    static final String VECTOR_DATA_CODEC_NAME = "Lucene104ScalarQuantizedVectorsFormatData";
    static final String META_EXTENSION = "vemq";
    static final String VECTOR_DATA_EXTENSION = "veq";
    static final int VERSION_CURRENT = 0;
    static final int DIRECT_MONOTONIC_BLOCK_SHIFT = 16;

    private final SegmentWriteState segmentWriteState;
    private final List<FieldWriter> fields = new ArrayList<>();
    private final IndexOutput meta, vectorData;
    private final FlatVectorsWriter rawVectorDelegate;
    private boolean finished;

    public KNN1040Int8ScalarQuantizedVectorsWriter(
        final SegmentWriteState state,
        final FlatVectorsWriter rawVectorDelegate,
        final FlatVectorsScorer vectorsScorer
    ) throws IOException {
        super(vectorsScorer);
        this.segmentWriteState = state;
        final String metaFileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, META_EXTENSION);
        final String vectorDataFileName = IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            VECTOR_DATA_EXTENSION
        );
        this.rawVectorDelegate = rawVectorDelegate;
        try {
            meta = state.directory.createOutput(metaFileName, state.context);
            vectorData = state.directory.createOutput(vectorDataFileName, state.context);

            CodecUtil.writeIndexHeader(meta, META_CODEC_NAME, VERSION_CURRENT, state.segmentInfo.getId(), state.segmentSuffix);
            CodecUtil.writeIndexHeader(vectorData, VECTOR_DATA_CODEC_NAME, VERSION_CURRENT, state.segmentInfo.getId(), state.segmentSuffix);
        } catch (Throwable t) {
            IOUtils.closeWhileHandlingException(this);
            throw t;
        }
    }

    @Override
    public FlatFieldVectorsWriter<?> addField(final FieldInfo fieldInfo) throws IOException {
        final FlatFieldVectorsWriter<?> rawFieldWriter = this.rawVectorDelegate.addField(fieldInfo);
        if (fieldInfo.getVectorEncoding().equals(VectorEncoding.FLOAT32)) {
            @SuppressWarnings("unchecked")
            final FieldWriter fieldWriter = new FieldWriter(fieldInfo, (FlatFieldVectorsWriter<float[]>) rawFieldWriter);
            fields.add(fieldWriter);
            return fieldWriter;
        }
        return rawFieldWriter;
    }

    @Override
    public void flush(final int maxDoc, final Sorter.DocMap sortMap) throws IOException {
        rawVectorDelegate.flush(maxDoc, sortMap);
        for (final FieldWriter field : fields) {
            final Int8ScalarQuantizer quantizer = new Int8ScalarQuantizer(field.fieldInfo.getVectorSimilarityFunction());
            if (sortMap == null) {
                writeField(field, maxDoc, quantizer);
            } else {
                writeSortingField(field, maxDoc, sortMap, quantizer);
            }
            field.finish();
        }
    }

    private void writeField(final FieldWriter fieldData, final int maxDoc, final Int8ScalarQuantizer quantizer) throws IOException {
        final long vectorDataOffset = vectorData.alignFilePointer(Float.BYTES);
        final int size = fieldData.getVectors().size();
        final int[] sequentialOrds = new int[size];
        for (int i = 0; i < size; i++) {
            sequentialOrds[i] = i;
        }
        writeVectors(fieldData, sequentialOrds, quantizer);
        final long vectorDataLength = vectorData.getFilePointer() - vectorDataOffset;
        writeMeta(fieldData.fieldInfo, maxDoc, vectorDataOffset, vectorDataLength, fieldData.getDocsWithFieldSet());
    }

    private void writeSortingField(
        final FieldWriter fieldData,
        final int maxDoc,
        final Sorter.DocMap sortMap,
        final Int8ScalarQuantizer quantizer
    ) throws IOException {
        final int[] ordMap = new int[fieldData.getDocsWithFieldSet().cardinality()]; // new ord to old ord
        final DocsWithFieldSet newDocsWithField = new DocsWithFieldSet();
        mapOldOrdToNewOrd(fieldData.getDocsWithFieldSet(), sortMap, null, ordMap, newDocsWithField);

        final long vectorDataOffset = vectorData.alignFilePointer(Float.BYTES);
        writeVectors(fieldData, ordMap, quantizer);
        final long vectorDataLength = vectorData.getFilePointer() - vectorDataOffset;
        writeMeta(fieldData.fieldInfo, maxDoc, vectorDataOffset, vectorDataLength, newDocsWithField);
    }

    // Writes the vectors at the given ordinals in the given order.
    private void writeVectors(final FieldWriter fieldData, final int[] ords, final Int8ScalarQuantizer quantizer) throws IOException {
        final int dimension = fieldData.fieldInfo.getVectorDimension();
        final boolean cosine = fieldData.fieldInfo.getVectorSimilarityFunction() == VectorSimilarityFunction.COSINE;
        final byte[] codes = new byte[ScalarEncoding.UNSIGNED_BYTE.getDiscreteDimensions(dimension)];
        final float[] scratch = cosine ? new float[dimension] : null;
        for (final int ord : ords) {
            float[] vector = fieldData.getVectors().get(ord);
            if (cosine) {
                // The raw vectors were already written unnormalized by the delegate; only the quantized copy is normalized.
                System.arraycopy(vector, 0, scratch, 0, dimension);
                VectorUtil.l2normalize(scratch);
                vector = scratch;
            }
            writeQuantizedVector(quantizer, vector, codes);
        }
    }

    // Layout per vector: [codes][lowerInterval][upperInterval][additionalCorrection][quantizedComponentSum].
    private void writeQuantizedVector(final Int8ScalarQuantizer quantizer, final float[] vector, final byte[] codes) throws IOException {
        final OptimizedScalarQuantizer.QuantizationResult corrections = quantizer.scalarQuantize(
            vector,
            codes,
            Int8ScalarQuantizer.BITS,
            null
        );
        vectorData.writeBytes(codes, codes.length);
        vectorData.writeInt(Float.floatToIntBits(corrections.lowerInterval()));
        vectorData.writeInt(Float.floatToIntBits(corrections.upperInterval()));
        vectorData.writeInt(Float.floatToIntBits(corrections.additionalCorrection()));
        vectorData.writeInt(corrections.quantizedComponentSum());
    }

    private void writeMeta(
        final FieldInfo field,
        final int maxDoc,
        final long vectorDataOffset,
        final long vectorDataLength,
        final DocsWithFieldSet docsWithField
    ) throws IOException {
        meta.writeInt(field.number);
        meta.writeInt(field.getVectorEncoding().ordinal());
        meta.writeInt(field.getVectorSimilarityFunction().ordinal());
        meta.writeVInt(field.getVectorDimension());
        meta.writeVLong(vectorDataOffset);
        meta.writeVLong(vectorDataLength);
        final int count = docsWithField.cardinality();
        meta.writeVInt(count);
        if (count > 0) {
            meta.writeVInt(ScalarEncoding.UNSIGNED_BYTE.getWireNumber());
            // The centroid is always the zero vector: int8 quantization does not center, and its dot product with itself is zero.
            final byte[] zeroCentroid = new byte[field.getVectorDimension() * Float.BYTES];
            meta.writeBytes(zeroCentroid, zeroCentroid.length);
            meta.writeInt(Float.floatToIntBits(0f));
        }
        OrdToDocDISIReaderConfiguration.writeStoredMeta(DIRECT_MONOTONIC_BLOCK_SHIFT, meta, vectorData, count, maxDoc, docsWithField);
    }

    @Override
    public void finish() throws IOException {
        if (finished) {
            throw new IllegalStateException("already finished");
        }
        finished = true;
        rawVectorDelegate.finish();
        if (meta != null) {
            // write end of fields marker
            meta.writeInt(-1);
            CodecUtil.writeFooter(meta);
        }
        if (vectorData != null) {
            CodecUtil.writeFooter(vectorData);
        }
    }

    @Override
    public void mergeOneFlatVectorField(final FieldInfo fieldInfo, final MergeState mergeState) throws IOException {
        // Don't need access to the random vectors, we can just use the merged
        rawVectorDelegate.mergeOneFlatVectorField(fieldInfo, mergeState);
        if (!fieldInfo.getVectorEncoding().equals(VectorEncoding.FLOAT32)) {
            return;
        }

        final FloatVectorValues floatVectorValues = MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
        final int dimension = fieldInfo.getVectorDimension();
        final boolean cosine = fieldInfo.getVectorSimilarityFunction() == VectorSimilarityFunction.COSINE;
        final Int8ScalarQuantizer quantizer = new Int8ScalarQuantizer(fieldInfo.getVectorSimilarityFunction());
        final byte[] codes = new byte[ScalarEncoding.UNSIGNED_BYTE.getDiscreteDimensions(dimension)];
        final float[] scratch = cosine ? new float[dimension] : null;

        final long vectorDataOffset = vectorData.alignFilePointer(Float.BYTES);
        final DocsWithFieldSet docsWithField = new DocsWithFieldSet();
        final KnnVectorValues.DocIndexIterator iterator = floatVectorValues.iterator();
        for (int doc = iterator.nextDoc(); doc != NO_MORE_DOCS; doc = iterator.nextDoc()) {
            float[] vector = floatVectorValues.vectorValue(iterator.index());
            if (cosine) {
                System.arraycopy(vector, 0, scratch, 0, dimension);
                VectorUtil.l2normalize(scratch);
                vector = scratch;
            }
            writeQuantizedVector(quantizer, vector, codes);
            docsWithField.add(doc);
        }
        final long vectorDataLength = vectorData.getFilePointer() - vectorDataOffset;
        writeMeta(fieldInfo, segmentWriteState.segmentInfo.maxDoc(), vectorDataOffset, vectorDataLength, docsWithField);
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(meta, vectorData, rawVectorDelegate);
    }

    @Override
    public long ramBytesUsed() {
        long total = SHALLOW_RAM_BYTES_USED;
        // The rawVectorDelegate tracks all vector data for both byte and float32 fields, and the field writers add no state of
        // their own beyond a shallow wrapper.
        total += rawVectorDelegate.ramBytesUsed();
        for (final FieldWriter field : fields) {
            total += field.quantizationOverheadBytesUsed();
        }
        return total;
    }

    static class FieldWriter extends FlatFieldVectorsWriter<float[]> {
        private static final long SHALLOW_SIZE = shallowSizeOfInstance(FieldWriter.class);
        private final FieldInfo fieldInfo;
        private boolean finished;
        private final FlatFieldVectorsWriter<float[]> flatFieldVectorsWriter;

        FieldWriter(final FieldInfo fieldInfo, final FlatFieldVectorsWriter<float[]> flatFieldVectorsWriter) {
            this.fieldInfo = fieldInfo;
            this.flatFieldVectorsWriter = flatFieldVectorsWriter;
        }

        @Override
        public List<float[]> getVectors() {
            return flatFieldVectorsWriter.getVectors();
        }

        @Override
        public DocsWithFieldSet getDocsWithFieldSet() {
            return flatFieldVectorsWriter.getDocsWithFieldSet();
        }

        @Override
        public void finish() throws IOException {
            if (finished) {
                return;
            }
            assert flatFieldVectorsWriter.isFinished();
            finished = true;
        }

        @Override
        public boolean isFinished() {
            return finished && flatFieldVectorsWriter.isFinished();
        }

        @Override
        public void addValue(final int docID, final float[] vectorValue) throws IOException {
            flatFieldVectorsWriter.addValue(docID, vectorValue);
        }

        @Override
        public float[] copyValue(final float[] vectorValue) {
            throw new UnsupportedOperationException();
        }

        // Excludes flatFieldVectorsWriter, whose data the writer-level rawVectorDelegate already accounts for.
        long quantizationOverheadBytesUsed() {
            return SHALLOW_SIZE;
        }

        @Override
        public long ramBytesUsed() {
            return quantizationOverheadBytesUsed() + flatFieldVectorsWriter.ramBytesUsed();
        }
    }
}
