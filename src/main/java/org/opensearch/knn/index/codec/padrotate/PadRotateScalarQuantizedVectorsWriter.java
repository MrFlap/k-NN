/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.padrotate;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorScorer;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat.ScalarEncoding;
import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.hnsw.CloseableRandomVectorScorerSupplier;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

/**
 * Writer for the pad-and-rotate scalar-quantized flat format. Streams input vectors to a per-field
 * temp file during indexing, then at flush time re-reads each vector, applies pad+rotate with a
 * segment-specific random rotation, and writes single-bit scalar-quantized bytes + corrective
 * factors to the {@code .peq} data file. Metadata (including the rotation seed per field) goes
 * to {@code .pemq}.
 *
 * <p>RAM during indexing is O(D) per field for the running centroid sum. No vectors are held in
 * memory.
 */
final class PadRotateScalarQuantizedVectorsWriter extends FlatVectorsWriter {

    private static final long SHALLOW_RAM_BYTES = RamUsageEstimator.shallowSizeOfInstance(PadRotateScalarQuantizedVectorsWriter.class);
    private static final ScalarEncoding ENCODING = ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE;

    private final SegmentWriteState segmentWriteState;
    private final IndexOutput meta;
    private final IndexOutput vectorData;
    private final List<StreamingFieldVectorsWriter> fields = new ArrayList<>();
    private boolean finished;

    PadRotateScalarQuantizedVectorsWriter(SegmentWriteState state, Lucene104ScalarQuantizedVectorScorer scorer) throws IOException {
        super(scorer);
        this.segmentWriteState = state;
        String metaFileName = IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            PadRotateScalarQuantizedVectorsFormat.META_EXTENSION
        );
        String dataFileName = IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            PadRotateScalarQuantizedVectorsFormat.DATA_EXTENSION
        );
        boolean success = false;
        IndexOutput metaOut = null;
        IndexOutput dataOut = null;
        try {
            metaOut = state.directory.createOutput(metaFileName, state.context);
            dataOut = state.directory.createOutput(dataFileName, state.context);
            CodecUtil.writeIndexHeader(
                metaOut,
                PadRotateScalarQuantizedVectorsFormat.META_CODEC_NAME,
                PadRotateScalarQuantizedVectorsFormat.VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix
            );
            CodecUtil.writeIndexHeader(
                dataOut,
                PadRotateScalarQuantizedVectorsFormat.DATA_CODEC_NAME,
                PadRotateScalarQuantizedVectorsFormat.VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix
            );
            this.meta = metaOut;
            this.vectorData = dataOut;
            success = true;
        } finally {
            if (success == false) {
                IOUtils.closeWhileHandlingException(metaOut, dataOut);
            }
        }
    }

    @Override
    public FlatFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
        if (fieldInfo.getVectorEncoding() != org.apache.lucene.index.VectorEncoding.FLOAT32) {
            throw new IllegalArgumentException(
                "PadRotateScalarQuantizedVectorsFormat only supports FLOAT32 encoding; got " + fieldInfo.getVectorEncoding()
            );
        }
        StreamingFieldVectorsWriter fw = new StreamingFieldVectorsWriter(fieldInfo, segmentWriteState.directory, segmentWriteState.context);
        fields.add(fw);
        return fw;
    }

    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
        if (sortMap != null) {
            // TODO Phase 2: support segment-sorted writes. The Lucene104 writer has a
            // writeSortingField path; we'd implement the same against our streaming file.
            throw new UnsupportedOperationException("sorted flushes are not yet supported by PadRotateScalarQuantizedVectorsFormat");
        }
        for (StreamingFieldVectorsWriter field : fields) {
            writeField(field, maxDoc);
        }
    }

    private void writeField(StreamingFieldVectorsWriter field, int maxDoc) throws IOException {
        // Finalize the temp file so we can open it for reading, but keep the delete until the
        // end so a failure mid-write doesn't leak the file.
        field.closeTempOutputForReading();
        try {
            long seed = pickSeed(field.fieldInfo());
            PadRotate padRotate = new PadRotate(field.dim(), seed);
            int originalDim = field.dim();
            int paddedDim = padRotate.paddedDimensions();
            float[] centroidD = field.computeCentroid();
            float[] scratchPadded = new float[paddedDim];
            float[] centroid4D = new float[paddedDim];
            padRotate.forward(centroidD, scratchPadded, centroid4D);

            OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(field.fieldInfo().getVectorSimilarityFunction());

            long dataOffset = vectorData.alignFilePointer(Float.BYTES);
            DocsWithFieldSet docsWithField = field.getDocsWithFieldSet();
            int vectorCount = field.vectorCount();

            if (vectorCount > 0) {
                try (IndexInput tempIn = field.openTempInput()) {
                    writeQuantizedVectors(tempIn, field, padRotate, quantizer, centroid4D);
                }
            }
            long dataLength = vectorData.getFilePointer() - dataOffset;

            float centroidDp = 0f;
            if (vectorCount > 0) {
                for (int i = 0; i < paddedDim; i++) {
                    centroidDp += centroid4D[i] * centroid4D[i];
                }
            }

            writeMeta(field.fieldInfo(), originalDim, paddedDim, seed, maxDoc, dataOffset, dataLength, centroid4D, centroidDp, docsWithField);
        } finally {
            field.deleteTempFile();
        }
    }

    private void writeQuantizedVectors(
        IndexInput tempIn,
        StreamingFieldVectorsWriter field,
        PadRotate padRotate,
        OptimizedScalarQuantizer quantizer,
        float[] centroid4D
    ) throws IOException {
        int originalDim = field.dim();
        int paddedDim = padRotate.paddedDimensions();
        int vectorCount = field.vectorCount();

        byte[] quantScratch = new byte[ENCODING.getDiscreteDimensions(paddedDim)];
        byte[] packed = new byte[ENCODING.getDocPackedLength(paddedDim)];

        float[] raw = new float[originalDim];
        float[] padScratch = new float[paddedDim];
        float[] rotated = new float[paddedDim];

        for (int v = 0; v < vectorCount; v++) {
            for (int i = 0; i < originalDim; i++) {
                raw[i] = Float.intBitsToFloat(tempIn.readInt());
            }
            padRotate.forward(raw, padScratch, rotated);

            OptimizedScalarQuantizer.QuantizationResult corrections = quantizer.scalarQuantize(
                rotated,
                quantScratch,
                ENCODING.getBits(),
                centroid4D
            );
            OptimizedScalarQuantizer.packAsBinary(quantScratch, packed);
            vectorData.writeBytes(packed, packed.length);
            vectorData.writeInt(Float.floatToIntBits(corrections.lowerInterval()));
            vectorData.writeInt(Float.floatToIntBits(corrections.upperInterval()));
            vectorData.writeInt(Float.floatToIntBits(corrections.additionalCorrection()));
            vectorData.writeInt(corrections.quantizedComponentSum());
        }
    }

    private void writeMeta(
        FieldInfo field,
        int originalDim,
        int paddedDim,
        long rotationSeed,
        int maxDoc,
        long dataOffset,
        long dataLength,
        float[] centroid4D,
        float centroidDp,
        DocsWithFieldSet docsWithField
    ) throws IOException {
        meta.writeInt(field.number);
        meta.writeInt(field.getVectorEncoding().ordinal());
        meta.writeInt(field.getVectorSimilarityFunction().ordinal());
        meta.writeVInt(originalDim);
        meta.writeVInt(paddedDim);
        meta.writeLong(rotationSeed);
        meta.writeVLong(dataOffset);
        meta.writeVLong(dataLength);
        int count = docsWithField.cardinality();
        meta.writeVInt(count);
        if (count > 0) {
            ByteBuffer buffer = ByteBuffer.allocate(paddedDim * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
            buffer.asFloatBuffer().put(centroid4D);
            meta.writeBytes(buffer.array(), buffer.array().length);
            meta.writeInt(Float.floatToIntBits(centroidDp));
        }
        OrdToDocDISIReaderConfiguration.writeStoredMeta(
            PadRotateScalarQuantizedVectorsFormat.DIRECT_MONOTONIC_BLOCK_SHIFT,
            meta,
            vectorData,
            count,
            maxDoc,
            docsWithField
        );
    }

    /**
     * Produce a 64-bit seed for the per-segment, per-field rotation. Derived from segment id
     * and field number so it is stable across reader open/close cycles for a given segment.
     * The seed is persisted in metadata so cross-version algorithm changes won't silently
     * corrupt old indices (readers go by the persisted value, not by re-deriving).
     */
    private long pickSeed(FieldInfo fieldInfo) {
        byte[] id = segmentWriteState.segmentInfo.getId();
        long base = 0L;
        for (int i = 0; i < id.length && i < 8; i++) {
            base = (base << 8) | (id[i] & 0xFFL);
        }
        // Mix in field number so different fields in the same segment get different rotations.
        return base ^ (0x9E3779B97F4A7C15L * (fieldInfo.number + 1));
    }

    @Override
    public CloseableRandomVectorScorerSupplier mergeOneFieldToIndex(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        // This method is called only by Lucene's native HNSW writer when it owns graph
        // construction and needs a scorer supplier over the merged quantized data.
        //
        // This format is consumed by the FAISS HNSW flow, which builds the graph natively and
        // drives merge via {@link #mergeOneField} (inherited). That path never requires a
        // merge-time scorer supplier. Supporting Lucene99HnswVectorsFormat on top of this
        // format would need a full implementation that opens a reader on the just-written
        // files and wraps its quantized values into a RandomVectorScorerSupplier.
        //
        // We raise a clear error so any future consumer that wires Lucene HNSW on top of this
        // format has an obvious signal rather than a silent failure.
        throw new UnsupportedOperationException(
            "mergeOneFieldToIndex is not implemented for "
                + PadRotateScalarQuantizedVectorsFormat.NAME
                + "; this format is designed to be consumed by the native FAISS HNSW flow, which drives merges via mergeOneField."
        );
    }

    @Override
    public void finish() throws IOException {
        if (finished) {
            throw new IllegalStateException("already finished");
        }
        finished = true;
        if (meta != null) {
            meta.writeInt(-1); // sentinel: no more fields
            CodecUtil.writeFooter(meta);
        }
        if (vectorData != null) {
            CodecUtil.writeFooter(vectorData);
        }
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(meta, vectorData);
        for (StreamingFieldVectorsWriter f : fields) {
            f.closeOnAbort();
        }
    }

    @Override
    public long ramBytesUsed() {
        long total = SHALLOW_RAM_BYTES;
        for (StreamingFieldVectorsWriter f : fields) {
            total += f.ramBytesUsed();
        }
        return total;
    }
}
