/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.apache.lucene.codecs.lucene104;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat.ScalarEncoding;
import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.FileDataHint;
import org.apache.lucene.store.FileTypeHint;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.opensearch.knn.index.codec.padrotate.PadRotate;
import org.opensearch.knn.index.codec.padrotate.PadRotateScalarQuantizedVectorsFormat;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Reader for the pad-and-rotate scalar-quantized flat format. Lives in the
 * {@code org.apache.lucene.codecs.lucene104} package so it can invoke
 * {@link OffHeapScalarQuantizedVectorValues#load}, which is package-private in Lucene.
 *
 * <p>Graph traversal scoring delegates to the standard Lucene104 quantized scorer, using a
 * pad+rotate of the query vector so both sides live in the same padded-rotated space.
 *
 * <p>Raw-float retrieval (for rescoring or value fetching) reconstructs {@code D}-dim vectors by
 * de-quantizing from {@code paddedDim}, applying the inverse rotation, and truncating to
 * {@code D}.
 */
public final class PadRotateScalarQuantizedVectorsReader extends FlatVectorsReader {

    private static final long SHALLOW_SIZE = RamUsageEstimator.shallowSizeOfInstance(PadRotateScalarQuantizedVectorsReader.class);
    private static final ScalarEncoding ENCODING = ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE;

    private final Map<String, FieldEntry> fields = new HashMap<>();
    private final IndexInput quantizedVectorData;
    private final Lucene104ScalarQuantizedVectorScorer scorer;

    public PadRotateScalarQuantizedVectorsReader(SegmentReadState state, Lucene104ScalarQuantizedVectorScorer scorer) throws IOException {
        super(scorer);
        this.scorer = scorer;
        int versionMeta = -1;
        String metaFileName = IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            PadRotateScalarQuantizedVectorsFormat.META_EXTENSION
        );
        boolean success = false;
        IndexInput dataInput = null;
        try (ChecksumIndexInput meta = state.directory.openChecksumInput(metaFileName)) {
            Throwable priorE = null;
            try {
                versionMeta = CodecUtil.checkIndexHeader(
                    meta,
                    PadRotateScalarQuantizedVectorsFormat.META_CODEC_NAME,
                    PadRotateScalarQuantizedVectorsFormat.VERSION_START,
                    PadRotateScalarQuantizedVectorsFormat.VERSION_CURRENT,
                    state.segmentInfo.getId(),
                    state.segmentSuffix
                );
                readFields(meta, state.fieldInfos);
            } catch (Throwable e) {
                priorE = e;
            } finally {
                CodecUtil.checkFooter(meta, priorE);
            }
            dataInput = openDataInput(state, versionMeta);
            this.quantizedVectorData = dataInput;
            success = true;
        } finally {
            if (success == false) {
                IOUtils.closeWhileHandlingException(dataInput);
            }
        }
    }

    private static IndexInput openDataInput(SegmentReadState state, int versionMeta) throws IOException {
        String fileName = IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            PadRotateScalarQuantizedVectorsFormat.DATA_EXTENSION
        );
        IndexInput in = state.directory.openInput(
            fileName,
            state.context.withHints(FileTypeHint.DATA, FileDataHint.KNN_VECTORS, DataAccessHint.RANDOM)
        );
        boolean success = false;
        try {
            int versionData = CodecUtil.checkIndexHeader(
                in,
                PadRotateScalarQuantizedVectorsFormat.DATA_CODEC_NAME,
                PadRotateScalarQuantizedVectorsFormat.VERSION_START,
                PadRotateScalarQuantizedVectorsFormat.VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix
            );
            if (versionMeta != versionData) {
                throw new CorruptIndexException(
                    "format versions mismatch: meta=" + versionMeta + ", data=" + versionData + " for " + fileName,
                    in
                );
            }
            CodecUtil.retrieveChecksum(in);
            success = true;
            return in;
        } finally {
            if (success == false) {
                IOUtils.closeWhileHandlingException(in);
            }
        }
    }

    private void readFields(ChecksumIndexInput meta, FieldInfos infos) throws IOException {
        for (int fieldNumber = meta.readInt(); fieldNumber != -1; fieldNumber = meta.readInt()) {
            FieldInfo info = infos.fieldInfo(fieldNumber);
            if (info == null) {
                throw new CorruptIndexException("invalid field number: " + fieldNumber, meta);
            }
            int encodingOrdinal = meta.readInt();
            int similarityOrdinal = meta.readInt();
            int originalDim = meta.readVInt();
            int paddedDim = meta.readVInt();
            long rotationSeed = meta.readLong();
            long dataOffset = meta.readVLong();
            long dataLength = meta.readVLong();
            int count = meta.readVInt();
            float[] centroid = null;
            float centroidDp = 0f;
            if (count > 0) {
                centroid = new float[paddedDim];
                byte[] buf = new byte[paddedDim * Float.BYTES];
                meta.readBytes(buf, 0, buf.length);
                java.nio.ByteBuffer.wrap(buf).order(java.nio.ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(centroid);
                centroidDp = Float.intBitsToFloat(meta.readInt());
            }
            OrdToDocDISIReaderConfiguration disi = OrdToDocDISIReaderConfiguration.fromStoredMeta(meta, count);
            VectorEncoding vectorEncoding = VectorEncoding.values()[encodingOrdinal];
            VectorSimilarityFunction similarity = VectorSimilarityFunction.values()[similarityOrdinal];
            fields.put(
                info.name,
                new FieldEntry(
                    vectorEncoding,
                    similarity,
                    originalDim,
                    paddedDim,
                    rotationSeed,
                    dataOffset,
                    dataLength,
                    count,
                    centroid,
                    centroidDp,
                    disi
                )
            );
        }
    }

    private OffHeapScalarQuantizedVectorValues loadQuantizedValues(FieldEntry fe) throws IOException {
        OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(fe.similarity);
        return OffHeapScalarQuantizedVectorValues.load(
            fe.disi,
            fe.paddedDim,
            fe.count,
            quantizer,
            ENCODING,
            fe.similarity,
            scorer,
            fe.centroid,
            fe.centroidDp,
            fe.dataOffset,
            fe.dataLength,
            quantizedVectorData
        );
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(String field, float[] target) throws IOException {
        FieldEntry fe = fields.get(field);
        if (fe == null) {
            return null;
        }
        // Pad+rotate the query into the same 4D space as the stored quantized vectors.
        PadRotate padRotate = new PadRotate(fe.originalDim, fe.rotationSeed);
        float[] scratch = new float[fe.paddedDim];
        float[] rotatedQuery = new float[fe.paddedDim];
        padRotate.forward(target, scratch, rotatedQuery);
        return scorer.getRandomVectorScorer(fe.similarity, loadQuantizedValues(fe), rotatedQuery);
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(String field, byte[] target) throws IOException {
        throw new UnsupportedOperationException("byte query vectors not supported by PadRotateScalarQuantizedVectorsFormat");
    }

    @Override
    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
        FieldEntry fe = fields.get(field);
        if (fe == null) {
            return null;
        }
        OffHeapScalarQuantizedVectorValues sq = loadQuantizedValues(fe);
        return new ReconstructingFloatVectorValues(sq, fe);
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) throws IOException {
        throw new UnsupportedOperationException("byte vector values not supported by PadRotateScalarQuantizedVectorsFormat");
    }

    @Override
    public void checkIntegrity() throws IOException {
        CodecUtil.checksumEntireFile(quantizedVectorData);
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(quantizedVectorData);
    }

    @Override
    public long ramBytesUsed() {
        long size = SHALLOW_SIZE;
        size += RamUsageEstimator.sizeOfMap(fields, RamUsageEstimator.shallowSizeOfInstance(FieldEntry.class));
        return size;
    }

    @Override
    public Map<String, Long> getOffHeapByteSize(FieldInfo fieldInfo) {
        Objects.requireNonNull(fieldInfo);
        FieldEntry fe = fields.get(fieldInfo.name);
        if (fe == null) {
            return Map.of();
        }
        return Map.of(PadRotateScalarQuantizedVectorsFormat.DATA_EXTENSION, fe.dataLength);
    }

    private record FieldEntry(
        VectorEncoding vectorEncoding,
        VectorSimilarityFunction similarity,
        int originalDim,
        int paddedDim,
        long rotationSeed,
        long dataOffset,
        long dataLength,
        int count,
        float[] centroid,
        float centroidDp,
        OrdToDocDISIReaderConfiguration disi
    ) {}

    /**
     * {@link FloatVectorValues} that reconstructs the original D-dim vector on demand. Used by
     * the rescore path and by KNN value fetchers. Also exposes the underlying quantized vector
     * values via the {@code quantizedVectorValues} field (same name Lucene uses internally for
     * its own wrapper, so existing reflection-based extractors work unchanged).
     */
    static final class ReconstructingFloatVectorValues extends FloatVectorValues {
        /** Exposed for reflection-based extractors (e.g. FAISS native HNSW build). */
        @SuppressWarnings("unused")
        private final OffHeapScalarQuantizedVectorValues quantizedVectorValues;

        private final OffHeapScalarQuantizedVectorValues sq;
        private final FieldEntry fe;
        private final PadRotate padRotate;
        private final byte[] packed;
        private final byte[] unpacked;
        private final float[] dequantizedPadded;
        private final float[] reverseScratch;
        private final float[] reconstructed;
        private int currentOrd = -1;

        ReconstructingFloatVectorValues(OffHeapScalarQuantizedVectorValues sq, FieldEntry fe) {
            this.sq = sq;
            this.quantizedVectorValues = sq;
            this.fe = fe;
            this.padRotate = new PadRotate(fe.originalDim, fe.rotationSeed);
            int packedLen = ENCODING.getDocPackedLength(fe.paddedDim);
            this.packed = new byte[packedLen];
            this.unpacked = new byte[fe.paddedDim];
            this.dequantizedPadded = new float[fe.paddedDim];
            this.reverseScratch = new float[fe.paddedDim];
            this.reconstructed = new float[fe.originalDim];
        }

        @Override
        public int dimension() {
            return fe.originalDim;
        }

        @Override
        public int size() {
            return fe.count;
        }

        @Override
        public float[] vectorValue(int ord) throws IOException {
            if (ord == currentOrd) {
                return reconstructed;
            }
            byte[] quantized = sq.vectorValue(ord);
            OptimizedScalarQuantizer.QuantizationResult corrections = sq.getCorrectiveTerms(ord);
            // 1-bit doc packing: unpack each bit back to 0/1.
            System.arraycopy(quantized, 0, packed, 0, packed.length);
            OptimizedScalarQuantizer.unpackBinary(packed, unpacked);
            OptimizedScalarQuantizer.deQuantize(
                unpacked,
                dequantizedPadded,
                ENCODING.getBits(),
                corrections.lowerInterval(),
                corrections.upperInterval(),
                fe.centroid
            );
            padRotate.reverseAndTruncate(dequantizedPadded, reverseScratch, reconstructed);
            currentOrd = ord;
            return reconstructed;
        }

        @Override
        public int ordToDoc(int ord) {
            return sq.ordToDoc(ord);
        }

        @Override
        public DocIndexIterator iterator() {
            return sq.iterator();
        }

        @Override
        public VectorScorer scorer(float[] target) throws IOException {
            // Route through the reader's scorer machinery instead; rescore uses vectorValue().
            throw new UnsupportedOperationException("scorer() not supported here; use getRandomVectorScorer on the reader");
        }

        @Override
        public FloatVectorValues copy() {
            return new ReconstructingFloatVectorValues(sq, fe);
        }
    }

    // Unused direct VectorScorer import path, but referenced via javadoc above. Keep symbol.
    @SuppressWarnings("unused")
    private static VectorScorer unusedVectorScorer() {
        return null;
    }
}
