/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene95.OffHeapByteVectorValues;
import org.apache.lucene.codecs.lucene95.OffHeapFloatVectorValues;
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
import org.apache.lucene.internal.hppc.IntObjectHashMap;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.FileDataHint;
import org.apache.lucene.store.FileTypeHint;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.FixedBlockSkipListIndexReader;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.ReorderedOffHeapFloatVectorValues111;

import java.io.IOException;
import java.util.Map;

import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.readSimilarityFunction;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.readVectorEncoding;

/**
 * Unified FlatVectorsReader that reads standard Lucene99 codec headers but dispatches per-field
 * based on {@code fieldInfo.getAttribute("knn_reordered")}. Reordered fields use skip list
 * metadata; standard fields use OrdToDocDISI metadata.
 */
public class UnifiedFlatVectorsReader extends FlatVectorsReader {

    static final String META_CODEC_NAME = "Lucene99FlatVectorsFormatMeta";
    static final String VECTOR_DATA_CODEC_NAME = "Lucene99FlatVectorsFormatData";
    static final String META_EXTENSION = "vemf";
    static final String VECTOR_DATA_EXTENSION = "vec";
    static final int VERSION_START = 0;
    static final int VERSION_CURRENT = 0;
    static final int DIRECT_MONOTONIC_BLOCK_SHIFT = 16;

    private final IntObjectHashMap<FieldEntry> fields = new IntObjectHashMap<>();
    private final IndexInput vectorData;
    private final FieldInfos fieldInfos;

    public UnifiedFlatVectorsReader(SegmentReadState state, FlatVectorsScorer scorer) throws IOException {
        super(scorer);
        int versionMeta = readMetadata(state);
        this.fieldInfos = state.fieldInfos;
        IOContext dataContext = state.context.withHints(FileTypeHint.DATA, FileDataHint.KNN_VECTORS, DataAccessHint.RANDOM);
        boolean success = false;
        try {
            String dataFileName = IndexFileNames.segmentFileName(
                state.segmentInfo.name, state.segmentSuffix, VECTOR_DATA_EXTENSION);
            IndexInput in = state.directory.openInput(dataFileName, dataContext);
            int versionData = CodecUtil.checkIndexHeader(in, VECTOR_DATA_CODEC_NAME,
                VERSION_START, VERSION_CURRENT, state.segmentInfo.getId(), state.segmentSuffix);
            if (versionMeta != versionData) {
                throw new CorruptIndexException("Format versions mismatch: meta=" + versionMeta
                    + ", data=" + versionData, in);
            }
            CodecUtil.retrieveChecksum(in);
            vectorData = in;
            success = true;
        } finally {
            if (!success) IOUtils.closeWhileHandlingException(this);
        }
    }

    private int readMetadata(SegmentReadState state) throws IOException {
        String metaFileName = IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, META_EXTENSION);
        try (ChecksumIndexInput meta = state.directory.openChecksumInput(metaFileName)) {
            int versionMeta = CodecUtil.checkIndexHeader(meta, META_CODEC_NAME,
                VERSION_START, VERSION_CURRENT, state.segmentInfo.getId(), state.segmentSuffix);
            readFields(meta, state.fieldInfos);
            return versionMeta;
        }
    }

    private void readFields(ChecksumIndexInput meta, FieldInfos infos) throws IOException {
        for (int fieldNumber = meta.readInt(); fieldNumber != -1; fieldNumber = meta.readInt()) {
            FieldInfo info = infos.fieldInfo(fieldNumber);
            if (info == null) {
                throw new CorruptIndexException("Invalid field number: " + fieldNumber, meta);
            }
            boolean isReordered = "true".equals(info.getAttribute("knn_reordered"));
//            System.out.println("[UnifiedReader] readFields field=" + info.name + " num=" + info.number + " reordered=" + isReordered + " attrs=" + info.attributes());
            FieldEntry entry = isReordered
                ? FieldEntry.createReordered(meta, info)
                : FieldEntry.createStandard(meta, info);
            fields.put(info.number, entry);
        }
    }

    @Override
    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
        FieldEntry entry = getFieldEntry(field, VectorEncoding.FLOAT32);
        if (entry.reordered) {
            if (entry.isDense) {
                return ReorderedOffHeapFloatVectorValues111.load(
                    entry.similarityFunction, vectorScorer, entry.skipListReader,
                    entry.dimension, entry.vectorDataOffset, entry.vectorDataLength,
                    vectorData, entry.skipListReader.maxDoc + 1, entry.ordToDocMap
                );
            } else {
                return ReorderedOffHeapFloatVectorValues111.loadSparse(
                    entry.similarityFunction, vectorScorer, entry.skipListReader,
                    entry.dimension, entry.vectorDataOffset, entry.vectorDataLength,
                    vectorData, entry.ordToDocMap.length, entry.maxDoc, entry.ordToDocMap
                );
            }
        }
//        System.out.println("[UnifiedReader] getFloatVectorValues field=" + field + " → STANDARD, vecLen=" + entry.vectorDataLength + " dim=" + entry.dimension);
        return OffHeapFloatVectorValues.load(
            entry.similarityFunction, vectorScorer, entry.ordToDocConfig,
            VectorEncoding.FLOAT32, entry.dimension,
            entry.vectorDataOffset, entry.vectorDataLength, vectorData
        );
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(String field, float[] target) throws IOException {
        FieldEntry entry = getFieldEntry(field, VectorEncoding.FLOAT32);
        FloatVectorValues values = getFloatVectorValues(field);
        return vectorScorer.getRandomVectorScorer(entry.similarityFunction, values, target);
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) throws IOException {
        FieldEntry entry = getFieldEntry(field, VectorEncoding.BYTE);
        return OffHeapByteVectorValues.load(
            entry.similarityFunction, vectorScorer, entry.ordToDocConfig,
            VectorEncoding.BYTE, entry.dimension,
            entry.vectorDataOffset, entry.vectorDataLength, vectorData
        );
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(String field, byte[] target) throws IOException {
        FieldEntry entry = getFieldEntry(field, VectorEncoding.BYTE);
        ByteVectorValues values = getByteVectorValues(field);
        return vectorScorer.getRandomVectorScorer(entry.similarityFunction, values, target);
    }

    @Override
    public void checkIntegrity() throws IOException {
        CodecUtil.checksumEntireFile(vectorData);
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(vectorData);
    }

    @Override
    public long ramBytesUsed() {
        return 0;
    }

    @Override
    public Map<String, Long> getOffHeapByteSize(FieldInfo fieldInfo) {
        FieldEntry entry = fields.get(fieldInfo.number);
        if (entry == null) return Map.of();
        return Map.of(VECTOR_DATA_EXTENSION, entry.vectorDataLength);
    }

    private FieldEntry getFieldEntry(String field, VectorEncoding expectedEncoding) {
        FieldInfo info = fieldInfos.fieldInfo(field);
        if (info == null) throw new IllegalArgumentException("field=\"" + field + "\" not found");
        FieldEntry entry = fields.get(info.number);
        if (entry == null) throw new IllegalArgumentException("field=\"" + field + "\" not found");
        if (entry.vectorEncoding != expectedEncoding) {
            throw new IllegalArgumentException("field=\"" + field + "\" encoding mismatch");
        }
        return entry;
    }

    record FieldEntry(
        VectorSimilarityFunction similarityFunction,
        VectorEncoding vectorEncoding,
        long vectorDataOffset,
        long vectorDataLength,
        int dimension,
        boolean reordered,
        boolean isDense,
        int maxDoc,
        FixedBlockSkipListIndexReader skipListReader,       // non-null for reordered
        OrdToDocDISIReaderConfiguration ordToDocConfig,     // non-null for standard
        int[] ordToDocMap                                   // non-null for reordered
    ) {
        static FieldEntry createReordered(IndexInput input, FieldInfo info) throws IOException {
            VectorEncoding enc = readVectorEncoding(input);
            VectorSimilarityFunction sim = readSimilarityFunction(input);
            long offset = input.readVLong();
            long length = input.readVLong();
            int dim = input.readVInt();

            boolean isDense = input.readByte() == 1;
            int maxDoc = input.readInt();
            input.readInt();   // numLevel
            input.readInt();   // numDocsForGrouping
            input.readInt();   // groupFactor

            FixedBlockSkipListIndexReader skipList = new FixedBlockSkipListIndexReader(input, maxDoc);

            // Read ord→doc array
            int n = input.readInt();
            int[] ordToDocMap = new int[n];
            for (int i = 0; i < n; i++) {
                ordToDocMap[i] = input.readInt();
            }

            return new FieldEntry(sim, enc, offset, length, dim, true, isDense, maxDoc, skipList, null, ordToDocMap);
        }

        static FieldEntry createStandard(IndexInput input, FieldInfo info) throws IOException {
            VectorEncoding enc = readVectorEncoding(input);
            VectorSimilarityFunction sim = readSimilarityFunction(input);
            long offset = input.readVLong();
            long length = input.readVLong();
            int dim = input.readVInt();

            int count = input.readInt();
            OrdToDocDISIReaderConfiguration config = OrdToDocDISIReaderConfiguration.fromStoredMeta(input, count);
            return new FieldEntry(sim, enc, offset, length, dim, false, true, 0, null, config, null);
        }
    }
}
