/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.store.ByteBuffersIndexOutput;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.StringHelper;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ReorderedFlatVectorsWriter implements Closeable {
    private final IndexOutput meta, vectorData;
    private final List<ReorderedFlatFieldVectorsWriter> fields = new ArrayList<>();
    private final int numDocs;
    private final int numLevel = 4;
    private final int numDocsForGrouping = 256;
    private final int groupFactor = 4;
    public final byte[] segmentId;
    public final String segmentSuffix;

    public ReorderedFlatVectorsWriter(
        final Directory directory,
        final String metaFileName,
        final String flatVectorFileName,
        final int numDocs,
        final IndexInput originalMetaInput
    ) throws IOException {
        this.numDocs = numDocs;
        meta = directory.createOutput(metaFileName, IOContext.DEFAULT);

        // Seek to the 0th
        originalMetaInput.seek(0);
        // Header
        final int headerMagicNumber = CodecUtil.readBEInt(originalMetaInput);
        // Codec name
        final String codec = originalMetaInput.readString();
        // Version
        final int version = CodecUtil.readBEInt(originalMetaInput);
        // Segment id
        segmentId = new byte[StringHelper.ID_LENGTH];
        originalMetaInput.readBytes(segmentId, 0, segmentId.length);
        final int suffixLength = originalMetaInput.readByte() & 0xFF;
        final byte[] suffixBytes = new byte[suffixLength];
        originalMetaInput.readBytes(suffixBytes, 0, suffixLength);
        segmentSuffix = new String(suffixBytes);
        // Read fields
        originalMetaInput.seek(0);
        CodecUtil.checkIndexHeader(
            originalMetaInput,
            "Lucene99FlatVectorsFormatMeta",
            Lucene99FlatVectorsFormat.VERSION_START,
            Lucene99FlatVectorsFormat.VERSION_CURRENT,
            segmentId,
            segmentSuffix
        );

        // Write index header
        CodecUtil.writeIndexHeader(meta, ReorderedLucene99FlatVectorsReader.META_CODEC_NAME, version, segmentId, segmentSuffix);

        vectorData = directory.createOutput(flatVectorFileName, IOContext.DEFAULT);
        CodecUtil.writeIndexHeader(
            vectorData,
            ReorderedLucene99FlatVectorsReader.VECTOR_DATA_CODEC_NAME,
            version,
            segmentId,
            segmentSuffix
        );
    }

    @Override
    public void close() throws IOException {
        if (meta != null) {
            // write end of fields marker
            meta.writeInt(-1);
            CodecUtil.writeFooter(meta);
        }
        if (vectorData != null) {
            CodecUtil.writeFooter(vectorData);
        }

        IOUtils.close(meta, vectorData);
    }

    public ReorderedFlatFieldVectorsWriter<?> addField(final FieldInfo fieldInfo) throws IOException {
        // TODO : Byte index (dense + sparse), Float index (sparse)
        fields.add(new ReorderedDenseFloatFlatFieldVectorsWriter(fieldInfo));
        return fields.getLast();
    }

    private class ReorderedDenseFloatFlatFieldVectorsWriter extends ReorderedFlatFieldVectorsWriter<float[]> {
        private FieldInfo fieldInfo;
        private Long[] docAndOrds;
        private int docIdx;
        private int ord;
        private long vecStartOffset;
        private final ByteBuffer buffer;

        public ReorderedDenseFloatFlatFieldVectorsWriter(final FieldInfo fieldInfo) {
            this.fieldInfo = fieldInfo;
            this.docAndOrds = new Long[numDocs];
            this.docIdx = 0;
            this.ord = 0;
            this.vecStartOffset = vectorData.getFilePointer();
            this.buffer = ByteBuffer.allocate(fieldInfo.getVectorDimension() * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        }

        @Override
        public void addValue(final int docID, final float[] vector) throws IOException {
            docAndOrds[docIdx++] = ((long) docID) << 32 | ord;
            ++ord;
            buffer.asFloatBuffer().put(vector);
            vectorData.writeBytes(buffer.array(), buffer.array().length);
        }

        @Override
        public void finish() throws IOException {
            // Write meta info
            meta.writeInt(fieldInfo.number);
            meta.writeInt(fieldInfo.getVectorEncoding().ordinal());
            meta.writeInt(fieldInfo.getVectorSimilarityFunction().ordinal());
            final long vectorDataOffset = vectorData.getFilePointer() - vecStartOffset;
            meta.writeVLong(vecStartOffset);
            meta.writeVLong(vectorDataOffset);
            meta.writeVInt(fieldInfo.getVectorDimension());

            // Sort by doc ids in asc
            Arrays.sort(docAndOrds, (a, b) -> {
                final int docA = (int) (a >>> 32); // extract upper 32 bits
                final int docB = (int) (b >>> 32);
                return Integer.compare(docA, docB);
            });

            // Is dense = true
            meta.writeByte((byte) 1);
            // Max doc id
            final int maxDoc = (int) (docAndOrds[docAndOrds.length - 1] >>> 32);
            meta.writeInt(maxDoc);
            // #level
            meta.writeInt(numLevel);
            // #docs for a group
            meta.writeInt(numDocsForGrouping);
            // gropu factor
            meta.writeInt(groupFactor);

            // Create skip list
            final ByteBuffersDataOutput bufferedDataOutput = ByteBuffersDataOutput.newResettableInstance();
            final ByteBuffersIndexOutput bufferedIndexOutput = new ByteBuffersIndexOutput(
                bufferedDataOutput,
                "DocId2OrdSkipList",
                "DocId2OrdSkipListWriter"
            );
//            final DocIdOrdSkipListIndexBuilder skipListIndexBuilder = new DocIdOrdSkipListIndexBuilder(
//                true,
//                numLevel,
//                numDocsForGrouping,
//                groupFactor,
//                bufferedIndexOutput
//            );

            final FixedBlockSkipListIndexBuilder skipListIndexBuilder = new FixedBlockSkipListIndexBuilder(meta, maxDoc);

            // Flush a skip list
            for (final long docAndOrd : docAndOrds) {
                skipListIndexBuilder.add((int) (docAndOrd >>> 32), (int) docAndOrd);
            }
            skipListIndexBuilder.finish();
//            long[] offsets = skipListIndexBuilder.finish();

            // e.g. offsets[0] -> starting offset of ords block, offset[1] -> starting offset of level-1 skip-list
            // Write skip list offsets
//            meta.writeVInt(offsets.length);
//            for (long offset : offsets) {
//                meta.writeLong(offset);
//            }

            // Then flush skip list bytes
//            bufferedDataOutput.copyTo(meta);
        }
    }

    public static abstract class ReorderedFlatFieldVectorsWriter<T> {
        public abstract void addValue(int docID, T vectorValue) throws IOException;

        public abstract void finish() throws IOException;
    }
}
