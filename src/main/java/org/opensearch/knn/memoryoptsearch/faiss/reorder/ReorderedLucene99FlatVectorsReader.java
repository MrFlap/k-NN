package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
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
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FileDataHint;
import org.apache.lucene.store.FileTypeHint;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.readSimilarityFunction;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.readVectorEncoding;

public class ReorderedLucene99FlatVectorsReader extends FlatVectorsReader {

    public static final String META_CODEC_NAME = "ReorderedLucene99FlatVectorsFormatMeta";
    public static final String VECTOR_DATA_CODEC_NAME = "ReorderedLucene99FlatVectorsFormatData";
    public static final String META_EXTENSION = "vemf";
    public static final String VECTOR_DATA_EXTENSION = "vec";
    public static final int VERSION_START = 0;
    public static final int VERSION_CURRENT = VERSION_START;

    private final IntObjectHashMap<ReorderedLucene99FlatVectorsReader.FieldEntry> fields = new IntObjectHashMap<>();
    private final IndexInput vectorData;
    private final FieldInfos fieldInfos;
    private final IOContext dataContext;
    private final String customSuffix;
    private MMapDirectory afterReorderingMMapDirectory;

    public ReorderedLucene99FlatVectorsReader(SegmentReadState state, FlatVectorsScorer scorer) throws IOException {
        this(state, scorer, false);
    }

    public ReorderedLucene99FlatVectorsReader(
        final SegmentReadState state,
        final FlatVectorsScorer scorer,
        final boolean useReorderedSuffix
    ) throws IOException {
        super(scorer);

        if (useReorderedSuffix == false && Files.exists(Path.of("/tmp/dododo"))) {
            final Path directory = (((FSDirectory) FilterDirectory.unwrap(state.directory))).getDirectory();
            System.out.println("############### Original directory -> " + directory);
            final String afterReorderingDirectory =
                directory.toAbsolutePath().toString().replaceAll("before-reordering", "after-reordering");
            System.out.println("############### New directory -> " + afterReorderingDirectory);
            afterReorderingMMapDirectory = new MMapDirectory(Path.of(afterReorderingDirectory));
        }

        this.customSuffix = useReorderedSuffix ? ".reorder" : "";
        final int versionMeta = readMetadata(state);
        this.fieldInfos = state.fieldInfos;
        boolean success = false;
        // Flat formats are used to randomly access vectors from their node ID that is stored
        // in the HNSW graph.
        dataContext = state.context.withHints(FileTypeHint.DATA, FileDataHint.KNN_VECTORS, DataAccessHint.RANDOM);

        try {
            vectorData = openDataInput(state, versionMeta, VECTOR_DATA_EXTENSION + customSuffix, VECTOR_DATA_CODEC_NAME, dataContext);
            success = true;
        } finally {
            if (success == false) {
                IOUtils.closeWhileHandlingException(this);
            }
        }
    }

    private IndexInput openDataInput(SegmentReadState state, int versionMeta, String fileExtension, String codecName, IOContext context)
        throws IOException {
        String fileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, fileExtension);
        final Directory directory = afterReorderingMMapDirectory != null ? afterReorderingMMapDirectory : state.directory;
        IndexInput in = directory.openInput(fileName, context);
        boolean success = false;
        try {
            int versionVectorData = CodecUtil.checkIndexHeader(
                in,
                codecName,
                Lucene99FlatVectorsFormat.VERSION_START,
                Lucene99FlatVectorsFormat.VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix
            );
            if (versionMeta != versionVectorData) {
                throw new CorruptIndexException(
                    "Format versions mismatch: meta=" + versionMeta + ", " + codecName + "=" + versionVectorData,
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
                throw new CorruptIndexException("Invalid field number: " + fieldNumber, meta);
            }
            ReorderedLucene99FlatVectorsReader.FieldEntry fieldEntry = ReorderedLucene99FlatVectorsReader.FieldEntry.create(meta, info);
            fields.put(info.number, fieldEntry);
        }
    }

    private int readMetadata(SegmentReadState state) throws IOException {
        final Directory directory = afterReorderingMMapDirectory != null ? afterReorderingMMapDirectory : state.directory;
        String metaFileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, META_EXTENSION + customSuffix);
        int versionMeta = -1;
        try (ChecksumIndexInput meta = directory.openChecksumInput(metaFileName)) {
            Throwable priorE = null;
            try {
                versionMeta = CodecUtil.checkIndexHeader(
                    meta,
                    META_CODEC_NAME,
                    VERSION_START,
                    VERSION_CURRENT,
                    state.segmentInfo.getId(),
                    state.segmentSuffix
                );
                readFields(meta, state.fieldInfos);
            } catch (Throwable exception) {
                priorE = exception;
            } finally {
                CodecUtil.checkFooter(meta, priorE);
            }
        }
        return versionMeta;
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(String field, float[] target) throws IOException {
        final FieldEntry fieldEntry = getFieldEntry(field, VectorEncoding.BYTE);
        return vectorScorer.getRandomVectorScorer(
            fieldEntry.similarityFunction, ReorderedOffHeapFloatVectorValues.load(
                fieldEntry.similarityFunction,
                vectorScorer,
                fieldEntry.doc2OrdSkipList,
                fieldEntry.dimension,
                fieldEntry.vectorDataOffset,
                fieldEntry.vectorDataLength,
                vectorData,
                fieldEntry.doc2OrdSkipList.getMaxDoc() + 1
            ), target
        );
    }

    @Override
    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
        final FieldEntry fieldEntry = getFieldEntry(field, VectorEncoding.FLOAT32);
        return ReorderedOffHeapFloatVectorValues.load(
            fieldEntry.similarityFunction,
            vectorScorer,
            fieldEntry.doc2OrdSkipList,
            fieldEntry.dimension,
            fieldEntry.vectorDataOffset,
            fieldEntry.vectorDataLength,
            vectorData,
            fieldEntry.doc2OrdSkipList.getMaxDoc() + 1
        );
    }

    private ReorderedLucene99FlatVectorsReader.FieldEntry getFieldEntry(String field, VectorEncoding expectedEncoding) {
        final ReorderedLucene99FlatVectorsReader.FieldEntry fieldEntry = getFieldEntryOrThrow(field);
        if (fieldEntry.vectorEncoding != expectedEncoding) {
            throw new IllegalArgumentException(
                "field=\"" + field + "\" is encoded as: " + fieldEntry.vectorEncoding + " expected: " + expectedEncoding);
        }
        return fieldEntry;
    }

    private ReorderedLucene99FlatVectorsReader.FieldEntry getFieldEntryOrThrow(String field) {
        final FieldInfo info = fieldInfos.fieldInfo(field);
        final ReorderedLucene99FlatVectorsReader.FieldEntry entry;
        if (info == null || (entry = fields.get(info.number)) == null) {
            throw new IllegalArgumentException("field=\"" + field + "\" not found");
        }
        return entry;
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
    public ByteVectorValues getByteVectorValues(String field) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void checkIntegrity() {}

    @Override
    public RandomVectorScorer getRandomVectorScorer(String field, byte[] target) {
        throw new UnsupportedOperationException();
    }

    private record FieldEntry(VectorSimilarityFunction similarityFunction, VectorEncoding vectorEncoding, long vectorDataOffset,
                              long vectorDataLength, int dimension, DocIdOrdSkipListIndex doc2OrdSkipList, FieldInfo info) {

        FieldEntry {
            if (similarityFunction != info.getVectorSimilarityFunction()) {
                throw new IllegalStateException(
                    "Inconsistent vector similarity function for field=\"" + info.name + "\"; " + similarityFunction + " != "
                    + info.getVectorSimilarityFunction());
            }
            int infoVectorDimension = info.getVectorDimension();
            if (infoVectorDimension != dimension) {
                throw new IllegalStateException(
                    "Inconsistent vector dimension for field=\"" + info.name + "\"; " + infoVectorDimension + " != " + dimension);
            }
        }

        static ReorderedLucene99FlatVectorsReader.FieldEntry create(IndexInput input, FieldInfo info) throws IOException {
            // Read meta
            final VectorEncoding vectorEncoding = readVectorEncoding(input);
            final VectorSimilarityFunction similarityFunction = readSimilarityFunction(input);
            final var vectorDataOffset = input.readVLong();
            final var vectorDataLength = input.readVLong();
            final var dimension = input.readVInt();

            // Read skip list meta
            // Is dense?
            final boolean isDense = input.readByte() == 1;
            // Max doc id
            final int maxDoc = input.readInt();
            // #level
            final int numLevel = input.readInt();
            // #docs for a group
            final int numDocsForGrouping = input.readInt();
            // group factor
            final int groupFactor = input.readInt();

            // offsets
            final int offsetLen = input.readVInt();
            final long[] skipListOffsets = new long[offsetLen];
            for (int i = 0; i < offsetLen; ++i) {
                skipListOffsets[i] = input.readLong();
            }

            // Skip list index
            final DocIdOrdSkipListIndex skipListIndex =
                new DocIdOrdSkipListIndex(input, isDense, numLevel, numDocsForGrouping, groupFactor, skipListOffsets, maxDoc);

            // Read skip list
            return new ReorderedLucene99FlatVectorsReader.FieldEntry(
                similarityFunction,
                                                                     vectorEncoding,
                                                                     vectorDataOffset,
                                                                     vectorDataLength,
                                                                     dimension,
                                                                     skipListIndex,
                                                                     info
            );
        }
    }
}
