/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsReader;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.kmeansreorder.KMeansReorderStrategy;

import java.io.IOException;
import java.util.Collections;

import static org.opensearch.knn.index.codec.util.KNNCodecUtil.buildEngineFileName;

/**
 * Orchestrates post-write reordering of .vec and .faiss segment files.
 * Opens the .vec via Lucene99FlatVectorsReader (mmap-backed) and passes FloatVectorValues
 * to the VectorReorderStrategy — BP uses zero-heap-copy, KMeans materializes to heap.
 */
@Log4j2
public class SegmentReorderService {

    public static final int MIN_VECTORS_FOR_REORDER = 10_000;
    public static final int DEFAULT_REORDER_THREADS = 4;

    private static final String REORDER_SUFFIX = ".reorder";

    private final SegmentWriteState state;
    private final FieldInfo fieldInfo;
    private final VectorReorderStrategy strategy;
    private final int numThreads;

    public SegmentReorderService(SegmentWriteState state, FieldInfo fieldInfo, VectorReorderStrategy strategy) {
        this(state, fieldInfo, strategy, DEFAULT_REORDER_THREADS);
    }

    public SegmentReorderService(SegmentWriteState state, FieldInfo fieldInfo, VectorReorderStrategy strategy, int numThreads) {
        this.state = state;
        this.fieldInfo = fieldInfo;
        this.strategy = strategy;
        this.numThreads = numThreads;
    }

    /**
     * Reorder .vec, .vemf, and .faiss files for the current segment/field.
     * Requires that .vec/.vemf have finalized codec footers (call after flatVectorsWriter.finish()).
     * Reads vectors from the .vec file via Lucene99FlatVectorsReader, computes the permutation,
     * then rewrites all three files with the reordered layout.
     */
    /**
     * Reorder using pre-loaded cluster summaries from source segments (merge-aware path).
     * Opens the finalized .vec file, computes permutation via centroid merging, writes .kcs,
     * then rewrites .vec, .vemf, and .faiss.
     */
    public void reorderSegmentFilesWithSummaries(
        java.util.List<org.opensearch.knn.memoryoptsearch.faiss.reorder.kmeansreorder.ClusterSummary> sourceSummaries,
        MergeAwareReorderStrategy mergeAware
    ) throws IOException {
        final Directory directory = state.directory;
        final String segmentName = state.segmentInfo.name;

        // Open finalized .vec to get FloatVectorValues
        final String vecDataFileName = findFileWithSuffix(directory, segmentName, ".vec");
        final String vecMetaFileName = findFileWithSuffix(directory, segmentName, ".vemf");
        if (vecDataFileName == null || vecMetaFileName == null) {
            throw new IOException("Cannot find .vec/.vemf files for segment " + segmentName);
        }

        final FieldInfo readFieldInfo = buildFieldInfoForRead();
        final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { readFieldInfo });
        final String segmentSuffix = state.segmentSuffix;

        final SegmentReadState readState = new SegmentReadState(
            directory, state.segmentInfo, fieldInfos, IOContext.DEFAULT, segmentSuffix
        );

        final int[] permutation;
        try (Lucene99FlatVectorsReader reader = new Lucene99FlatVectorsReader(readState, DefaultFlatVectorScorer.INSTANCE)) {
            final FloatVectorValues vectorValues = reader.getFloatVectorValues(fieldInfo.name);
            if (vectorValues == null) {
                throw new IOException("No float vector values for field " + fieldInfo.name);
            }

            var result = mergeAware.computePermutationFromMergedSummaries(
                sourceSummaries, vectorValues, numThreads, fieldInfo.getVectorSimilarityFunction()
            );
            permutation = result.permutation();
            log.info("Merge-aware permutation size={}, vectorValues.size()={}", permutation.length, vectorValues.size());

            // Write .kcs for the merged segment
            org.opensearch.knn.memoryoptsearch.faiss.reorder.kmeansreorder.ClusterSummaryWriter.writeAndGetFileName(
                state, fieldInfo, result.summary()
            );
        }

        final ReorderOrdMap reorderOrdMap = new ReorderOrdMap(permutation);

        final Directory writeDir = new FilterDirectory(directory) {
            @Override
            public IndexOutput createOutput(String name, IOContext context) throws IOException {
                return in.createOutput(name, state.context);
            }
        };

        rewriteVecFile(directory, writeDir, vecDataFileName, vecMetaFileName, reorderOrdMap);

        final KNNEngine knnEngine = KNNEngine.FAISS;
        final String engineFileName = buildEngineFileName(
            segmentName, knnEngine.getVersion(), fieldInfo.name, knnEngine.getExtension()
        );
        rewriteFaissFile(directory, writeDir, engineFileName, reorderOrdMap);

        log.info("Merge-aware reorder complete for segment {}, field [{}]", segmentName, fieldInfo.name);
    }

    public void reorderSegmentFiles() throws IOException {
        final Directory directory = state.directory;
        final String segmentName = state.segmentInfo.name;

        // Step 1: Compute permutation from finalized .vec file
        final int[] permutation = computePermutationFromVecFile(directory, segmentName);
        final ReorderOrdMap reorderOrdMap = new ReorderOrdMap(permutation);

        // Wrap directory to use the correct IOContext (MERGE during merge, DEFAULT during flush).
        // ConcurrentMergeScheduler asserts createOutput uses IOContext.MERGE.
        final Directory writeDir = new FilterDirectory(directory) {
            @Override
            public IndexOutput createOutput(String name, IOContext context) throws IOException {
                return in.createOutput(name, state.context);
            }
        };

        // Step 2: Rewrite .vec and .vemf in reordered layout
        final String vecDataFileName = findFileWithSuffix(directory, segmentName, ".vec");
        final String vecMetaFileName = findFileWithSuffix(directory, segmentName, ".vemf");
        rewriteVecFile(directory, writeDir, vecDataFileName, vecMetaFileName, reorderOrdMap);

        // Step 3: Rewrite .faiss file with remapped neighbor lists
        final KNNEngine knnEngine = KNNEngine.FAISS;
        final String engineFileName = buildEngineFileName(
            segmentName, knnEngine.getVersion(), fieldInfo.name, knnEngine.getExtension()
        );
        rewriteFaissFile(directory, writeDir, engineFileName, reorderOrdMap);
    }

    private int[] computePermutationFromVecFile(Directory directory, String segmentName) throws IOException {
        final String vecDataFileName = findFileWithSuffix(directory, segmentName, ".vec");
        final String vecMetaFileName = findFileWithSuffix(directory, segmentName, ".vemf");
        if (vecDataFileName == null || vecMetaFileName == null) {
            throw new IOException("Cannot find .vec/.vemf files for segment " + segmentName);
        }

        // Build a SegmentReadState to open the Lucene99FlatVectorsReader
        final FieldInfo readFieldInfo = buildFieldInfoForRead();
        final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { readFieldInfo });
        final String segmentSuffix = extractSegmentSuffix(vecDataFileName, segmentName);

        final SegmentReadState readState = new SegmentReadState(
            directory, state.segmentInfo, fieldInfos, IOContext.DEFAULT, segmentSuffix
        );

        try (Lucene99FlatVectorsReader reader = new Lucene99FlatVectorsReader(readState, DefaultFlatVectorScorer.INSTANCE)) {
            final FloatVectorValues vectorValues = reader.getFloatVectorValues(fieldInfo.name);
            if (vectorValues == null) {
                throw new IOException("No float vector values for field " + fieldInfo.name);
            }

            // Try mmap passthrough for KMeans — avoids ~8GB heap + ~8GB native allocation
            if (strategy instanceof KMeansReorderStrategy kmeansStrategy) {
                int[] mmapResult = tryMMapKMeans(kmeansStrategy, directory, vecDataFileName, vectorValues);
                if (mmapResult != null) {
                    return mmapResult;
                }
                log.info("mmap KMeans unavailable, falling back to heap materialization");
            }

            log.info("Computing reorder permutation for {} vectors, field [{}]", vectorValues.size(), fieldInfo.name);
            int[] perm = strategy.computePermutation(vectorValues, numThreads, fieldInfo.getVectorSimilarityFunction());
            return perm;
        }
    }

    /**
     * Try to run KMeans via mmap passthrough. Returns null if mmap extraction fails.
     */
    private int[] tryMMapKMeans(
        KMeansReorderStrategy kmeansStrategy, Directory directory,
        String vecDataFileName, FloatVectorValues vectorValues
    ) {
        try (IndexInput rawVecInput = directory.openInput(vecDataFileName, IOContext.DEFAULT)) {
            // Skip the codec header to find where vector data starts
            CodecUtil.checkIndexHeader(
                rawVecInput,
                "Lucene99FlatVectorsFormatData",
                0, 1,
                state.segmentInfo.getId(),
                state.segmentSuffix
            );
            long headerSize = rawVecInput.getFilePointer();

            // Lucene99FlatVectorsWriter aligns to Float.BYTES after the header
            long vectorDataOffset = alignUp(headerSize, Float.BYTES);

            long vectorDataLength = (long) vectorValues.size() * vectorValues.dimension() * Float.BYTES;

            log.info("mmap KMeans: file={}, fileLen={}, headerSize={}, vectorDataOffset={}, vectorDataLen={}, n={}, dim={}",
                vecDataFileName, rawVecInput.length(), headerSize, vectorDataOffset, vectorDataLength,
                vectorValues.size(), vectorValues.dimension());

            // Open a FRESH input for address extraction (no header consumed)
            try (IndexInput mmapInput = directory.openInput(vecDataFileName, IOContext.DEFAULT)) {
                return kmeansStrategy.computePermutationMMap(
                    mmapInput, vectorDataOffset,
                    vectorValues.size(), vectorValues.dimension(),
                    fieldInfo.getVectorSimilarityFunction()
                );
            }
        } catch (Exception e) {
            log.warn("mmap KMeans attempt failed", e);
            return null;
        }
    }

    private static long alignUp(long position, int alignment) {
        long remainder = position % alignment;
        return remainder == 0 ? position : position + alignment - remainder;
    }

    private void rewriteVecFile(
        Directory readDir, Directory writeDir, String vecDataFileName, String vecMetaFileName, ReorderOrdMap reorderOrdMap
    ) throws IOException {
        final String reorderedVecData = vecDataFileName + REORDER_SUFFIX;
        final String reorderedVecMeta = vecMetaFileName + REORDER_SUFFIX;

        // Clean up any leftover reorder files
        deleteIfExists(readDir, reorderedVecData);
        deleteIfExists(readDir, reorderedVecMeta);

        final int numVectors = reorderOrdMap.newOrd2Old.length;

        // Build reader state
        final FieldInfo readFieldInfo = buildFieldInfoForRead();
        final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { readFieldInfo });
        final String segmentSuffix = extractSegmentSuffix(vecDataFileName, state.segmentInfo.name);

        final SegmentReadState readState = new SegmentReadState(
            readDir, state.segmentInfo, fieldInfos, IOContext.DEFAULT, segmentSuffix
        );
        // write the new .vemf file
        try (IndexInput vecMetaInput = readDir.openInput(vecMetaFileName, IOContext.DEFAULT)) {
            final ReorderedFlatVectorsWriter writer = new ReorderedFlatVectorsWriter(
                writeDir, reorderedVecMeta, reorderedVecData, numVectors, vecMetaInput
            );

            try (
                writer;
                Lucene99FlatVectorsReader reader = new Lucene99FlatVectorsReader(readState, DefaultFlatVectorScorer.INSTANCE)
            ) {
                final FloatVectorValues vectorValues = reader.getFloatVectorValues(fieldInfo.name);
                final ReorderedFlatVectorsWriter.ReorderedFlatFieldVectorsWriter fieldWriter = writer.addField(readFieldInfo);

                for (int newOrd = 0; newOrd < numVectors; newOrd++) {
                    int oldOrd = reorderOrdMap.newOrd2Old[newOrd];
                    float[] vector = vectorValues.vectorValue(oldOrd);
                    fieldWriter.addValue(oldOrd, vector);
                }
                fieldWriter.finish();
            }
        }

        // Atomic replace: delete originals, rename reordered files
        readDir.deleteFile(vecDataFileName);
        readDir.deleteFile(vecMetaFileName);
        readDir.rename(reorderedVecData, vecDataFileName);
        readDir.rename(reorderedVecMeta, vecMetaFileName);

        log.info("Reordered .vec file for segment {}, field [{}]", state.segmentInfo.name, fieldInfo.name);
    }

    private void rewriteFaissFile(Directory readDir, Directory writeDir, String engineFileName, ReorderOrdMap reorderOrdMap) throws IOException {
        final String reorderedFaiss = engineFileName + REORDER_SUFFIX;
        deleteIfExists(readDir, reorderedFaiss);

        try (IndexInput faissInput = readDir.openInput(engineFileName, IOContext.DEFAULT)) {
            final FaissIndex faissIndex = FaissIndex.load(faissInput);

            try (IndexOutput faissOutput = writeDir.createOutput(reorderedFaiss, IOContext.DEFAULT)) {
                FaissIndexReorderTransformer.transform(faissIndex, faissInput, faissOutput, reorderOrdMap);
            }
        }

        readDir.deleteFile(engineFileName);
        readDir.rename(reorderedFaiss, engineFileName);

        log.info("Reordered .faiss file for segment {}, field [{}]", state.segmentInfo.name, fieldInfo.name);
    }

    private FieldInfo buildFieldInfoForRead() {
        return new FieldInfo(
            fieldInfo.name,
            fieldInfo.number,
            false, false, false,
            IndexOptions.NONE,
            DocValuesType.NONE,
            DocValuesSkipIndexType.NONE,
            -1,
            Collections.emptyMap(),
            0, 0, 0,
            fieldInfo.getVectorDimension(),
            VectorEncoding.FLOAT32,
            fieldInfo.getVectorSimilarityFunction(),
            false, false
        );
    }

    /**
     * Extract the segment suffix from a .vec filename.
     * Filename format: {segmentName}_{suffix}_{fieldName}.vec
     * e.g. "_0_NativeEngines990KnnVectorsFormat_0.vec" → "NativeEngines990KnnVectorsFormat_0"
     */
    private String extractSegmentSuffix(String vecFileName, String segmentName) {
        // Strip segment prefix (e.g. "_0_") and the ".vec" extension
        String withoutPrefix = vecFileName.substring(segmentName.length() + 1); // skip "{segmentName}_"
        // The suffix is everything between the segment prefix and the field-specific part
        // Lucene99FlatVectorsFormat uses: segmentName + "_" + segmentSuffix + "_" + ...
        // But the actual suffix used in SegmentReadState is what was passed to the format.
        // For NativeEngines990KnnVectorsFormat, the suffix comes from SegmentWriteState.
        return state.segmentSuffix;
    }

    private String findFileWithSuffix(Directory directory, String segmentName, String extension) throws IOException {
        for (String file : directory.listAll()) {
            if (file.startsWith(segmentName + "_") && file.endsWith(extension)) {
                return file;
            }
        }
        return null;
    }

    private static void deleteIfExists(Directory directory, String fileName) {
        try {
            directory.deleteFile(fileName);
        } catch (IOException ignored) {}
    }
}
