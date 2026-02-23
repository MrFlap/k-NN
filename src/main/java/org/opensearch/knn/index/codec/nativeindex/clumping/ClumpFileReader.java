/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.clumping;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.jni.SimdVectorComputeService;
import org.opensearch.knn.memoryoptsearch.MemorySegmentAddressExtractorUtil;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Reads a .clump sidecar file (v2 format with inline vectors) to expand marker
 * vector results. Reads hidden vectors sequentially from the clump file and scores
 * them directly against the query vector, avoiding random access into Lucene's
 * vector storage.
 * <p>
 * See {@link ClumpFileFormat} for the binary layout.
 */
@Log4j2
public final class ClumpFileReader {

    private ClumpFileReader() {}

    /**
     * Cached marker table data for a segment+field pair. Segments are immutable once
     * committed, so cached entries never go stale. Eviction happens via
     * {@link #evictMarkerTableCache(String, String)} when segments are deleted during merge.
     */
    static final class CachedMarkerTable {
        final int numMarkers;
        final int dimension;
        final byte vectorDataType;
        final int vectorSize;
        final int[] markerDocIds;
        final int[] numHidden;
        final long[] clumpDataOffsets;
        final String clumpFileName;

        CachedMarkerTable(
            int numMarkers, int dimension, byte vectorDataType, int vectorSize,
            int[] markerDocIds, int[] numHidden, long[] clumpDataOffsets, String clumpFileName
        ) {
            this.numMarkers = numMarkers;
            this.dimension = dimension;
            this.vectorDataType = vectorDataType;
            this.vectorSize = vectorSize;
            this.markerDocIds = markerDocIds;
            this.numHidden = numHidden;
            this.clumpDataOffsets = clumpDataOffsets;
            this.clumpFileName = clumpFileName;
        }
    }

    /** Cache of parsed marker tables keyed by "segmentName/fieldName". */
    private static final ConcurrentHashMap<String, CachedMarkerTable> MARKER_TABLE_CACHE = new ConcurrentHashMap<>();

    private static String cacheKey(String segmentName, String fieldName) {
        return segmentName + "/" + fieldName;
    }

    /**
     * Evicts the cached marker table for a segment+field pair. Should be called
     * when a segment is deleted (e.g., after merge).
     */
    public static void evictMarkerTableCache(String segmentName, String fieldName) {
        MARKER_TABLE_CACHE.remove(cacheKey(segmentName, fieldName));
    }

    /**
     * Clears the entire marker table cache. Useful for testing.
     */
    public static void clearMarkerTableCache() {
        MARKER_TABLE_CACHE.clear();
    }

    /**
     * Loads or retrieves the cached marker table for the given segment and field.
     * Returns null if no clump file exists.
     */
    private static CachedMarkerTable getOrLoadMarkerTable(
        Directory directory, String segmentName, String fieldName
    ) throws IOException {
        String key = cacheKey(segmentName, fieldName);
        CachedMarkerTable cached = MARKER_TABLE_CACHE.get(key);
        if (cached != null) {
            return cached;
        }

        String clumpFileName = resolveClumpFileName(directory, segmentName, fieldName);
        if (clumpFileName == null) {
            return null;
        }

        try (IndexInput input = directory.openInput(clumpFileName, IOContext.DEFAULT)) {
            int numMarkers = input.readInt();
            if (numMarkers == 0) {
                return null;
            }
            int dimension = input.readInt();
            byte vectorDataType = input.readByte();
            int vectorSize = ClumpFileFormat.vectorBytes(dimension, vectorDataType);

            int[] markerDocIds = new int[numMarkers];
            int[] numHidden = new int[numMarkers];
            long[] clumpDataOffsets = new long[numMarkers];

            for (int i = 0; i < numMarkers; i++) {
                markerDocIds[i] = input.readInt();
                numHidden[i] = input.readInt();
                clumpDataOffsets[i] = input.readLong();
            }

            cached = new CachedMarkerTable(
                numMarkers, dimension, vectorDataType, vectorSize,
                markerDocIds, numHidden, clumpDataOffsets, clumpFileName
            );
            MARKER_TABLE_CACHE.putIfAbsent(key, cached);
            return cached;
        }
    }

    /**
     * Checks whether a clump file exists for the given segment and field.
     * Uses the marker table cache when available to avoid directory listing.
     */
    public static boolean clumpFileExists(Directory directory, String segmentName, String fieldName) throws IOException {
        if (MARKER_TABLE_CACHE.containsKey(cacheKey(segmentName, fieldName))) {
            return true;
        }
        String clumpFileName = resolveClumpFileName(directory, segmentName, fieldName);
        return clumpFileName != null;
    }

    /**
     * Resolves the actual clump file name, checking for both the compound (.clumpc)
     * and original (.clump) variants. Returns null if neither exists.
     */
    private static String resolveClumpFileName(Directory directory, String segmentName, String fieldName) throws IOException {
        String baseName = ClumpFileWriter.buildClumpFileName(segmentName, fieldName);
        String compoundName = baseName + KNNConstants.COMPOUND_EXTENSION;
        for (String file : directory.listAll()) {
            if (file.equals(compoundName)) {
                return compoundName;
            }
            if (file.equals(baseName)) {
                return baseName;
            }
        }
        return null;
    }

    /**
     * Expands marker doc IDs by reading their hidden vectors from the .clump file,
     * scoring each hidden vector against the query vector, and returning scored results.
     * <p>
     * The read phase is parallelized: each matched marker's raw hidden entry bytes are
     * read concurrently via cloned {@link IndexInput} handles on a parallel stream.
     * The scoring phase runs sequentially over the pre-read byte buffers to avoid
     * thread-safety concerns with the similarity function and to keep CPU work on the
     * calling thread.
     *
     * @param directory          The segment directory
     * @param segmentName        The segment name
     * @param fieldName          The vector field name
     * @param markerDocIds       The marker doc IDs from the ANN search results
     * @param floatQueryVector   The float query vector (null for byte vectors)
     * @param byteQueryVector    The byte query vector (null for float vectors)
     * @param similarityFunction The similarity function for scoring
     * @return List of ScoreDoc for hidden vectors. Does NOT include the markers themselves.
     */
    public static List<ScoreDoc> getHiddenVectorsScored(
        Directory directory,
        String segmentName,
        String fieldName,
        int[] markerDocIds,
        float[] floatQueryVector,
        byte[] byteQueryVector,
        KNNVectorSimilarityFunction similarityFunction
    ) throws IOException {
        CachedMarkerTable table = getOrLoadMarkerTable(directory, segmentName, fieldName);
        if (table == null) {
            return Collections.emptyList();
        }

        // Resolve which marker table indices match the query markers
        int[] matchedIndices = Arrays.stream(markerDocIds)
            .map(docId -> Arrays.binarySearch(table.markerDocIds, docId))
            .filter(idx -> idx >= 0 && table.numHidden[idx] > 0)
            .toArray();

        if (matchedIndices.length == 0) {
            return Collections.emptyList();
        }

        // Sort matched indices by file offset for better I/O locality
        Arrays.sort(matchedIndices);

        // v3 layout: per marker, doc IDs and vectors are in separate contiguous blocks.
        // docIdBlockStart = clumpDataOffset + vectorSize
        // vectorBlockStart = docIdBlockStart + numHidden * 4
        boolean isFloat = (table.vectorDataType == ClumpFileFormat.VECTOR_TYPE_FLOAT);
        boolean isFp16 = (table.vectorDataType == ClumpFileFormat.VECTOR_TYPE_FP16);

        // For FP16, try to get mmap addresses for SIMD bulk scoring
        SimdVectorComputeService.SimilarityFunctionType simdFunctionType = null;
        if (isFp16) {
            simdFunctionType = toSimdFunctionType(similarityFunction);
        }

        // Phase 1: READ doc IDs in parallel. For non-SIMD paths, also read vector blocks.
        int[][] markerDocIdArrays = new int[matchedIndices.length][];
        byte[][] markerVectorBlocks = (simdFunctionType == null) ? new byte[matchedIndices.length][] : null;
        // For SIMD path, store vector block file offsets and sizes instead of reading bytes
        long[] vecBlockOffsets = (simdFunctionType != null) ? new long[matchedIndices.length] : null;
        int[] vecBlockSizes = (simdFunctionType != null) ? new int[matchedIndices.length] : null;

        try (IndexInput input = directory.openInput(table.clumpFileName, IOContext.DEFAULT)) {
            final byte[][] finalVectorBlocks = markerVectorBlocks;
            final long[] finalVecOffsets = vecBlockOffsets;
            final int[] finalVecSizes = vecBlockSizes;

            Arrays.stream(matchedIndices).parallel().forEach(markerIndex -> {
                int idx = Arrays.binarySearch(matchedIndices, markerIndex);
                int numHidden = table.numHidden[markerIndex];
                long docIdStart = table.clumpDataOffsets[markerIndex] + table.vectorSize;
                int docIdBlockSize = numHidden * Integer.BYTES;
                int vecBlockSize = numHidden * table.vectorSize;

                if (finalVectorBlocks != null) {
                    // Non-SIMD path: read doc IDs + vector block in a single I/O operation
                    // since they are contiguous in the v3 layout.
                    int totalSize = docIdBlockSize + vecBlockSize;
                    byte[] combined = new byte[totalSize];
                    try (IndexInput clonedInput = input.clone()) {
                        clonedInput.seek(docIdStart);
                        clonedInput.readBytes(combined, 0, totalSize);
                    } catch (IOException e) {
                        log.warn("Error reading clump data for marker index {}", markerIndex, e);
                    }

                    // Parse doc IDs from the first portion of the combined buffer
                    int[] docIds = new int[numHidden];
                    java.nio.ByteBuffer dbuf = java.nio.ByteBuffer.wrap(combined, 0, docIdBlockSize)
                        .order(java.nio.ByteOrder.LITTLE_ENDIAN);
                    for (int j = 0; j < numHidden; j++) {
                        docIds[j] = dbuf.getInt();
                    }
                    markerDocIdArrays[idx] = docIds;

                    // Extract vector block from the remainder
                    byte[] vecBlock = new byte[vecBlockSize];
                    System.arraycopy(combined, docIdBlockSize, vecBlock, 0, vecBlockSize);
                    finalVectorBlocks[idx] = vecBlock;
                } else {
                    // SIMD path: read only doc IDs; vectors will be scored directly from mmap
                    int[] docIds = new int[numHidden];
                    byte[] docIdBytes = new byte[docIdBlockSize];
                    try (IndexInput clonedInput = input.clone()) {
                        clonedInput.seek(docIdStart);
                        clonedInput.readBytes(docIdBytes, 0, docIdBytes.length);
                        java.nio.ByteBuffer dbuf = java.nio.ByteBuffer.wrap(docIdBytes)
                            .order(java.nio.ByteOrder.LITTLE_ENDIAN);
                        for (int j = 0; j < numHidden; j++) {
                            docIds[j] = dbuf.getInt();
                        }
                    } catch (IOException e) {
                        log.warn("Error reading doc IDs for marker index {}", markerIndex, e);
                    }
                    markerDocIdArrays[idx] = docIds;

                    // Record the offset and size for SIMD scoring later
                    long vecStart = docIdStart + (long) docIdBlockSize;
                    finalVecOffsets[idx] = vecStart;
                    finalVecSizes[idx] = vecBlockSize;
                }
            });

            // Phase 2: SCORE
            List<ScoreDoc> scoredHidden = new ArrayList<>();

            if (simdFunctionType != null) {
                // SIMD bulk scoring for FP16 vectors via native code.
                // For each marker, extract mmap addresses for its vector block and score in bulk.
                for (int m = 0; m < matchedIndices.length; m++) {
                    int[] docIds = markerDocIdArrays[m];
                    if (docIds == null || docIds.length == 0) {
                        continue;
                    }
                    int numHidden = docIds.length;

                    long[] addressAndSize = MemorySegmentAddressExtractorUtil.tryExtractAddressAndSize(
                        input, vecBlockOffsets[m], vecBlockSizes[m]
                    );

                    if (addressAndSize != null) {
                        // SIMD path: bulk score all hidden vectors for this marker
                        SimdVectorComputeService.saveSearchContext(
                            floatQueryVector, addressAndSize, simdFunctionType.ordinal()
                        );
                        int[] ordinals = new int[numHidden];
                        for (int j = 0; j < numHidden; j++) {
                            ordinals[j] = j;
                        }
                        float[] scores = new float[numHidden];
                        SimdVectorComputeService.scoreSimilarityInBulk(ordinals, scores, numHidden);
                        for (int j = 0; j < numHidden; j++) {
                            scoredHidden.add(new ScoreDoc(docIds[j], scores[j]));
                        }
                    } else {
                        // Fallback: read vector block and score with Java
                        byte[] vecBlock = new byte[vecBlockSizes[m]];
                        try (IndexInput clonedInput = input.clone()) {
                            clonedInput.seek(vecBlockOffsets[m]);
                            clonedInput.readBytes(vecBlock, 0, vecBlock.length);
                        }
                        scoreFp16VectorBlock(docIds, vecBlock, table.dimension, floatQueryVector, similarityFunction, scoredHidden);
                    }
                }
            } else {
                // Java scoring path for float and byte vectors
                for (int m = 0; m < matchedIndices.length; m++) {
                    int[] docIds = markerDocIdArrays[m];
                    byte[] vecBlock = markerVectorBlocks[m];
                    if (docIds == null || vecBlock == null) {
                        continue;
                    }
                    int numHidden = docIds.length;
                    java.nio.ByteBuffer vb = java.nio.ByteBuffer.wrap(vecBlock)
                        .order(java.nio.ByteOrder.LITTLE_ENDIAN);

                    if (isFloat) {
                        float[] reusableVector = new float[table.dimension];
                        for (int j = 0; j < numHidden; j++) {
                            for (int d = 0; d < table.dimension; d++) {
                                reusableVector[d] = vb.getFloat();
                            }
                            float score = similarityFunction.compare(floatQueryVector, reusableVector);
                            scoredHidden.add(new ScoreDoc(docIds[j], score));
                        }
                    } else {
                        byte[] reusableVector = new byte[table.dimension];
                        for (int j = 0; j < numHidden; j++) {
                            vb.get(reusableVector);
                            float score = similarityFunction.compare(byteQueryVector, reusableVector);
                            scoredHidden.add(new ScoreDoc(docIds[j], score));
                        }
                    }
                }
            }

            return scoredHidden;
        }
    }

    /**
     * Maps a KNNVectorSimilarityFunction to the corresponding SIMD function type for FP16.
     * Returns null if the similarity function is not supported by the SIMD service.
     */
    private static SimdVectorComputeService.SimilarityFunctionType toSimdFunctionType(
        KNNVectorSimilarityFunction similarityFunction
    ) {
        if (similarityFunction == KNNVectorSimilarityFunction.MAXIMUM_INNER_PRODUCT
            || similarityFunction == KNNVectorSimilarityFunction.DOT_PRODUCT) {
            return SimdVectorComputeService.SimilarityFunctionType.FP16_MAXIMUM_INNER_PRODUCT;
        } else if (similarityFunction == KNNVectorSimilarityFunction.EUCLIDEAN) {
            return SimdVectorComputeService.SimilarityFunctionType.FP16_L2;
        }
        return null;
    }

    /**
     * Fallback Java scoring for FP16 vector blocks when SIMD is not available.
     */
    private static void scoreFp16VectorBlock(
        int[] docIds,
        byte[] vecBlock,
        int dimension,
        float[] queryVector,
        KNNVectorSimilarityFunction similarityFunction,
        List<ScoreDoc> results
    ) {
        java.nio.ByteBuffer vb = java.nio.ByteBuffer.wrap(vecBlock).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        float[] reusableVector = new float[dimension];
        for (int j = 0; j < docIds.length; j++) {
            for (int d = 0; d < dimension; d++) {
                reusableVector[d] = Float.float16ToFloat(vb.getShort());
            }
            float score = similarityFunction.compare(queryVector, reusableVector);
            results.add(new ScoreDoc(docIds[j], score));
        }
    }

    /**
     * Returns just the hidden doc IDs (without scoring) for backward compatibility
     * or cases where only doc IDs are needed.
     */
    public static List<Integer> getHiddenDocIds(
        Directory directory,
        String segmentName,
        String fieldName,
        int[] markerDocIds
    ) throws IOException {
        CachedMarkerTable table = getOrLoadMarkerTable(directory, segmentName, fieldName);
        if (table == null) {
            return Collections.emptyList();
        }

        try (IndexInput input = directory.openInput(table.clumpFileName, IOContext.DEFAULT)) {
            List<Integer> hiddenDocIds = new ArrayList<>();

            for (int queryMarkerDocId : markerDocIds) {
                int markerIndex = Arrays.binarySearch(table.markerDocIds, queryMarkerDocId);
                if (markerIndex < 0) {
                    continue;
                }

                int numHidden = table.numHidden[markerIndex];
                if (numHidden == 0) {
                    continue;
                }

                // v3: doc IDs are in a contiguous block after the marker vector
                long docIdStart = table.clumpDataOffsets[markerIndex] + table.vectorSize;
                input.seek(docIdStart);

                for (int j = 0; j < numHidden; j++) {
                    hiddenDocIds.add(input.readInt());
                }
            }

            return hiddenDocIds;
        }
    }

}
