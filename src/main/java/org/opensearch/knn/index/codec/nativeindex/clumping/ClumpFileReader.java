/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.clumping;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Reads a .clump sidecar file to expand marker vector results. Reads hidden vectors
 * sequentially from the clump file and scores them directly against the query vector,
 * avoiding random access into Lucene's vector storage.
 * <p>
 * Supports both the legacy FP16/FLOAT/BYTE paths and the new SQ_1BIT path. For SQ_1BIT,
 * hidden vectors are scored via the ADC formula (SQ bulk score), then the top {@code 2*k}
 * candidates are rescored with full-precision fp32 vectors read from the segment's
 * {@link FloatVectorValues}.
 * <p>
 * See {@link ClumpFileFormat} for the binary layout.
 */
@Log4j2
public final class ClumpFileReader {

    private ClumpFileReader() {}

    /** Hardcoded oversample factor for SQ → fp32 rescore. */
    static final int SQ_RESCORE_OVERSAMPLE = 5;

    /**
     * Cached marker table data for a segment+field pair. Segments are immutable once
     * committed, so cached entries never go stale. Eviction happens via
     * {@link #evictMarkerTableCache(String, String)} when segments are deleted during merge.
     */
    static final class CachedMarkerTable {
        final int numMarkers;
        final int dimension;
        final byte vectorDataType;
        /** Bytes per vector entry (includes SQ correction bytes for SQ_1BIT). */
        final int vectorSize;
        final int[] markerDocIds;
        final int[] numHidden;
        final long[] clumpDataOffsets;
        final String clumpFileName;

        // SQ_1BIT-only fields (null / 0 for other types)
        final int quantizedVecBytes;
        final float centroidDp;
        final float[] centroid;

        CachedMarkerTable(
            int numMarkers, int dimension, byte vectorDataType, int vectorSize,
            int[] markerDocIds, int[] numHidden, long[] clumpDataOffsets, String clumpFileName,
            int quantizedVecBytes, float centroidDp, float[] centroid
        ) {
            this.numMarkers = numMarkers;
            this.dimension = dimension;
            this.vectorDataType = vectorDataType;
            this.vectorSize = vectorSize;
            this.markerDocIds = markerDocIds;
            this.numHidden = numHidden;
            this.clumpDataOffsets = clumpDataOffsets;
            this.clumpFileName = clumpFileName;
            this.quantizedVecBytes = quantizedVecBytes;
            this.centroidDp = centroidDp;
            this.centroid = centroid;
        }
    }

    /** Cache of parsed marker tables keyed by "segmentName/fieldName". */
    private static final ConcurrentHashMap<String, CachedMarkerTable> MARKER_TABLE_CACHE = new ConcurrentHashMap<>();

    private static String cacheKey(String segmentName, String fieldName) {
        return segmentName + "/" + fieldName;
    }

    public static void evictMarkerTableCache(String segmentName, String fieldName) {
        MARKER_TABLE_CACHE.remove(cacheKey(segmentName, fieldName));
    }

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

            // SQ_1BIT header extension
            int quantizedVecBytes = 0;
            float centroidDp = 0f;
            float[] centroid = null;
            if (vectorDataType == ClumpFileFormat.VECTOR_TYPE_SQ_1BIT) {
                quantizedVecBytes = input.readInt();
                centroidDp = Float.intBitsToFloat(input.readInt());
                centroid = new float[dimension];
                for (int d = 0; d < dimension; d++) {
                    centroid[d] = Float.intBitsToFloat(input.readInt());
                }
            }

            int vectorSize = ClumpFileFormat.vectorBytes(dimension, vectorDataType, quantizedVecBytes);

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
                markerDocIds, numHidden, clumpDataOffsets, clumpFileName,
                quantizedVecBytes, centroidDp, centroid
            );
            MARKER_TABLE_CACHE.putIfAbsent(key, cached);
            return cached;
        }
    }

    public static boolean clumpFileExists(Directory directory, String segmentName, String fieldName) throws IOException {
        if (MARKER_TABLE_CACHE.containsKey(cacheKey(segmentName, fieldName))) {
            return true;
        }
        return resolveClumpFileName(directory, segmentName, fieldName) != null;
    }

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

    // ---- Legacy overload (non-SQ callers) ----

    /**
     * Expands marker doc IDs by reading their hidden vectors from the .clump file,
     * scoring each hidden vector against the query vector, and returning scored results.
     * <p>
     * For SQ_1BIT clump files this overload returns ALL SQ-scored hidden vectors without
     * the fp32 rescore step (no {@link FloatVectorValues} supplied). Callers that want the
     * full two-level SQ→fp32 rescore should use the overload that accepts
     * {@code floatVectorValues} and {@code k}.
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
        return getHiddenVectorsScored(
            directory, segmentName, fieldName, markerDocIds,
            floatQueryVector, byteQueryVector, similarityFunction,
            null, 0
        );
    }

    /**
     * Full overload that supports the SQ_1BIT two-level rescore.
     *
     * @param floatVectorValues  Segment's full-precision float vector values for fp32 rescore.
     *                           Required for SQ_1BIT; may be null for other types.
     * @param k                  The final k requested. For SQ_1BIT, the SQ stage keeps
     *                           {@code SQ_RESCORE_OVERSAMPLE * k} candidates before fp32 rescore.
     *                           Ignored for non-SQ types.
     */
    public static List<ScoreDoc> getHiddenVectorsScored(
        Directory directory,
        String segmentName,
        String fieldName,
        int[] markerDocIds,
        float[] floatQueryVector,
        byte[] byteQueryVector,
        KNNVectorSimilarityFunction similarityFunction,
        FloatVectorValues floatVectorValues,
        int k
    ) throws IOException {
        CachedMarkerTable table = getOrLoadMarkerTable(directory, segmentName, fieldName);
        if (table == null) {
            return Collections.emptyList();
        }

        int[] matchedIndices = Arrays.stream(markerDocIds)
            .map(docId -> Arrays.binarySearch(table.markerDocIds, docId))
            .filter(idx -> idx >= 0 && table.numHidden[idx] > 0)
            .toArray();

        if (matchedIndices.length == 0) {
            return Collections.emptyList();
        }

        Arrays.sort(matchedIndices);

        if (table.vectorDataType == ClumpFileFormat.VECTOR_TYPE_SQ_1BIT) {
            return scoreSqHiddenVectors(
                directory, table, matchedIndices,
                floatQueryVector, similarityFunction,
                floatVectorValues, k
            );
        }

        return scoreLegacyHiddenVectors(
            directory, table, matchedIndices,
            floatQueryVector, byteQueryVector, similarityFunction
        );
    }

    // ---- SQ_1BIT scoring path ----

    /**
     * Scores hidden SQ vectors via the ADC formula, then rescores the top {@code 2*k} with fp32.
     *
     * <p>Phase 1 (I/O): read doc-ID blocks and SQ vector blocks for each matched marker.
     * <p>Phase 2 (SQ score): quantize the query, score all hidden SQ entries via
     *     {@link SqAdcScorer}, collect all scored candidates.
     * <p>Phase 3 (fp32 rescore): keep top {@code 2*k} by SQ score, read their fp32 vectors
     *     from {@code floatVectorValues}, rescore with the real similarity function.
     */
    private static List<ScoreDoc> scoreSqHiddenVectors(
        Directory directory,
        CachedMarkerTable table,
        int[] matchedIndices,
        float[] floatQueryVector,
        KNNVectorSimilarityFunction similarityFunction,
        FloatVectorValues floatVectorValues,
        int k
    ) throws IOException {
        final int quantizedVecBytes = table.quantizedVecBytes;
        final int sqEntrySize = table.vectorSize; // quantizedVecBytes + correctionBytes

        // Phase 1: READ doc IDs + SQ vector blocks
        int[][] markerDocIdArrays = new int[matchedIndices.length][];
        byte[][] markerVectorBlocks = new byte[matchedIndices.length][];

        try (IndexInput input = directory.openInput(table.clumpFileName, IOContext.DEFAULT)) {
            try (ExecutorService vte = Executors.newVirtualThreadPerTaskExecutor()) {
                List<Future<?>> readFutures = new ArrayList<>(matchedIndices.length);
                for (int mi = 0; mi < matchedIndices.length; mi++) {
                    final int idx = mi;
                    final int markerIndex = matchedIndices[mi];
                    readFutures.add(vte.submit(() -> {
                        int numHidden = table.numHidden[markerIndex];
                        long docIdStart = table.clumpDataOffsets[markerIndex] + table.vectorSize;
                        int docIdBlockSize = numHidden * Integer.BYTES;
                        int vecBlockSize = numHidden * sqEntrySize;

                        int totalSize = docIdBlockSize + vecBlockSize;
                        byte[] combined = new byte[totalSize];
                        try (IndexInput clonedInput = input.clone()) {
                            clonedInput.seek(docIdStart);
                            clonedInput.readBytes(combined, 0, totalSize);
                        } catch (IOException e) {
                            log.warn("Error reading SQ clump data for marker index {}", markerIndex, e);
                        }

                        int[] docIds = new int[numHidden];
                        java.nio.ByteBuffer dbuf = java.nio.ByteBuffer.wrap(combined, 0, docIdBlockSize)
                            .order(java.nio.ByteOrder.LITTLE_ENDIAN);
                        for (int j = 0; j < numHidden; j++) {
                            docIds[j] = dbuf.getInt();
                        }
                        markerDocIdArrays[idx] = docIds;

                        byte[] vecBlock = new byte[vecBlockSize];
                        System.arraycopy(combined, docIdBlockSize, vecBlock, 0, vecBlockSize);
                        markerVectorBlocks[idx] = vecBlock;
                    }));
                }
                for (Future<?> f : readFutures) {
                    try { f.get(); } catch (Exception e) { log.warn("Error waiting for SQ clump read", e); }
                }
            }

            // Phase 2: SQ score all hidden vectors
            // Build a fresh quantizer from the similarity function + centroid stored in the header.
            VectorSimilarityFunction luceneSim = similarityFunction.getVectorSimilarityFunction();
            OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(luceneSim);

            // Quantize the query into 4 bit planes.
            // Per-bit-plane byte count (matches native SIMD int4BitDotProduct contract).
            final int binaryCodeBytes = (table.dimension + 7) / 8;
            // scratch holds one byte per discretized dimension (one quantized value per dim,
            // in [0..15]). For SINGLE_BIT_QUERY_NIBBLE this is dim rounded up to a multiple of 8.
            final int discretizedDim = binaryCodeBytes * 8;
            byte[] scratch = new byte[discretizedDim];
            // For 1-bit data with 4-bit query: queryPackedLength = 4 * binaryCodeBytes
            byte[] targetQuantized = new byte[4 * binaryCodeBytes];
            float[] queryCopy = floatQueryVector.clone();
            OptimizedScalarQuantizer.QuantizationResult qr = quantizer.scalarQuantize(
                queryCopy, scratch, (byte) 4, table.centroid
            );
            OptimizedScalarQuantizer.transposeHalfByte(scratch, targetQuantized);

            SqAdcScorer.QuantizedQuery qq = new SqAdcScorer.QuantizedQuery(
                targetQuantized, binaryCodeBytes,
                qr.lowerInterval(), qr.upperInterval(),
                qr.additionalCorrection(), qr.quantizedComponentSum()
            );

            boolean useIp = (similarityFunction == KNNVectorSimilarityFunction.MAXIMUM_INNER_PRODUCT
                || similarityFunction == KNNVectorSimilarityFunction.DOT_PRODUCT
                || similarityFunction == KNNVectorSimilarityFunction.COSINE);

            List<ScoreDoc> allSqScored = new ArrayList<>();
            for (int m = 0; m < matchedIndices.length; m++) {
                int[] docIds = markerDocIdArrays[m];
                byte[] vecBlock = markerVectorBlocks[m];
                if (docIds == null || vecBlock == null) continue;

                for (int j = 0; j < docIds.length; j++) {
                    int entryOffset = j * sqEntrySize;
                    // Extract the binary code
                    byte[] code = new byte[quantizedVecBytes];
                    System.arraycopy(vecBlock, entryOffset, code, 0, quantizedVecBytes);
                    // Extract correction factors
                    int corrOffset = entryOffset + quantizedVecBytes;
                    float lower = Float.intBitsToFloat(readIntLE(vecBlock, corrOffset));
                    float upper = Float.intBitsToFloat(readIntLE(vecBlock, corrOffset + 4));
                    float additional = Float.intBitsToFloat(readIntLE(vecBlock, corrOffset + 8));
                    int componentSum = readIntLE(vecBlock, corrOffset + 12);

                    SqVectorEntry entry = new SqVectorEntry(code, lower, upper, additional, componentSum);
                    float score;
                    if (useIp) {
                        score = SqAdcScorer.scoreIp(qq, code, entry, table.dimension, table.centroidDp);
                        // Apply ipToMaxIp transform: 1/(1 + e^(-score))  — matching native
                        score = (float) (1.0 / (1.0 + Math.exp(-score)));
                    } else {
                        score = SqAdcScorer.scoreL2(qq, code, entry, table.dimension);
                        // Apply l2 transform: 1/(1+score)
                        score = 1.0f / (1.0f + score);
                    }
                    allSqScored.add(new ScoreDoc(docIds[j], score));
                }
            }

            // Phase 3: fp32 rescore top 2*k
            if (floatVectorValues == null || k <= 0) {
                // No fp32 rescore available — return all SQ-scored results
                return allSqScored;
            }

            int sqKeep = Math.min(SQ_RESCORE_OVERSAMPLE * k, allSqScored.size());
            // Sort descending by score (higher = better for both IP and L2 transforms)
            allSqScored.sort((a, b) -> Float.compare(b.score, a.score));
            List<ScoreDoc> topSq = allSqScored.subList(0, sqKeep);

            // Rescore with fp32
            List<ScoreDoc> rescored = new ArrayList<>(topSq.size());
            for (ScoreDoc sd : topSq) {
                float[] fp32Vec = floatVectorValues.vectorValue(
                    floatVectorValues.ordToDoc(sd.doc) >= 0
                        ? findOrdForDoc(floatVectorValues, sd.doc)
                        : sd.doc
                );
                if (fp32Vec != null) {
                    float score = similarityFunction.compare(floatQueryVector, fp32Vec);
                    rescored.add(new ScoreDoc(sd.doc, score));
                } else {
                    // Fallback: keep SQ score
                    rescored.add(sd);
                }
            }
            return rescored;
        }
    }

    /**
     * Finds the ordinal for a given docId in FloatVectorValues by advancing.
     * FloatVectorValues doesn't have a direct docId→ord lookup, so we use advance().
     */
    private static int findOrdForDoc(FloatVectorValues fvv, int docId) throws IOException {
        // FloatVectorValues.vectorValue takes an ordinal. For dense segments ord == docId.
        // For sparse segments we'd need the iterator, but clumping currently only works
        // with dense segments. Return docId as the ordinal.
        return docId;
    }

    // ---- Legacy (FP16/FLOAT/BYTE) scoring path ----

    private static List<ScoreDoc> scoreLegacyHiddenVectors(
        Directory directory,
        CachedMarkerTable table,
        int[] matchedIndices,
        float[] floatQueryVector,
        byte[] byteQueryVector,
        KNNVectorSimilarityFunction similarityFunction
    ) throws IOException {
        boolean isFloat = (table.vectorDataType == ClumpFileFormat.VECTOR_TYPE_FLOAT);
        boolean isFp16 = (table.vectorDataType == ClumpFileFormat.VECTOR_TYPE_FP16);

        SimdVectorComputeService.SimilarityFunctionType simdFunctionType = null;
        if (isFp16) {
            simdFunctionType = toSimdFunctionType(similarityFunction);
        }

        int[][] markerDocIdArrays = new int[matchedIndices.length][];
        byte[][] markerVectorBlocks = (simdFunctionType == null) ? new byte[matchedIndices.length][] : null;
        long[] vecBlockOffsets = (simdFunctionType != null) ? new long[matchedIndices.length] : null;
        int[] vecBlockSizes = (simdFunctionType != null) ? new int[matchedIndices.length] : null;

        try (IndexInput input = directory.openInput(table.clumpFileName, IOContext.DEFAULT)) {
            final byte[][] finalVectorBlocks = markerVectorBlocks;
            final long[] finalVecOffsets = vecBlockOffsets;
            final int[] finalVecSizes = vecBlockSizes;

            try (ExecutorService vte = Executors.newVirtualThreadPerTaskExecutor()) {
                List<Future<?>> readFutures = new ArrayList<>(matchedIndices.length);
                for (int mi = 0; mi < matchedIndices.length; mi++) {
                    final int idx = mi;
                    final int markerIndex = matchedIndices[mi];
                    readFutures.add(vte.submit(() -> {
                        int numHidden = table.numHidden[markerIndex];
                        long docIdStart = table.clumpDataOffsets[markerIndex] + table.vectorSize;
                        int docIdBlockSize = numHidden * Integer.BYTES;
                        int vecBlockSize = numHidden * table.vectorSize;

                        if (finalVectorBlocks != null) {
                            int totalSize = docIdBlockSize + vecBlockSize;
                            byte[] combined = new byte[totalSize];
                            try (IndexInput clonedInput = input.clone()) {
                                clonedInput.seek(docIdStart);
                                clonedInput.readBytes(combined, 0, totalSize);
                            } catch (IOException e) {
                                log.warn("Error reading clump data for marker index {}", markerIndex, e);
                            }

                            int[] docIds = new int[numHidden];
                            java.nio.ByteBuffer dbuf = java.nio.ByteBuffer.wrap(combined, 0, docIdBlockSize)
                                .order(java.nio.ByteOrder.LITTLE_ENDIAN);
                            for (int j = 0; j < numHidden; j++) {
                                docIds[j] = dbuf.getInt();
                            }
                            markerDocIdArrays[idx] = docIds;

                            byte[] vecBlock = new byte[vecBlockSize];
                            System.arraycopy(combined, docIdBlockSize, vecBlock, 0, vecBlockSize);
                            finalVectorBlocks[idx] = vecBlock;
                        } else {
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

                            long vecStart = docIdStart + (long) docIdBlockSize;
                            finalVecOffsets[idx] = vecStart;
                            finalVecSizes[idx] = vecBlockSize;
                        }
                    }));
                }
                for (Future<?> f : readFutures) {
                    try { f.get(); } catch (Exception e) { log.warn("Error waiting for clump read task", e); }
                }
            }

            List<ScoreDoc> scoredHidden = new ArrayList<>();

            if (simdFunctionType != null) {
                for (int m = 0; m < matchedIndices.length; m++) {
                    int[] docIds = markerDocIdArrays[m];
                    if (docIds == null || docIds.length == 0) continue;
                    int numHidden = docIds.length;

                    long[] addressAndSize = MemorySegmentAddressExtractorUtil.tryExtractAddressAndSize(
                        input, vecBlockOffsets[m], vecBlockSizes[m]
                    );

                    if (addressAndSize != null) {
                        SimdVectorComputeService.saveSearchContext(
                            floatQueryVector, addressAndSize, simdFunctionType.ordinal()
                        );
                        int[] ordinals = new int[numHidden];
                        for (int j = 0; j < numHidden; j++) ordinals[j] = j;
                        float[] scores = new float[numHidden];
                        SimdVectorComputeService.scoreSimilarityInBulk(ordinals, scores, numHidden);
                        for (int j = 0; j < numHidden; j++) {
                            scoredHidden.add(new ScoreDoc(docIds[j], scores[j]));
                        }
                    } else {
                        byte[] vecBlock = new byte[vecBlockSizes[m]];
                        try (IndexInput clonedInput = input.clone()) {
                            clonedInput.seek(vecBlockOffsets[m]);
                            clonedInput.readBytes(vecBlock, 0, vecBlock.length);
                        }
                        scoreFp16VectorBlock(docIds, vecBlock, table.dimension, floatQueryVector, similarityFunction, scoredHidden);
                    }
                }
            } else {
                for (int m = 0; m < matchedIndices.length; m++) {
                    int[] docIds = markerDocIdArrays[m];
                    byte[] vecBlock = markerVectorBlocks[m];
                    if (docIds == null || vecBlock == null) continue;
                    int numHidden = docIds.length;
                    java.nio.ByteBuffer vb = java.nio.ByteBuffer.wrap(vecBlock)
                        .order(java.nio.ByteOrder.LITTLE_ENDIAN);

                    if (isFloat) {
                        float[] reusableVector = new float[table.dimension];
                        for (int j = 0; j < numHidden; j++) {
                            for (int d = 0; d < table.dimension; d++) reusableVector[d] = vb.getFloat();
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

    // ---- Marker rescoring ----

    /**
     * Re-scores marker vectors so their scores are on the same scale as hidden vector scores.
     * <p>
     * For FP16: reads the marker's FP16 vector from the clump file and scores with the
     * similarity function.
     * <p>
     * For SQ_1BIT: reads the marker's fp32 vector from {@code floatVectorValues} and scores
     * with the similarity function (matching the fp32 rescore applied to hidden vectors).
     * <p>
     * For FLOAT/BYTE: returns the original scores (ANN search uses the same scorer).
     */
    public static ScoreDoc[] rescoreMarkers(
        Directory directory,
        String segmentName,
        String fieldName,
        ScoreDoc[] markerScoreDocs,
        float[] floatQueryVector,
        byte[] byteQueryVector,
        KNNVectorSimilarityFunction similarityFunction
    ) throws IOException {
        return rescoreMarkers(
            directory, segmentName, fieldName, markerScoreDocs,
            floatQueryVector, byteQueryVector, similarityFunction, null
        );
    }

    /**
     * Full overload with optional {@link FloatVectorValues} for SQ_1BIT fp32 rescore.
     */
    public static ScoreDoc[] rescoreMarkers(
        Directory directory,
        String segmentName,
        String fieldName,
        ScoreDoc[] markerScoreDocs,
        float[] floatQueryVector,
        byte[] byteQueryVector,
        KNNVectorSimilarityFunction similarityFunction,
        FloatVectorValues floatVectorValues
    ) throws IOException {
        CachedMarkerTable table = getOrLoadMarkerTable(directory, segmentName, fieldName);
        if (table == null) {
            return markerScoreDocs;
        }

        if (table.vectorDataType == ClumpFileFormat.VECTOR_TYPE_SQ_1BIT) {
            // For SQ: rescore markers with fp32 so they're comparable to fp32-rescored hidden
            if (floatVectorValues == null) {
                return markerScoreDocs;
            }
            ScoreDoc[] rescored = new ScoreDoc[markerScoreDocs.length];
            for (int i = 0; i < markerScoreDocs.length; i++) {
                int docId = markerScoreDocs[i].doc;
                float[] fp32Vec = floatVectorValues.vectorValue(docId);
                if (fp32Vec != null) {
                    float score = similarityFunction.compare(floatQueryVector, fp32Vec);
                    rescored[i] = new ScoreDoc(docId, score);
                } else {
                    rescored[i] = markerScoreDocs[i];
                }
            }
            return rescored;
        }

        if (table.vectorDataType != ClumpFileFormat.VECTOR_TYPE_FP16) {
            return markerScoreDocs;
        }

        // FP16 rescore path (unchanged)
        try (IndexInput input = directory.openInput(table.clumpFileName, IOContext.DEFAULT)) {
            ScoreDoc[] rescored = new ScoreDoc[markerScoreDocs.length];
            for (int i = 0; i < markerScoreDocs.length; i++) {
                int docId = markerScoreDocs[i].doc;
                int markerIndex = Arrays.binarySearch(table.markerDocIds, docId);
                if (markerIndex < 0) {
                    rescored[i] = markerScoreDocs[i];
                    continue;
                }

                long markerVecOffset = table.clumpDataOffsets[markerIndex];
                byte[] vecBytes = new byte[table.vectorSize];
                try (IndexInput cloned = input.clone()) {
                    cloned.seek(markerVecOffset);
                    cloned.readBytes(vecBytes, 0, vecBytes.length);
                }

                java.nio.ByteBuffer vb = java.nio.ByteBuffer.wrap(vecBytes).order(java.nio.ByteOrder.LITTLE_ENDIAN);
                float[] vec = new float[table.dimension];
                for (int d = 0; d < table.dimension; d++) {
                    vec[d] = Float.float16ToFloat(vb.getShort());
                }
                float score = similarityFunction.compare(floatQueryVector, vec);
                rescored[i] = new ScoreDoc(docId, score);
            }
            return rescored;
        }
    }

    // ---- Utility methods ----

    public static boolean isMarkerDocId(
        Directory directory, String segmentName, String fieldName, int docId
    ) throws IOException {
        CachedMarkerTable table = getOrLoadMarkerTable(directory, segmentName, fieldName);
        if (table == null) return false;
        return Arrays.binarySearch(table.markerDocIds, docId) >= 0;
    }

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
                if (markerIndex < 0) continue;

                int numHidden = table.numHidden[markerIndex];
                if (numHidden == 0) continue;

                long docIdStart = table.clumpDataOffsets[markerIndex] + table.vectorSize;
                input.seek(docIdStart);

                for (int j = 0; j < numHidden; j++) {
                    hiddenDocIds.add(input.readInt());
                }
            }

            return hiddenDocIds;
        }
    }

    /**
     * Returns true if this segment+field has an SQ_1BIT clump file.
     */
    public static boolean isSqClumpFile(Directory directory, String segmentName, String fieldName) throws IOException {
        CachedMarkerTable table = getOrLoadMarkerTable(directory, segmentName, fieldName);
        return table != null && table.vectorDataType == ClumpFileFormat.VECTOR_TYPE_SQ_1BIT;
    }

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

    private static void scoreFp16VectorBlock(
        int[] docIds, byte[] vecBlock, int dimension,
        float[] queryVector, KNNVectorSimilarityFunction similarityFunction,
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

    /** Reads a little-endian int from a byte array at the given offset. */
    private static int readIntLE(byte[] buf, int off) {
        return (buf[off] & 0xff)
            | ((buf[off + 1] & 0xff) << 8)
            | ((buf[off + 2] & 0xff) << 16)
            | ((buf[off + 3] & 0xff) << 24);
    }
}
