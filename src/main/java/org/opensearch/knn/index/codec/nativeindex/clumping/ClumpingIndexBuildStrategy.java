/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.clumping;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.lucene104.QuantizedByteVectorValues;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategy;
import org.opensearch.knn.index.codec.nativeindex.MemOptimizedNativeIndexBuildStrategy;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.jni.JNIService;

import java.io.IOException;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.IntStream;

import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.initializeVectorValues;

/**
 * A build strategy that wraps an underlying {@link NativeIndexBuildStrategy} and adds clumping.
 * <p>
 * Every nth vector (starting from 0) is designated a "marker" vector and is inserted into the
 * native index via the delegate strategy. The remaining "hidden" vectors are not inserted into
 * the native index. Instead, after the marker-only index is built, each hidden vector is assigned
 * to its nearest marker by searching the native index with k=1 via JNI. The mapping — including
 * all vector data — is written to a .clump sidecar file so that expansion at query time can read
 * vectors sequentially without random access into Lucene's vector storage.
 * <p>
 * Memory optimization: marker-to-hidden assignments are tracked in a flat int array indexed by
 * doc ID, where {@code assignMap[docId]} holds the marker doc ID for that vector. Marker vectors
 * point to themselves. After the assignment pass completes, this array is flushed to a temporary
 * file ({@code .clumpassign}) and freed from heap. The {@link ClumpFileWriter} then reads the
 * assign file from disk when assembling the final .clump file, avoiding the need to hold
 * per-hidden-vector metadata (doc IDs, offsets, marker indices) in heap simultaneously.
 * <p>
 * Hidden vector data is spilled to a separate temp file ({@code .clumptmp}) in encounter order.
 * Since each entry is fixed-size ({@code 4 + dimension * elementSize} bytes), the byte offset
 * of the Nth hidden entry can be computed as {@code N * entrySize} without storing offsets.
 */
@Log4j2
public class ClumpingIndexBuildStrategy implements NativeIndexBuildStrategy {

    private final NativeIndexBuildStrategy delegate;
    private final int clumpingFactor;

    /** Value written to the assign map for doc IDs that have no assignment (sparse gaps). */
    static final int UNASSIGNED = -1;

    /** Number of hidden vectors to batch before dispatching parallel searches. */
    static final int ASSIGNMENT_BATCH_SIZE = 256;

    public ClumpingIndexBuildStrategy(NativeIndexBuildStrategy delegate, int clumpingFactor) {
        this.delegate = delegate;
        this.clumpingFactor = clumpingFactor;
    }

    @Override
    public void buildAndWriteIndex(BuildIndexParams indexInfo) throws IOException {
        // Pass 1: collect only doc IDs to classify markers vs hidden (no vector data held)
        final KNNVectorValues<?> idScan = indexInfo.getKnnVectorValuesSupplier().get();
        initializeVectorValues(idScan);

        final List<Integer> allDocIds = new ArrayList<>();
        while (idScan.docId() != NO_MORE_DOCS) {
            allDocIds.add(idScan.docId());
            idScan.nextDoc();
        }

        if (allDocIds.isEmpty()) {
            return;
        }

        // Separate markers by insertion order; track max doc ID for the assign array size.
        final List<Integer> markerDocIds = new ArrayList<>();
        int maxDocId = 0;

        for (int i = 0; i < allDocIds.size(); i++) {
            int docId = allDocIds.get(i);
            if (docId > maxDocId) {
                maxDocId = docId;
            }
            if (i % clumpingFactor == 0) {
                markerDocIds.add(docId);
            }
        }

        int totalHidden = allDocIds.size() - markerDocIds.size();
        allDocIds.clear();

        log.debug(
            "Clumping: {} markers, {} hidden (factor={})",
            markerDocIds.size(),
            totalHidden,
            clumpingFactor
        );

        if (totalHidden == 0) {
            delegate.buildAndWriteIndex(indexInfo);
            return;
        }

        final KNNEngine knnEngine = indexInfo.getKnnEngine();
        final String segmentName = indexInfo.getSegmentWriteState().segmentInfo.name;
        final String fieldName = indexInfo.getField();

        // Build marker index into a temp file so we can load it for searching
        String tempEngineFileName = segmentName + "_" + fieldName + ".clumpidx";
        buildMarkerIndexToTempFile(indexInfo, markerDocIds, tempEngineFileName);

        // Assign hidden vectors to markers, spilling vector data to .clumptmp
        String assignFileName = buildAssignFileName(segmentName, fieldName);
        String tempHiddenFileName = ClumpFileWriter.buildTempFileName(segmentName, fieldName);
        List<Object> markerVectors;
        int dimension;
        byte vectorDataType;
        int quantizedVecBytes;
        float[] centroid;
        float centroidDp;

        try (
            IndexOutput tempHiddenOutput = indexInfo.getSegmentWriteState().directory.createOutput(
                tempHiddenFileName, indexInfo.getSegmentWriteState().context
            )
        ) {
            AssignmentResult result = assignHiddenToMarkersViaIndex(
                indexInfo, markerDocIds, maxDocId,
                tempHiddenOutput, tempEngineFileName, knnEngine
            );
            dimension = result.dimension;
            vectorDataType = result.vectorDataType;
            markerVectors = result.markerVectors;
            quantizedVecBytes = result.quantizedVecBytes;
            centroid = result.centroid;
            centroidDp = result.centroidDp;

            // Flush the assign map to disk and free from heap
            writeAssignFile(indexInfo, assignFileName, result.assignMap);
        }

        // Copy the temp engine file to the real output, then clean it up
        copyTempToRealOutput(indexInfo, tempEngineFileName);
        deleteTempFile(indexInfo, tempEngineFileName);

        // Write the final .clump file using the assign file and hidden spill file
        try (
            IndexInput assignInput = indexInfo.getSegmentWriteState().directory.openInput(
                assignFileName, indexInfo.getSegmentWriteState().context
            );
            IndexInput tempHiddenInput = indexInfo.getSegmentWriteState().directory.openInput(
                tempHiddenFileName, indexInfo.getSegmentWriteState().context
            )
        ) {
            if (vectorDataType == ClumpFileFormat.VECTOR_TYPE_SQ_1BIT) {
                ClumpFileWriter.writeClumpFileSq(
                    indexInfo.getSegmentWriteState(),
                    fieldName,
                    dimension,
                    quantizedVecBytes,
                    centroid,
                    centroidDp,
                    markerDocIds,
                    markerVectors,
                    assignInput,
                    tempHiddenInput,
                    totalHidden
                );
            } else {
                ClumpFileWriter.writeClumpFile(
                    indexInfo.getSegmentWriteState(),
                    fieldName,
                    dimension,
                    vectorDataType,
                    markerDocIds,
                    markerVectors,
                    assignInput,
                    tempHiddenInput,
                    totalHidden
                );
            }
        } finally {
            deleteTempFile(indexInfo, assignFileName);
            deleteTempFile(indexInfo, tempHiddenFileName);
        }
    }

    /**
     * Writes the assign map to a temp file. The file contains {@code (maxDocId + 1)} int32
     * entries, where the int at offset {@code 4 * docId} is the marker doc ID for that vector.
     * Markers point to themselves; unoccupied slots contain {@link #UNASSIGNED}.
     */
    private void writeAssignFile(BuildIndexParams indexInfo, String assignFileName, int[] assignMap) throws IOException {
        try (IndexOutput out = indexInfo.getSegmentWriteState().directory.createOutput(
            assignFileName, indexInfo.getSegmentWriteState().context
        )) {
            for (int markerDocId : assignMap) {
                out.writeInt(markerDocId);
            }
        }
    }

    /**
     * Builds the marker-only native index into a temporary file in the segment directory.
     * Always uses full fp32 vectors for the marker index, regardless of whether the parent
     * field uses SQ. For SQ fields the outer delegate is {@link MemOptimizedScalarQuantizedIndexBuildStrategy}
     * which requires {@code quantizedByteVectorValues} — so we substitute
     * {@link MemOptimizedNativeIndexBuildStrategy} for the marker index build instead.
     */
    private void buildMarkerIndexToTempFile(
        BuildIndexParams indexInfo,
        List<Integer> markerDocIds,
        String tempEngineFileName
    ) throws IOException {
        // For SQ fields the outer delegate needs quantizedByteVectorValues and writes a binary
        // Faiss index. We want a plain fp32 HNSW for the marker index, so use the fp32 strategy.
        final NativeIndexBuildStrategy markerDelegate = (indexInfo.getQuantizedByteVectorValues() != null)
            ? MemOptimizedNativeIndexBuildStrategy.getInstance()
            : delegate;

        try (IndexOutput tempOutput = indexInfo.getSegmentWriteState().directory.createOutput(
            tempEngineFileName, indexInfo.getSegmentWriteState().context
        )) {
            IndexOutputWithBuffer tempOutputWithBuffer = new IndexOutputWithBuffer(tempOutput);

            BuildIndexParams markerIndexParams = BuildIndexParams.builder()
                .field(indexInfo.getField())
                .knnEngine(indexInfo.getKnnEngine())
                .indexOutputWithBuffer(tempOutputWithBuffer)
                .vectorDataType(indexInfo.getVectorDataType())
                .indexParameters(indexInfo.getIndexParameters())
                .quantizationState(indexInfo.getQuantizationState())
                .knnVectorValuesSupplier(() -> {
                    try {
                        return new FilteredKNNVectorValues(markerDocIds, indexInfo);
                    } catch (IOException e) {
                        throw new RuntimeException("Failed to create filtered vector values", e);
                    }
                })
                .totalLiveDocs(markerDocIds.size())
                .segmentWriteState(indexInfo.getSegmentWriteState())
                .isFlush(indexInfo.isFlush())
                .quantizedByteVectorValues(null)
                .build();

            markerDelegate.buildAndWriteIndex(markerIndexParams);
        }
    }

    /**
     * Assigns each hidden vector to its nearest marker by querying the loaded fp32 HNSW index
     * with k=1 via JNI. Both SQ and non-SQ fields use this path — the marker index is always
     * built with full fp32 vectors. The only difference is how hidden vectors are spilled to the
     * temp file: SQ fields store SQ entries (code + corrections); non-SQ fields store raw vectors.
     */
    private AssignmentResult assignHiddenToMarkersViaIndex(
        BuildIndexParams indexInfo,
        List<Integer> markerDocIds,
        int maxDocId,
        IndexOutput tempHiddenOutput,
        String tempEngineFileName,
        KNNEngine knnEngine
    ) throws IOException {
        return assignHiddenNonSq(indexInfo, markerDocIds, maxDocId, tempHiddenOutput, tempEngineFileName, knnEngine);
    }

    /**
     * Assigns hidden vectors to markers by querying the loaded fp32 HNSW index with k=1,
     * batched and parallelised.
     * <p>
     * For SQ fields ({@code indexInfo.getQuantizedByteVectorValues() != null}), marker vectors
     * are collected as {@link SqVectorEntry} objects and hidden vectors are spilled as SQ entries
     * so the clump file stores compact quantized data. For non-SQ fields, raw fp32/byte vectors
     * are used throughout.
     */
    private AssignmentResult assignHiddenNonSq(
        BuildIndexParams indexInfo,
        List<Integer> markerDocIds,
        int maxDocId,
        IndexOutput tempHiddenOutput,
        String tempEngineFileName,
        KNNEngine knnEngine
    ) throws IOException {
        final boolean isSq = indexInfo.getQuantizedByteVectorValues() != null;
        final QuantizedByteVectorValues qbvv = isSq ? indexInfo.getQuantizedByteVectorValues() : null;

        // Build a reverse map from marker doc ID to marker index for translating JNI results
        Map<Integer, Integer> markerDocIdToIndex = new HashMap<>(markerDocIds.size());
        for (int i = 0; i < markerDocIds.size(); i++) {
            markerDocIdToIndex.put(markerDocIds.get(i), i);
        }

        // Read marker vectors into memory (1/N of total) for the clump file.
        // IMPORTANT: For the List-backed KNNVectorValues, getVector() uses a sequential
        // counter (index[0]++) that advances on every call regardless of docId. We must call
        // getVector() for EVERY vector in the scan — not just markers — to keep the counter in
        // sync with the docId iterator. Skipping hidden vectors without calling getVector() would
        // cause the counter to fall behind, giving marker slots the wrong vectors.
        final KNNVectorValues<?> markerScan = indexInfo.getKnnVectorValuesSupplier().get();
        // Use nextDoc() directly instead of initializeVectorValues() — the latter calls getVector()
        // which would consume index[0] in FieldWriterIteratorValues before our loop does, causing
        // every subsequent getVector() call to return the wrong vector (off-by-one) and eventually
        // an IndexOutOfBoundsException when the counter exceeds the list size.
        markerScan.nextDoc();

        Set<Integer> markerSet = new HashSet<>(markerDocIds);
        Object[] markerVectorArray = new Object[markerDocIds.size()];
        int markerIdx = 0;
        int dimension = 0;
        byte vectorDataType;
        int quantizedVecBytes = 0;
        float[] centroid = null;
        float centroidDp = 0f;

        if (isSq) {
            // SQ path: collect SqVectorEntry for each marker; also keep fp32 for JNI fallback.
            // The fp32 scan drives the sequential counter; qbvv is accessed by segment ordinal.
            quantizedVecBytes = qbvv.vectorValue(0).length;
            centroid = qbvv.getCentroid();
            centroidDp = qbvv.getCentroidDP();
            dimension = qbvv.dimension();
            vectorDataType = ClumpFileFormat.VECTOR_TYPE_SQ_1BIT;

            int segmentOrdinal = 0;
            while (markerScan.docId() != NO_MORE_DOCS) {
                // Always call getVector() to advance any sequential counter inside the iterator.
                Object rawVec = markerScan.getVector();
                if (markerSet.contains(markerScan.docId())) {
                    // Store fp32 in markerVectorArray for JNI brute-force fallback distance.
                    markerVectorArray[markerIdx] = ((float[]) rawVec).clone();
                    // Collect SQ entry for the clump file.
                    byte[] code = qbvv.vectorValue(segmentOrdinal).clone();
                    OptimizedScalarQuantizer.QuantizationResult q = qbvv.getCorrectiveTerms(segmentOrdinal);
                    markerVectorArray[markerIdx] = new SqVectorEntry(
                        code, q.lowerInterval(), q.upperInterval(),
                        q.additionalCorrection(), q.quantizedComponentSum()
                    );
                    markerIdx++;
                }
                segmentOrdinal++;
                markerScan.nextDoc();
            }
        } else {
            // Non-SQ path: collect raw fp32/byte vectors.
            vectorDataType = ClumpFileFormat.VECTOR_TYPE_FLOAT; // refined below on first marker

            while (markerScan.docId() != NO_MORE_DOCS) {
                Object vec = cloneVector(markerScan);
                if (markerSet.contains(markerScan.docId())) {
                    markerVectorArray[markerIdx] = vec;
                    if (dimension == 0) {
                        dimension = (vec instanceof float[]) ? ((float[]) vec).length : ((byte[]) vec).length;
                        vectorDataType = (vec instanceof float[])
                            ? ClumpFileFormat.VECTOR_TYPE_FP16
                            : ClumpFileFormat.VECTOR_TYPE_BYTE;
                    }
                    markerIdx++;
                }
                markerScan.nextDoc();
            }
        }

        List<Object> markerVectors = new ArrayList<>(markerDocIds.size());
        for (Object mv : markerVectorArray) {
            markerVectors.add(mv);
        }

        // Build the assign map: assignMap[docId] = markerDocId. Markers point to themselves.
        int[] assignMap = new int[maxDocId + 1];
        Arrays.fill(assignMap, UNASSIGNED);
        for (int mDocId : markerDocIds) {
            assignMap[mDocId] = mDocId;
        }

        // For JNI queries we need fp32 vectors; for SQ markers we stored SqVectorEntry above,
        // so build a parallel fp32 array for the brute-force fallback.
        final Object[] jniFallbackVectors;
        if (isSq) {
            // Re-scan fp32 vectors for the brute-force fallback only.
            jniFallbackVectors = new Object[markerDocIds.size()];
            final KNNVectorValues<?> fp32Scan = indexInfo.getKnnVectorValuesSupplier().get();
            fp32Scan.nextDoc();
            int fi = 0;
            while (fp32Scan.docId() != NO_MORE_DOCS) {
                Object vec = fp32Scan.getVector();
                if (markerSet.contains(fp32Scan.docId())) {
                    jniFallbackVectors[fi++] = ((float[]) vec).clone();
                }
                fp32Scan.nextDoc();
            }
        } else {
            jniFallbackVectors = markerVectorArray;
        }

        final byte finalVectorDataType = vectorDataType;
        final int finalQuantizedVecBytes = quantizedVecBytes;
        final QuantizedByteVectorValues finalQbvv = qbvv;

        long indexPointer = loadTempIndex(indexInfo, tempEngineFileName, knnEngine);
        try {
            assignHiddenViaJni(
                indexInfo, markerSet, jniFallbackVectors, markerDocIds,
                markerDocIdToIndex, finalVectorDataType, assignMap, tempHiddenOutput,
                indexPointer, knnEngine, finalQbvv, finalQuantizedVecBytes
            );
        } finally {
            freeTempIndex(indexPointer, knnEngine, false);
        }

        return new AssignmentResult(markerVectors, assignMap, dimension, vectorDataType,
            quantizedVecBytes, centroid, centroidDp);
    }

    /**
     * Assigns hidden vectors to markers by querying the loaded JNI index with k=1,
     * batched and parallelised.
     */
    private void assignHiddenViaJni(
        BuildIndexParams indexInfo,
        Set<Integer> markerSet,
        Object[] markerVectorArray,
        List<Integer> markerDocIds,
        Map<Integer, Integer> markerDocIdToIndex,
        byte vectorDataType,
        int[] assignMap,
        IndexOutput tempHiddenOutput,
        long indexPointer,
        KNNEngine knnEngine,
        QuantizedByteVectorValues qbvv,
        int quantizedVecBytes
    ) throws IOException {
        final KNNVectorValues<?> hiddenScan = indexInfo.getKnnVectorValuesSupplier().get();
        hiddenScan.nextDoc();

        List<int[]> batchDocIds = new ArrayList<>(ASSIGNMENT_BATCH_SIZE);
        List<Object> batchVectors = new ArrayList<>(ASSIGNMENT_BATCH_SIZE);
        // For SQ: track segment ordinals so we can read SQ entries for spilled hidden vectors.
        List<Integer> batchOrdinals = (qbvv != null) ? new ArrayList<>(ASSIGNMENT_BATCH_SIZE) : null;
        int segmentOrdinal = 0;

        while (hiddenScan.docId() != NO_MORE_DOCS) {
            int docId = hiddenScan.docId();
            // Always call getVector() to advance the sequential counter in FieldWriterIteratorValues.
            Object vec = cloneVectorRaw(hiddenScan.getVector());
            if (markerSet.contains(docId) == false) {
                batchDocIds.add(new int[] { docId });
                batchVectors.add(vec);
                if (batchOrdinals != null) {
                    batchOrdinals.add(segmentOrdinal);
                }

                if (batchDocIds.size() >= ASSIGNMENT_BATCH_SIZE) {
                    processBatch(
                        batchDocIds, batchVectors, batchOrdinals,
                        indexPointer, knnEngine,
                        markerDocIdToIndex, markerVectorArray, markerDocIds,
                        vectorDataType, assignMap, tempHiddenOutput,
                        qbvv, quantizedVecBytes
                    );
                    batchDocIds.clear();
                    batchVectors.clear();
                    if (batchOrdinals != null) batchOrdinals.clear();
                }
            }
            segmentOrdinal++;
            hiddenScan.nextDoc();
        }

        if (batchDocIds.isEmpty() == false) {
            processBatch(
                batchDocIds, batchVectors, batchOrdinals,
                indexPointer, knnEngine,
                markerDocIdToIndex, markerVectorArray, markerDocIds,
                vectorDataType, assignMap, tempHiddenOutput,
                qbvv, quantizedVecBytes
            );
        }
    }

    /**
     * Processes a batch of hidden vectors: searches for nearest markers in parallel,
     * then writes assignments and spills vector data sequentially.
     * For SQ fields, hidden vectors are spilled as {@link SqVectorEntry} objects read from
     * {@code qbvv} by segment ordinal.
     */
    private void processBatch(
        List<int[]> batchDocIds,
        List<Object> batchVectors,
        List<Integer> batchOrdinals,
        long indexPointer,
        KNNEngine knnEngine,
        Map<Integer, Integer> markerDocIdToIndex,
        Object[] markerVectorArray,
        List<Integer> markerDocIds,
        byte vectorDataType,
        int[] assignMap,
        IndexOutput tempHiddenOutput,
        QuantizedByteVectorValues qbvv,
        int quantizedVecBytes
    ) throws IOException {
        int batchSize = batchDocIds.size();
        int[] nearestMarkers = new int[batchSize];

        // Parallel search: each hidden vector finds its nearest marker concurrently
        IntStream.range(0, batchSize).parallel().forEach(i -> {
            nearestMarkers[i] = findNearestMarkerDocIdViaIndex(
                batchVectors.get(i), indexPointer, knnEngine,
                markerDocIdToIndex, markerVectorArray, markerDocIds
            );
        });

        // Sequential write: assign map updates and spill file writes
        for (int i = 0; i < batchSize; i++) {
            int docId = batchDocIds.get(i)[0];
            assignMap[docId] = nearestMarkers[i];
            if (qbvv != null) {
                // SQ path: spill the hidden vector's SQ entry (code + corrections).
                int ord = batchOrdinals.get(i);
                byte[] code = qbvv.vectorValue(ord).clone();
                OptimizedScalarQuantizer.QuantizationResult q = qbvv.getCorrectiveTerms(ord);
                SqVectorEntry entry = new SqVectorEntry(
                    code, q.lowerInterval(), q.upperInterval(),
                    q.additionalCorrection(), q.quantizedComponentSum()
                );
                ClumpFileWriter.writeHiddenEntryToTemp(
                    tempHiddenOutput, docId, entry, ClumpFileFormat.VECTOR_TYPE_SQ_1BIT
                );
            } else {
                ClumpFileWriter.writeHiddenEntryToTemp(
                    tempHiddenOutput, docId, batchVectors.get(i), vectorDataType
                );
            }
        }
    }

    /**
     * Searches the loaded native index for the nearest marker to the given hidden vector.
     * Falls back to brute-force if the JNI search returns no results.
     *
     * @return the doc ID of the nearest marker
     */
    private int findNearestMarkerDocIdViaIndex(
        Object hiddenVector,
        long indexPointer,
        KNNEngine knnEngine,
        Map<Integer, Integer> markerDocIdToIndex,
        Object[] markerVectorArray,
        List<Integer> markerDocIds
    ) {
        float[] queryVector;
        if (hiddenVector instanceof float[]) {
            queryVector = (float[]) hiddenVector;
        } else {
            queryVector = toFloatArray((byte[]) hiddenVector);
        }

        KNNQueryResult[] results = AccessController.doPrivileged((PrivilegedAction<KNNQueryResult[]>) () ->
            JNIService.queryIndex(
                indexPointer, queryVector, 1, null, knnEngine,
                null, 0, null
            )
        );

        if (results != null && results.length > 0) {
            int nearestDocId = results[0].getId();
            if (markerDocIdToIndex.containsKey(nearestDocId)) {
                return nearestDocId;
            }
            log.warn("JNI search returned doc ID {} not in marker set, falling back to brute-force", nearestDocId);
        } else {
            log.warn("JNI search returned no results, falling back to brute-force");
        }

        // Fallback: brute-force L2 distance
        int bestIdx = findNearestMarkerBruteForce(hiddenVector, markerVectorArray);
        return markerDocIds.get(bestIdx);
    }

    /**
     * Brute-force fallback for finding the nearest marker using squared L2 distance.
     */
    private int findNearestMarkerBruteForce(Object hiddenVector, Object[] markerVectors) {
        int bestIdx = 0;
        float bestDist = Float.MAX_VALUE;

        for (int i = 0; i < markerVectors.length; i++) {
            float dist = computeDistance(hiddenVector, markerVectors[i]);
            if (dist < bestDist) {
                bestDist = dist;
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    /**
     * Loads the temp marker index file via JNI. The marker index is always built with full fp32
     * vectors, so no data-type override is needed.
     */
    private long loadTempIndex(BuildIndexParams indexInfo, String tempEngineFileName, KNNEngine knnEngine) throws IOException {
        try (
            IndexInput tempInput = indexInfo.getSegmentWriteState().directory.openInput(
                tempEngineFileName, IOContext.DEFAULT
            )
        ) {
            IndexInputWithBuffer readStream = new IndexInputWithBuffer(tempInput);
            return AccessController.doPrivileged((PrivilegedAction<Long>) () ->
                JNIService.loadIndex(readStream, indexInfo.getIndexParameters(), knnEngine)
            );
        }
    }

    /**
     * Frees a loaded native index.
     */
    private void freeTempIndex(long indexPointer, KNNEngine knnEngine, boolean isBinaryIndex) {
        try {
            AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
                JNIService.free(indexPointer, knnEngine, false);
                return null;
            });
        } catch (Exception e) {
            log.warn("Failed to free temp index pointer", e);
        }
    }

    /**
     * Copies the temp engine file contents to the real engine output.
     */
    private void copyTempToRealOutput(BuildIndexParams indexInfo, String tempEngineFileName) throws IOException {
        IndexOutput realOutput = indexInfo.getIndexOutputWithBuffer().getIndexOutput();
        try (
            IndexInput tempInput = indexInfo.getSegmentWriteState().directory.openInput(
                tempEngineFileName, IOContext.DEFAULT
            )
        ) {
            long remaining = tempInput.length();
            byte[] copyBuffer = new byte[64 * 1024];
            while (remaining > 0) {
                int toRead = (int) Math.min(remaining, copyBuffer.length);
                tempInput.readBytes(copyBuffer, 0, toRead);
                realOutput.writeBytes(copyBuffer, 0, toRead);
                remaining -= toRead;
            }
        }
    }

    private void deleteTempFile(BuildIndexParams indexInfo, String fileName) {
        try {
            indexInfo.getSegmentWriteState().directory.deleteFile(fileName);
        } catch (IOException e) {
            log.warn("Failed to delete temp file {}", fileName, e);
        }
    }

    private float computeDistance(Object a, Object b) {
        if (a instanceof float[] && b instanceof float[]) {
            float[] fa = (float[]) a;
            float[] fb = (float[]) b;
            float sum = 0;
            for (int i = 0; i < fa.length; i++) {
                float diff = fa[i] - fb[i];
                sum += diff * diff;
            }
            return sum;
        } else if (a instanceof byte[] && b instanceof byte[]) {
            byte[] ba = (byte[]) a;
            byte[] bb = (byte[]) b;
            float sum = 0;
            for (int i = 0; i < ba.length; i++) {
                float diff = ba[i] - bb[i];
                sum += diff * diff;
            }
            return sum;
        }
        throw new IllegalArgumentException("Unsupported vector types: " + a.getClass() + ", " + b.getClass());
    }

    private static float[] toFloatArray(byte[] bytes) {
        float[] result = new float[bytes.length];
        for (int i = 0; i < bytes.length; i++) {
            result[i] = bytes[i];
        }
        return result;
    }

    private Object cloneVector(KNNVectorValues<?> vectorValues) throws IOException {
        Object vector = vectorValues.getVector();
        if (vector instanceof float[]) {
            return ((float[]) vector).clone();
        } else if (vector instanceof byte[]) {
            return ((byte[]) vector).clone();
        }
        throw new IllegalArgumentException("Unsupported vector type: " + vector.getClass());
    }

    /**
     * Clones a raw vector object (float[] or byte[]). Unlike {@link #cloneVector}, this
     * operates on an already-extracted vector rather than a KNNVectorValues instance.
     * Needed because {@code KNNVectorValues.getVector()} reuses its internal buffer.
     */
    private Object cloneVectorRaw(Object vector) {
        if (vector instanceof float[]) {
            return ((float[]) vector).clone();
        } else if (vector instanceof byte[]) {
            return ((byte[]) vector).clone();
        }
        throw new IllegalArgumentException("Unsupported vector type: " + vector.getClass());
    }

    static String buildAssignFileName(String segmentName, String fieldName) {
        return segmentName + "_" + fieldName + ".clumpassign";
    }

    /**
     * Result of the streaming assignment pass. Contains marker vectors (held in memory),
     * the flat assign map (docId → markerDocId), and dimension/type metadata. For the SQ_1BIT
     * path, also carries the quantizer's centroid + centroidDp + quantized-code byte length so
     * the clump writer can persist them in the file header. {@code centroid} is null and
     * {@code quantizedVecBytes} is 0 for non-SQ paths.
     *
     * <p>The assignMap is flushed to disk after this result is returned, then freed.
     */
    private static class AssignmentResult {
        final List<Object> markerVectors;
        final int[] assignMap;
        final int dimension;
        final byte vectorDataType;
        final int quantizedVecBytes;
        final float[] centroid;
        final float centroidDp;

        AssignmentResult(
            List<Object> markerVectors, int[] assignMap,
            int dimension, byte vectorDataType,
            int quantizedVecBytes, float[] centroid, float centroidDp
        ) {
            this.markerVectors = markerVectors;
            this.assignMap = assignMap;
            this.dimension = dimension;
            this.vectorDataType = vectorDataType;
            this.quantizedVecBytes = quantizedVecBytes;
            this.centroid = centroid;
            this.centroidDp = centroidDp;
        }
    }
}
