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
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.KNNQueryResult;
import org.opensearch.knn.index.store.IndexInputWithBuffer;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.jni.JNIService;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.VectorDataType;

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
        // Also track the original ordinal (position in the full vector sequence) for each marker
        // so we can build a FilteredQuantizedByteVectorValues that remaps marker ordinals correctly.
        final List<Integer> markerDocIds = new ArrayList<>();
        final List<Integer> markerOrdinals = new ArrayList<>();
        int maxDocId = 0;

        for (int i = 0; i < allDocIds.size(); i++) {
            int docId = allDocIds.get(i);
            if (docId > maxDocId) {
                maxDocId = docId;
            }
            if (i % clumpingFactor == 0) {
                markerDocIds.add(docId);
                markerOrdinals.add(i);
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
        buildMarkerIndexToTempFile(indexInfo, markerDocIds, markerOrdinals, tempEngineFileName);

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
     * If the parent {@link BuildIndexParams} carries a {@link QuantizedByteVectorValues} (SQ path),
     * a {@link FilteredQuantizedByteVectorValues} is created that remaps compact marker ordinals
     * (0, 1, 2, ...) back to the original full-segment ordinals, so that
     * {@link org.opensearch.knn.index.codec.nativeindex.MemOptimizedScalarQuantizedIndexBuildStrategy}
     * reads the correct quantized codes and correction factors for each marker.
     */
    private void buildMarkerIndexToTempFile(
        BuildIndexParams indexInfo,
        List<Integer> markerDocIds,
        List<Integer> markerOrdinals,
        String tempEngineFileName
    ) throws IOException {
        try (IndexOutput tempOutput = indexInfo.getSegmentWriteState().directory.createOutput(
            tempEngineFileName, indexInfo.getSegmentWriteState().context
        )) {
            IndexOutputWithBuffer tempOutputWithBuffer = new IndexOutputWithBuffer(tempOutput);

            // If the parent params carry QuantizedByteVectorValues (SQ/1-bit path), wrap them
            // so that marker ordinal i maps to the original full-segment ordinal markerOrdinals[i].
            final QuantizedByteVectorValues filteredQBVV = indexInfo.getQuantizedByteVectorValues() != null
                ? new FilteredQuantizedByteVectorValues(indexInfo.getQuantizedByteVectorValues(), markerOrdinals)
                : null;

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
                .quantizedByteVectorValues(filteredQBVV)
                .build();

            delegate.buildAndWriteIndex(markerIndexParams);
        }
    }

    /**
     * Assigns each hidden vector to its nearest marker.
     * <p>
     * For the SQ/1-bit path ({@code indexInfo.getQuantizedByteVectorValues() != null}), the temp
     * marker index is written with {@code IO_FLAG_SKIP_STORAGE} and contains a {@code "null"}
     * storage sentinel that the C++ {@code read_index_binary} cannot parse. In that case we use a
     * two-level SQ-then-fp32 rescore that mirrors the query-time strategy: quantize the hidden
     * vector, SQ-score against all marker codes, take the top-16, then pick the best by fp32
     * distance. Hidden entries are spilled as {@link SqVectorEntry}.
     * <p>
     * For all other paths the just-built marker index is loaded via JNI and queried with k=1 per
     * hidden vector (parallelised in batches).
     */
    private AssignmentResult assignHiddenToMarkersViaIndex(
        BuildIndexParams indexInfo,
        List<Integer> markerDocIds,
        int maxDocId,
        IndexOutput tempHiddenOutput,
        String tempEngineFileName,
        KNNEngine knnEngine
    ) throws IOException {
        if (indexInfo.getQuantizedByteVectorValues() != null) {
            return assignHiddenSqTwoLevel(indexInfo, markerDocIds, maxDocId, tempHiddenOutput);
        }
        return assignHiddenNonSq(indexInfo, markerDocIds, maxDocId, tempHiddenOutput, tempEngineFileName, knnEngine);
    }

    /**
     * SQ_1BIT assignment path.
     *
     * <p>Harvests markers directly from {@link QuantizedByteVectorValues} — both the packed 1-bit
     * codes plus four correction factors (as {@link SqVectorEntry}) and their fp32 counterparts
     * (for the rescore step and the later query-time fp32 rescore). For each hidden vector:
     * <ol>
     *   <li>Quantize the hidden fp32 into 4 bit planes (via
     *       {@link org.apache.lucene.util.quantization.OptimizedScalarQuantizer}) — same primitives
     *       as {@code KNN1040ScalarQuantizedVectorScorer} uses for query vectors.</li>
     *   <li>Score it against every marker SQ code with the ADC L2 formula via {@link SqAdcScorer},
     *       keep the top {@value #SQ_RESCORE_SHORTLIST} candidates.</li>
     *   <li>Rescore those {@value #SQ_RESCORE_SHORTLIST} with fp32 L2 and pick the best.</li>
     * </ol>
     * The hidden vector's own SQ code is harvested from {@code QuantizedByteVectorValues} by
     * parallel-iterating its ordinal with the fp32 iterator — guaranteeing the spilled SQ bytes
     * are byte-identical to what Lucene wrote to the {@code .veb} file.
     */
    private AssignmentResult assignHiddenSqTwoLevel(
        BuildIndexParams indexInfo,
        List<Integer> markerDocIds,
        int maxDocId,
        IndexOutput tempHiddenOutput
    ) throws IOException {
        final QuantizedByteVectorValues qbvv = indexInfo.getQuantizedByteVectorValues();
        final float[] centroid = qbvv.getCentroid();
        final float centroidDp = qbvv.getCentroidDP();
        final int dimension = qbvv.dimension();
        // vectorValue(0) gives the true quantized-code length (handles dimension padding).
        final int quantizedVecBytes = qbvv.vectorValue(0).length;

        final Set<Integer> markerSet = new HashSet<>(markerDocIds);

        // Pass 1: harvest markers. We need both fp32 (for rescore) and SQ (for SQ scoring)
        // representations, in markerDocIds insertion order. The fp32 iterator doubles as the
        // source of truth for segmentOrdinal — it must be traversed completely to keep any
        // internal counters (e.g. FieldWriterIteratorValues) in sync.
        final float[][] markerFloatVectors = new float[markerDocIds.size()][];
        final SqVectorEntry[] markerSqEntries = new SqVectorEntry[markerDocIds.size()];
        int markerIdx = 0;

        final KNNVectorValues<?> markerScan = indexInfo.getKnnVectorValuesSupplier().get();
        markerScan.nextDoc();
        int segmentOrdinal = 0;
        while (markerScan.docId() != NO_MORE_DOCS) {
            // Always call getVector() to advance any sequential counter inside the iterator.
            Object rawVec = markerScan.getVector();
            if (markerSet.contains(markerScan.docId())) {
                markerFloatVectors[markerIdx] = ((float[]) rawVec).clone();
                markerSqEntries[markerIdx] = readSqEntry(qbvv, segmentOrdinal);
                markerIdx++;
            }
            segmentOrdinal++;
            markerScan.nextDoc();
        }

        // Build marker-vectors list for the writer (SQ entries go into the clump file).
        final List<Object> markerVectors = new ArrayList<>(markerSqEntries.length);
        for (SqVectorEntry e : markerSqEntries) {
            markerVectors.add(e);
        }

        // Build the assign map.
        int[] assignMap = new int[maxDocId + 1];
        Arrays.fill(assignMap, UNASSIGNED);
        for (int mDocId : markerDocIds) {
            assignMap[mDocId] = mDocId;
        }

        // Pass 2: assign hidden vectors to markers via the SQ-then-fp32 two-level rescore,
        // spilling each hidden vector's SQ entry to tempHiddenOutput.
        final KNNVectorValues<?> hiddenScan = indexInfo.getKnnVectorValuesSupplier().get();
        hiddenScan.nextDoc();
        int hiddenSegmentOrdinal = 0;
        while (hiddenScan.docId() != NO_MORE_DOCS) {
            int docId = hiddenScan.docId();
            float[] hiddenFp32 = ((float[]) hiddenScan.getVector()).clone();

            if (markerSet.contains(docId) == false) {
                SqVectorEntry hiddenSq = readSqEntry(qbvv, hiddenSegmentOrdinal);
                int bestMarkerIdx = findNearestMarkerSqTwoLevel(
                    hiddenFp32, markerSqEntries, markerFloatVectors,
                    centroid, centroidDp, dimension, qbvv
                );
                assignMap[docId] = markerDocIds.get(bestMarkerIdx);
                ClumpFileWriter.writeHiddenEntryToTemp(
                    tempHiddenOutput, docId, hiddenSq, ClumpFileFormat.VECTOR_TYPE_SQ_1BIT
                );
            }
            hiddenSegmentOrdinal++;
            hiddenScan.nextDoc();
        }

        return new AssignmentResult(
            markerVectors, assignMap, dimension,
            ClumpFileFormat.VECTOR_TYPE_SQ_1BIT,
            quantizedVecBytes, centroid, centroidDp
        );
    }

    /**
     * Finds the best marker for a hidden fp32 vector using a two-level SQ→fp32 rescore,
     * mirroring the query-time strategy.
     */
    private int findNearestMarkerSqTwoLevel(
        float[] hiddenFp32,
        SqVectorEntry[] markerSqEntries,
        float[][] markerFloatVectors,
        float[] centroid,
        float centroidDp,
        int dimension,
        QuantizedByteVectorValues qbvv
    ) throws IOException {
        // Quantize the hidden vector as if it were a query.
        final float[] queryCopy = hiddenFp32.clone(); // scalarQuantize mutates input
        final SqAdcScorer.QuantizedQuery q = SqAdcScorer.quantizeQuery(qbvv, queryCopy);

        // SQ score all markers. Use L2 here — it matches the semantics of "nearest marker".
        final int m = markerSqEntries.length;
        final int shortlist = Math.min(SQ_RESCORE_SHORTLIST, m);
        // Track top-K smallest scores. Simple O(m * log K) insertion via a bounded scan.
        final float[] bestScores = new float[shortlist];
        final int[] bestIdx = new int[shortlist];
        Arrays.fill(bestScores, Float.POSITIVE_INFINITY);
        Arrays.fill(bestIdx, -1);

        for (int i = 0; i < m; i++) {
            float score = SqAdcScorer.scoreL2(q, markerSqEntries[i].code, markerSqEntries[i], dimension);
            // Insert into top-K if smaller than the current worst kept.
            if (score < bestScores[shortlist - 1]) {
                int pos = shortlist - 1;
                while (pos > 0 && bestScores[pos - 1] > score) {
                    bestScores[pos] = bestScores[pos - 1];
                    bestIdx[pos] = bestIdx[pos - 1];
                    pos--;
                }
                bestScores[pos] = score;
                bestIdx[pos] = i;
            }
        }

        // Rescore shortlist with fp32 L2 and pick the best.
        int winner = bestIdx[0];
        float winnerDist = fp32L2(hiddenFp32, markerFloatVectors[winner]);
        for (int i = 1; i < shortlist; i++) {
            if (bestIdx[i] < 0) break;
            float d = fp32L2(hiddenFp32, markerFloatVectors[bestIdx[i]]);
            if (d < winnerDist) {
                winnerDist = d;
                winner = bestIdx[i];
            }
        }
        return winner;
    }

    /**
     * Reads one SQ entry (code + corrections) from the segment's {@link QuantizedByteVectorValues}
     * by segment ordinal.
     */
    private static SqVectorEntry readSqEntry(QuantizedByteVectorValues qbvv, int segmentOrdinal) throws IOException {
        byte[] code = qbvv.vectorValue(segmentOrdinal).clone();
        OptimizedScalarQuantizer.QuantizationResult q = qbvv.getCorrectiveTerms(segmentOrdinal);
        return new SqVectorEntry(
            code,
            q.lowerInterval(),
            q.upperInterval(),
            q.additionalCorrection(),
            q.quantizedComponentSum()
        );
    }

    private static float fp32L2(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            float d = a[i] - b[i];
            sum += d * d;
        }
        return sum;
    }

    /**
     * Non-SQ assignment path. Unchanged from the pre-SQ-rewrite behavior.
     */
    private AssignmentResult assignHiddenNonSq(
        BuildIndexParams indexInfo,
        List<Integer> markerDocIds,
        int maxDocId,
        IndexOutput tempHiddenOutput,
        String tempEngineFileName,
        KNNEngine knnEngine
    ) throws IOException {
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
        byte vectorDataType = ClumpFileFormat.VECTOR_TYPE_FLOAT;

        while (markerScan.docId() != NO_MORE_DOCS) {
            // Always call getVector() to advance the sequential counter in FieldWriterIteratorValues.
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

        long indexPointer = loadTempIndex(indexInfo, tempEngineFileName, knnEngine);
        try {
            assignHiddenViaJni(
                indexInfo, markerSet, markerVectorArray, markerDocIds,
                markerDocIdToIndex, vectorDataType, assignMap, tempHiddenOutput,
                indexPointer, knnEngine
            );
        } finally {
            freeTempIndex(indexPointer, knnEngine, false);
        }

        return new AssignmentResult(markerVectors, assignMap, dimension, vectorDataType, 0, null, 0.0f);
    }

    /**
     * Shortlist size for the SQ → fp32 rescore. Matches the query-time strategy: SQ picks the
     * top 16 candidates, fp32 rescores to choose the single winner.
     */
    private static final int SQ_RESCORE_SHORTLIST = 512;

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
        KNNEngine knnEngine
    ) throws IOException {
        final KNNVectorValues<?> hiddenScan = indexInfo.getKnnVectorValuesSupplier().get();
        hiddenScan.nextDoc();

        List<int[]> batchDocIds = new ArrayList<>(ASSIGNMENT_BATCH_SIZE);
        List<Object> batchVectors = new ArrayList<>(ASSIGNMENT_BATCH_SIZE);

        while (hiddenScan.docId() != NO_MORE_DOCS) {
            int docId = hiddenScan.docId();
            // Always call getVector() to advance the sequential counter in FieldWriterIteratorValues.
            Object vec = cloneVectorRaw(hiddenScan.getVector());
            if (markerSet.contains(docId) == false) {
                batchDocIds.add(new int[] { docId });
                batchVectors.add(vec);

                if (batchDocIds.size() >= ASSIGNMENT_BATCH_SIZE) {
                    processBatch(
                        batchDocIds, batchVectors, indexPointer, knnEngine,
                        markerDocIdToIndex, markerVectorArray, markerDocIds,
                        vectorDataType, assignMap, tempHiddenOutput
                    );
                    batchDocIds.clear();
                    batchVectors.clear();
                }
            }
            hiddenScan.nextDoc();
        }

        if (batchDocIds.isEmpty() == false) {
            processBatch(
                batchDocIds, batchVectors, indexPointer, knnEngine,
                markerDocIdToIndex, markerVectorArray, markerDocIds,
                vectorDataType, assignMap, tempHiddenOutput
            );
        }
    }

    /**
     * Processes a batch of hidden vectors: searches for nearest markers in parallel,
     * then writes assignments and spills vector data sequentially.
     */
    private void processBatch(
        List<int[]> batchDocIds,
        List<Object> batchVectors,
        long indexPointer,
        KNNEngine knnEngine,
        Map<Integer, Integer> markerDocIdToIndex,
        Object[] markerVectorArray,
        List<Integer> markerDocIds,
        byte vectorDataType,
        int[] assignMap,
        IndexOutput tempHiddenOutput
    ) throws IOException {
        int batchSize = batchDocIds.size();
        int[] nearestMarkers = new int[batchSize];

        // Parallel search: each hidden vector finds its nearest marker concurrently
        IntStream.range(0, batchSize).parallel().forEach(i -> {
            nearestMarkers[i] = findNearestMarkerDocIdViaIndex(
                batchVectors.get(i), indexPointer, knnEngine,
                markerDocIdToIndex, markerVectorArray, markerDocIds, vectorDataType
            );
        });

        // Sequential write: assign map updates and spill file writes
        for (int i = 0; i < batchSize; i++) {
            int docId = batchDocIds.get(i)[0];
            assignMap[docId] = nearestMarkers[i];
            ClumpFileWriter.writeHiddenEntryToTemp(
                tempHiddenOutput, docId, batchVectors.get(i), vectorDataType
            );
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
        List<Integer> markerDocIds,
        byte vectorDataType
    ) {
        float[] queryVector;
        if (vectorDataType == ClumpFileFormat.VECTOR_TYPE_FLOAT || vectorDataType == ClumpFileFormat.VECTOR_TYPE_FP16) {
            queryVector = (hiddenVector instanceof float[])
                ? (float[]) hiddenVector
                : toFloatArray((byte[]) hiddenVector);
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
     * Loads the temp marker index file via JNI.
     * For the SQ/1-bit path, the marker index is written as a binary Faiss index
     * (IndexBinaryIDMap), so we must pass VECTOR_DATA_TYPE=binary to route to
     * {@code FaissService.loadBinaryIndexWithStream} instead of the float reader.
     */
    private long loadTempIndex(BuildIndexParams indexInfo, String tempEngineFileName, KNNEngine knnEngine) throws IOException {
        final Map<String, Object> loadParameters;
        if (indexInfo.getQuantizedByteVectorValues() != null) {
            // SQ path: the marker index was written as a binary Faiss index.
            // Override the data type so JNIService routes to the binary load path.
            loadParameters = new HashMap<>(indexInfo.getIndexParameters());
            loadParameters.put(KNNConstants.VECTOR_DATA_TYPE_FIELD, VectorDataType.BINARY.getValue());
        } else {
            loadParameters = indexInfo.getIndexParameters();
        }

        try (
            IndexInput tempInput = indexInfo.getSegmentWriteState().directory.openInput(
                tempEngineFileName, IOContext.DEFAULT
            )
        ) {
            IndexInputWithBuffer readStream = new IndexInputWithBuffer(tempInput);
            return AccessController.doPrivileged((PrivilegedAction<Long>) () ->
                JNIService.loadIndex(readStream, loadParameters, knnEngine)
            );
        }
    }

    /**
     * Frees a loaded native index. For the SQ/binary path the index is an
     * {@code IndexBinaryIDMap}, so {@code isBinaryIndex=true} must be passed.
     */
    private void freeTempIndex(long indexPointer, KNNEngine knnEngine, boolean isBinaryIndex) {
        try {
            AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
                JNIService.free(indexPointer, knnEngine, isBinaryIndex);
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
