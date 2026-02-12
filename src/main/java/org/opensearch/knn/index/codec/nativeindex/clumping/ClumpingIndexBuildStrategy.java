/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.clumping;

import lombok.extern.log4j.Log4j2;
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

        // Separate markers by insertion order; track max doc ID for the assign array size
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
        final String fieldName = indexInfo.getFieldName();

        // Build marker index into a temp file so we can load it for searching
        String tempEngineFileName = segmentName + "_" + fieldName + ".clumpidx";
        buildMarkerIndexToTempFile(indexInfo, markerDocIds, tempEngineFileName);

        // Assign hidden vectors to markers, spilling vector data to .clumptmp
        String assignFileName = buildAssignFileName(segmentName, fieldName);
        String tempHiddenFileName = ClumpFileWriter.buildTempFileName(segmentName, fieldName);
        List<Object> markerVectors;
        int dimension;
        byte vectorDataType;

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
     */
    private void buildMarkerIndexToTempFile(
        BuildIndexParams indexInfo,
        List<Integer> markerDocIds,
        String tempEngineFileName
    ) throws IOException {
        try (IndexOutput tempOutput = indexInfo.getSegmentWriteState().directory.createOutput(
            tempEngineFileName, indexInfo.getSegmentWriteState().context
        )) {
            IndexOutputWithBuffer tempOutputWithBuffer = new IndexOutputWithBuffer(tempOutput);

            BuildIndexParams markerIndexParams = BuildIndexParams.builder()
                .fieldName(indexInfo.getFieldName())
                .knnEngine(indexInfo.getKnnEngine())
                .indexOutputWithBuffer(tempOutputWithBuffer)
                .vectorDataType(indexInfo.getVectorDataType())
                .parameters(indexInfo.getParameters())
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
                .build();

            delegate.buildAndWriteIndex(markerIndexParams);
        }
    }

    /**
     * Assigns each hidden vector to its nearest marker by loading the just-built marker index
     * via JNI and querying it with k=1 for each hidden vector.
     * <p>
     * Marker vectors are cloned into memory (1/N of total) for the clump file.
     * Hidden vectors are scored via the native index, then immediately spilled to
     * {@code tempHiddenOutput}. Assignments are recorded in a flat int array where
     * {@code assignMap[docId] = markerDocId}. Markers point to themselves.
     */
    private AssignmentResult assignHiddenToMarkersViaIndex(
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

        // Read marker vectors into memory (1/N of total) for the clump file
        final KNNVectorValues<?> markerScan = indexInfo.getKnnVectorValuesSupplier().get();
        initializeVectorValues(markerScan);

        Set<Integer> markerSet = new HashSet<>(markerDocIds);
        Object[] markerVectorArray = new Object[markerDocIds.size()];
        int markerIdx = 0;
        int dimension = 0;
        byte vectorDataType = ClumpFileFormat.VECTOR_TYPE_FLOAT;

        while (markerScan.docId() != NO_MORE_DOCS && markerIdx < markerDocIds.size()) {
            if (markerSet.contains(markerScan.docId())) {
                Object cloned = cloneVector(markerScan);
                markerVectorArray[markerIdx] = cloned;
                if (dimension == 0) {
                    dimension = (cloned instanceof float[]) ? ((float[]) cloned).length : ((byte[]) cloned).length;
                    vectorDataType = (cloned instanceof float[])
                        ? ClumpFileFormat.VECTOR_TYPE_FLOAT
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

        // Load the temp marker index via JNI
        long indexPointer = loadTempIndex(indexInfo, tempEngineFileName, knnEngine);

        try {
            // Stream hidden vectors: read each one, search the native index for nearest marker,
            // record assignment, spill vector to temp file
            final KNNVectorValues<?> hiddenScan = indexInfo.getKnnVectorValuesSupplier().get();
            initializeVectorValues(hiddenScan);

            final byte finalVectorDataType = vectorDataType;

            while (hiddenScan.docId() != NO_MORE_DOCS) {
                int docId = hiddenScan.docId();

                if (markerSet.contains(docId) == false) {
                    Object hiddenVector = hiddenScan.getVector();

                    // Search the native index with k=1 to find the nearest marker
                    int nearestMarkerDocId = findNearestMarkerDocIdViaIndex(
                        hiddenVector, indexPointer, knnEngine, markerDocIdToIndex,
                        markerVectorArray, markerDocIds, finalVectorDataType
                    );

                    // Record assignment in the flat array
                    assignMap[docId] = nearestMarkerDocId;

                    // Spill vector to temp file immediately (docId + vector bytes)
                    ClumpFileWriter.writeHiddenEntryToTemp(
                        tempHiddenOutput, docId, hiddenVector, finalVectorDataType
                    );
                }
                hiddenScan.nextDoc();
            }

            return new AssignmentResult(markerVectors, assignMap, dimension, vectorDataType);
        } finally {
            freeTempIndex(indexPointer, knnEngine);
        }
    }

    /**
     * Searches the loaded native index for the nearest marker to the given hidden vector.
     * Falls back to brute-force L2 if the JNI search returns no results.
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
        if (vectorDataType == ClumpFileFormat.VECTOR_TYPE_FLOAT) {
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
     */
    private long loadTempIndex(BuildIndexParams indexInfo, String tempEngineFileName, KNNEngine knnEngine) throws IOException {
        try (
            IndexInput tempInput = indexInfo.getSegmentWriteState().directory.openInput(
                tempEngineFileName, IOContext.DEFAULT
            )
        ) {
            IndexInputWithBuffer readStream = new IndexInputWithBuffer(tempInput);
            return AccessController.doPrivileged((PrivilegedAction<Long>) () ->
                JNIService.loadIndex(readStream, indexInfo.getParameters(), knnEngine)
            );
        }
    }

    /**
     * Frees a loaded native index.
     */
    private void freeTempIndex(long indexPointer, KNNEngine knnEngine) {
        try {
            AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
                JNIService.free(indexPointer, knnEngine);
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

    static String buildAssignFileName(String segmentName, String fieldName) {
        return segmentName + "_" + fieldName + ".clumpassign";
    }

    /**
     * Result of the streaming assignment pass. Contains marker vectors (held in memory),
     * the flat assign map (docId → markerDocId), and dimension/type metadata.
     * The assignMap is flushed to disk after this result is returned, then freed.
     */
    private static class AssignmentResult {
        final List<Object> markerVectors;
        final int[] assignMap;
        final int dimension;
        final byte vectorDataType;

        AssignmentResult(List<Object> markerVectors, int[] assignMap,
                         int dimension, byte vectorDataType) {
            this.markerVectors = markerVectors;
            this.assignMap = assignMap;
            this.dimension = dimension;
            this.vectorDataType = vectorDataType;
        }
    }
}
