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
 * The marker index is first built into a temporary file so it can be loaded and searched via JNI.
 * After hidden vector assignment is complete, the temp file contents are copied to the real engine
 * output and the temp file is deleted.
 */
@Log4j2
public class ClumpingIndexBuildStrategy implements NativeIndexBuildStrategy {

    private final NativeIndexBuildStrategy delegate;
    private final int clumpingFactor;

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

        // Separate markers and hidden by insertion order
        final List<Integer> markerDocIds = new ArrayList<>();
        final List<Integer> hiddenDocIds = new ArrayList<>();

        for (int i = 0; i < allDocIds.size(); i++) {
            if (i % clumpingFactor == 0) {
                markerDocIds.add(allDocIds.get(i));
            } else {
                hiddenDocIds.add(allDocIds.get(i));
            }
        }
        allDocIds.clear();

        log.debug(
            "Clumping: {} markers, {} hidden (factor={})",
            markerDocIds.size(),
            hiddenDocIds.size(),
            clumpingFactor
        );

        if (hiddenDocIds.isEmpty()) {
            delegate.buildAndWriteIndex(indexInfo);
            return;
        }

        final KNNEngine knnEngine = indexInfo.getKnnEngine();
        final String segmentName = indexInfo.getSegmentWriteState().segmentInfo.name;
        final String fieldName = indexInfo.getFieldName();

        // Build marker index into a temp file so we can load it for searching
        String tempEngineFileName = segmentName + "_" + fieldName + ".clumpidx";
        buildMarkerIndexToTempFile(indexInfo, markerDocIds, tempEngineFileName);

        // Load the temp index, assign hidden vectors, then clean up
        int[] numHiddenPerMarker = new int[markerDocIds.size()];
        List<HiddenEntryLocation> hiddenEntryLocations;
        List<Object> markerVectors;
        int dimension;
        byte vectorDataType;

        String tempHiddenFileName = ClumpFileWriter.buildTempFileName(segmentName, fieldName);

        try (
            IndexOutput tempHiddenOutput = indexInfo.getSegmentWriteState().directory.createOutput(
                tempHiddenFileName, indexInfo.getSegmentWriteState().context
            )
        ) {
            AssignmentResult result = assignHiddenToMarkersViaIndex(
                indexInfo, markerDocIds, hiddenDocIds, numHiddenPerMarker,
                tempHiddenOutput, tempEngineFileName, knnEngine
            );
            dimension = result.dimension;
            vectorDataType = result.vectorDataType;
            hiddenEntryLocations = result.hiddenEntryLocations;
            markerVectors = result.markerVectors;
        }

        // Copy the temp engine file to the real output, then clean it up
        copyTempToRealOutput(indexInfo, tempEngineFileName);
        deleteTempFile(indexInfo, tempEngineFileName);

        // Write the final .clump file, reading hidden vectors back from the temp file
        try (
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
                numHiddenPerMarker,
                hiddenEntryLocations,
                tempHiddenInput
            );
        } finally {
            deleteTempFile(indexInfo, tempHiddenFileName);
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
     * Marker vectors are cloned into memory (1/N of total).
     * Hidden vectors are scored via the native index, then immediately spilled to
     * {@code tempHiddenOutput} and discarded.
     */
    private AssignmentResult assignHiddenToMarkersViaIndex(
        BuildIndexParams indexInfo,
        List<Integer> markerDocIds,
        List<Integer> hiddenDocIds,
        int[] numHiddenPerMarker,
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

        // Load the temp marker index via JNI
        long indexPointer = loadTempIndex(indexInfo, tempEngineFileName, knnEngine);

        try {
            // Stream hidden vectors: read each one, search the native index for nearest marker,
            // spill to temp file
            final KNNVectorValues<?> hiddenScan = indexInfo.getKnnVectorValuesSupplier().get();
            initializeVectorValues(hiddenScan);

            Set<Integer> hiddenSet = new HashSet<>(hiddenDocIds);
            List<HiddenEntryLocation> hiddenEntryLocations = new ArrayList<>(hiddenDocIds.size());
            final byte finalVectorDataType = vectorDataType;
            final int finalDimension = dimension;

            while (hiddenScan.docId() != NO_MORE_DOCS) {
                if (hiddenSet.contains(hiddenScan.docId())) {
                    Object hiddenVector = hiddenScan.getVector();

                    // Search the native index with k=1 to find the nearest marker
                    int nearestMarkerIdx = findNearestMarkerViaIndex(
                        hiddenVector, indexPointer, knnEngine, markerDocIdToIndex,
                        markerVectorArray, finalVectorDataType
                    );

                    // Spill to temp file immediately
                    long tempOffset = ClumpFileWriter.writeHiddenEntryToTemp(
                        tempHiddenOutput, hiddenScan.docId(), hiddenVector, finalVectorDataType
                    );

                    hiddenEntryLocations.add(new HiddenEntryLocation(nearestMarkerIdx, tempOffset));
                    numHiddenPerMarker[nearestMarkerIdx]++;
                }
                hiddenScan.nextDoc();
            }

            return new AssignmentResult(markerVectors, hiddenEntryLocations, finalDimension, finalVectorDataType);
        } finally {
            // Always free the loaded index — the temp engine file is deleted by the caller
            // after it has been copied to the real output.
            freeTempIndex(indexPointer, knnEngine);
        }
    }

    /**
     * Searches the loaded native index for the nearest marker to the given hidden vector.
     * Falls back to brute-force L2 if the JNI search returns no results.
     *
     * @return the index into the markerDocIds list of the nearest marker
     */
    private int findNearestMarkerViaIndex(
        Object hiddenVector,
        long indexPointer,
        KNNEngine knnEngine,
        Map<Integer, Integer> markerDocIdToIndex,
        Object[] markerVectorArray,
        byte vectorDataType
    ) {
        KNNQueryResult[] results;
        if (vectorDataType == ClumpFileFormat.VECTOR_TYPE_FLOAT) {
            float[] queryVector = (hiddenVector instanceof float[])
                ? (float[]) hiddenVector
                : toFloatArray((byte[]) hiddenVector);
            results = AccessController.doPrivileged((PrivilegedAction<KNNQueryResult[]>) () ->
                JNIService.queryIndex(
                    indexPointer, queryVector, 1, null, knnEngine,
                    null, 0, null
                )
            );
        } else {
            // For byte vectors, convert to float for querying (JNI queryIndex takes float[])
            float[] queryVector = toFloatArray((byte[]) hiddenVector);
            results = AccessController.doPrivileged((PrivilegedAction<KNNQueryResult[]>) () ->
                JNIService.queryIndex(
                    indexPointer, queryVector, 1, null, knnEngine,
                    null, 0, null
                )
            );
        }

        if (results != null && results.length > 0) {
            int nearestDocId = results[0].getId();
            Integer markerIndex = markerDocIdToIndex.get(nearestDocId);
            if (markerIndex != null) {
                return markerIndex;
            }
            log.warn("JNI search returned doc ID {} not in marker set, falling back to brute-force", nearestDocId);
        } else {
            log.warn("JNI search returned no results, falling back to brute-force");
        }

        // Fallback: brute-force L2 distance
        return findNearestMarkerBruteForce(hiddenVector, markerVectorArray);
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

    /**
     * Result of the streaming assignment pass. Contains marker vectors (held in memory),
     * dimension/type metadata, and the list of hidden entry locations pointing into the temp file.
     */
    private static class AssignmentResult {
        final List<Object> markerVectors;
        final List<HiddenEntryLocation> hiddenEntryLocations;
        final int dimension;
        final byte vectorDataType;

        AssignmentResult(List<Object> markerVectors, List<HiddenEntryLocation> hiddenEntryLocations,
                         int dimension, byte vectorDataType) {
            this.markerVectors = markerVectors;
            this.hiddenEntryLocations = hiddenEntryLocations;
            this.dimension = dimension;
            this.vectorDataType = vectorDataType;
        }
    }
}
