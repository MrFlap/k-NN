/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.clumping;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategy;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.initializeVectorValues;

/**
 * A build strategy that wraps an underlying {@link NativeIndexBuildStrategy} and adds clumping.
 * <p>
 * Every nth vector (starting from 0) is designated a "marker" vector and is inserted into the
 * native index via the delegate strategy. The remaining "hidden" vectors are not inserted into
 * the native index. Instead, after all marker vectors are indexed, each hidden vector is assigned
 * to its nearest marker via brute-force L2 distance. The mapping — including all vector data —
 * is written to a .clump sidecar file so that expansion at query time can read vectors
 * sequentially without random access into Lucene's vector storage.
 * <p>
 * Memory budget: O(marker vectors) = O(1/clumpingFactor × total). Hidden vectors are spilled
 * to a temp file during assignment and never held in heap simultaneously.
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

        // Build native index with only marker vectors
        BuildIndexParams markerIndexParams = BuildIndexParams.builder()
            .fieldName(indexInfo.getFieldName())
            .knnEngine(indexInfo.getKnnEngine())
            .indexOutputWithBuffer(indexInfo.getIndexOutputWithBuffer())
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

        // Pass 2: assign hidden vectors to nearest markers.
        // Marker vectors are held in memory (1/N of total).
        // Hidden vectors are spilled to a temp file immediately after scoring — never held in heap.
        String tempFileName = ClumpFileWriter.buildTempFileName(
            indexInfo.getSegmentWriteState().segmentInfo.name,
            indexInfo.getFieldName()
        );

        int dimension;
        byte vectorDataType;
        int[] numHiddenPerMarker = new int[markerDocIds.size()];
        List<HiddenEntryLocation> hiddenEntryLocations;
        List<Object> markerVectors;

        try (
            IndexOutput tempOutput = indexInfo.getSegmentWriteState().directory.createOutput(
                tempFileName, indexInfo.getSegmentWriteState().context
            )
        ) {
            AssignmentResult result = assignHiddenToMarkersStreaming(
                indexInfo, markerDocIds, hiddenDocIds, numHiddenPerMarker, tempOutput
            );
            dimension = result.dimension;
            vectorDataType = result.vectorDataType;
            hiddenEntryLocations = result.hiddenEntryLocations;
            markerVectors = result.markerVectors;
        }

        // Write the final .clump file, reading hidden vectors back from the temp file
        try (
            IndexInput tempInput = indexInfo.getSegmentWriteState().directory.openInput(
                tempFileName, indexInfo.getSegmentWriteState().context
            )
        ) {
            ClumpFileWriter.writeClumpFile(
                indexInfo.getSegmentWriteState(),
                indexInfo.getFieldName(),
                dimension,
                vectorDataType,
                markerDocIds,
                markerVectors,
                numHiddenPerMarker,
                hiddenEntryLocations,
                tempInput
            );
        } finally {
            // Clean up temp file
            try {
                indexInfo.getSegmentWriteState().directory.deleteFile(tempFileName);
            } catch (IOException e) {
                log.warn("Failed to delete temp clump file {}", tempFileName, e);
            }
        }
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

    /**
     * Assigns each hidden vector to its nearest marker using streaming.
     * <p>
     * Marker vectors are cloned into memory (1/N of total — same budget as v1).
     * Hidden vectors are scored against all markers, then immediately spilled to
     * {@code tempOutput} and discarded. Only a lightweight {@link HiddenEntryLocation}
     * (12 bytes) is retained per hidden vector.
     *
     * @param indexInfo           Build parameters with vector supplier
     * @param markerDocIds        Ordered marker doc IDs
     * @param hiddenDocIds        Ordered hidden doc IDs
     * @param numHiddenPerMarker  Output array: count of hidden vectors per marker (filled by this method)
     * @param tempOutput          Temp file to spill hidden vector data into
     * @return Assignment result with marker vectors and hidden entry locations
     */
    private AssignmentResult assignHiddenToMarkersStreaming(
        BuildIndexParams indexInfo,
        List<Integer> markerDocIds,
        List<Integer> hiddenDocIds,
        int[] numHiddenPerMarker,
        IndexOutput tempOutput
    ) throws IOException {
        // First, read marker vectors into memory (only the marker subset — 1/N of total)
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

        // Now stream hidden vectors: read each one, find nearest marker, spill to temp file
        final KNNVectorValues<?> hiddenScan = indexInfo.getKnnVectorValuesSupplier().get();
        initializeVectorValues(hiddenScan);

        Set<Integer> hiddenSet = new HashSet<>(hiddenDocIds);
        List<HiddenEntryLocation> hiddenEntryLocations = new ArrayList<>(hiddenDocIds.size());
        final byte finalVectorDataType = vectorDataType;

        while (hiddenScan.docId() != NO_MORE_DOCS) {
            if (hiddenSet.contains(hiddenScan.docId())) {
                // Get vector reference (do NOT clone — we'll write it immediately and discard)
                Object hiddenVector = hiddenScan.getVector();
                int nearestIdx = findNearestMarker(hiddenVector, markerVectorArray);

                // Spill to temp file immediately
                long tempOffset = ClumpFileWriter.writeHiddenEntryToTemp(
                    tempOutput, hiddenScan.docId(), hiddenVector, finalVectorDataType
                );

                hiddenEntryLocations.add(new HiddenEntryLocation(nearestIdx, tempOffset));
                numHiddenPerMarker[nearestIdx]++;
            }
            hiddenScan.nextDoc();
        }

        return new AssignmentResult(markerVectors, hiddenEntryLocations, dimension, vectorDataType);
    }

    /**
     * Finds the index of the nearest marker vector using squared L2 distance.
     */
    private int findNearestMarker(Object hiddenVector, Object[] markerVectors) {
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
     * Computes squared L2 distance between two vectors. Works for both float[] and byte[].
     */
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

    private Object cloneVector(KNNVectorValues<?> vectorValues) throws IOException {
        Object vector = vectorValues.getVector();
        if (vector instanceof float[]) {
            return ((float[]) vector).clone();
        } else if (vector instanceof byte[]) {
            return ((byte[]) vector).clone();
        }
        throw new IllegalArgumentException("Unsupported vector type: " + vector.getClass());
    }
}
