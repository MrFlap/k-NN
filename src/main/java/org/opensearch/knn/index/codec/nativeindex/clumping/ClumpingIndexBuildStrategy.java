/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.clumping;

import lombok.extern.log4j.Log4j2;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategy;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.initializeVectorValues;

/**
 * A build strategy that wraps an underlying {@link NativeIndexBuildStrategy} and adds clumping.
 * <p>
 * Every nth vector (starting from 0) is designated a "marker" vector and is inserted into the
 * native index via the delegate strategy. The remaining "hidden" vectors are not inserted into
 * the native index. Instead, after all marker vectors are indexed, each hidden vector is assigned
 * to its nearest marker via a 1-NN search on the native index. The mapping is written to a .clump
 * sidecar file.
 * <p>
 * This strategy uses composition: it delegates the actual native index building to the wrapped
 * strategy, only changing which vectors get inserted.
 * <p>
 * Memory efficiency: vectors are never all held in memory simultaneously. The first pass collects
 * only doc IDs (ints). The delegate build re-reads marker vectors from a fresh supplier via
 * {@link FilteredKNNVectorValues}. The hidden-to-marker assignment holds only marker vectors
 * in memory (1/clumpingFactor of total) and streams hidden vectors one at a time.
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
        // allDocIds no longer needed
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

        // Build native index with only marker vectors using a filtered supplier that
        // re-reads from the original supplier (no vector data cached in memory)
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
        // Load only marker vectors into memory (1/clumpingFactor of total).
        Map<Integer, List<Integer>> markerToHidden = new HashMap<>(markerDocIds.size());
        for (int markerDocId : markerDocIds) {
            markerToHidden.put(markerDocId, new ArrayList<>());
        }

        assignHiddenToMarkers(indexInfo, markerDocIds, hiddenDocIds, markerToHidden);

        // Write the .clump file
        ClumpFileWriter.writeClumpFile(
            indexInfo.getSegmentWriteState(),
            indexInfo.getFieldName(),
            markerDocIds,
            markerToHidden
        );
    }

    /**
     * Assigns each hidden vector to its nearest marker. Holds marker vectors in memory
     * (1/clumpingFactor of total) and streams hidden vectors one at a time from a fresh supplier.
     */
    private void assignHiddenToMarkers(
        BuildIndexParams indexInfo,
        List<Integer> markerDocIds,
        List<Integer> hiddenDocIds,
        Map<Integer, List<Integer>> markerToHidden
    ) throws IOException {
        // Read marker vectors into memory (only the marker subset)
        final KNNVectorValues<?> markerScan = indexInfo.getKnnVectorValuesSupplier().get();
        initializeVectorValues(markerScan);

        int markerIdx = 0;
        java.util.Set<Integer> markerSet = new java.util.HashSet<>(markerDocIds);
        Object[] markerVectors = new Object[markerDocIds.size()];
        int[] markerDocIdArray = new int[markerDocIds.size()];

        while (markerScan.docId() != NO_MORE_DOCS && markerIdx < markerDocIds.size()) {
            if (markerSet.contains(markerScan.docId())) {
                markerDocIdArray[markerIdx] = markerScan.docId();
                markerVectors[markerIdx] = cloneVector(markerScan);
                markerIdx++;
            }
            markerScan.nextDoc();
        }

        // Stream hidden vectors one at a time from a fresh supplier
        final KNNVectorValues<?> hiddenScan = indexInfo.getKnnVectorValuesSupplier().get();
        initializeVectorValues(hiddenScan);

        java.util.Set<Integer> hiddenSet = new java.util.HashSet<>(hiddenDocIds);

        while (hiddenScan.docId() != NO_MORE_DOCS) {
            if (hiddenSet.contains(hiddenScan.docId())) {
                int nearestIdx = findNearestMarker(hiddenScan.getVector(), markerVectors);
                markerToHidden.get(markerDocIdArray[nearestIdx]).add(hiddenScan.docId());
            }
            hiddenScan.nextDoc();
        }
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
