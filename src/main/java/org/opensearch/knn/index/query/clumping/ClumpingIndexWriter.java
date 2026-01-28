/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clumping;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.Directory;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Handles writing clumping data during index creation.
 * 
 * This class is responsible for:
 * 1. Selecting marker vectors from the input vectors
 * 2. Writing hidden vectors to the clumping file
 * 3. Returning only marker vectors for the main index
 * 
 * Usage during indexing:
 * <pre>
 * ClumpingIndexWriter writer = new ClumpingIndexWriter(clumpingContext, directory, segmentName, fieldName);
 * Map<Integer, float[]> markersOnly = writer.processVectors(allVectors, spaceType, vectorDataType);
 * // markersOnly contains only the vectors that should go into the main index
 * // hidden vectors are automatically written to disk
 * </pre>
 */
@Log4j2
public class ClumpingIndexWriter {

    private final ClumpingContext clumpingContext;
    private final Directory directory;
    private final String segmentName;
    private final String fieldName;

    public ClumpingIndexWriter(
        ClumpingContext clumpingContext,
        Directory directory,
        String segmentName,
        String fieldName
    ) {
        this.clumpingContext = clumpingContext;
        this.directory = directory;
        this.segmentName = segmentName;
        this.fieldName = fieldName;
    }

    /**
     * Processes vectors for clumping, selecting markers and writing hidden vectors to disk.
     * 
     * @param docIdToVector map from document ID to its vector
     * @param spaceType the space type for distance calculation
     * @param vectorDataType the vector data type
     * @param dimension the vector dimension
     * @return map containing only marker vectors (to be indexed in main index)
     * @throws IOException if writing hidden vectors fails
     */
    public Map<Integer, float[]> processVectors(
        Map<Integer, float[]> docIdToVector,
        SpaceType spaceType,
        VectorDataType vectorDataType,
        int dimension
    ) throws IOException {
        if (!isClumpingEnabled() || docIdToVector.isEmpty()) {
            return docIdToVector;
        }

        // Select markers and assign hidden vectors
        MarkerVectorSelector selector = new MarkerVectorSelector(clumpingContext);
        MarkerVectorSelector.SelectionResult selection = selector.selectMarkers(
            docIdToVector,
            spaceType,
            vectorDataType
        );

        // Write hidden vectors to disk
        if (!selection.hiddenVectors.isEmpty()) {
            writeHiddenVectors(selection, vectorDataType, dimension);
        }

        // Return only marker vectors for the main index
        Map<Integer, float[]> markerVectors = new HashMap<>();
        for (int markerDocId : selection.markerDocIds) {
            markerVectors.put(markerDocId, docIdToVector.get(markerDocId));
        }

        log.info(
            "Clumping: {} total vectors -> {} markers (indexed), {} hidden (on disk) for segment {}",
            docIdToVector.size(),
            markerVectors.size(),
            selection.hiddenVectors.size(),
            segmentName
        );

        return markerVectors;
    }

    /**
     * Checks if clumping is enabled.
     */
    public boolean isClumpingEnabled() {
        return clumpingContext != null && clumpingContext.isEffectivelyEnabled();
    }

    /**
     * Gets the set of marker document IDs without processing.
     * Useful for checking which docs will be markers before full processing.
     */
    public Set<Integer> getMarkerDocIds(
        Map<Integer, float[]> docIdToVector,
        SpaceType spaceType,
        VectorDataType vectorDataType
    ) {
        if (!isClumpingEnabled()) {
            return docIdToVector.keySet();
        }

        MarkerVectorSelector selector = new MarkerVectorSelector(clumpingContext);
        MarkerVectorSelector.SelectionResult selection = selector.selectMarkers(
            docIdToVector,
            spaceType,
            vectorDataType
        );

        return selection.markerDocIds;
    }

    private void writeHiddenVectors(
        MarkerVectorSelector.SelectionResult selection,
        VectorDataType vectorDataType,
        int dimension
    ) throws IOException {
        ClumpingVectorStore store = new ClumpingVectorStore(directory, segmentName, fieldName);
        store.writeHiddenVectors(
            selection.hiddenVectors,
            selection.hiddenToMarkerAssignment,
            vectorDataType,
            dimension
        );
    }
}
