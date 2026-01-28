/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clumping;

import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Represents the mapping between marker vectors and their associated hidden vectors.
 * This is used during search to expand marker vector results to include hidden vectors.
 */
@Getter
@AllArgsConstructor
public class HiddenVectorMapping {

    /**
     * Map from marker document ID to list of hidden document IDs associated with that marker.
     */
    private final Map<Integer, List<Integer>> markerToHiddenDocs;

    /**
     * Set of all marker document IDs for quick lookup.
     */
    private final Set<Integer> markerDocIds;

    /**
     * Empty mapping singleton for when clumping is disabled.
     */
    public static final HiddenVectorMapping EMPTY = new HiddenVectorMapping(
        Collections.emptyMap(),
        Collections.emptySet()
    );

    /**
     * Gets the hidden document IDs associated with a marker document.
     * 
     * @param markerDocId the marker document ID
     * @return list of hidden document IDs, or empty list if none
     */
    public List<Integer> getHiddenDocsForMarker(int markerDocId) {
        return markerToHiddenDocs.getOrDefault(markerDocId, Collections.emptyList());
    }

    /**
     * Checks if a document ID is a marker.
     * 
     * @param docId the document ID to check
     * @return true if the document is a marker
     */
    public boolean isMarker(int docId) {
        return markerDocIds.contains(docId);
    }

    /**
     * Gets the total number of hidden vectors across all markers.
     * 
     * @return total hidden vector count
     */
    public int getTotalHiddenCount() {
        return markerToHiddenDocs.values().stream()
            .mapToInt(List::size)
            .sum();
    }

    /**
     * Checks if this mapping is empty (no clumping).
     * 
     * @return true if empty
     */
    public boolean isEmpty() {
        return markerToHiddenDocs.isEmpty();
    }
}
