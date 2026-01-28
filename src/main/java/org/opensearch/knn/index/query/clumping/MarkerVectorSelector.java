/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clumping;

import lombok.extern.log4j.Log4j2;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * Selects marker vectors from a set of vectors based on the clumping factor.
 * 
 * The selection process:
 * 1. Randomly select 1/clumpingFactor vectors as markers
 * 2. For each non-marker (hidden) vector, find its nearest marker
 * 3. Associate hidden vectors with their nearest markers
 * 
 * This is used during indexing to determine which vectors go into the main index
 * (markers) and which are stored separately (hidden).
 */
@Log4j2
public class MarkerVectorSelector {

    private final ClumpingContext clumpingContext;
    private final Random random;

    public MarkerVectorSelector(ClumpingContext clumpingContext) {
        this.clumpingContext = clumpingContext;
        this.random = new Random();
    }

    public MarkerVectorSelector(ClumpingContext clumpingContext, long seed) {
        this.clumpingContext = clumpingContext;
        this.random = new Random(seed);
    }

    /**
     * Result of marker selection containing markers, hidden vectors, and their assignments.
     */
    public static class SelectionResult {
        /**
         * Set of document IDs selected as markers.
         */
        public final Set<Integer> markerDocIds;

        /**
         * Map from hidden doc ID to its vector.
         */
        public final Map<Integer, float[]> hiddenVectors;

        /**
         * Map from hidden doc ID to its assigned marker doc ID.
         */
        public final Map<Integer, Integer> hiddenToMarkerAssignment;

        public SelectionResult(
            Set<Integer> markerDocIds,
            Map<Integer, float[]> hiddenVectors,
            Map<Integer, Integer> hiddenToMarkerAssignment
        ) {
            this.markerDocIds = markerDocIds;
            this.hiddenVectors = hiddenVectors;
            this.hiddenToMarkerAssignment = hiddenToMarkerAssignment;
        }
    }

    /**
     * Selects markers from the given vectors and assigns hidden vectors to markers.
     * 
     * @param docIdToVector map from document ID to its vector
     * @param spaceType the space type for distance calculation
     * @param vectorDataType the vector data type
     * @return selection result with markers and hidden vector assignments
     */
    public SelectionResult selectMarkers(
        Map<Integer, float[]> docIdToVector,
        SpaceType spaceType,
        VectorDataType vectorDataType
    ) {
        if (!clumpingContext.isEffectivelyEnabled() || docIdToVector.isEmpty()) {
            // No clumping - all vectors are markers
            return new SelectionResult(
                new HashSet<>(docIdToVector.keySet()),
                new HashMap<>(),
                new HashMap<>()
            );
        }

        int totalVectors = docIdToVector.size();
        int targetMarkerCount = clumpingContext.getExpectedMarkerCount(totalVectors);

        // Randomly select markers
        List<Integer> allDocIds = new ArrayList<>(docIdToVector.keySet());
        Set<Integer> markerDocIds = selectRandomMarkers(allDocIds, targetMarkerCount);

        // Separate markers and hidden vectors
        Map<Integer, float[]> markerVectors = new HashMap<>();
        Map<Integer, float[]> hiddenVectors = new HashMap<>();

        for (Map.Entry<Integer, float[]> entry : docIdToVector.entrySet()) {
            int docId = entry.getKey();
            float[] vector = entry.getValue();
            
            if (markerDocIds.contains(docId)) {
                markerVectors.put(docId, vector);
            } else {
                hiddenVectors.put(docId, vector);
            }
        }

        // Assign each hidden vector to its nearest marker
        Map<Integer, Integer> hiddenToMarkerAssignment = assignHiddenToMarkers(
            hiddenVectors,
            markerVectors,
            spaceType,
            vectorDataType
        );

        log.debug(
            "Marker selection: {} total vectors -> {} markers, {} hidden",
            totalVectors,
            markerDocIds.size(),
            hiddenVectors.size()
        );

        return new SelectionResult(markerDocIds, hiddenVectors, hiddenToMarkerAssignment);
    }

    /**
     * Randomly selects marker document IDs.
     */
    private Set<Integer> selectRandomMarkers(List<Integer> allDocIds, int targetCount) {
        Set<Integer> markers = new HashSet<>();
        
        if (targetCount >= allDocIds.size()) {
            markers.addAll(allDocIds);
            return markers;
        }

        // Fisher-Yates shuffle to select random markers
        List<Integer> shuffled = new ArrayList<>(allDocIds);
        for (int i = shuffled.size() - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            Integer temp = shuffled.get(i);
            shuffled.set(i, shuffled.get(j));
            shuffled.set(j, temp);
        }

        for (int i = 0; i < targetCount; i++) {
            markers.add(shuffled.get(i));
        }

        return markers;
    }

    /**
     * Assigns each hidden vector to its nearest marker.
     */
    private Map<Integer, Integer> assignHiddenToMarkers(
        Map<Integer, float[]> hiddenVectors,
        Map<Integer, float[]> markerVectors,
        SpaceType spaceType,
        VectorDataType vectorDataType
    ) {
        Map<Integer, Integer> assignments = new HashMap<>();

        for (Map.Entry<Integer, float[]> hiddenEntry : hiddenVectors.entrySet()) {
            int hiddenDocId = hiddenEntry.getKey();
            float[] hiddenVector = hiddenEntry.getValue();

            int nearestMarker = findNearestMarker(hiddenVector, markerVectors, spaceType);
            assignments.put(hiddenDocId, nearestMarker);
        }

        return assignments;
    }

    /**
     * Finds the nearest marker to a given vector.
     */
    private int findNearestMarker(
        float[] vector,
        Map<Integer, float[]> markerVectors,
        SpaceType spaceType
    ) {
        int nearestMarker = -1;
        float bestDistance = Float.MAX_VALUE;

        for (Map.Entry<Integer, float[]> markerEntry : markerVectors.entrySet()) {
            int markerDocId = markerEntry.getKey();
            float[] markerVector = markerEntry.getValue();

            float distance = computeDistance(vector, markerVector, spaceType);
            
            if (distance < bestDistance) {
                bestDistance = distance;
                nearestMarker = markerDocId;
            }
        }

        return nearestMarker;
    }

    /**
     * Computes distance between two vectors based on space type.
     */
    private float computeDistance(float[] a, float[] b, SpaceType spaceType) {
        switch (spaceType) {
            case L2:
                return computeL2Distance(a, b);
            case COSINESIMIL:
                // For cosine, use 1 - similarity as distance
                return 1.0f - computeCosineSimilarity(a, b);
            case INNER_PRODUCT:
                // For inner product, negate (higher is better, so lower negative is closer)
                return -computeInnerProduct(a, b);
            case L1:
                return computeL1Distance(a, b);
            case LINF:
                return computeLinfDistance(a, b);
            case HAMMING:
                return computeHammingDistance(a, b);
            default:
                return computeL2Distance(a, b);
        }
    }

    private float computeL2Distance(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum; // Return squared distance for efficiency
    }

    private float computeCosineSimilarity(float[] a, float[] b) {
        float dotProduct = 0;
        float normA = 0;
        float normB = 0;
        
        for (int i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        
        if (normA == 0 || normB == 0) {
            return 0;
        }
        
        return dotProduct / (float) (Math.sqrt(normA) * Math.sqrt(normB));
    }

    private float computeInnerProduct(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    private float computeL1Distance(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.abs(a[i] - b[i]);
        }
        return sum;
    }

    private float computeLinfDistance(float[] a, float[] b) {
        float max = 0;
        for (int i = 0; i < a.length; i++) {
            max = Math.max(max, Math.abs(a[i] - b[i]));
        }
        return max;
    }

    private float computeHammingDistance(float[] a, float[] b) {
        int distance = 0;
        for (int i = 0; i < a.length; i++) {
            if ((int) a[i] != (int) b[i]) {
                distance++;
            }
        }
        return distance;
    }
}
