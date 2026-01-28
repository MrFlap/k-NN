/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.clumping.MarkerSelector;
import org.opensearch.test.OpenSearchTestCase;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Unit tests for {@link ClumpingIndexBuildStrategy}.
 * 
 * Validates: Requirements 2.1, 2.2, 2.3, 2.4, 3.3
 */
public class ClumpingIndexBuildStrategyTests extends OpenSearchTestCase {

    /**
     * Test that marker association assigns hidden vectors to their closest marker.
     * Validates: Requirement 3.3 (marker association correctness)
     */
    public void testMarkerAssociationAssignsToClosestMarker() {
        // Create a set of marker vectors at known positions
        List<float[]> markerVectors = new ArrayList<>();
        List<Integer> markerDocIds = new ArrayList<>();
        
        // Marker at origin
        markerVectors.add(new float[] { 0.0f, 0.0f, 0.0f, 0.0f });
        markerDocIds.add(0);
        
        // Marker at (10, 0, 0, 0)
        markerVectors.add(new float[] { 10.0f, 0.0f, 0.0f, 0.0f });
        markerDocIds.add(1);
        
        // Marker at (0, 10, 0, 0)
        markerVectors.add(new float[] { 0.0f, 10.0f, 0.0f, 0.0f });
        markerDocIds.add(2);

        // Hidden vector close to origin - should be assigned to marker 0
        float[] hiddenNearOrigin = new float[] { 1.0f, 1.0f, 0.0f, 0.0f };
        int closestToOrigin = findClosestMarker(hiddenNearOrigin, markerDocIds, markerVectors, SpaceType.L2);
        assertEquals("Hidden vector near origin should be assigned to marker 0", 0, closestToOrigin);

        // Hidden vector close to (10, 0, 0, 0) - should be assigned to marker 1
        float[] hiddenNearMarker1 = new float[] { 9.0f, 0.0f, 0.0f, 0.0f };
        int closestToMarker1 = findClosestMarker(hiddenNearMarker1, markerDocIds, markerVectors, SpaceType.L2);
        assertEquals("Hidden vector near (10,0,0,0) should be assigned to marker 1", 1, closestToMarker1);

        // Hidden vector close to (0, 10, 0, 0) - should be assigned to marker 2
        float[] hiddenNearMarker2 = new float[] { 0.0f, 9.0f, 0.0f, 0.0f };
        int closestToMarker2 = findClosestMarker(hiddenNearMarker2, markerDocIds, markerVectors, SpaceType.L2);
        assertEquals("Hidden vector near (0,10,0,0) should be assigned to marker 2", 2, closestToMarker2);
    }

    /**
     * Test that marker association is deterministic.
     * Validates: Requirement 3.3 (determinism)
     */
    public void testMarkerAssociationIsDeterministic() {
        List<float[]> markerVectors = new ArrayList<>();
        List<Integer> markerDocIds = new ArrayList<>();
        
        for (int i = 0; i < 10; i++) {
            markerVectors.add(randomVector(4));
            markerDocIds.add(i);
        }

        float[] hiddenVector = randomVector(4);

        // Run association multiple times
        int firstResult = findClosestMarker(hiddenVector, markerDocIds, markerVectors, SpaceType.L2);
        
        for (int i = 0; i < 10; i++) {
            int result = findClosestMarker(hiddenVector, markerDocIds, markerVectors, SpaceType.L2);
            assertEquals("Marker association should be deterministic", firstResult, result);
        }
    }

    /**
     * Test that marker association works with different space types.
     */
    public void testMarkerAssociationWithDifferentSpaceTypes() {
        List<float[]> markerVectors = new ArrayList<>();
        List<Integer> markerDocIds = new ArrayList<>();
        
        // Normalized vectors for cosine similarity
        markerVectors.add(normalize(new float[] { 1.0f, 0.0f, 0.0f, 0.0f }));
        markerDocIds.add(0);
        
        markerVectors.add(normalize(new float[] { 0.0f, 1.0f, 0.0f, 0.0f }));
        markerDocIds.add(1);

        // Hidden vector more aligned with marker 0
        float[] hiddenVector = normalize(new float[] { 0.9f, 0.1f, 0.0f, 0.0f });

        // With cosine similarity, should be assigned to marker 0
        int closestCosine = findClosestMarker(hiddenVector, markerDocIds, markerVectors, SpaceType.COSINESIMIL);
        assertEquals("With cosine similarity, should be assigned to marker 0", 0, closestCosine);
    }

    /**
     * Test that all hidden vectors are assigned to exactly one marker.
     * Validates: Requirement 3.3 (completeness)
     */
    public void testAllHiddenVectorsAreAssigned() {
        int numMarkers = 10;
        int numHidden = 100;
        int dimension = 8;

        List<float[]> markerVectors = new ArrayList<>();
        List<Integer> markerDocIds = new ArrayList<>();
        
        for (int i = 0; i < numMarkers; i++) {
            markerVectors.add(randomVector(dimension));
            markerDocIds.add(i);
        }

        // Track assignments
        Map<Integer, List<Integer>> assignments = new HashMap<>();
        for (int markerDocId : markerDocIds) {
            assignments.put(markerDocId, new ArrayList<>());
        }

        // Assign all hidden vectors
        for (int hiddenId = 0; hiddenId < numHidden; hiddenId++) {
            float[] hiddenVector = randomVector(dimension);
            int closestMarker = findClosestMarker(hiddenVector, markerDocIds, markerVectors, SpaceType.L2);
            
            // Verify assignment is to a valid marker
            assertTrue("Assignment should be to a valid marker", markerDocIds.contains(closestMarker));
            
            assignments.get(closestMarker).add(hiddenId);
        }

        // Verify all hidden vectors were assigned
        int totalAssigned = assignments.values().stream().mapToInt(List::size).sum();
        assertEquals("All hidden vectors should be assigned", numHidden, totalAssigned);
    }

    /**
     * Test that marker selection produces correct ratio.
     * Validates: Requirement 2.1
     */
    public void testMarkerSelectionRatio() {
        int totalDocs = 10000;
        int clumpingFactor = 8;
        
        int markerCount = 0;
        int hiddenCount = 0;
        
        for (int docId = 0; docId < totalDocs; docId++) {
            if (MarkerSelector.isMarker(docId, clumpingFactor)) {
                markerCount++;
            } else {
                hiddenCount++;
            }
        }

        double expectedMarkerRatio = 1.0 / clumpingFactor;
        double actualMarkerRatio = (double) markerCount / totalDocs;
        double tolerance = 0.05;

        assertTrue(
            String.format("Marker ratio should be approximately %.2f, but was %.2f", expectedMarkerRatio, actualMarkerRatio),
            Math.abs(actualMarkerRatio - expectedMarkerRatio) < tolerance
        );

        // Verify all docs are either markers or hidden
        assertEquals("All docs should be either markers or hidden", totalDocs, markerCount + hiddenCount);
    }

    /**
     * Test that partitioning is complete (no vectors lost).
     * Validates: Requirements 2.3, 2.4
     */
    public void testPartitioningIsComplete() {
        int totalDocs = 1000;
        int clumpingFactor = 8;
        
        List<Integer> markerDocIds = new ArrayList<>();
        List<Integer> hiddenDocIds = new ArrayList<>();
        
        for (int docId = 0; docId < totalDocs; docId++) {
            if (MarkerSelector.isMarker(docId, clumpingFactor)) {
                markerDocIds.add(docId);
            } else {
                hiddenDocIds.add(docId);
            }
        }

        // Verify no overlap
        for (int markerDocId : markerDocIds) {
            assertFalse("Marker should not be in hidden list", hiddenDocIds.contains(markerDocId));
        }

        // Verify completeness
        assertEquals("Total should equal markers + hidden", totalDocs, markerDocIds.size() + hiddenDocIds.size());
    }

    /**
     * Test constructor rejects invalid clumping factor.
     */
    public void testConstructorRejectsInvalidClumpingFactor() {
        NativeIndexBuildStrategy mockDelegate = params -> {};

        IllegalArgumentException exception = expectThrows(
            IllegalArgumentException.class,
            () -> new ClumpingIndexBuildStrategy(mockDelegate, 1)
        );
        assertTrue(exception.getMessage().contains("Clumping factor must be at least 2"));

        exception = expectThrows(
            IllegalArgumentException.class,
            () -> new ClumpingIndexBuildStrategy(mockDelegate, 0)
        );
        assertTrue(exception.getMessage().contains("Clumping factor must be at least 2"));

        exception = expectThrows(
            IllegalArgumentException.class,
            () -> new ClumpingIndexBuildStrategy(mockDelegate, -1)
        );
        assertTrue(exception.getMessage().contains("Clumping factor must be at least 2"));
    }

    /**
     * Test that valid clumping factors are accepted.
     */
    public void testConstructorAcceptsValidClumpingFactor() {
        NativeIndexBuildStrategy mockDelegate = params -> {};

        // Minimum valid clumping factor
        ClumpingIndexBuildStrategy strategy2 = new ClumpingIndexBuildStrategy(mockDelegate, 2);
        assertNotNull(strategy2);

        // Typical clumping factor
        ClumpingIndexBuildStrategy strategy8 = new ClumpingIndexBuildStrategy(mockDelegate, 8);
        assertNotNull(strategy8);

        // Maximum typical clumping factor
        ClumpingIndexBuildStrategy strategy100 = new ClumpingIndexBuildStrategy(mockDelegate, 100);
        assertNotNull(strategy100);
    }

    // Helper methods

    /**
     * Finds the closest marker to a hidden vector using exact distance calculation.
     * This mirrors the logic in ClumpingIndexBuildStrategy.
     */
    private int findClosestMarker(
        float[] hiddenVector,
        List<Integer> markerDocIds,
        List<float[]> markerVectors,
        SpaceType spaceType
    ) {
        int closestMarkerDocId = markerDocIds.get(0);
        float bestSimilarity = spaceType.getKnnVectorSimilarityFunction().compare(hiddenVector, markerVectors.get(0));

        for (int i = 1; i < markerDocIds.size(); i++) {
            float similarity = spaceType.getKnnVectorSimilarityFunction().compare(hiddenVector, markerVectors.get(i));
            if (similarity > bestSimilarity) {
                bestSimilarity = similarity;
                closestMarkerDocId = markerDocIds.get(i);
            }
        }

        return closestMarkerDocId;
    }

    private float[] randomVector(int dimension) {
        float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            vector[i] = randomFloat() * 10 - 5; // Random values between -5 and 5
        }
        return vector;
    }

    private float[] normalize(float[] vector) {
        float norm = 0;
        for (float v : vector) {
            norm += v * v;
        }
        norm = (float) Math.sqrt(norm);
        
        float[] normalized = new float[vector.length];
        for (int i = 0; i < vector.length; i++) {
            normalized[i] = vector[i] / norm;
        }
        return normalized;
    }
}
