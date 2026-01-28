/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clumping;

import org.opensearch.test.OpenSearchTestCase;

/**
 * Unit tests for {@link MarkerSelector}.
 */
public class MarkerSelectorTests extends OpenSearchTestCase {

    /**
     * Test that isMarker returns consistent results for the same inputs.
     * This validates the determinism requirement (Requirement 2.2).
     */
    public void testIsMarkerDeterministic() {
        int clumpingFactor = 8;
        
        // Test multiple document IDs
        for (int docId = 0; docId < 1000; docId++) {
            boolean firstResult = MarkerSelector.isMarker(docId, clumpingFactor);
            boolean secondResult = MarkerSelector.isMarker(docId, clumpingFactor);
            
            assertEquals(
                "isMarker should return the same result for the same docId and clumpingFactor",
                firstResult,
                secondResult
            );
        }
    }

    /**
     * Test that marker selection produces approximately 1/clumpingFactor markers.
     * This validates the selection ratio requirement (Requirement 2.1).
     */
    public void testMarkerSelectionRatio() {
        int clumpingFactor = 8;
        int totalDocs = 10000;
        int markerCount = 0;
        
        for (int docId = 0; docId < totalDocs; docId++) {
            if (MarkerSelector.isMarker(docId, clumpingFactor)) {
                markerCount++;
            }
        }
        
        double expectedRatio = 1.0 / clumpingFactor;
        double actualRatio = (double) markerCount / totalDocs;
        double tolerance = 0.05; // 5% tolerance
        
        assertTrue(
            String.format(
                "Marker ratio should be approximately %.2f, but was %.2f (markers: %d, total: %d)",
                expectedRatio, actualRatio, markerCount, totalDocs
            ),
            Math.abs(actualRatio - expectedRatio) < tolerance
        );
    }

    /**
     * Test marker selection with different clumping factors.
     */
    public void testMarkerSelectionWithDifferentFactors() {
        int totalDocs = 10000;
        int[] clumpingFactors = { 2, 4, 8, 16, 32, 50, 100 };
        
        for (int clumpingFactor : clumpingFactors) {
            int markerCount = 0;
            
            for (int docId = 0; docId < totalDocs; docId++) {
                if (MarkerSelector.isMarker(docId, clumpingFactor)) {
                    markerCount++;
                }
            }
            
            double expectedRatio = 1.0 / clumpingFactor;
            double actualRatio = (double) markerCount / totalDocs;
            double tolerance = 0.10; // 10% tolerance for smaller sample sizes
            
            assertTrue(
                String.format(
                    "For clumpingFactor=%d: marker ratio should be approximately %.2f, but was %.2f",
                    clumpingFactor, expectedRatio, actualRatio
                ),
                Math.abs(actualRatio - expectedRatio) < tolerance
            );
        }
    }

    /**
     * Test that clumping factor less than 2 throws an exception.
     */
    public void testInvalidClumpingFactorThrowsException() {
        IllegalArgumentException exception = expectThrows(
            IllegalArgumentException.class,
            () -> MarkerSelector.isMarker(0, 1)
        );
        assertTrue(exception.getMessage().contains("Clumping factor must be at least 2"));
        
        exception = expectThrows(
            IllegalArgumentException.class,
            () -> MarkerSelector.isMarker(0, 0)
        );
        assertTrue(exception.getMessage().contains("Clumping factor must be at least 2"));
        
        exception = expectThrows(
            IllegalArgumentException.class,
            () -> MarkerSelector.isMarker(0, -1)
        );
        assertTrue(exception.getMessage().contains("Clumping factor must be at least 2"));
    }

    /**
     * Test that the minimum valid clumping factor (2) works correctly.
     */
    public void testMinimumClumpingFactor() {
        int clumpingFactor = 2;
        int totalDocs = 10000;
        int markerCount = 0;
        
        for (int docId = 0; docId < totalDocs; docId++) {
            if (MarkerSelector.isMarker(docId, clumpingFactor)) {
                markerCount++;
            }
        }
        
        // With clumping factor 2, approximately half should be markers
        double expectedRatio = 0.5;
        double actualRatio = (double) markerCount / totalDocs;
        double tolerance = 0.05;
        
        assertTrue(
            String.format(
                "With clumpingFactor=2, marker ratio should be approximately 0.5, but was %.2f",
                actualRatio
            ),
            Math.abs(actualRatio - expectedRatio) < tolerance
        );
    }

    /**
     * Test that the maximum valid clumping factor (100) works correctly.
     */
    public void testMaximumClumpingFactor() {
        int clumpingFactor = 100;
        int totalDocs = 100000; // Need more docs for accurate ratio with high clumping factor
        int markerCount = 0;
        
        for (int docId = 0; docId < totalDocs; docId++) {
            if (MarkerSelector.isMarker(docId, clumpingFactor)) {
                markerCount++;
            }
        }
        
        double expectedRatio = 0.01; // 1/100
        double actualRatio = (double) markerCount / totalDocs;
        double tolerance = 0.005; // 0.5% tolerance
        
        assertTrue(
            String.format(
                "With clumpingFactor=100, marker ratio should be approximately 0.01, but was %.4f",
                actualRatio
            ),
            Math.abs(actualRatio - expectedRatio) < tolerance
        );
    }

    /**
     * Test that negative document IDs are handled correctly.
     * While Lucene doc IDs are typically non-negative, the hash function should handle any int.
     */
    public void testNegativeDocIds() {
        int clumpingFactor = 8;
        
        // Should not throw exception
        boolean result1 = MarkerSelector.isMarker(-1, clumpingFactor);
        boolean result2 = MarkerSelector.isMarker(Integer.MIN_VALUE, clumpingFactor);
        
        // Results should be deterministic
        assertEquals(result1, MarkerSelector.isMarker(-1, clumpingFactor));
        assertEquals(result2, MarkerSelector.isMarker(Integer.MIN_VALUE, clumpingFactor));
    }

    /**
     * Test that large document IDs are handled correctly.
     */
    public void testLargeDocIds() {
        int clumpingFactor = 8;
        
        // Should not throw exception
        boolean result1 = MarkerSelector.isMarker(Integer.MAX_VALUE, clumpingFactor);
        boolean result2 = MarkerSelector.isMarker(Integer.MAX_VALUE - 1, clumpingFactor);
        
        // Results should be deterministic
        assertEquals(result1, MarkerSelector.isMarker(Integer.MAX_VALUE, clumpingFactor));
        assertEquals(result2, MarkerSelector.isMarker(Integer.MAX_VALUE - 1, clumpingFactor));
    }

    /**
     * Test that the hash function provides good distribution.
     * Sequential doc IDs should not produce sequential marker selections.
     */
    public void testHashDistribution() {
        int clumpingFactor = 8;
        int windowSize = 100;
        int markerCount = 0;
        
        // Count markers in a small window of sequential doc IDs
        for (int docId = 0; docId < windowSize; docId++) {
            if (MarkerSelector.isMarker(docId, clumpingFactor)) {
                markerCount++;
            }
        }
        
        // Even in a small window, we should see some markers
        assertTrue("Should have at least some markers in a window of " + windowSize, markerCount > 0);
        
        // And not all should be markers
        assertTrue("Should have at least some hidden vectors in a window of " + windowSize, markerCount < windowSize);
    }
}
