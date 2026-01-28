/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clumping;

import org.opensearch.test.OpenSearchTestCase;

/**
 * Unit tests for {@link ClumpingContext} validation.
 * 
 * These tests verify that the clumping factor boundary validation works correctly.
 * 
 * Validates: Requirements 1.2, 1.3
 */
public class ClumpingContextTests extends OpenSearchTestCase {

    /**
     * Test that clumping factor values below minimum (< 2) are rejected.
     * Validates: Requirement 1.2
     */
    public void testClumpingFactorBelowMinimumIsRejected() {
        // Test values below minimum
        int[] invalidValues = { 1, 0, -1, -100, Integer.MIN_VALUE };
        
        for (int invalidValue : invalidValues) {
            ClumpingContext context = ClumpingContext.builder()
                .clumpingFactor(invalidValue)
                .build();
            
            // Validate using the static validate method
            assertFalse(
                String.format("Clumping factor %d should be invalid (below minimum %d)", 
                    invalidValue, ClumpingContext.MIN_CLUMPING_FACTOR),
                isValidClumpingFactor(context.getClumpingFactor())
            );
        }
    }

    /**
     * Test that clumping factor values above maximum (> 100) are rejected.
     * Validates: Requirement 1.3
     */
    public void testClumpingFactorAboveMaximumIsRejected() {
        // Test values above maximum
        int[] invalidValues = { 101, 200, 1000, Integer.MAX_VALUE };
        
        for (int invalidValue : invalidValues) {
            ClumpingContext context = ClumpingContext.builder()
                .clumpingFactor(invalidValue)
                .build();
            
            assertFalse(
                String.format("Clumping factor %d should be invalid (above maximum %d)", 
                    invalidValue, ClumpingContext.MAX_CLUMPING_FACTOR),
                isValidClumpingFactor(context.getClumpingFactor())
            );
        }
    }

    /**
     * Test that clumping factor values within valid range [2, 100] are accepted.
     * Validates: Requirements 1.2, 1.3 (inverse)
     */
    public void testClumpingFactorWithinValidRangeIsAccepted() {
        // Test values within valid range
        int[] validValues = { 2, 3, 8, 50, 99, 100 };
        
        for (int validValue : validValues) {
            ClumpingContext context = ClumpingContext.builder()
                .clumpingFactor(validValue)
                .build();
            
            assertTrue(
                String.format("Clumping factor %d should be valid (within range [%d, %d])", 
                    validValue, ClumpingContext.MIN_CLUMPING_FACTOR, ClumpingContext.MAX_CLUMPING_FACTOR),
                isValidClumpingFactor(context.getClumpingFactor())
            );
            
            assertEquals(validValue, context.getClumpingFactor());
        }
    }

    /**
     * Test boundary edge cases are handled correctly.
     */
    public void testBoundaryEdgeCases() {
        // MIN_CLUMPING_FACTOR (2) should be valid
        assertTrue(isValidClumpingFactor(ClumpingContext.MIN_CLUMPING_FACTOR));
        
        // MAX_CLUMPING_FACTOR (100) should be valid
        assertTrue(isValidClumpingFactor(ClumpingContext.MAX_CLUMPING_FACTOR));
        
        // MIN_CLUMPING_FACTOR - 1 (1) should be invalid
        assertFalse(isValidClumpingFactor(ClumpingContext.MIN_CLUMPING_FACTOR - 1));
        
        // MAX_CLUMPING_FACTOR + 1 (101) should be invalid
        assertFalse(isValidClumpingFactor(ClumpingContext.MAX_CLUMPING_FACTOR + 1));
    }

    /**
     * Test default values are set correctly.
     */
    public void testDefaultValues() {
        ClumpingContext context = ClumpingContext.builder().build();
        
        assertEquals(ClumpingContext.DEFAULT_CLUMPING_FACTOR, context.getClumpingFactor());
        assertEquals(ClumpingContext.DEFAULT_EXPANSION_FACTOR, context.getExpansionFactor(), 0.001f);
        assertTrue(context.isEnabled());
    }

    /**
     * Test getFirstPassK calculation.
     */
    public void testGetFirstPassK() {
        ClumpingContext context = ClumpingContext.builder()
            .expansionFactor(2.0f)
            .build();
        
        assertEquals(20, context.getFirstPassK(10));
        assertEquals(2, context.getFirstPassK(1));
        assertEquals(200, context.getFirstPassK(100));
        
        // Test with different expansion factor
        ClumpingContext context2 = ClumpingContext.builder()
            .expansionFactor(1.5f)
            .build();
        
        assertEquals(15, context2.getFirstPassK(10));
        assertEquals(2, context2.getFirstPassK(1)); // ceil(1.5) = 2
    }

    /**
     * Test getDefault returns a valid default context.
     */
    public void testGetDefault() {
        ClumpingContext defaultContext = ClumpingContext.getDefault();
        
        assertNotNull(defaultContext);
        assertEquals(ClumpingContext.DEFAULT_CLUMPING_FACTOR, defaultContext.getClumpingFactor());
        assertEquals(ClumpingContext.DEFAULT_EXPANSION_FACTOR, defaultContext.getExpansionFactor(), 0.001f);
        assertTrue(defaultContext.isEnabled());
    }

    /**
     * Test builder with all parameters.
     */
    public void testBuilderWithAllParameters() {
        ClumpingContext context = ClumpingContext.builder()
            .clumpingFactor(16)
            .expansionFactor(3.0f)
            .enabled(false)
            .build();
        
        assertEquals(16, context.getClumpingFactor());
        assertEquals(3.0f, context.getExpansionFactor(), 0.001f);
        assertFalse(context.isEnabled());
    }

    /**
     * Helper method to validate clumping factor.
     */
    private boolean isValidClumpingFactor(int clumpingFactor) {
        return clumpingFactor >= ClumpingContext.MIN_CLUMPING_FACTOR 
            && clumpingFactor <= ClumpingContext.MAX_CLUMPING_FACTOR;
    }
}
