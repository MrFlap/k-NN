/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clumping;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;

/**
 * Context for clumping-based search optimization.
 * 
 * Clumping is a technique where only a fraction (1/clumpingFactor) of vectors are stored
 * as "marker" vectors in the main index. The remaining "hidden" vectors are stored separately
 * on disk and are associated with their nearest marker vector.
 * 
 * During search:
 * 1. k-NN search is performed on marker vectors only
 * 2. For each marker vector found, its associated hidden vectors are retrieved
 * 3. All vectors (markers + hidden) are scored against the query
 * 4. Top-k results are returned from the combined set
 */
@Getter
@AllArgsConstructor
@Builder
@EqualsAndHashCode
public final class ClumpingContext {

    /**
     * Default clumping factor - no clumping (all vectors are markers)
     */
    public static final int DEFAULT_CLUMPING_FACTOR = 1;

    /**
     * Minimum clumping factor
     */
    public static final int MIN_CLUMPING_FACTOR = 1;

    /**
     * Maximum clumping factor - at most 1/100 vectors are markers
     */
    public static final int MAX_CLUMPING_FACTOR = 100;

    /**
     * The clumping factor determines what fraction of vectors are stored as markers.
     * A clumping factor of N means 1/N vectors are markers, and (N-1)/N are hidden.
     * For example, clumpingFactor=8 means 1/8 vectors are markers.
     */
    @Builder.Default
    private int clumpingFactor = DEFAULT_CLUMPING_FACTOR;

    /**
     * Flag to track whether clumping is enabled.
     */
    @Builder.Default
    private boolean clumpingEnabled = false;

    /**
     * Disabled clumping context singleton.
     */
    public static final ClumpingContext DISABLED = ClumpingContext.builder()
        .clumpingFactor(DEFAULT_CLUMPING_FACTOR)
        .clumpingEnabled(false)
        .build();

    /**
     * Creates a default ClumpingContext with clumping disabled.
     * 
     * @return default ClumpingContext
     */
    public static ClumpingContext getDefault() {
        return DISABLED;
    }

    /**
     * Creates a ClumpingContext with the specified clumping factor.
     * 
     * @param clumpingFactor the clumping factor (must be between MIN and MAX)
     * @return ClumpingContext with clumping enabled
     */
    public static ClumpingContext withFactor(int clumpingFactor) {
        if (clumpingFactor < MIN_CLUMPING_FACTOR || clumpingFactor > MAX_CLUMPING_FACTOR) {
            throw new IllegalArgumentException(
                String.format(
                    "Clumping factor must be between %d and %d, got %d",
                    MIN_CLUMPING_FACTOR,
                    MAX_CLUMPING_FACTOR,
                    clumpingFactor
                )
            );
        }
        return ClumpingContext.builder()
            .clumpingFactor(clumpingFactor)
            .clumpingEnabled(clumpingFactor > 1)
            .build();
    }

    /**
     * Checks if clumping is effectively enabled (factor > 1).
     * 
     * @return true if clumping is enabled and factor > 1
     */
    public boolean isEffectivelyEnabled() {
        return clumpingEnabled && clumpingFactor > 1;
    }

    /**
     * Calculates the expected number of marker vectors given a total vector count.
     * 
     * @param totalVectors total number of vectors
     * @return expected number of marker vectors
     */
    public int getExpectedMarkerCount(int totalVectors) {
        if (!isEffectivelyEnabled()) {
            return totalVectors;
        }
        return Math.max(1, (int) Math.ceil((double) totalVectors / clumpingFactor));
    }

    /**
     * Calculates the expected number of hidden vectors given a total vector count.
     * 
     * @param totalVectors total number of vectors
     * @return expected number of hidden vectors
     */
    public int getExpectedHiddenCount(int totalVectors) {
        return totalVectors - getExpectedMarkerCount(totalVectors);
    }
}
