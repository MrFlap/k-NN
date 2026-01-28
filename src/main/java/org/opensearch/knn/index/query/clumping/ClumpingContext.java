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
 * Configuration object for clumping parameters, similar to RescoreContext.
 * Clumping is an optimization technique that reduces the size of the main k-NN index
 * by only indexing a subset of vectors (marker vectors) while storing the remaining
 * vectors (hidden vectors) in a separate file on disk.
 */
@Getter
@AllArgsConstructor
@Builder
@EqualsAndHashCode
public final class ClumpingContext {

    /**
     * Default clumping factor - determines the ratio of total vectors to marker vectors.
     * A clumping factor of 8 means approximately 1/8 vectors become markers.
     */
    public static final int DEFAULT_CLUMPING_FACTOR = 8;

    /**
     * Minimum allowed clumping factor. Must be at least 2 to have any effect.
     */
    public static final int MIN_CLUMPING_FACTOR = 2;

    /**
     * Maximum allowed clumping factor.
     */
    public static final int MAX_CLUMPING_FACTOR = 100;

    /**
     * Default expansion factor for first-pass search.
     * The first pass retrieves k * expansionFactor results from marker vectors.
     */
    public static final float DEFAULT_EXPANSION_FACTOR = 2.0f;

    /**
     * Sentinel value to indicate that clumping has been explicitly disabled by the user.
     * This is different from null (which means use default behavior).
     */
    public static final ClumpingContext EXPLICITLY_DISABLED_CLUMPING_CONTEXT = ClumpingContext.builder().enabled(false).build();

    /**
     * The clumping factor that determines the ratio of total vectors to marker vectors.
     * A clumping factor of N means approximately 1/N vectors become markers.
     */
    @Builder.Default
    private int clumpingFactor = DEFAULT_CLUMPING_FACTOR;

    /**
     * The expansion factor used to determine how many marker results to retrieve
     * in the first pass of search. The first pass retrieves k * expansionFactor results.
     */
    @Builder.Default
    private float expansionFactor = DEFAULT_EXPANSION_FACTOR;

    /**
     * Flag indicating whether clumping is enabled for this query.
     * When false, hidden vector expansion is skipped and only marker results are returned.
     */
    @Builder.Default
    private boolean enabled = true;

    /**
     * Calculates the number of results to retrieve in the first pass of clumping search.
     * This is the number of marker vectors to retrieve before expanding to include hidden vectors.
     *
     * @param finalK The final number of results desired after rescoring.
     * @return The number of marker results to retrieve in the first pass.
     */
    public int getFirstPassK(int finalK) {
        return (int) Math.ceil(finalK * expansionFactor);
    }

    /**
     * Returns a default ClumpingContext with default values for all parameters.
     *
     * @return A new ClumpingContext with default settings.
     */
    public static ClumpingContext getDefault() {
        return ClumpingContext.builder().build();
    }
}
