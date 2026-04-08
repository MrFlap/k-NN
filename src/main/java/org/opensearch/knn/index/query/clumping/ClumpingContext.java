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
 * Context for clumping-based search. Clumping inserts every nth vector as a "marker" into the
 * native index and stores the remaining "hidden" vectors in a sidecar .clump file. During search,
 * only marker vectors are searched; results are then expanded to include hidden vectors associated
 * with each marker, and the final top-k is selected from the combined set.
 */
@Getter
@AllArgsConstructor
@Builder
@EqualsAndHashCode
public final class ClumpingContext {

    public static final int DEFAULT_CLUMPING_FACTOR = 8;
    public static final int MIN_CLUMPING_FACTOR = 2;
    public static final int MAX_CLUMPING_FACTOR = 1024;

    /**
     * The clumping factor n. Every nth vector (starting from 0) is a marker vector.
     * The remaining vectors are hidden and associated with their nearest marker.
     */
    @Builder.Default
    private int clumpingFactor = DEFAULT_CLUMPING_FACTOR;

    /**
     * Whether clumping is enabled.
     */
    @Builder.Default
    private boolean enabled = true;

    public static ClumpingContext getDefault() {
        return ClumpingContext.builder().clumpingFactor(DEFAULT_CLUMPING_FACTOR).build();
    }
}
