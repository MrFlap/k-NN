/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.clumping;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * Holds a hidden vector's doc ID and its vector data for writing into the .clump file.
 */
@Getter
@AllArgsConstructor
public final class HiddenVectorEntry {
    private final int docId;
    /** The vector data — either float[] or byte[]. */
    private final Object vector;
}
