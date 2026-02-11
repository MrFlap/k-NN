/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.clumping;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * Lightweight record that maps a spilled hidden vector entry in the temp file
 * to its assigned marker. Holds only the marker index and the byte offset in
 * the temp file — no vector data is retained in heap.
 * <p>
 * Memory cost: 12 bytes per hidden vector (int + long), compared to the previous
 * {@link HiddenVectorEntry} which held the full vector array in heap.
 */
@Getter
@AllArgsConstructor
public final class HiddenEntryLocation {
    /** Index into the markerDocIds list that this hidden vector is assigned to. */
    private final int markerIndex;
    /** Byte offset in the temp spill file where this entry's (docId + vector) data starts. */
    private final long tempFileOffset;
}
