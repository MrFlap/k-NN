/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.locality;

import org.opensearch.knn.KNNTestCase;

public class RecordLayoutTests extends KNNTestCase {

    public void testOf_whenNoNeighbors_thenPackedLikeVersion0() {
        final LocalityOrderedQuantizedVectorsWriter.RecordLayout layout = LocalityOrderedQuantizedVectorsWriter.RecordLayout.of(128, 0);
        assertEquals(144, layout.recordSize());
        for (int p = 0; p < 1000; p++) {
            assertEquals((long) p * 144, layout.offset(p));
        }
        assertEquals(1000L * 144, layout.length(1000));
    }

    public void testOf_whenInlineNeighbors_thenNoRecordCrossesAPage() {
        // 1024-dim 1-bit code (128 B) + corrections + 32 neighbors: 272 B, 15 per page.
        assertLayout(128, 32, 272, 15);
        assertLayout(96, 32, 240, 17); // 768-dim
        assertLayout(2000, 64, 2272, 1); // 16000-dim, M=32
        final LocalityOrderedQuantizedVectorsWriter.RecordLayout big = LocalityOrderedQuantizedVectorsWriter.RecordLayout.of(4096, 32);
        assertEquals(8192, big.pageBytes()); // a record larger than a page starts on its own page
        assertEquals(8192L * 3, big.offset(3));
    }

    private static void assertLayout(final int codeLength, final int slots, final int recordSize, final int perPage) {
        final LocalityOrderedQuantizedVectorsWriter.RecordLayout layout = LocalityOrderedQuantizedVectorsWriter.RecordLayout.of(
            codeLength,
            slots
        );
        assertEquals(recordSize, layout.recordSize());
        assertEquals(perPage, layout.recordsPerPage());
        assertEquals(4096, layout.pageBytes());
        for (int p = 0; p < 10_000; p++) {
            final long start = layout.offset(p);
            assertEquals("record " + p + " crosses a page", start / 4096, (start + recordSize - 1) / 4096);
            if (p > 0) {
                assertTrue(start >= layout.offset(p - 1) + recordSize);
            }
        }
        assertEquals(4096L * ((10_000 + perPage - 1) / perPage), layout.length(10_000));
    }
}
