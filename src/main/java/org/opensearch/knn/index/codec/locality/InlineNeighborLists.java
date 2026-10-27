/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.locality;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.IOSupplier;
import org.opensearch.knn.index.codec.scorer.PrefetchHelper;

import java.io.IOException;
import java.io.UncheckedIOException;

/**
 * Reads the layer-0 neighbor lists stored in the locality-ordered records (see
 * {@link LocalityOrderedQuantizedVectorsWriter}). A node's record is usually read to score it before the node is
 * expanded, so its neighbor list is then already in the page cache and needs no read from {@code .faiss}.
 * <p>
 * Neighbors are stored as physical positions and returned as original ordinals, the graph's ordinal space.
 * Not thread safe: {@link #clone()} per search, which also picks the search's I/O backend.
 */
public final class InlineNeighborLists implements Cloneable {
    private final IOSupplier<IndexInput> recordsForSearch;
    private final IndexInput records;
    private final LocalityOrderedQuantizedVectorsWriter.RecordLayout layout;
    private final int listOffset;
    private final int[] ordToPhysicalOrdMap;
    private final int[] physicalToOrdMap;
    private final int[] list;

    InlineNeighborLists(
        final IOSupplier<IndexInput> recordsForSearch,
        final LocalityOrderedQuantizedVectorsWriter.RecordLayout layout,
        final int codeLength,
        final int neighborSlots,
        final int[] ordToPhysicalOrdMap,
        final int[] physicalToOrdMap
    ) {
        this.recordsForSearch = recordsForSearch;
        try {
            this.records = recordsForSearch.get();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        this.layout = layout;
        this.listOffset = codeLength + LocalityOrderedQuantizedVectorsWriter.CORRECTION_BYTES;
        this.ordToPhysicalOrdMap = ordToPhysicalOrdMap;
        this.physicalToOrdMap = physicalToOrdMap;
        this.list = new int[neighborSlots];
    }

    /** @return the maximum number of neighbors a node has at layer 0 */
    public int maxNeighbors() {
        return list.length;
    }

    /**
     * Reads the layer-0 neighbors of {@code ord} into {@code neighbors}, which must hold {@link #maxNeighbors()}.
     *
     * @return the number of neighbors
     */
    public int read(final int ord, final int[] neighbors) throws IOException {
        records.seek(recordOffset(ord) + listOffset);
        records.readInts(list, 0, list.length);
        int count = 0;
        while (count < list.length && list[count] >= 0) {
            neighbors[count] = physicalToOrdMap[list[count]];
            count++;
        }
        return count;
    }

    /** Prefetches the records holding the neighbor lists of {@code ords[0..numOrds)}. */
    public void prefetch(final int[] ords, final int numOrds) throws IOException {
        final long[] starts = new long[numOrds];
        for (int i = 0; i < numOrds; i++) {
            starts[i] = recordOffset(ords[i]);
        }
        PrefetchHelper.prefetchRanges(records, starts, numOrds, layout.recordSize(), PrefetchHelper.Grouping.BLOCK_32KB);
    }

    private long recordOffset(final int ord) {
        return layout.offset(ordToPhysicalOrdMap[ord]);
    }

    @Override
    public InlineNeighborLists clone() {
        return new InlineNeighborLists(
            recordsForSearch,
            layout,
            listOffset - LocalityOrderedQuantizedVectorsWriter.CORRECTION_BYTES,
            list.length,
            ordToPhysicalOrdMap,
            physicalToOrdMap
        );
    }
}
