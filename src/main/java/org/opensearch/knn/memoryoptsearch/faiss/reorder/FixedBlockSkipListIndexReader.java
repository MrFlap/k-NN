/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.store.IndexInput;

import java.io.IOException;

public class FixedBlockSkipListIndexReader {
    private final long[] blocks;
    public final int maxDoc;
    private final int numBytesPerValue;
    private int doc;
    private int mask;

    public FixedBlockSkipListIndexReader(final IndexInput metaInput, int maxDoc) throws IOException {
        final long numBytes = Integer.BYTES - (Integer.numberOfLeadingZeros(maxDoc) / Byte.SIZE);
        this.maxDoc = metaInput.readInt();
        this.numBytesPerValue = metaInput.readInt();

        final long tmp = (numBytes * (maxDoc + 1));
        long blockSizes = (tmp / Long.BYTES) + ((tmp % Long.BYTES) > 0 ? 1 : 0);
        this.blocks = new long[Math.toIntExact(blockSizes)];
        metaInput.readLongs(blocks, 0, blocks.length);
        mask = (1 << (8 * numBytesPerValue)) - 1;
    }

    public int skipTo(int doc) {
        return this.doc = doc;
    }

    public int getOrd() {
        final long bitPos = doc * 8L * numBytesPerValue;
        final int word = (int) (bitPos >>> 6);
        final int relaBitPos = (int) (bitPos % 64);
        final int endBitPos = relaBitPos + 8 * numBytesPerValue;
        if (endBitPos <= 64) {
            return (int) ((blocks[word] >>> relaBitPos)) & mask;
        } else {
            final long upper = blocks[word] >>> relaBitPos;
            final long lower = (blocks[word + 1] << (64 - relaBitPos)) & mask;
            return (int) (upper | lower);
        }
    }
}
