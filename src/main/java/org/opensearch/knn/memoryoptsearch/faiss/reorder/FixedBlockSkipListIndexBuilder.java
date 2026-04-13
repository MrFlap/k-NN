/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.store.IndexOutput;

import java.io.IOException;

public class FixedBlockSkipListIndexBuilder {
    private final IndexOutput indexOutput;
    private final int numBytes;
    private final byte[] buffer;
    private final long startOffset;

    public FixedBlockSkipListIndexBuilder(final IndexOutput indexOutput, final int maxDoc) throws IOException {
        this.indexOutput = indexOutput;

        // Determine bytes
        this.numBytes = Integer.BYTES - (Integer.numberOfLeadingZeros(maxDoc) / Byte.SIZE);

        buffer = new byte[numBytes];

        indexOutput.writeInt(maxDoc);
        indexOutput.writeInt(numBytes);
        startOffset = indexOutput.getFilePointer();
    }

    public void add(int doc, int ord) throws IOException {
        for (int i = 0; i < numBytes; ++i) {
            buffer[i] = (byte) ((ord >>> (8 * i)) & 0xFF);
        }
        indexOutput.writeBytes(buffer, 0, numBytes);
    }

    public void finish() throws IOException {
        final long endOffset = indexOutput.getFilePointer();
        final long writtenBytes = endOffset - startOffset;
        final int padding = Long.BYTES - Math.toIntExact(writtenBytes % Long.BYTES);
        if (padding != Long.BYTES) {
            for (int i = 0 ; i < padding ; ++i) {
                indexOutput.writeByte((byte) 0xFF);
            }
        }
    }
}
