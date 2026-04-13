/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import lombok.Getter;
import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.store.IndexOutput;

import java.io.IOException;
import java.util.Arrays;

public class DocIdOrdSkipListIndexBuilder {
    // e.g. Dense -> doc[i] must have a vector, sparse -> doc[i] may not have a vector
    private final boolean isDense;
    // Number of level. Must be >= 2
    private final int numLevel;
    // TODO : Remove this and use 256 magic number
    private final int numDocsForGrouping;
    // How many blocks need to create a parent block?
    // Max groupFactor is 15.
    private final int groupFactor;
    // Buffer for doc, ord until the buffer size reaches 256.
    private final int[] docBuffer;
    private int docBufferUpto;
    private final int[] ordBuffer;
    private int ordBufferUpto;
    // SIMD encoding util
    // private Lucene101PForUtil pforUtil;
    // #Flushed blocks. Once the #Flushed block reaches `groupFactor`, it creates a parent block
    private int numFlushedBlocks;
    // Accumulated leaf block sizes for jump table which has the starting offset of each sub-block.
    // For example, leaf block sizes=[5, 5, 10] then `accumulatedLeafBlockSizes` should be [5, 10],
    // meaning the second sub-block's starting offset is 5 and 10 for the third. (Of course, 0 for the first sub-block)
    // Note that we don't save the first starting offset which is known during runtime.
    private int[] accumulatedLeafBlockSizes;
    private int leafBlockSizeUpto;
    private int leafBlockSizeSoFar;
    // Index output
    private IndexOutput indexOutput;
    // Memory buffer for each level > 0
    private ByteBuffersDataOutput[] leafBlockBufferOutPerLevel;
    // Base offset of this skip list index.
    private long level0StartOffset;
    private long[] startOffsetOfGroup;
    @Getter
    private long level0EndOffset;
    private byte[] bitPackingBuffer;

    public DocIdOrdSkipListIndexBuilder(
        final boolean isDense,
        final int numLevel,
        final int numDocsForGrouping,
        final int groupFactor,
        final IndexOutput indexOutput
    ) {
        this.isDense = isDense;
        this.numLevel = numLevel;
        this.numDocsForGrouping = numDocsForGrouping;
        this.groupFactor = groupFactor;
        if (isDense == false) {
            this.docBuffer = new int[numDocsForGrouping];
        } else {
            this.docBuffer = null;
        }
        this.docBufferUpto = 0;

        this.ordBuffer = new int[numDocsForGrouping];
        this.ordBufferUpto = 0;

        // this.pforUtil = new Lucene101PForUtil(new Lucene101ForUtil());

        this.leafBlockBufferOutPerLevel = new ByteBuffersDataOutput[numLevel];
        // We don't need memory buffer for level-0
        for (int i = 1; i < numLevel; i++) {
            leafBlockBufferOutPerLevel[i] = ByteBuffersDataOutput.newResettableInstance();
        }

        this.numFlushedBlocks = 0;

        this.accumulatedLeafBlockSizes = new int[groupFactor];
        this.leafBlockSizeUpto = 0;
        this.leafBlockSizeSoFar = 0;

        this.indexOutput = indexOutput;
        this.level0StartOffset = indexOutput.getFilePointer();
        this.level0EndOffset = 0;

        this.startOffsetOfGroup = new long[numLevel];

        this.bitPackingBuffer = IntValuesBitPackingUtil.allocateBuffer(numDocsForGrouping);
    }

    public void add(int doc, int ord) throws IOException {
        if (isDense == false) {
            docBuffer[docBufferUpto++] = doc;
        }
        ordBuffer[ordBufferUpto++] = ord;

        if (ordBufferUpto == numDocsForGrouping) {
            pushLeafSubBlock(false);
        }
    }

    private void pushLeafSubBlock(final boolean force) throws IOException {
        // Write a leaf sub block
        final long startOffsetOfBlock = indexOutput.getFilePointer();

        if (force == false) {
            // Pack doc ids + ords
            if (isDense == false) {
                // pforUtil.encode(docBuffer, indexOutput);
                IntValuesBitPackingUtil.writePackedInts(docBuffer, bitPackingBuffer, indexOutput);
            }
            // pforUtil.encode(ordBuffer, indexOutput);
            IntValuesBitPackingUtil.writePackedInts(ordBuffer, bitPackingBuffer, indexOutput);
        } else {
            assert ordBufferUpto > 0;
            if (isDense == false) {
                Arrays.fill(docBuffer, ordBufferUpto, docBuffer.length, 0);
                IntValuesBitPackingUtil.writePackedInts(docBuffer, bitPackingBuffer, indexOutput);
            }
            Arrays.fill(ordBuffer, ordBufferUpto, ordBuffer.length, 0);
            IntValuesBitPackingUtil.writePackedInts(ordBuffer, bitPackingBuffer, indexOutput);
        }
        level0EndOffset = indexOutput.getFilePointer();

        // Try to build a jump table
        // TODO : non-dense case
        numFlushedBlocks += 1;
        final int leafBlockSize = Math.toIntExact(indexOutput.getFilePointer() - startOffsetOfBlock);
        if (leafBlockSizeSoFar > 0) {
            accumulatedLeafBlockSizes[leafBlockSizeUpto++] = leafBlockSizeSoFar;
        }
        leafBlockSizeSoFar += leafBlockSize;

        // Time to create parent blocks
        // Recursively add non-leaf level
        addNonLeafLevel(1, force);

        if (numFlushedBlocks > 0 && (numFlushedBlocks % groupFactor) == 0) {
            // Reset tracking variables.
            resetAfterLeafBlock();
            startOffsetOfGroup[0] = indexOutput.getFilePointer() - level0StartOffset;
        }

        // Reset tracking variable at sub-block level.
        resetAfterSubBlock();
    }

    private void addNonLeafLevel(final int level, final boolean force) throws IOException {
        if (level >= numLevel) {
            return;
        }

        if (force == false && (numFlushedBlocks % ((int) Math.pow(groupFactor, level))) != 0) {
            // For example assuming group-factor=4, #flushed blocks 8, then level-1 should be created, but not yet for level-2.
            // It should add 8 more leaf block to have leaf-2 block. Equally, level-3 would need 56 = 64 - 8 more blocks.
            return;
        }

        // Write lower level file offset.
        // Ex: `groupFactor` == 3
        // 0
        // 0 - 0 - 0
        // ^------ staring offset = 100
        leafBlockBufferOutPerLevel[level].writeVLong(startOffsetOfGroup[level - 1]);

        if (level == 1) {
            // Level-1, write a jump table
            for (int i = 0; i < leafBlockSizeUpto; ++i) {
                // Since a single block can have 2048 bytes at the maximum (256 doc_ids + ords)
                // Relative offset should be within 2 bytes range.
                // Technically, we have spare 4 bits. (2048 << 5 == 65536 > short's max number)
                int accumLeafBlockSize = accumulatedLeafBlockSizes[i];
                if (i == 0) {
                    // Save `leafBlockSizeUpto` for the last block.
                    accumLeafBlockSize = ((accumLeafBlockSize << 4) | (force ? leafBlockSizeUpto : 0));
                }
                leafBlockBufferOutPerLevel[1].writeShort((short) accumLeafBlockSize);
            }
        } else {
            startOffsetOfGroup[level - 1] = leafBlockBufferOutPerLevel[level - 1].size();
        }

        addNonLeafLevel(level + 1, force);
    }

    private void resetAfterLeafBlock() {
        leafBlockSizeSoFar = 0;
        leafBlockSizeUpto = 0;
    }

    private void resetAfterSubBlock() {
        docBufferUpto = ordBufferUpto = 0;
    }

    // Returns the starting offsets of each level
    public long[] finish() throws IOException {
        if (ordBufferUpto > 0) {
            pushLeafSubBlock(true);
        } else if ((numFlushedBlocks % groupFactor) > 0) {
            addNonLeafLevel(1, true);
        }

        // Level-0 is already persisted.
        long[] offsets = new long[numLevel + 1];
        offsets[0] = level0StartOffset;

        for (int level = 1; level < numLevel; ++level) {
            // Write the starting offset of each level
            offsets[level] = indexOutput.getFilePointer();
            leafBlockBufferOutPerLevel[level].copyTo(indexOutput);
            leafBlockBufferOutPerLevel[level] = null;
        }
        offsets[numLevel] = indexOutput.getFilePointer();

        return offsets;
    }
}
