package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import lombok.Getter;
import org.apache.lucene.store.IndexInput;
import org.opensearch.common.lucene.store.ByteArrayIndexInput;

import java.io.IOException;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

public class DocIdOrdSkipListIndex {
    private final int numDocsForGrouping;
    private final int groupFactor;
    private long level0StartOffset;
    @Getter
    private final int maxDoc;
    private final byte[] skipListBytes;
    private final byte[] compressedOrdBytes;
    private final int[] skipListOffsetPerLevels;
    private final int[] skipListSizes;
    private final int maxLevel;

    public DocIdOrdSkipListIndex(
        final IndexInput metaInput,
        boolean isDense,
        int numLevel,  // Ex: if numLevel=4, then there's level-0 (which has ords), level-1, level-2 and level3.
        int numDocsForGrouping,
        int groupFactor,
        long[] offsets,
        int maxDoc
    ) throws IOException {
        this.numDocsForGrouping = numDocsForGrouping;
        this.groupFactor = groupFactor;
        this.maxDoc = maxDoc;

        // Load ord blocks
        // TODO : Adding extra 10 bytes for bit unpacking now, but this needs more robus solution.
        level0StartOffset = metaInput.getFilePointer();
        final long skipListStartOffset = offsets[1];
        compressedOrdBytes = new byte[Math.toIntExact(skipListStartOffset + 10)];
        metaInput.readBytes(compressedOrdBytes, 0, compressedOrdBytes.length - 10);

        // Load non-leaf level
        int skipListBytesLen = 0;
        int maxLevel = 1;
        for (int level = 1; level < numLevel; ++level) {
            final int skipListSize = Math.toIntExact(offsets[level + 1] - offsets[level]);
            if (skipListSize >= 0) {
                maxLevel = level;
            } else {
                break;
            }
            skipListBytesLen += skipListSize;
        }
        this.maxLevel = maxLevel;
        // 1-based
        skipListOffsetPerLevels = new int[maxLevel + 1];
        skipListSizes = new int[maxLevel + 1];
        skipListBytes = new byte[skipListBytesLen];

        for (int level = 1, offset = 0; level <= maxLevel; ++level) {
            final int skipListSize = Math.toIntExact(offsets[level + 1] - offsets[level]);
            skipListSizes[level] = skipListSize;
            skipListOffsetPerLevels[level] = offset;

            metaInput.readBytes(skipListBytes, offset, skipListSize);
            offset += skipListSize;
        }
    }

    public Reader createReader() {
        return new Reader();
    }

    private class SkipListSkipper {
        public long lowerLevelStartOffset;
        public int numSkippedDocs;
        public int maxDocUpperBound;
        protected ByteArrayIndexInput skipInput;
        protected final int level;
        protected final int numDocsToSkip;
        protected final int readUntil;
        protected boolean isLastBlock;

        private SkipListSkipper(final int level, final int skipListOffset, final int skipListSize) {
            this.skipInput = new ByteArrayIndexInput("SkipInput", skipListBytes, skipListOffset, skipListSize);
            this.level = level;
            // D * G^(level) -> #docs to skip at this level
            // Ex: #docs_in_block=256, group_factor=4 then level-1 = 256 * 4 docs, level-2 = 256 * 4 * 4 docs
            this.numDocsToSkip = numDocsForGrouping * ((int) Math.pow(groupFactor, level));
            this.lowerLevelStartOffset = 0;
            this.numSkippedDocs = 0;
            this.readUntil = skipListSize;
            this.maxDocUpperBound = 0;
            this.isLastBlock = false;
        }

        public boolean skipTo(int doc) {
            if (maxDocUpperBound > 0 && doc < maxDocUpperBound) {
                return false;
            }

            int readCount = 0;
            while (skipInput.getFilePointer() < readUntil) {
                try {
                    lowerLevelStartOffset = skipInput.readVLong();
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                ++readCount;
                maxDocUpperBound += numDocsToSkip;
                numSkippedDocs = maxDocUpperBound - numDocsToSkip;
                isLastBlock = maxDocUpperBound > maxDoc;
                if (doc < maxDocUpperBound) {
                    break;
                }
            }

            return readCount > 0;
        }

        public void prepare(final long lowerLevelStartOffset, final int skippedDoc) {
            try {
                skipInput.seek(lowerLevelStartOffset);
                this.numSkippedDocs = skippedDoc;
                this.maxDocUpperBound = skippedDoc;
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    private class Level1Skipper extends SkipListSkipper {
        private final ByteArrayIndexInput ordsInput;
        private final int maxJumpTableSize = groupFactor - 1;
        private boolean jumpTableLoaded = false;
        private int[] jumpTable;
        private int jumpTableSize;

        private Level1Skipper(final int skipListOffset, final int skipListSize) {
            super(1, skipListOffset, skipListSize);
            ordsInput = new ByteArrayIndexInput("DocId2OrdIndexInput", compressedOrdBytes, 0, compressedOrdBytes.length);
            jumpTable = new int[groupFactor];
            jumpTableSize = 0;
        }

        @Override
        public boolean skipTo(int doc) {
            if (maxDocUpperBound > 0 && doc < maxDocUpperBound) {
                return false;
            }

            while (skipInput.getFilePointer() < readUntil) {
                // Load starting offset
                try {
                    lowerLevelStartOffset = skipInput.readVLong();
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }

                // Doc range update
                maxDocUpperBound += numDocsToSkip;
                numSkippedDocs = maxDocUpperBound - numDocsToSkip;
                isLastBlock = maxDocUpperBound > maxDoc;

                // Jump table is not loaded ye
                jumpTableLoaded = false;

                // Did we find the block?
                if (doc < maxDocUpperBound) {
                    return true;
                }

                // Skip a jump table. We don't skip for the last block.
                if (isLastBlock == false) {
                    try {
                        skipInput.seek(skipInput.getFilePointer() + Short.BYTES * maxJumpTableSize);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                }
            }

            return false;
        }

        @Override
        public void prepare(long lowerLevelStartOffset, int skippedDoc) {
            // No ops
        }

        public int findOrd(int doc) throws IOException {
            // Load jump table
            loadJumpTable();

            // Which leaf block?
            final int leafBlockIndex = getLeafBlockIndex(doc);

            // Get the actual leaf block offset
            final long leafBlockOffset;
            if (leafBlockIndex == 0) {
                leafBlockOffset = lowerLevelStartOffset;
            } else {
                leafBlockOffset = lowerLevelStartOffset + jumpTable[leafBlockIndex - 1];
            }

            // Seek to the leaf block
            ordsInput.seek(leafBlockOffset);

            // Find ord
            final int ordIndex = doc - (numSkippedDocs + numDocsForGrouping * leafBlockIndex);
            return IntValuesBitPackingUtil.getValue(ordsInput, ordIndex, compressedOrdBytes);
        }

        private void loadJumpTable() {
            if (jumpTableLoaded) {
                return;
            }

            try {
                if (skipInput.getFilePointer() == readUntil) {
                    // Edge case: There's only one single remaining block.
                    // In this case, we don't have a jump table.
                    // For example, #docs = 1074 = 256 * 4 + 50. (#docs_in_group = 256, group_factor = 4)
                    // Then, there will be two level-1 blocks, and the last one will not have the jump table.
                    // Why? Since we don't record the starting offset of the first block in the group,
                    // we've already known it as `lowerLevelStartOffset`.
                    jumpTableSize = 0;
                    jumpTableLoaded = true;
                    return;
                }

                // We have more than one block in the group
                final int firstBlockSizeAndNumSubBlocks = skipInput.readShort();

                // It's valid only when isLastBlock == true
                final int jumpTableSizeInLastBlock = firstBlockSizeAndNumSubBlocks & 0b1111;
                jumpTable[0] = firstBlockSizeAndNumSubBlocks >> 4;

                if (isLastBlock == false) {
                    jumpTableSize = groupFactor - 1;
                } else {
                    jumpTableSize = jumpTableSizeInLastBlock;
                }

                for (int i = 1; i < jumpTableSize; ++i) {
                    jumpTable[i] = skipInput.readShort();
                }

                jumpTableLoaded = true;
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        private int getLeafBlockIndex(final int doc) {
            return (doc - numSkippedDocs) / numDocsForGrouping;
        }
    }

    public class Reader {
        private final SkipListSkipper[] skippers;
        private int currDoc = -1;

        public Reader() {
            skippers = new SkipListSkipper[maxLevel + 1];
            skippers[1] = new Level1Skipper(skipListOffsetPerLevels[1], skipListSizes[1]);
            for (int i = 2; i <= maxLevel; ++i) {
                skippers[i] = new SkipListSkipper(i, skipListOffsetPerLevels[i], skipListSizes[i]);
            }
        }

        public int skipTo(final int doc) {
            // All exhausted
            if (doc > maxDoc) {
                return currDoc = NO_MORE_DOCS;
            }
            // Fast track : No need to advance
            if (doc < skippers[1].maxDocUpperBound) {
                return currDoc = doc;
            }

            for (int level = maxLevel; level >= 1; --level) {
                if (skippers[level].skipTo(doc)) {
                    if (level > 1) {
                        skippers[level - 1].prepare(skippers[level].lowerLevelStartOffset, skippers[level].numSkippedDocs);
                    }
                }
            }

            // It's a dense case, we always has `doc`.
            return currDoc = doc;
        }

        public int getOrd() throws IOException {
            final Level1Skipper level1Skipper = (Level1Skipper) skippers[1];
            return level1Skipper.findOrd(currDoc);
        }
    }
}
