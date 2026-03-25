/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.store.IndexOutput;

import java.io.IOException;
import java.util.Arrays;

/**
 * Writes reordered field metadata and doc-to-ord skip list into an {@link IndexOutput}.
 * The format matches what {@code ReorderedLucene99FlatVectorsReader111.FieldEntry.create()} reads.
 */
public class ReorderedFieldMetaWriter {

    /**
     * Write reordered field metadata + skip list into the meta IndexOutput.
     *
     * @param meta              the .vemf IndexOutput to write into
     * @param fieldInfo         field being written
     * @param vectorDataOffset  byte offset of this field's vectors in .vec
     * @param vectorDataLength  byte length of this field's vectors in .vec
     * @param mergedOrdToDocId  mergedOrdToDocId[mergedOrd] = merged doc ID
     * @param permutation       permutation[newOrd] = oldMergedOrd (from VectorReorderStrategy)
     */
    public static void writeReorderedMeta(
        IndexOutput meta,
        FieldInfo fieldInfo,
        long vectorDataOffset,
        long vectorDataLength,
        int[] mergedOrdToDocId,
        int[] permutation
    ) throws IOException {
        int n = permutation.length;

        // Field metadata (matches FieldEntry.create() read order)
        meta.writeInt(fieldInfo.number);
        meta.writeInt(fieldInfo.getVectorEncoding().ordinal());
        meta.writeInt(fieldInfo.getVectorSimilarityFunction().ordinal());
        meta.writeVLong(vectorDataOffset);
        meta.writeVLong(vectorDataLength);
        meta.writeVInt(fieldInfo.getVectorDimension());

        // Build oldOrd2New from permutation (newOrd2Old)
        int[] oldOrd2New = new int[n];
        for (int newOrd = 0; newOrd < n; newOrd++) {
            oldOrd2New[permutation[newOrd]] = newOrd;
        }

        // Build (docId, reorderedOrd) pairs sorted by docId ascending
        long[] docAndOrds = new long[n];
        int maxDoc = 0;
        for (int mergedOrd = 0; mergedOrd < n; mergedOrd++) {
            int docId = mergedOrdToDocId[mergedOrd];
            int reorderedOrd = oldOrd2New[mergedOrd];
            docAndOrds[mergedOrd] = ((long) docId << 32) | (reorderedOrd & 0xFFFFFFFFL);
            if (docId > maxDoc) maxDoc = docId;
        }
        Arrays.sort(docAndOrds);

        boolean isDense = (n == maxDoc + 1);

        // Skip list header (matches reader)
        meta.writeByte(isDense ? (byte) 1 : (byte) 0);
        meta.writeInt(maxDoc);       // maxDoc
        meta.writeInt(4);            // numLevel
        meta.writeInt(256);          // numDocsForGrouping
        meta.writeInt(4);            // groupFactor

        // Skip list body (doc→ord, used by search path)
        // For sparse, fill sentinel for docs without vectors
        FixedBlockSkipListIndexBuilder skipListBuilder = new FixedBlockSkipListIndexBuilder(meta, maxDoc);
        int numBytesPerValue = Integer.BYTES - (Integer.numberOfLeadingZeros(maxDoc) / Byte.SIZE);
        int sentinel = (1 << (8 * numBytesPerValue)) - 1;
        int vecIdx = 0;
        for (int doc = 0; doc <= maxDoc; doc++) {
            if (vecIdx < n && (int)(docAndOrds[vecIdx] >>> 32) == doc) {
                skipListBuilder.add(doc, (int) docAndOrds[vecIdx]);
                vecIdx++;
            } else {
                skipListBuilder.add(doc, sentinel);
            }
        }
        skipListBuilder.finish();

        // Ord→doc array (used by merge path for ordToDoc())
        // ordToDoc[newOrd] = docId for the vector at position newOrd in .vec
        meta.writeInt(n);
        for (int newOrd = 0; newOrd < n; newOrd++) {
            meta.writeInt(mergedOrdToDocId[permutation[newOrd]]);
        }
    }
}
