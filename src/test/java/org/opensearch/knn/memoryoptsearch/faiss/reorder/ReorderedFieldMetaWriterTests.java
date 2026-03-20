/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.Collections;

public class ReorderedFieldMetaWriterTests extends OpenSearchTestCase {

    private FieldInfo makeFieldInfo(int dim) {
        return new FieldInfo(
            "test_field", 0, false, false, false,
            IndexOptions.NONE, DocValuesType.NONE, DocValuesSkipIndexType.NONE,
            -1, Collections.emptyMap(), 0, 0, 0,
            dim, VectorEncoding.FLOAT32, VectorSimilarityFunction.EUCLIDEAN,
            false, false
        );
    }

    /**
     * 3 vectors, permutation=[2,0,1], dense docIds=[0,1,2].
     * oldOrd2New = [1, 2, 0]
     * doc 0 → mergedOrd 0 → reorderedOrd 1
     * doc 1 → mergedOrd 1 → reorderedOrd 2
     * doc 2 → mergedOrd 2 → reorderedOrd 0
     */
    public void testSimplePermutation() throws IOException {
        int[] permutation = { 2, 0, 1 };
        int[] mergedOrdToDocId = { 0, 1, 2 };

        verifySkipList(permutation, mergedOrdToDocId, 3, new int[] { 1, 2, 0 });
    }

    /**
     * Identity permutation: skipTo(d).getOrd() == d for all d.
     */
    public void testIdentityPermutation() throws IOException {
        int n = 100;
        int[] permutation = new int[n];
        int[] mergedOrdToDocId = new int[n];
        int[] expectedOrds = new int[n];
        for (int i = 0; i < n; i++) {
            permutation[i] = i;
            mergedOrdToDocId[i] = i;
            expectedOrds[i] = i;
        }

        verifySkipList(permutation, mergedOrdToDocId, n, expectedOrds);
    }

    /**
     * Reverse permutation: skipTo(d).getOrd() == N-1-d.
     */
    public void testReversePermutation() throws IOException {
        int n = 50;
        int[] permutation = new int[n];
        int[] mergedOrdToDocId = new int[n];
        int[] expectedOrds = new int[n];
        for (int i = 0; i < n; i++) {
            permutation[i] = n - 1 - i;
            mergedOrdToDocId[i] = i;
            expectedOrds[i] = n - 1 - i;  // oldOrd2New[i] = n-1-i
        }

        verifySkipList(permutation, mergedOrdToDocId, n, expectedOrds);
    }

    /**
     * Write metadata + skip list, then read back and verify doc→ord mapping.
     */
    private void verifySkipList(int[] permutation, int[] mergedOrdToDocId, int dim, int[] expectedOrdPerDoc) throws IOException {
        FieldInfo fieldInfo = makeFieldInfo(dim);

        try (ByteBuffersDirectory dir = new ByteBuffersDirectory()) {
            // Write
            try (IndexOutput out = dir.createOutput("test.vemf", IOContext.DEFAULT)) {
                ReorderedFieldMetaWriter.writeReorderedMeta(out, fieldInfo, 0L, 100L, mergedOrdToDocId, permutation);
            }

            // Read back — skip past field metadata to reach skip list
            try (IndexInput in = dir.openInput("test.vemf", IOContext.DEFAULT)) {
                // Skip field metadata: fieldNumber(int) + encoding(int) + similarity(int) + offset(vlong) + length(vlong) + dim(vint)
                in.readInt();   // fieldNumber
                in.readInt();   // encoding
                in.readInt();   // similarity
                in.readVLong(); // vectorDataOffset
                in.readVLong(); // vectorDataLength
                in.readVInt();  // dimension

                // Skip list header
                in.readByte();  // isDense
                int maxDoc = in.readInt();
                in.readInt();   // numLevel
                in.readInt();   // numDocsForGrouping
                in.readInt();   // groupFactor

                // Read skip list
                FixedBlockSkipListIndexReader reader = new FixedBlockSkipListIndexReader(in, maxDoc);

                // Verify each doc maps to expected ord
                for (int docId = 0; docId < expectedOrdPerDoc.length; docId++) {
                    reader.skipTo(docId);
                    assertEquals("Wrong ord for doc " + docId, expectedOrdPerDoc[docId], reader.getOrd());
                }
            }
        }
    }
}
