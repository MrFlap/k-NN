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
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

/**
 * Tests for ReorderedFieldMetaWriter round-trip with dense and sparse vectors.
 * Dense tests serve as ground truth; sparse tests use 10x docs with vectors in random 1/10.
 */
public class SparseReorderedFieldMetaWriterTests extends OpenSearchTestCase {

    private static final int DIM = 4;

    private FieldInfo makeFieldInfo() {
        return new FieldInfo(
            "test_field", 0, false, false, false,
            IndexOptions.NONE, DocValuesType.NONE, DocValuesSkipIndexType.NONE,
            -1, Collections.emptyMap(), 0, 0, 0,
            DIM, VectorEncoding.FLOAT32, VectorSimilarityFunction.EUCLIDEAN,
            false, false
        );
    }

    /**
     * Helper: write metadata, read back skip list and ordToDoc array, verify.
     *
     * @param permutation       newOrd2Old permutation
     * @param mergedOrdToDocId  mergedOrd -> docId
     * @param docsWithVectors   set of doc IDs that have vectors (for sparse sentinel check)
     * @param maxDoc            max doc ID
     */
    private void verifyRoundTrip(int[] permutation, int[] mergedOrdToDocId, Set<Integer> docsWithVectors, int maxDoc)
        throws IOException {
        int n = permutation.length;
        FieldInfo fieldInfo = makeFieldInfo();

        // Compute expected: oldOrd2New
        int[] oldOrd2New = new int[n];
        for (int newOrd = 0; newOrd < n; newOrd++) {
            oldOrd2New[permutation[newOrd]] = newOrd;
        }

        try (ByteBuffersDirectory dir = new ByteBuffersDirectory()) {
            try (IndexOutput out = dir.createOutput("test.vemf", IOContext.DEFAULT)) {
                ReorderedFieldMetaWriter.writeReorderedMeta(out, fieldInfo, 0L, 100L, mergedOrdToDocId, permutation);
            }

            try (IndexInput in = dir.openInput("test.vemf", IOContext.DEFAULT)) {
                // Skip field metadata
                in.readInt();   // fieldNumber
                in.readInt();   // encoding
                in.readInt();   // similarity
                in.readVLong(); // vectorDataOffset
                in.readVLong(); // vectorDataLength
                in.readVInt();  // dimension

                // Skip list header
                byte isDenseByte = in.readByte();
                int readMaxDoc = in.readInt();
                in.readInt();   // numLevel
                in.readInt();   // numDocsForGrouping
                in.readInt();   // groupFactor

                assertEquals(maxDoc, readMaxDoc);

                // Read skip list
                FixedBlockSkipListIndexReader reader = new FixedBlockSkipListIndexReader(in, readMaxDoc);

                // Compute sentinel
                int numBytesPerValue = Integer.BYTES - (Integer.numberOfLeadingZeros(readMaxDoc) / Byte.SIZE);
                int sentinel = (1 << (8 * numBytesPerValue)) - 1;

                // Verify doc->ord mapping
                for (int doc = 0; doc <= maxDoc; doc++) {
                    reader.skipTo(doc);
                    int ord = reader.getOrd();
                    if (docsWithVectors.contains(doc)) {
                        // Find which mergedOrd has this docId
                        int mergedOrd = -1;
                        for (int m = 0; m < n; m++) {
                            if (mergedOrdToDocId[m] == doc) {
                                mergedOrd = m;
                                break;
                            }
                        }
                        assertNotEquals("Doc " + doc + " should have a mergedOrd", -1, mergedOrd);
                        assertEquals("Wrong ord for doc " + doc, oldOrd2New[mergedOrd], ord);
                    } else {
                        assertEquals("Doc " + doc + " without vector should have sentinel", sentinel, ord);
                    }
                }

                // Read and verify ordToDoc array
                int readN = in.readInt();
                assertEquals(n, readN);
                for (int newOrd = 0; newOrd < n; newOrd++) {
                    int expectedDocId = mergedOrdToDocId[permutation[newOrd]];
                    int actualDocId = in.readInt();
                    assertEquals("Wrong ordToDoc for newOrd " + newOrd, expectedDocId, actualDocId);
                }
            }
        }
    }

    // --- Test 1: Dense round-trip ---

    public void testDenseRoundTrip() throws IOException {
        int n = 100;
        int[] permutation = new int[n];
        int[] mergedOrdToDocId = new int[n];
        Set<Integer> docsWithVectors = new HashSet<>();
        for (int i = 0; i < n; i++) {
            permutation[i] = i;
            mergedOrdToDocId[i] = i;
            docsWithVectors.add(i);
        }
        // Shuffle permutation
        Random rng = new Random(42);
        for (int i = n - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int tmp = permutation[i];
            permutation[i] = permutation[j];
            permutation[j] = tmp;
        }
        verifyRoundTrip(permutation, mergedOrdToDocId, docsWithVectors, n - 1);
    }

    // --- Test 1: Sparse round-trip ---

    public void testSparseRoundTrip() throws IOException {
        int n = 100;
        int totalDocs = 1000;
        Random rng = new Random(42);

        // Pick n random doc IDs out of totalDocs
        Set<Integer> docsWithVectors = new HashSet<>();
        while (docsWithVectors.size() < n) {
            docsWithVectors.add(rng.nextInt(totalDocs));
        }
        int[] sortedDocs = docsWithVectors.stream().mapToInt(Integer::intValue).sorted().toArray();

        int[] mergedOrdToDocId = new int[n];
        for (int i = 0; i < n; i++) {
            mergedOrdToDocId[i] = sortedDocs[i];
        }

        int[] permutation = new int[n];
        for (int i = 0; i < n; i++) permutation[i] = i;
        for (int i = n - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int tmp = permutation[i];
            permutation[i] = permutation[j];
            permutation[j] = tmp;
        }

        int maxDoc = sortedDocs[n - 1];
        verifyRoundTrip(permutation, mergedOrdToDocId, docsWithVectors, maxDoc);
    }

    // --- Test 9: Dense-as-sparse (all docs have vectors) ---

    public void testAllDocsHaveVectorsDetectedAsDense() throws IOException {
        int n = 50;
        int[] permutation = new int[n];
        int[] mergedOrdToDocId = new int[n];
        Set<Integer> docsWithVectors = new HashSet<>();
        for (int i = 0; i < n; i++) {
            permutation[i] = n - 1 - i; // reverse
            mergedOrdToDocId[i] = i;
            docsWithVectors.add(i);
        }
        // maxDoc = n-1, numVectors = n, so n == maxDoc+1 → dense
        verifyRoundTrip(permutation, mergedOrdToDocId, docsWithVectors, n - 1);
    }

    // --- Test 10: Vectors only at high doc IDs ---

    public void testVectorsOnlyAtHighDocIds() throws IOException {
        int n = 100;
        int totalDocs = 1000;
        // Vectors in docs [900..999]
        int[] mergedOrdToDocId = new int[n];
        Set<Integer> docsWithVectors = new HashSet<>();
        for (int i = 0; i < n; i++) {
            mergedOrdToDocId[i] = 900 + i;
            docsWithVectors.add(900 + i);
        }

        int[] permutation = new int[n];
        for (int i = 0; i < n; i++) permutation[i] = i;
        Random rng = new Random(99);
        for (int i = n - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int tmp = permutation[i];
            permutation[i] = permutation[j];
            permutation[j] = tmp;
        }

        verifyRoundTrip(permutation, mergedOrdToDocId, docsWithVectors, 999);
    }

    // --- Test 11: Single vector (sparse) ---

    public void testSingleVectorSparse() throws IOException {
        int[] permutation = { 0 };
        int[] mergedOrdToDocId = { 999 };
        Set<Integer> docsWithVectors = new HashSet<>(Arrays.asList(999));
        verifyRoundTrip(permutation, mergedOrdToDocId, docsWithVectors, 999);
    }

    // --- Test 11: Single vector (dense) ---

    public void testSingleVectorDense() throws IOException {
        int[] permutation = { 0, 1 };
        int[] mergedOrdToDocId = { 0, 1 };
        Set<Integer> docsWithVectors = new HashSet<>(Arrays.asList(0, 1));
        verifyRoundTrip(permutation, mergedOrdToDocId, docsWithVectors, 1);
    }

    // --- Test 12: Consecutive vectors at end ---

    public void testConsecutiveVectorsAtEnd() throws IOException {
        int n = 100;
        int totalDocs = 1000;
        int[] mergedOrdToDocId = new int[n];
        Set<Integer> docsWithVectors = new HashSet<>();
        for (int i = 0; i < n; i++) {
            mergedOrdToDocId[i] = 900 + i;
            docsWithVectors.add(900 + i);
        }
        // Identity permutation
        int[] permutation = new int[n];
        for (int i = 0; i < n; i++) permutation[i] = i;
        verifyRoundTrip(permutation, mergedOrdToDocId, docsWithVectors, 999);
    }
}
