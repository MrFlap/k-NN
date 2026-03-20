/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ReorderedKNNFloatVectorValuesTests extends OpenSearchTestCase {

    /**
     * 3 vectors, permutation=[2,0,1], docIds=[10,20,30].
     * Iteration order: newOrd 0→oldOrd 2 (doc 30, vec[2]),
     *                  newOrd 1→oldOrd 0 (doc 10, vec[0]),
     *                  newOrd 2→oldOrd 1 (doc 20, vec[1])
     */
    public void testPermutedIteration() throws IOException {
        float[][] vecs = { { 1, 0 }, { 0, 1 }, { 1, 1 } };
        int[] permutation = { 2, 0, 1 };
        int[] mergedOrdToDocId = { 10, 20, 30 };

        FloatVectorValues fvv = FloatVectorValues.fromFloats(toList(vecs), 2);
        ReorderedKNNFloatVectorValues values = new ReorderedKNNFloatVectorValues(fvv, permutation, mergedOrdToDocId);

        // Before first nextDoc
        assertEquals(-1, values.docId());

        // newOrd 0 → oldOrd 2 → doc 30, vec [1,1]
        values.nextDoc();
        assertEquals(30, values.docId());
        assertArrayEquals(new float[] { 1, 1 }, values.getVector(), 0f);

        // newOrd 1 → oldOrd 0 → doc 10, vec [1,0]
        values.nextDoc();
        assertEquals(10, values.docId());
        assertArrayEquals(new float[] { 1, 0 }, values.getVector(), 0f);

        // newOrd 2 → oldOrd 1 → doc 20, vec [0,1]
        values.nextDoc();
        assertEquals(20, values.docId());
        assertArrayEquals(new float[] { 0, 1 }, values.getVector(), 0f);

        // Exhausted
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, values.nextDoc());
    }

    public void testDimensionAndBytesPerVector() throws IOException {
        float[][] vecs = { { 1, 2, 3 }, { 4, 5, 6 } };
        int[] permutation = { 0, 1 };
        int[] mergedOrdToDocId = { 0, 1 };

        FloatVectorValues fvv = FloatVectorValues.fromFloats(toList(vecs), 3);
        ReorderedKNNFloatVectorValues values = new ReorderedKNNFloatVectorValues(fvv, permutation, mergedOrdToDocId);

        values.nextDoc();
        values.getVector();
        assertEquals(3, values.dimension());
        assertEquals(12, values.bytesPerVector());
    }

    public void testConditionalCloneVector() throws IOException {
        float[][] vecs = { { 1, 2 } };
        int[] permutation = { 0 };
        int[] mergedOrdToDocId = { 0 };

        FloatVectorValues fvv = FloatVectorValues.fromFloats(toList(vecs), 2);
        ReorderedKNNFloatVectorValues values = new ReorderedKNNFloatVectorValues(fvv, permutation, mergedOrdToDocId);

        values.nextDoc();
        float[] v1 = values.conditionalCloneVector();
        float[] v2 = values.conditionalCloneVector();
        assertArrayEquals(v1, v2, 0f);
        assertNotSame(v1, v2); // should be cloned
    }

    public void testLiveDocs() throws IOException {
        float[][] vecs = new float[5][2];
        int[] permutation = { 0, 1, 2, 3, 4 };
        int[] mergedOrdToDocId = { 0, 1, 2, 3, 4 };

        FloatVectorValues fvv = FloatVectorValues.fromFloats(toList(vecs), 2);
        ReorderedKNNFloatVectorValues values = new ReorderedKNNFloatVectorValues(fvv, permutation, mergedOrdToDocId);

        assertEquals(5, values.totalLiveDocs());
    }

    private static List<float[]> toList(float[][] vecs) {
        List<float[]> list = new ArrayList<>(vecs.length);
        for (float[] v : vecs) list.add(v);
        return list;
    }
}
