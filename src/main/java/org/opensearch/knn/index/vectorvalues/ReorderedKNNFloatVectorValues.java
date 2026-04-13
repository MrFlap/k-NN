/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.DocIdSetIterator;

import java.io.IOException;

/**
 * A {@link KNNVectorValues} that iterates vectors in reordered (permuted) order.
 * For newOrd 0..N-1, yields docId = mergedOrdToDocId[permutation[newOrd]]
 * and vector = mergedRandomAccess.vectorValue(permutation[newOrd]).
 */
public class ReorderedKNNFloatVectorValues extends KNNVectorValues<float[]> {

    private final FloatVectorValues mergedRandomAccess;
    private final int[] permutation;
    private final int[] mergedOrdToDocId;

    public ReorderedKNNFloatVectorValues(
        FloatVectorValues mergedRandomAccess,
        int[] permutation,
        int[] mergedOrdToDocId
    ) {
        super(new ReorderedIterator(permutation, mergedOrdToDocId));
        this.mergedRandomAccess = mergedRandomAccess;
        this.permutation = permutation;
        this.mergedOrdToDocId = mergedOrdToDocId;
    }

    @Override
    public float[] getVector() throws IOException {
        int newOrd = ((ReorderedIterator) vectorValuesIterator).currentNewOrd();
        float[] vector = mergedRandomAccess.vectorValue(permutation[newOrd]);
        this.dimension = vector.length;
        this.bytesPerVector = vector.length * Float.BYTES;
        return vector;
    }

    @Override
    public float[] conditionalCloneVector() throws IOException {
        float[] vector = getVector();
        return vector.clone();
    }

    /**
     * Iterator that walks newOrd 0..N-1, yielding the permuted doc IDs.
     */
    static class ReorderedIterator implements KNNVectorValuesIterator {
        private final int[] permutation;
        private final int[] mergedOrdToDocId;
        private int newOrd = -1;
        private int docId = -1;

        ReorderedIterator(int[] permutation, int[] mergedOrdToDocId) {
            this.permutation = permutation;
            this.mergedOrdToDocId = mergedOrdToDocId;
        }

        int currentNewOrd() {
            return newOrd;
        }

        @Override
        public int docId() {
            return docId;
        }

        @Override
        public int nextDoc() {
            newOrd++;
            if (newOrd >= permutation.length) {
                docId = DocIdSetIterator.NO_MORE_DOCS;
            } else {
                docId = mergedOrdToDocId[permutation[newOrd]];
            }
            return docId;
        }

        @Override
        public int advance(int target) {
            throw new UnsupportedOperationException();
        }

        @Override
        public DocIdSetIterator getDocIdSetIterator() {
            throw new UnsupportedOperationException();
        }

        @Override
        public long liveDocs() {
            return permutation.length;
        }

        @Override
        public VectorValueExtractorStrategy getVectorExtractorStrategy() {
            throw new UnsupportedOperationException();
        }
    }
}
