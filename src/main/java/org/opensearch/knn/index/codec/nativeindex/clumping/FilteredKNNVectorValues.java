/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.clumping;

import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;

import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.initializeVectorValues;

/**
 * A KNNVectorValues implementation that only iterates over a filtered subset of doc IDs.
 * Used by {@link ClumpingIndexBuildStrategy} to feed only marker vectors to the delegate strategy.
 * <p>
 * This wraps a fresh KNNVectorValues from the supplier and skips doc IDs not in the allowed set,
 * avoiding the need to hold all vectors in memory.
 */
class FilteredKNNVectorValues extends KNNVectorValues<Object> {

    private final KNNVectorValues<?> delegate;
    private final Set<Integer> allowedDocIds;

    FilteredKNNVectorValues(List<Integer> docIds, BuildIndexParams indexInfo) throws IOException {
        super(null);
        this.allowedDocIds = new HashSet<>(docIds);
        this.delegate = indexInfo.getKnnVectorValuesSupplier().get();
        initializeVectorValues(this.delegate);
        this.dimension = delegate.dimension();
        this.bytesPerVector = delegate.bytesPerVector();
    }

    @Override
    public int docId() {
        return delegate.docId();
    }

    @Override
    public int nextDoc() throws IOException {
        int doc = delegate.nextDoc();
        while (doc != NO_MORE_DOCS && allowedDocIds.contains(doc) == false) {
            doc = delegate.nextDoc();
        }
        return doc;
    }

    @Override
    public int advance(int target) throws IOException {
        int doc = delegate.advance(target);
        while (doc != NO_MORE_DOCS && allowedDocIds.contains(doc) == false) {
            doc = delegate.nextDoc();
        }
        return doc;
    }

    @Override
    public Object getVector() throws IOException {
        return delegate.getVector();
    }

    @Override
    public Object conditionalCloneVector() throws IOException {
        return delegate.conditionalCloneVector();
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public int bytesPerVector() {
        return bytesPerVector;
    }
}
