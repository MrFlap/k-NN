package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

import java.io.IOException;

public abstract class ReorderedOffHeapFloatVectorValues111 extends FloatVectorValues {
    protected final int dimension;
    protected final IndexInput slice;
    protected final int byteSize;
    protected final int numVectors;
    protected int lastOrd = -1;
    protected final float[] value;
    protected final VectorSimilarityFunction similarityFunction;
    protected final FlatVectorsScorer flatVectorsScorer;

    ReorderedOffHeapFloatVectorValues111(
        int dimension,
        IndexInput slice,
        int byteSize,
        int numVectors,
        FlatVectorsScorer flatVectorsScorer,
        VectorSimilarityFunction similarityFunction
    ) {
        this.dimension = dimension;
        this.slice = slice;
        this.byteSize = byteSize;
        this.numVectors = numVectors;
        this.similarityFunction = similarityFunction;
        this.flatVectorsScorer = flatVectorsScorer;
        value = new float[dimension];
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public int size() {
        return numVectors;
    }

    @Override
    public float[] vectorValue(int targetOrd) throws IOException {
        if (lastOrd == targetOrd) {
            return value;
        }
        slice.seek((long) targetOrd * byteSize);
        slice.readFloats(value, 0, value.length);
        lastOrd = targetOrd;
        return value;
    }

    public static ReorderedOffHeapFloatVectorValues111 load(
        VectorSimilarityFunction vectorSimilarityFunction,
        FlatVectorsScorer flatVectorsScorer,
        FixedBlockSkipListIndexReader docIdOrdSkipListIndex,
        int dimension,
        long vectorDataOffset,
        long vectorDataLength,
        IndexInput vectorData,
        int numVectors
    ) throws IOException {
        return load(vectorSimilarityFunction, flatVectorsScorer, docIdOrdSkipListIndex,
            dimension, vectorDataOffset, vectorDataLength, vectorData, numVectors, null);
    }

    public static ReorderedOffHeapFloatVectorValues111 load(
        VectorSimilarityFunction vectorSimilarityFunction,
        FlatVectorsScorer flatVectorsScorer,
        FixedBlockSkipListIndexReader docIdOrdSkipListIndex,
        int dimension,
        long vectorDataOffset,
        long vectorDataLength,
        IndexInput vectorData,
        int numVectors,
        int[] ordToDocMap
    ) throws IOException {

        final IndexInput bytesSlice = vectorData.slice("vector-data", vectorDataOffset, vectorDataLength);
        final int byteSize = dimension * Float.BYTES;
        return new ReorderedOffHeapFloatVectorValues111.DenseOffHeapVectorValues(
            dimension, bytesSlice, byteSize, numVectors,
            flatVectorsScorer, vectorSimilarityFunction,
            docIdOrdSkipListIndex, ordToDocMap
        );
    }

    public static class DenseOffHeapVectorValues extends ReorderedOffHeapFloatVectorValues111 {
        private final FixedBlockSkipListIndexReader docIdOrdSkipListIndex;
        private final int[] ordToDocMap;  // pre-built from .vemf, or null

        public DenseOffHeapVectorValues(
            int dimension,
            IndexInput slice,
            int byteSize,
            int totalNumVectors,
            FlatVectorsScorer flatVectorsScorer,
            VectorSimilarityFunction similarityFunction,
            FixedBlockSkipListIndexReader docIdOrdSkipListIndex
        ) {
            this(dimension, slice, byteSize, totalNumVectors, flatVectorsScorer, similarityFunction, docIdOrdSkipListIndex, null);
        }

        public DenseOffHeapVectorValues(
            int dimension,
            IndexInput slice,
            int byteSize,
            int totalNumVectors,
            FlatVectorsScorer flatVectorsScorer,
            VectorSimilarityFunction similarityFunction,
            FixedBlockSkipListIndexReader docIdOrdSkipListIndex,
            int[] ordToDocMap
        ) {
            super(dimension, slice, byteSize, totalNumVectors, flatVectorsScorer, similarityFunction);
            this.docIdOrdSkipListIndex = docIdOrdSkipListIndex;
            this.ordToDocMap = ordToDocMap;
        }

        @Override
        public DenseOffHeapVectorValues copy() throws IOException {
            return new DenseOffHeapVectorValues(
                dimension, slice.clone(), byteSize, numVectors,
                flatVectorsScorer, similarityFunction, docIdOrdSkipListIndex, ordToDocMap
            );
        }

        @Override
        public int ordToDoc(int ord) {
            if (ordToDocMap != null) return ordToDocMap[ord];
            return ord;
        }

        @Override
        public Bits getAcceptOrds(final Bits acceptDocs) {
            return acceptDocs;
        }

        @Override
        public DocIndexIterator iterator() {
            return new DocIndexIterator() {
                int doc = -1;
                int ordDoc = -1;
                int ord = -1;

                @Override
                public int index() {
                    if (doc != ordDoc) {
                        docIdOrdSkipListIndex.skipTo(doc);
                        ordDoc = doc;
                        ord = docIdOrdSkipListIndex.getOrd();
                    }

                    return ord;
                }

                @Override
                public int docID() {
                    return doc;
                }

                @Override
                public int nextDoc() throws IOException {
                    if (doc >= size() - 1) {
                        return doc = NO_MORE_DOCS;
                    } else {
                        return ++doc;
                    }
                }

                @Override
                public int advance(int target) throws IOException {
                    if (target >= size()) {
                        return doc = NO_MORE_DOCS;
                    }
                    return doc = target;
                }

                @Override
                public long cost() {
                    return size();
                }
            };
        }

        @Override
        public VectorScorer scorer(float[] query) throws IOException {
            DenseOffHeapVectorValues copy = copy();
            DocIndexIterator iterator = copy.iterator();
            RandomVectorScorer randomVectorScorer = flatVectorsScorer.getRandomVectorScorer(similarityFunction, copy, query);
            return new VectorScorer() {
                @Override
                public float score() throws IOException {
                    return randomVectorScorer.score(iterator.docID());
                }

                @Override
                public DocIdSetIterator iterator() {
                    return iterator;
                }
            };
        }
    }
}
