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

        final IndexInput bytesSlice = vectorData.slice("vector-data", vectorDataOffset, vectorDataLength);
        final int byteSize = dimension * Float.BYTES;
        return new ReorderedOffHeapFloatVectorValues111.DenseOffHeapVectorValues(
            dimension,
                                                                                 bytesSlice,
                                                                                 byteSize,
                                                                                 numVectors,
                                                                                 flatVectorsScorer,
                                                                                 vectorSimilarityFunction,
                                                                                 docIdOrdSkipListIndex
        );
    }

    public static class DenseOffHeapVectorValues extends ReorderedOffHeapFloatVectorValues111 {
        private final FixedBlockSkipListIndexReader docIdOrdSkipListIndex;

        private final int[][] sharedOrdToDocMap; // single-element array as mutable holder

        public DenseOffHeapVectorValues(
            int dimension,
            IndexInput slice,
            int byteSize,
            int totalNumVectors,
            FlatVectorsScorer flatVectorsScorer,
            VectorSimilarityFunction similarityFunction,
            FixedBlockSkipListIndexReader docIdOrdSkipListIndex
        ) {
            this(dimension, slice, byteSize, totalNumVectors, flatVectorsScorer, similarityFunction, docIdOrdSkipListIndex, new int[1][]);
        }

        private DenseOffHeapVectorValues(
            int dimension,
            IndexInput slice,
            int byteSize,
            int totalNumVectors,
            FlatVectorsScorer flatVectorsScorer,
            VectorSimilarityFunction similarityFunction,
            FixedBlockSkipListIndexReader docIdOrdSkipListIndex,
            int[][] sharedOrdToDocMap
        ) {
            super(dimension, slice, byteSize, totalNumVectors, flatVectorsScorer, similarityFunction);
            this.docIdOrdSkipListIndex = docIdOrdSkipListIndex;
            this.sharedOrdToDocMap = sharedOrdToDocMap;
        }

        @Override
        public DenseOffHeapVectorValues copy() throws IOException {
            return new DenseOffHeapVectorValues(
                dimension, slice.clone(), byteSize, numVectors,
                flatVectorsScorer, similarityFunction, docIdOrdSkipListIndex, sharedOrdToDocMap
            );
        }

        @Override
        public int ordToDoc(int ord) {
            if (docIdOrdSkipListIndex == null) return ord;
            if (sharedOrdToDocMap[0] == null) {
                int[] map = new int[numVectors];
                for (int doc = 0; doc <= docIdOrdSkipListIndex.maxDoc; doc++) {
                    docIdOrdSkipListIndex.skipTo(doc);
                    int o = docIdOrdSkipListIndex.getOrd();
                    if (o >= 0 && o < numVectors) {
                        map[o] = doc;
                    }
                }
                sharedOrdToDocMap[0] = map;
                System.out.println("[ReorderedFVV] Built ordToDocMap: numVectors=" + numVectors
                    + ", maxDoc=" + docIdOrdSkipListIndex.maxDoc
                    + ", sample ordToDoc[0]=" + map[0]
                    + ", ordToDoc[1]=" + map[1]);
            }
            return sharedOrdToDocMap[0][ord];
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
