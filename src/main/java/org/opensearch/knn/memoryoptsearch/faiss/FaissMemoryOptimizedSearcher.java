/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import com.google.common.annotations.VisibleForTesting;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.KnnVectorValues.DocIndexIterator;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;
import org.apache.lucene.util.hnsw.OrdinalTranslatedKnnCollector;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.common.RobustUniqueRandomIterator;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.locality.InlineNeighborLists;
import org.opensearch.knn.index.io.KnnIo;
import org.opensearch.knn.index.codec.scorer.PageTouchTracker;
import org.opensearch.knn.index.util.WarmupUtil;
import org.opensearch.knn.memoryoptsearch.VectorSearcher;
import org.opensearch.knn.memoryoptsearch.faiss.cagra.FaissCagraHNSW;

import java.io.IOException;
import java.util.Arrays;

import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.EXHAUSTIVE_BULK_SCORE_ORDS;
import static org.opensearch.knn.common.KNNConstants.SPACE_TYPE;

/**
 * This searcher directly reads FAISS index file via the provided {@link IndexInput} then perform vector search on it.
 */
@Log4j2
public class FaissMemoryOptimizedSearcher implements VectorSearcher {
    private final IndexInput indexInput;
    private final FaissIndex faissIndex;
    private final FlatVectorsScorer flatVectorsScorer;
    private final FaissHNSW hnsw;
    private final VectorSimilarityFunction vectorSimilarityFunction;
    private final InlineNeighborLists inlineNeighbors;
    private final KnnIo.LazyCachedInput cachedIndexInput;
    private boolean isAdc;

    /**
     * Constructor that accepts a pre-loaded {@link FaissIndex}. The factory is responsible for
     * loading the index and applying any transformations (e.g., replacing null flat storage for Faiss SQ (for 1 bit)).
     */
    public FaissMemoryOptimizedSearcher(
        final IndexInput indexInput,
        final FaissIndex faissIndex,
        final FieldInfo fieldInfo,
        final FlatVectorsScorer flatVectorsScorer
    ) {
        this(indexInput, faissIndex, fieldInfo, flatVectorsScorer, null, null);
    }

    /**
     * @param inlineNeighbors layer-0 neighbor lists stored with the vectors, used instead of the FAISS file's while
     *                        {@code knn.locality.inline_neighbors.enabled} is true; null if there are none
     * @param cachedIndexInput the same FAISS file through the block cache, for the io_uring backend; null for none
     */
    public FaissMemoryOptimizedSearcher(
        final IndexInput indexInput,
        final FaissIndex faissIndex,
        final FieldInfo fieldInfo,
        final FlatVectorsScorer flatVectorsScorer,
        final InlineNeighborLists inlineNeighbors,
        final KnnIo.LazyCachedInput cachedIndexInput
    ) {
        this.indexInput = indexInput;
        this.inlineNeighbors = inlineNeighbors;
        this.cachedIndexInput = cachedIndexInput;
        this.faissIndex = faissIndex;
        // Faiss's on-disk format only stores METRIC_INNER_PRODUCT or METRIC_L2; there is no
        // cosine metric. Cosinesimil indices are written as IP on L2-normalized vectors, so
        // faissIndex.getVectorSimilarityFunction() reports MAXIMUM_INNER_PRODUCT for them.
        // Recover the user-declared space type from FieldInfo so cosine routes to the
        // dedicated FP16_COSINE / SQ_COSINE SIMD kernels in NativeEngines990KnnVectorsScorer
        // and KNN1040ScalarQuantizedVectorScorer, which emit (1 + dot) / 2 directly.
        final KNNVectorSimilarityFunction knnVectorSimilarityFunction = resolveKnnVectorSimilarityFunction(fieldInfo, faissIndex);

        if (knnVectorSimilarityFunction != KNNVectorSimilarityFunction.HAMMING) {
            vectorSimilarityFunction = knnVectorSimilarityFunction.getVectorSimilarityFunction();
        } else {
            vectorSimilarityFunction = null;
        }

        this.isAdc = FieldInfoExtractor.isAdc(fieldInfo);
        this.flatVectorsScorer = flatVectorsScorer;
        this.hnsw = extractFaissHnsw(faissIndex);
    }

    /**
     * Resolve the similarity function for this field. For {@code cosinesimil} we prefer the
     * user-declared space type recorded in FieldInfo over the metric derived from the Faiss
     * header; for every other space we trust the Faiss header.
     *
     * <p>The override is intentionally limited to cosine. Faiss only knows
     * {@code METRIC_INNER_PRODUCT} and {@code METRIC_L2}; cosine indices are written as IP on
     * normalized vectors, so reading back from the header collapses cosine to IP and the
     * FP16_COSINE / SQ_COSINE kernel arms in the downstream scorers never fire. Every other
     * supported space maps to the same metric the header reports, so overriding them would only
     * risk masking a genuine on-disk mismatch behind a stale declared attribute.
     */
    private static KNNVectorSimilarityFunction resolveKnnVectorSimilarityFunction(final FieldInfo fieldInfo, final FaissIndex faissIndex) {
        final String spaceTypeString = fieldInfo.getAttribute(SPACE_TYPE);
        if (spaceTypeString != null && spaceTypeString.isEmpty() == false) {
            try {
                final SpaceType declared = SpaceType.getSpace(spaceTypeString);
                // Only cosine legitimately disagrees with the Faiss header: cosine indices are written as
                // METRIC_INNER_PRODUCT on normalized vectors, so the header collapses cosine to IP and the
                // COSINE kernel arms would never fire. For every other space the declared value and the
                // Faiss-reported metric agree, so we keep trusting the on-disk header there rather than let
                // a (possibly stale or mismatched) declared attribute silently override the real metric.
                if (declared == SpaceType.COSINESIMIL) {
                    final KNNVectorSimilarityFunction declaredFn = declared.getKnnVectorSimilarityFunction();
                    if (declaredFn != null) {
                        return declaredFn;
                    }
                }
            } catch (IllegalArgumentException e) {
                // SpaceType.getSpace throws on an unrecognized space string. The attribute is normally
                // written by k-NN itself so this should not happen, but rather than fail searcher
                // construction we fall back to the similarity reported by the Faiss header.
                log.warn(
                    "Unrecognized space type attribute [{}] for field [{}]; falling back to Faiss-reported similarity.",
                    spaceTypeString,
                    fieldInfo.name,
                    e
                );
            }
        }
        return faissIndex.getVectorSimilarityFunction();
    }

    private static FaissHNSW extractFaissHnsw(final FaissIndex faissIndex) {
        if (faissIndex instanceof FaissIdMapIndex idMapIndex) {
            return idMapIndex.getFaissHnsw();
        }

        throw new IllegalArgumentException("Faiss index [" + faissIndex.getIndexType() + "] does not have HNSW as an index.");
    }

    @Override
    public void search(float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        final KnnVectorValues knnVectorValues = isAdc
            ? faissIndex.getByteValues(indexInput.clone())
            : faissIndex.getFloatValues(indexInput.clone());

        search(
            VectorEncoding.FLOAT32,
            flatVectorsScorer.getRandomVectorScorer(vectorSimilarityFunction, knnVectorValues, target),
            knnCollector,
            acceptDocs
        );
    }

    @Override
    public void search(byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        search(
            VectorEncoding.BYTE,
            flatVectorsScorer.getRandomVectorScorer(vectorSimilarityFunction, faissIndex.getByteValues(indexInput.clone()), target),
            knnCollector,
            acceptDocs
        );
    }

    /**
     * Returns a {@link FaissScorableByteVectorValues} that wraps the raw byte vectors from the
     * FAISS index with scoring support via {@link FlatVectorsScorer}.
     * <p>Each call creates a new instance backed by a fresh index input slice.
     */
    @Override
    public ByteVectorValues getByteVectorValues(DocIndexIterator iterator) throws IOException {
        return new FaissScorableByteVectorValues(
            faissIndex.getByteValues(indexInput.clone()),
            flatVectorsScorer,
            vectorSimilarityFunction,
            iterator
        );
    }

    @Override
    public void warmUp() throws IOException {
        // Warm up graph
        final IndexInput warmUpIndexInput = indexInput.clone();
        WarmupUtil.readAll(warmUpIndexInput);

        // Warm up flat vectors
        // This can warm up .veb, .vec or .faiss
        if (faissIndex.getVectorEncoding() == VectorEncoding.FLOAT32) {
            WarmupUtil.readAll(faissIndex.getFloatValues(warmUpIndexInput));
        } else if (faissIndex.getVectorEncoding() == VectorEncoding.BYTE) {
            WarmupUtil.readAll(faissIndex.getByteValues(warmUpIndexInput));
        }
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(indexInput, cachedIndexInput);
    }

    private void search(
        final VectorEncoding vectorEncoding,
        final RandomVectorScorer scorer,
        final KnnCollector knnCollector,
        final AcceptDocs acceptDocs
    ) throws IOException {
        if (faissIndex.getTotalNumberOfVectors() == 0 || knnCollector.k() == 0) {
            return;
        }

        if (!this.isAdc && faissIndex.getVectorEncoding() != vectorEncoding) {
            throw new IllegalArgumentException(
                "Search for vector encoding ["
                    + vectorEncoding
                    + "] is not supported in "
                    + "an index vector whose encoding is ["
                    + faissIndex.getVectorEncoding()
                    + "]"
            );
        }

        // Set up required components for vector search
        final KnnCollector collector = createKnnCollector(knnCollector, scorer);
        final Bits acceptedOrds = scorer.getAcceptOrds(acceptDocs.bits());

        // Per-query distinct-page instrumentation (no-op unless -Dknn.pageTouch.enabled=true). Brackets
        // this segment's traversal so PrefetchHelper accounting reflects exactly this query's reads.
        final PageTouchTracker pageTouchTracker = PageTouchTracker.get();
        pageTouchTracker.begin();
        try {
            if (knnCollector.k() < scorer.maxOrd()) {
                final FaissHnswGraph graph = new FaissHnswGraph(
                    hnsw,
                    cachedIndexInput == null ? indexInput.clone() : cachedIndexInput.select(indexInput, 0, indexInput.length()),
                    inlineNeighbors != null && KNNSettings.isInlineNeighborsEnabled() ? inlineNeighbors.clone() : null
                );
                final int beamWidth = KNNSettings.getMemoryOptimizedBeamWidth();
                final int[] seeds = beamSearchSeeds(beamWidth, collector, acceptedOrds);
                if (seeds != null) {
                    BeamHnswGraphSearcher.search(scorer, collector, graph, acceptedOrds, seeds, beamWidth);
                } else {
                    // Do ANN search with Lucene's HNSW graph searcher.
                    HnswGraphSearcher.search(scorer, collector, graph, acceptedOrds);
                }
            } else {
                // if k is larger than the number of vectors we expect to visit in an HNSW search,
                // we can just iterate over all vectors and collect them.
                int numVectors = scorer.maxOrd();
                int[] ords = new int[EXHAUSTIVE_BULK_SCORE_ORDS];
                float[] scores = new float[EXHAUSTIVE_BULK_SCORE_ORDS];
                int numOrds = 0;
                for (int i = 0; i < numVectors; i++) {
                    if (acceptedOrds == null || acceptedOrds.get(i)) {
                        if (knnCollector.earlyTerminated()) {
                            break;
                        }
                        ords[numOrds++] = i;
                        if (numOrds == ords.length) {
                            knnCollector.incVisitedCount(numOrds);
                            if (scorer.bulkScore(ords, scores, numOrds) > knnCollector.minCompetitiveSimilarity()) {
                                for (int j = 0; j < numOrds; j++) {
                                    knnCollector.collect(scorer.ordToDoc(ords[j]), scores[j]);
                                }
                            }
                            numOrds = 0;
                        }
                    }
                }

                if (numOrds > 0) {
                    knnCollector.incVisitedCount(numOrds);
                    if (scorer.bulkScore(ords, scores, numOrds) > knnCollector.minCompetitiveSimilarity()) {
                        for (int j = 0; j < numOrds; j++) {
                            knnCollector.collect(scorer.ordToDoc(ords[j]), scores[j]);
                        }
                    }
                }
            }
        } finally {
            pageTouchTracker.end("nvec=" + faissIndex.getTotalNumberOfVectors() + " k=" + knnCollector.k());
        }
    }

    /**
     * Entry ordinals for {@link BeamHnswGraphSearcher}, or null to use Lucene's searcher. The beam only takes seeded
     * searches (the locality-reordered format always seeds from its closest hub) without a filter, since filtered
     * searches go through Lucene's filtered searcher.
     */
    private static int[] beamSearchSeeds(final int beamWidth, final KnnCollector collector, final Bits acceptedOrds) throws IOException {
        if (beamWidth <= 1
            || acceptedOrds instanceof BitSet
            || !(collector.getSearchStrategy() instanceof KnnSearchStrategy.Seeded seeded)
            || seeded.numberOfEntryPoints() <= 0) {
            return null;
        }
        final int[] seeds = new int[seeded.numberOfEntryPoints()];
        final DocIdSetIterator entryPoints = seeded.entryPoints();
        int numSeeds = 0;
        for (int ord; numSeeds < seeds.length && (ord = entryPoints.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS;) {
            seeds[numSeeds++] = ord;
        }
        return numSeeds == seeds.length ? seeds : Arrays.copyOf(seeds, numSeeds);
    }

    @VisibleForTesting
    KnnCollector createKnnCollector(final KnnCollector knnCollector, final RandomVectorScorer scorer) {
        final KnnCollector ordinalTranslatedKnnCollector = new OrdinalTranslatedKnnCollector(knnCollector, scorer::ordToDoc);

        if (hnsw instanceof FaissCagraHNSW cagraHNSW && (knnCollector.getSearchStrategy() instanceof KnnSearchStrategy.Seeded) == false) {
            // If there are provided entry points, then we should honor it and ensure searching to start based on them instead of
            // search with randomly selected points.
            return new KnnCollector.Decorator(ordinalTranslatedKnnCollector) {
                @Override
                public KnnSearchStrategy getSearchStrategy() {
                    return RandomEntryPointsKnnSearchStrategy.getInstance(
                        cagraHNSW.getNumBaseLevelSearchEntryPoints(),
                        cagraHNSW.getTotalNumberOfVectors(),
                        knnCollector.getSearchStrategy()
                    );
                }
            };
        }

        return ordinalTranslatedKnnCollector;
    }

    /**
     * Knn search strategy having a doc-id-iterator returning random document ids.
     * This is not designed for general purpose, it is particularly designed for populating random document ids for Cagra index.
     * Note that doc-id-iterator returns a random ids in `nextDoc` method without sorting, and might return duplicated ids.
     */
    static class RandomEntryPointsKnnSearchStrategy extends KnnSearchStrategy.Seeded {

        public static RandomEntryPointsKnnSearchStrategy getInstance(
            final int numberOfEntryPoints,
            final long totalNumberOfVectors,
            final KnnSearchStrategy originalStrategy
        ) {

            int entryPoints = getTotalNumberOfEntryPoints(numberOfEntryPoints, Math.toIntExact(totalNumberOfVectors));

            final DocIdSetIterator docIdSetIterator = generateRandomEntryPoints(entryPoints, Math.toIntExact(totalNumberOfVectors));

            return new RandomEntryPointsKnnSearchStrategy(docIdSetIterator, entryPoints, originalStrategy);
        }

        private RandomEntryPointsKnnSearchStrategy(
            final DocIdSetIterator entryPoints,
            final int numberOfEntryPoints,
            final KnnSearchStrategy originalStrategy
        ) {
            super(entryPoints, numberOfEntryPoints, originalStrategy);
        }

        private static int getTotalNumberOfEntryPoints(int numberOfEntryPoints, int totalVectors) {
            return numberOfEntryPoints >= totalVectors ? totalVectors : numberOfEntryPoints;
        }

        private static DocIdSetIterator generateRandomEntryPoints(final int numberOfEntryPoints, int totalNumberOfVectors) {
            if (numberOfEntryPoints >= totalNumberOfVectors) {
                return DocIdSetIterator.all(totalNumberOfVectors);
            }
            return new DocIdSetIterator() {
                final RobustUniqueRandomIterator robustUniqueRandomIterator = new RobustUniqueRandomIterator(
                    totalNumberOfVectors,
                    numberOfEntryPoints
                );

                @Override
                public int docID() {
                    throw new UnsupportedOperationException("DISI in RandomEntryPointsKnnSearchStrategy does not support docID()");
                }

                @Override
                public int nextDoc() {
                    return robustUniqueRandomIterator.next();
                }

                @Override
                public int advance(int targetDoc) {
                    throw new UnsupportedOperationException("DISI in RandomEntryPointsKnnSearchStrategy does not support advance(int)");
                }

                @Override
                public long cost() {
                    throw new UnsupportedOperationException("DISI in RandomEntryPointsKnnSearchStrategy does not support cost()");
                }
            };
        }
    }
}
