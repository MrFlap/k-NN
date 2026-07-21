/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.search.knn.TopKnnCollectorManager;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.index.codec.KNN1040Codec.Faiss1040ScalarQuantizedKnnVectorsReader;
import org.opensearch.knn.index.codec.KNN1040Codec.QefContext;
import org.opensearch.knn.index.query.common.QueryUtils;
import org.opensearch.knn.index.query.memoryoptsearch.RadiusVectorSimilarityCollector;
import org.opensearch.knn.memoryoptsearch.faiss.FaissMemoryOptimizedSearcher;
import org.opensearch.lucene.SeededTopDocsDISI;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Callable;

/**
 * Bounded two-phase radial search query.
 * <p>
 * Phase 1: Raw quantized top-ef_search on full index (no QEF).
 * Phase 2: Full-precision radial graph walk, filtered to Phase 1 subset.
 *          Terminates when no more neighbors pass the similarity threshold,
 *          bounding full-precision computations to the radius neighborhood.
 */
@Log4j2
public class BoundedRadialSearchQuery extends Query {
    private static final KnnSearchStrategy.Hnsw DEFAULT_HNSW_SEARCH_STRATEGY = new KnnSearchStrategy.Hnsw(0);

    private final String field;
    private final float[] target;
    private final float similarity;
    private final int efSearch;
    private final Query filter;

    public BoundedRadialSearchQuery(String field, float[] target, float similarity, int efSearch, Query filter) {
        this.field = Objects.requireNonNull(field);
        this.target = Objects.requireNonNull(target);
        this.similarity = similarity;
        this.efSearch = efSearch;
        this.filter = filter;
    }

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {
        final Weight filterWeight;
        if (filter != null) {
            Query rewritten = searcher.rewrite(filter);
            filterWeight = searcher.createWeight(rewritten, ScoreMode.COMPLETE_NO_SCORES, 1f);
        } else {
            filterWeight = null;
        }

        List<LeafReaderContext> leaves = searcher.getIndexReader().leaves();
        List<Callable<TopDocs>> tasks = new ArrayList<>(leaves.size());
        for (LeafReaderContext ctx : leaves) {
            tasks.add(() -> searchLeaf(ctx, filterWeight, searcher));
        }

        List<TopDocs> leafResults = searcher.getTaskExecutor().invokeAll(tasks);

        int totalHits = 0;
        for (int i = 0; i < leafResults.size(); i++) {
            TopDocs leafTopDocs = leafResults.get(i);
            int docBase = leaves.get(i).docBase;
            for (ScoreDoc sd : leafTopDocs.scoreDocs) {
                sd.doc += docBase;
            }
            totalHits += leafTopDocs.scoreDocs.length;
        }

        ScoreDoc[] allDocs = new ScoreDoc[totalHits];
        int idx = 0;
        for (TopDocs leafTopDocs : leafResults) {
            for (ScoreDoc sd : leafTopDocs.scoreDocs) {
                allDocs[idx++] = sd;
            }
        }
        TopDocs merged = new TopDocs(new TotalHits(totalHits, TotalHits.Relation.EQUAL_TO), allDocs);

        if (merged.scoreDocs.length == 0) {
            return new MatchNoDocsQuery().createWeight(searcher, scoreMode, boost);
        }

        return QueryUtils.getInstance().createDocAndScoreQuery(searcher.getIndexReader(), merged).createWeight(searcher, scoreMode, boost);
    }

    private TopDocs searchLeaf(LeafReaderContext ctx, Weight filterWeight, IndexSearcher searcher) throws IOException {
        final LeafReader reader = ctx.reader();

        if (reader.getFloatVectorValues(field) == null) {
            return new TopDocs(new TotalHits(0, TotalHits.Relation.EQUAL_TO), new ScoreDoc[0]);
        }

        final AcceptDocs acceptDocs;
        final int visitLimit;
        if (filterWeight != null) {
            Scorer scorer = filterWeight.scorer(ctx);
            if (scorer == null) {
                return new TopDocs(new TotalHits(0, TotalHits.Relation.EQUAL_TO), new ScoreDoc[0]);
            }
            Bits liveDocs = reader.getLiveDocs();
            acceptDocs = AcceptDocs.fromIteratorSupplier(() -> scorer.iterator(), liveDocs, reader.maxDoc());
            visitLimit = acceptDocs.cost();
            if (visitLimit == 0) {
                return new TopDocs(new TotalHits(0, TotalHits.Relation.EQUAL_TO), new ScoreDoc[0]);
            }
        } else {
            acceptDocs = AcceptDocs.fromLiveDocs(reader.getLiveDocs(), reader.maxDoc());
            visitLimit = Integer.MAX_VALUE;
        }

        // Adjust ef_search and seed count per segment using optimistic formula (lambda=2).
        float leafProportion = ctx.reader().maxDoc() / (float) ctx.parent.reader().maxDoc();
        int perLeafEfSearch = perLeafK(efSearch, leafProportion);
        int perLeafSeeds = perLeafK(100, leafProportion);

        // === Phase 1: Raw quantized top-ef_search (no QEF) ===
        QefContext.CURRENT.set(null);
        TopDocs phase1Results;
        try {
            KnnCollector phase1Collector = new TopKnnCollectorManager(perLeafEfSearch, searcher).newCollector(
                visitLimit, DEFAULT_HNSW_SEARCH_STRATEGY, ctx
            );
            reader.searchNearestVectors(field, target, phase1Collector, acceptDocs);
            phase1Results = phase1Collector.topDocs();
        } finally {
            QefContext.CURRENT.remove();
        }

        if (phase1Results.scoreDocs.length == 0) {
            return new TopDocs(new TotalHits(0, TotalHits.Relation.EQUAL_TO), new ScoreDoc[0]);
        }

        // === Phase 2: Full-precision radial walk, filtered to Phase 1 subset ===
        LeafReader baseReader = reader;
        while (baseReader instanceof FilterLeafReader filterReader) {
            baseReader = filterReader.getDelegate();
        }
        final SegmentReader segReader = (SegmentReader) baseReader;
        final KnnVectorsReader vectorReader = segReader.getVectorReader();
        final PerFieldKnnVectorsFormat.FieldsReader fieldsReader = (PerFieldKnnVectorsFormat.FieldsReader) vectorReader;
        final Faiss1040ScalarQuantizedKnnVectorsReader fieldReader =
            (Faiss1040ScalarQuantizedKnnVectorsReader) fieldsReader.getFieldReader(field);
        final FaissMemoryOptimizedSearcher mosSearcher =
            (FaissMemoryOptimizedSearcher) fieldReader.getMemoryOptimizedSearcher(field);

        // Build subset filter from Phase 1 candidates.
        final FixedBitSet subsetBits = new FixedBitSet(reader.maxDoc());
        for (ScoreDoc sd : phase1Results.scoreDocs) {
            subsetBits.set(sd.doc);
        }

        final FloatVectorValues fvv = reader.getFloatVectorValues(field);
        final VectorSimilarityFunction simFunc = mosSearcher.getVectorSimilarityFunction();
        final RandomVectorScorer baseScorer =
            DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorer(simFunc, fvv, target);

        // Cache scores from Phase 2a so Phase 2b doesn't recompute them.
        final java.util.HashMap<Integer, Float> scoreCache = new java.util.HashMap<>();
        final int[] fullPrecisionCount = {0};

        // Wrap scorer: returns -infinity for non-subset nodes, caches all full-precision results.
        final RandomVectorScorer fullPrecisionScorer = new RandomVectorScorer() {
            @Override
            public float score(int node) throws IOException {
                if (!subsetBits.get(node)) {
                    return Float.NEGATIVE_INFINITY;
                }
                Float cached = scoreCache.get(node);
                if (cached != null) {
                    return cached;
                }
                fullPrecisionCount[0]++;
                float score = baseScorer.score(node);
                scoreCache.put(node, score);
                return score;
            }

            @Override
            public int maxOrd() {
                return baseScorer.maxOrd();
            }

            @Override
            public int ordToDoc(int ord) {
                return baseScorer.ordToDoc(ord);
            }

            @Override
            public Bits getAcceptOrds(Bits acceptDocs) {
                return baseScorer.getAcceptOrds(acceptDocs);
            }
        };

        // Phase 2a: Top-10 full-precision search on the subset to find true closest neighbors.
        int numSeeds = Math.min(perLeafSeeds, phase1Results.scoreDocs.length);
        ScoreDoc[] seedDocs = Arrays.copyOf(phase1Results.scoreDocs, numSeeds);
        TopDocs seedTopDocs = new TopDocs(new TotalHits(numSeeds, TotalHits.Relation.EQUAL_TO), seedDocs);

        KnnCollector top10Collector = new TopKnnCollectorManager(10, searcher).newCollector(
            Integer.MAX_VALUE,
            new KnnSearchStrategy.Seeded(
                new SeededTopDocsDISI(seedTopDocs),
                numSeeds,
                DEFAULT_HNSW_SEARCH_STRATEGY
            ),
            ctx
        );
        mosSearcher.searchWithScorer(fullPrecisionScorer, top10Collector, subsetBits);
        TopDocs top10Results = top10Collector.topDocs();

        if (top10Results.scoreDocs.length == 0 || top10Results.scoreDocs[0].score < similarity) {
            log.info("[BOUNDED-RADIAL] efSearch={}, phase1={}, noValidSeeds, fullPrecisionScored={}",
                efSearch, phase1Results.scoreDocs.length, fullPrecisionCount[0]);
            return new TopDocs(new TotalHits(0, TotalHits.Relation.EQUAL_TO), new ScoreDoc[0]);
        }

        // Filter to only seeds that pass the radius.
        List<ScoreDoc> validSeeds = new ArrayList<>();
        for (ScoreDoc sd : top10Results.scoreDocs) {
            if (sd.score >= similarity) {
                validSeeds.add(sd);
            }
        }

        if (validSeeds.isEmpty()) {
            log.info("[BOUNDED-RADIAL] efSearch={}, phase1={}, noValidSeeds, fullPrecisionScored={}",
                efSearch, phase1Results.scoreDocs.length, fullPrecisionCount[0]);
            return new TopDocs(new TotalHits(0, TotalHits.Relation.EQUAL_TO), new ScoreDoc[0]);
        }

        ScoreDoc[] seedArray = validSeeds.toArray(ScoreDoc[]::new);
        TopDocs trueSeed = new TopDocs(new TotalHits(seedArray.length, TotalHits.Relation.EQUAL_TO), seedArray);
        int phase2aFpCount = fullPrecisionCount[0];

        RadiusVectorSimilarityCollector radialCollector = new RadiusVectorSimilarityCollector(
            similarity, Integer.MAX_VALUE,
            new KnnSearchStrategy.Seeded(
                new SeededTopDocsDISI(trueSeed),
                trueSeed.scoreDocs.length,
                DEFAULT_HNSW_SEARCH_STRATEGY
            )
        );
        mosSearcher.searchWithScorer(fullPrecisionScorer, radialCollector, subsetBits);
        TopDocs results = radialCollector.topDocs();

        log.info("[BOUNDED-RADIAL] efSearch={}, perLeafEf={}, perLeafSeeds={}, phase1={}, phase2aFP={}, phase2bFP={}, totalFP={}, passed={}",
            efSearch, perLeafEfSearch, perLeafSeeds, phase1Results.scoreDocs.length, phase2aFpCount,
            fullPrecisionCount[0] - phase2aFpCount, fullPrecisionCount[0],
            results == null ? 0 : results.scoreDocs.length);

        if (results == null || results.scoreDocs.length == 0) {
            return new TopDocs(new TotalHits(0, TotalHits.Relation.EQUAL_TO), new ScoreDoc[0]);
        }

        return results;
    }

    @Override
    public String toString(String field) {
        return "BoundedRadialSearchQuery[field=" + this.field + " similarity=" + similarity + " efSearch=" + efSearch + "]";
    }

    @Override
    public void visit(QueryVisitor visitor) {
        if (visitor.acceptField(field)) {
            visitor.visitLeaf(this);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        BoundedRadialSearchQuery that = (BoundedRadialSearchQuery) o;
        return Float.compare(that.similarity, similarity) == 0
            && efSearch == that.efSearch
            && field.equals(that.field)
            && Arrays.equals(target, that.target)
            && Objects.equals(filter, that.filter);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(field, similarity, efSearch, filter);
        result = 31 * result + Arrays.hashCode(target);
        return result;
    }

    private static final int LAMBDA = 2;

    private static int perLeafK(int k, float leafProportion) {
        return (int) Math.max(1, k * leafProportion + LAMBDA * Math.sqrt(k * leafProportion * (1 - leafProportion)));
    }
}
