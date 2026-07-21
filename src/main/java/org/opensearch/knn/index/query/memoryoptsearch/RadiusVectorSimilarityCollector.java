/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.memoryoptsearch;

import org.apache.lucene.search.AbstractKnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.knn.KnnSearchStrategy;

import java.util.ArrayList;
import java.util.List;

/**
 * Collector for radial (similarity-threshold) search over an HNSW graph.
 * <p>
 * Uses a fixed similarity threshold as a hard cutoff: collects all visited nodes at or above
 * the threshold and refuses to explore below it. This is designed for the two-phase radial
 * search where phase-1 seeds provide good entry points, so no exploration beyond the threshold
 * is needed.
 * <p>
 * Compatible with Lucene 10.5's {@code HnswGraphSearcher} which computes the traversal bound as
 * {@code Math.nextUp(minCompetitiveSimilarity())}. We return {@code Math.nextDown(similarity)}
 * so the effective cutoff equals the threshold exactly.
 */
public class RadiusVectorSimilarityCollector extends AbstractKnnCollector {
    private static final KnnSearchStrategy.Hnsw DEFAULT_STRATEGY = new KnnSearchStrategy.Hnsw(0);
    private static final double DECAY_MAX_QUALITY = 1.0;
    private static final double DEFAULT_DECAY = 0.95;

    private final float resultSimilarity;
    private final double decay;
    private final List<ScoreDoc> scoreDocList;
    private float minCompetitiveSimilarity;

    /**
     * @param similarity minimum similarity for both traversal and collection.
     * @param visitLimit limit on number of nodes to visit.
     */
    public RadiusVectorSimilarityCollector(float similarity, long visitLimit) {
        this(similarity, visitLimit, DEFAULT_STRATEGY, DEFAULT_DECAY);
    }

    /**
     * @param similarity     minimum similarity for both traversal and collection.
     * @param visitLimit     limit on number of nodes to visit.
     * @param searchStrategy the HNSW search strategy (e.g. {@link KnnSearchStrategy.Seeded}).
     */
    public RadiusVectorSimilarityCollector(float similarity, long visitLimit, KnnSearchStrategy searchStrategy) {
        this(similarity, visitLimit, searchStrategy, DEFAULT_DECAY);
    }

    /**
     * @param similarity     minimum similarity for both traversal and collection.
     * @param visitLimit     limit on number of nodes to visit.
     * @param searchStrategy the HNSW search strategy (e.g. {@link KnnSearchStrategy.Seeded}).
     * @param decay          decay factor for the traversal bound (0-1). Lower values terminate sooner.
     */
    public RadiusVectorSimilarityCollector(float similarity, long visitLimit, KnnSearchStrategy searchStrategy, double decay) {
        super(1, visitLimit, searchStrategy);
        this.resultSimilarity = similarity;
        this.decay = decay;
        this.scoreDocList = new ArrayList<>();
        this.minCompetitiveSimilarity = Math.nextDown(similarity);
    }

    @Override
    public boolean collect(int docId, float similarity) {
        if (similarity >= resultSimilarity) {
            scoreDocList.add(new ScoreDoc(docId, similarity));
            return true;
        } else if (decay < DECAY_MAX_QUALITY) {
            minCompetitiveSimilarity = (float) (similarity + ((double) minCompetitiveSimilarity - similarity) * decay);
            return true;
        }
        return false;
    }

    @Override
    public float minCompetitiveSimilarity() {
        return minCompetitiveSimilarity;
    }

    @Override
    public TopDocs topDocs() {
        TotalHits.Relation relation = earlyTerminated() ? TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO : TotalHits.Relation.EQUAL_TO;
        return new TopDocs(new TotalHits(visitedCount(), relation), scoreDocList.toArray(ScoreDoc[]::new));
    }

    @Override
    public int numCollected() {
        return scoreDocList.size();
    }
}
