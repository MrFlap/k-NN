/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.search.AbstractKnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.knn.KnnSearchStrategy;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Collector for bounded radial search that terminates once all allowed docs have been visited.
 * <p>
 * The allowed set is the ef_search candidates from phase 1. As the graph walk visits nodes,
 * we track how many allowed docs remain unseen. Once all have been found, we raise
 * {@code minCompetitiveSimilarity()} to MAX_VALUE, which signals the graph searcher to stop.
 * <p>
 * This gives O(ef_search + neighborhood) visits in phase 2 rather than re-exploring the entire
 * graph region, because the allowed docs are clustered near the query.
 */
public class BoundedRadialCollector extends AbstractKnnCollector {

    private final Set<Integer> allowedDocs;
    private final float minQuantizedScore;
    private final List<ScoreDoc> collected;
    private int remaining;

    /**
     * @param allowedDocs       the set of doc IDs from phase 1 (ef_search candidates)
     * @param minQuantizedScore the lowest quantized score from phase 1; initial exploration floor
     * @param visitLimit        max nodes to visit in the graph walk
     * @param searchStrategy    HNSW search strategy (e.g., seeded)
     */
    public BoundedRadialCollector(
        Set<Integer> allowedDocs,
        float minQuantizedScore,
        long visitLimit,
        KnnSearchStrategy searchStrategy
    ) {
        super(1, visitLimit, searchStrategy);
        this.allowedDocs = allowedDocs;
        this.minQuantizedScore = minQuantizedScore;
        this.collected = new ArrayList<>();
        this.remaining = allowedDocs.size();
    }

    @Override
    public boolean collect(int docId, float quantizedScore) {
        if (allowedDocs.contains(docId)) {
            collected.add(new ScoreDoc(docId, quantizedScore));
            remaining--;
        }
        return false;
    }

    @Override
    public float minCompetitiveSimilarity() {
        if (remaining <= 0) {
            return Float.MAX_VALUE;
        }
        // Use the similarity threshold as the exploration floor rather than the min
        // quantized score. This constrains the walk to the above-threshold region of the
        // graph, terminating early when the frontier drops below the radius.
        // Allowed docs with quantized scores below threshold but true scores above it
        // will be missed — those are covered by the batch rescore of all collected docs.
        return Math.nextDown(minQuantizedScore);
    }

    @Override
    public TopDocs topDocs() {
        TotalHits.Relation relation = earlyTerminated() ? TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO : TotalHits.Relation.EQUAL_TO;
        return new TopDocs(new TotalHits(collected.size(), relation), collected.toArray(ScoreDoc[]::new));
    }

    @Override
    public int numCollected() {
        return collected.size();
    }
}
