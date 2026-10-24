/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.hnsw.NeighborQueue;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

import java.io.IOException;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

/**
 * Level-0 HNSW search that expands the best {@code beamWidth} candidates per round instead of one.
 * <p>
 * Lucene's {@code HnswGraphSearcher.searchLevel} pops one candidate, reads its neighbor list, then scores its
 * neighbors, so a cold query pays two dependent storage reads per hop. Here each round prefetches the neighbor
 * lists of all popped candidates together, then scores the union of their unvisited neighbors in one
 * {@link RandomVectorScorer#bulkScore} call (which prefetches those vectors together). On high-latency storage
 * a round then costs about two round trips regardless of {@code beamWidth}.
 * <p>
 * Termination and collection mirror {@code HnswGraphSearcher.searchLevel}: a candidate is expanded only while
 * its score is at least the collector's minimum competitive similarity, and only accepted ordinals are collected.
 * With {@code beamWidth == 1} this is the same algorithm as Lucene's.
 */
final class BeamHnswGraphSearcher {

    private final RandomVectorScorer scorer;
    private final KnnCollector results;
    private final Bits acceptOrds;
    private final FixedBitSet visited;
    private final NeighborQueue candidates;
    private int[] bulkNodes;
    private float[] bulkScores;
    private float minAcceptedSimilarity;
    private boolean shouldExploreMinSim = true;

    private BeamHnswGraphSearcher(RandomVectorScorer scorer, KnnCollector results, Bits acceptOrds, int graphSize, int batchSize) {
        this.scorer = scorer;
        this.results = results;
        this.acceptOrds = acceptOrds;
        this.visited = new FixedBitSet(graphSize);
        this.candidates = new NeighborQueue(results.k(), true);
        this.bulkNodes = new int[batchSize];
        this.bulkScores = new float[batchSize];
        this.minAcceptedSimilarity = Math.nextUp(results.minCompetitiveSimilarity());
    }

    static void search(
        final RandomVectorScorer scorer,
        final KnnCollector results,
        final FaissHnswGraph graph,
        final Bits acceptOrds,
        final int[] entryPoints,
        final int beamWidth
    ) throws IOException {
        final int batchSize = Math.max(entryPoints.length, beamWidth * graph.maxConn() * 2);
        new BeamHnswGraphSearcher(scorer, results, acceptOrds, graph.size(), batchSize).run(graph, entryPoints, beamWidth);
    }

    private void run(final FaissHnswGraph graph, final int[] entryPoints, final int beamWidth) throws IOException {
        int numNodes = 0;
        for (int ep : entryPoints) {
            if (visited.getAndSet(ep) == false) {
                numNodes = append(numNodes, ep);
            }
        }
        scoreBatch(numNodes);

        final int[] frontier = new int[beamWidth];
        while (candidates.size() > 0 && results.earlyTerminated() == false) {
            final int frontierSize = popFrontier(frontier);
            if (frontierSize == 0) {
                break;
            }
            graph.prefetchNeighborLists(0, frontier, frontierSize);
            numNodes = 0;
            for (int i = 0; i < frontierSize; i++) {
                graph.seek(0, frontier[i]);
                int friendOrd;
                while ((friendOrd = graph.nextNeighbor()) != NO_MORE_DOCS) {
                    if (visited.getAndSet(friendOrd) == false) {
                        numNodes = append(numNodes, friendOrd);
                    }
                }
            }
            scoreBatch(numNodes);
            if (results.getSearchStrategy() != null) {
                results.getSearchStrategy().nextVectorsBlock();
            }
        }
    }

    /** Pops up to {@code frontier.length} candidates that are still competitive; returns how many. */
    private int popFrontier(final int[] frontier) {
        int size = 0;
        while (size < frontier.length && candidates.size() > 0) {
            final float topScore = candidates.topScore();
            if (topScore < minAcceptedSimilarity) {
                // Allow exploring an equivalent minimum score once, as Lucene does.
                if (shouldExploreMinSim && Math.nextUp(topScore) == minAcceptedSimilarity) {
                    shouldExploreMinSim = false;
                } else {
                    break;
                }
            }
            frontier[size++] = candidates.pop();
        }
        return size;
    }

    private int append(final int numNodes, final int node) {
        if (numNodes == bulkNodes.length) {
            bulkNodes = ArrayUtil.grow(bulkNodes, numNodes + 1);
            bulkScores = ArrayUtil.growExact(bulkScores, bulkNodes.length);
        }
        bulkNodes[numNodes] = node;
        return numNodes + 1;
    }

    /** Scores {@code bulkNodes[0..numNodes)} in one bulk call and feeds competitive ones to candidates and results. */
    private void scoreBatch(int numNodes) throws IOException {
        numNodes = (int) Math.min((long) numNodes, results.visitLimit() - results.visitedCount());
        if (numNodes <= 0) {
            return;
        }
        results.incVisitedCount(numNodes);
        scorer.bulkScore(bulkNodes, bulkScores, numNodes);
        for (int i = 0; i < numNodes; i++) {
            final int node = bulkNodes[i];
            final float score = bulkScores[i];
            if (score < minAcceptedSimilarity) {
                continue;
            }
            candidates.add(node, score);
            if ((acceptOrds == null || acceptOrds.get(node)) && results.collect(node, score)) {
                final float oldMinAcceptedSimilarity = minAcceptedSimilarity;
                minAcceptedSimilarity = Math.nextUp(results.minCompetitiveSimilarity());
                if (minAcceptedSimilarity > oldMinAcceptedSimilarity) {
                    shouldExploreMinSim = true;
                }
            }
        }
    }
}
