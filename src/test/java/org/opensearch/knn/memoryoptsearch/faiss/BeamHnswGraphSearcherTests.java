/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.SneakyThrows;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.KNNTestCase;

import java.util.Arrays;
import java.util.Random;

import static org.opensearch.knn.memoryoptsearch.FaissHNSWTests.loadHnswBinary;

public class BeamHnswGraphSearcherTests extends KNNTestCase {
    private static final int NUM_VECTORS = 100;
    private static final int K = 10;

    @SneakyThrows
    public void testBeamWidthOneMatchesLuceneSeededSearch() {
        final RandomVectorScorer scorer = randomScorer(42);
        final int entry = graph().entryNode();

        final TopKnnCollector lucene = seededCollector(entry);
        HnswGraphSearcher.search(scorer, lucene, graph(), null);

        final TopKnnCollector beam = seededCollector(entry);
        BeamHnswGraphSearcher.search(scorer, beam, graph(), null, new int[] { entry }, 1);

        assertArrayEquals(docs(lucene.topDocs().scoreDocs), docs(beam.topDocs().scoreDocs));
        assertEquals(lucene.visitedCount(), beam.visitedCount());
    }

    @SneakyThrows
    public void testWideBeamReturnsTopKWithTrueScores() {
        final RandomVectorScorer scorer = randomScorer(7);
        final TopKnnCollector beam = seededCollector(graph().entryNode());
        BeamHnswGraphSearcher.search(scorer, beam, graph(), null, new int[] { graph().entryNode() }, 8);

        final ScoreDoc[] top = beam.topDocs().scoreDocs;
        assertEquals(K, top.length);
        for (ScoreDoc sd : top) {
            assertEquals(scorer.score(sd.doc), sd.score, 0f);
        }
        // A wider beam explores at least as much of the graph as Lucene's single-candidate search.
        final TopKnnCollector lucene = seededCollector(graph().entryNode());
        HnswGraphSearcher.search(scorer, lucene, graph(), null);
        assertTrue(beam.visitedCount() >= lucene.visitedCount());
    }

    @SneakyThrows
    public void testAcceptOrdsFilterCollection() {
        final RandomVectorScorer scorer = randomScorer(3);
        final TopKnnCollector beam = seededCollector(graph().entryNode());
        // Odd ordinals only: they are still traversed but never collected.
        BeamHnswGraphSearcher.search(scorer, beam, graph(), new Bits() {
            @Override
            public boolean get(int index) {
                return index % 2 == 1;
            }

            @Override
            public int length() {
                return NUM_VECTORS;
            }
        }, new int[] { graph().entryNode() }, 4);

        for (ScoreDoc sd : beam.topDocs().scoreDocs) {
            assertEquals(1, sd.doc % 2);
        }
    }

    @SneakyThrows
    private static FaissHnswGraph graph() {
        final FaissHNSW faissHNSW = new FaissHNSW();
        faissHNSW.load(loadHnswBinary("data/memoryoptsearch/faiss_hnsw_100_vectors.bin"), NUM_VECTORS);
        return new FaissHnswGraph(faissHNSW, loadHnswBinary("data/memoryoptsearch/faiss_hnsw_100_vectors.bin"));
    }

    @SneakyThrows
    private static TopKnnCollector seededCollector(final int entry) {
        return new TopKnnCollector(
            K,
            Integer.MAX_VALUE,
            new KnnSearchStrategy.Seeded(DocIdSetIterator.range(entry, entry + 1), 1, KnnSearchStrategy.Hnsw.DEFAULT)
        );
    }

    private static RandomVectorScorer randomScorer(final long seed) {
        final Random random = new Random(seed);
        final float[] scores = new float[NUM_VECTORS];
        for (int i = 0; i < NUM_VECTORS; i++) {
            scores[i] = random.nextFloat();
        }
        return new RandomVectorScorer() {
            @Override
            public float score(int node) {
                return scores[node];
            }

            @Override
            public int maxOrd() {
                return NUM_VECTORS;
            }
        };
    }

    private static int[] docs(final ScoreDoc[] scoreDocs) {
        return Arrays.stream(scoreDocs).mapToInt(sd -> sd.doc).toArray();
    }
}
