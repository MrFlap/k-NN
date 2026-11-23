/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec.rabitq;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.util.UnitTestCodec;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

/**
 * End to end through Lucene: HNSW traversal on the 1-bit coarse tier plus the 7-bit refine finds the true nearest
 * neighbors, before and after merging, for every similarity, on a sparse field, and the scores it returns are the
 * refine estimates (close to the exact similarity), not the coarse ones.
 */
public class KNN1040HnswRaBitQVectorsFormatTests extends KNNTestCase {

    private static final String FIELD = "v";
    private static final int DIM = 100; // not a multiple of 8 or 16: hits every bit-plane and SIMD tail
    private static final int N = 3000;
    private static final int K = 10;
    // Traversal scores 1-bit estimates, so it needs ef well above k (ef = 64 gives ~0.8 recall@10 here; fp32 ~0.98).
    private static final int EF = 200;
    private static final int QUERIES = 30;

    public void testInnerProduct() throws Exception {
        assertRecallAndScores(VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT);
    }

    public void testEuclidean() throws Exception {
        assertRecallAndScores(VectorSimilarityFunction.EUCLIDEAN);
    }

    public void testCosine() throws Exception {
        assertRecallAndScores(VectorSimilarityFunction.COSINE);
    }

    private void assertRecallAndScores(VectorSimilarityFunction similarity) throws Exception {
        final Random random = new Random(random().nextLong());
        // Every 7th doc has no vector, so ordinals differ from doc ids.
        final float[][] vectors = new float[N][];
        for (int i = 0; i < N; i++) {
            if (i % 7 == 3) {
                continue;
            }
            vectors[i] = new float[DIM];
            for (int d = 0; d < DIM; d++) {
                vectors[i][d] = (float) random.nextGaussian();
            }
        }
        try (Directory dir = newDirectory()) {
            final IndexWriterConfig iwc = new IndexWriterConfig().setCodec(new UnitTestCodec(KNN1040HnswRaBitQVectorsFormat::new))
                .setMaxBufferedDocs(700)
                .setMergePolicy(newLogMergePolicy(100)); // keep several segments until the explicit merge
            try (IndexWriter writer = new IndexWriter(dir, iwc)) {
                for (int i = 0; i < N; i++) {
                    final Document doc = new Document();
                    doc.add(new StringField("id", Integer.toString(i), Field.Store.YES));
                    if (vectors[i] != null) {
                        doc.add(new KnnFloatVectorField(FIELD, vectors[i], similarity));
                    }
                    writer.addDocument(doc);
                }
                writer.commit();
                try (DirectoryReader reader = DirectoryReader.open(writer)) {
                    assertTrue("expected several segments", reader.leaves().size() > 1);
                    check(reader, vectors, similarity, random);
                }
                writer.forceMerge(1);
                try (DirectoryReader reader = DirectoryReader.open(writer)) {
                    assertEquals(1, reader.leaves().size());
                    check(reader, vectors, similarity, random);
                }
            }
        }
    }

    private void check(DirectoryReader reader, float[][] vectors, VectorSimilarityFunction similarity, Random random) throws Exception {
        final IndexSearcher searcher = new IndexSearcher(reader);
        final StoredFields stored = reader.storedFields();
        double recall = 0;
        double absError = 0;
        int scored = 0;
        for (int q = 0; q < QUERIES; q++) {
            final float[] query = new float[DIM];
            final float[] base = vectors[random.nextInt(N / 7) * 7];
            for (int d = 0; d < DIM; d++) {
                query[d] = base[d] + (float) random.nextGaussian() * 0.5f;
            }
            final TopDocs hits = searcher.search(new KnnFloatVectorQuery(FIELD, query, EF), K);
            final Set<Integer> found = new HashSet<>();
            for (ScoreDoc hit : hits.scoreDocs) {
                final int id = Integer.parseInt(stored.document(hit.doc).get("id"));
                found.add(id);
                absError += Math.abs(hit.score - similarity.compare(query, vectors[id]));
                scored++;
            }
            for (int id : exactTopK(vectors, query, similarity)) {
                if (found.contains(id)) {
                    recall++;
                }
            }
        }
        recall /= (double) QUERIES * K;
        assertTrue(similarity + " recall@" + K + " = " + recall, recall >= 0.9);
        // Refined scores track the exact similarity far better than the 1-bit tier alone would.
        final double meanAbsError = absError / scored;
        final double scale = scoreScale(vectors, similarity);
        assertTrue(similarity + " mean |score - exact| = " + meanAbsError + " vs scale " + scale, meanAbsError < 0.05 * scale);
    }

    /** Spread of the exact similarities of the top hits, as a yardstick for the score error. */
    private static double scoreScale(float[][] vectors, VectorSimilarityFunction similarity) {
        final float[] q = vectors[0];
        float lo = Float.POSITIVE_INFINITY;
        float hi = Float.NEGATIVE_INFINITY;
        for (float[] v : vectors) {
            if (v != null) {
                final float s = similarity.compare(q, v);
                lo = Math.min(lo, s);
                hi = Math.max(hi, s);
            }
        }
        return hi - lo;
    }

    private static int[] exactTopK(float[][] vectors, float[] query, VectorSimilarityFunction similarity) {
        final Integer[] ids = new Integer[vectors.length];
        int n = 0;
        for (int i = 0; i < vectors.length; i++) {
            if (vectors[i] != null) {
                ids[n++] = i;
            }
        }
        final Integer[] live = Arrays.copyOf(ids, n);
        Arrays.sort(live, (a, b) -> Float.compare(similarity.compare(query, vectors[b]), similarity.compare(query, vectors[a])));
        final int[] top = new int[K];
        for (int i = 0; i < K; i++) {
            top[i] = live[i];
        }
        return top;
    }
}
