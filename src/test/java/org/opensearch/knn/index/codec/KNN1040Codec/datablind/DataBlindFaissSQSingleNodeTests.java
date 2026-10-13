/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec.datablind;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;
import org.opensearch.action.bulk.BulkRequestBuilder;
import org.opensearch.action.bulk.BulkResponse;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.IndexService;
import org.opensearch.index.engine.Engine;
import org.opensearch.index.shard.IndexShard;
import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.knn.index.codec.KNN1040Codec.Faiss1040ScalarQuantizedFlatVectorsReader;
import org.opensearch.knn.index.codec.KNN1040Codec.Faiss1040ScalarQuantizedKnnVectorsReader;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.search.SearchHit;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * POC end-to-end check that Faiss SQ 4-bit (x8) fields are written data-blind with floats, are searchable (including
 * rescoring), and merge without changing quantized bytes (zero centroid, so re-quantization is deterministic).
 */
public class DataBlindFaissSQSingleNodeTests extends KNNSingleNodeTestCase {

    private static final String INDEX = "datablind";
    private static final String FIELD = "v";
    private static final int NUM_DOCS = 2000;
    private static final int NUM_SEGMENTS = 4;
    private static final int NUM_QUERIES = 20;
    private static final int K = 10;

    public void testL2() throws Exception {
        runEndToEnd("l2", 128);
    }

    public void testInnerProduct() throws Exception {
        runEndToEnd("innerproduct", 128);
    }

    public void testCosine() throws Exception {
        // Lucene 10.5's OptimizedScalarQuantizer asserts a unit-length centroid for COSINE; the zero centroid only
        // passes with the #16030 change to that (unvendorable) class. Runs fine without -ea, e.g. on benchmark nodes.
        boolean assertionsEnabled = false;
        assert assertionsEnabled = true;
        assumeFalse("COSINE + zero centroid trips Lucene's -ea assertion", assertionsEnabled);
        runEndToEnd("cosinesimil", 128);
    }

    public void testOddDimension() throws Exception {
        runEndToEnd("l2", 33);
    }

    private void runEndToEnd(final String spaceType, final int dim) throws Exception {
        final IndexService indexService = createIndex(
            INDEX,
            Settings.builder().put(getKNNDefaultIndexSettingsBuildsGraphAlways()).put("index.refresh_interval", "-1").build()
        );
        final XContentBuilder mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD)
            .field("type", "knn_vector")
            .field("dimension", dim)
            .field("space_type", spaceType)
            .field("compression_level", "8x")
            .startObject("method")
            .field("name", "hnsw")
            .field("engine", "faiss")
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        client().admin().indices().preparePutMapping(INDEX).setSource(mapping).get();

        final Random random = new Random(42);
        final float[][] docs = randomVectors(random, NUM_DOCS, dim, spaceType);
        final float[][] queries = randomVectors(random, NUM_QUERIES, dim, spaceType);

        final int perSegment = NUM_DOCS / NUM_SEGMENTS;
        for (int s = 0; s < NUM_SEGMENTS; s++) {
            final BulkRequestBuilder bulk = client().prepareBulk();
            for (int i = s * perSegment; i < (s + 1) * perSegment; i++) {
                bulk.add(client().prepareIndex(INDEX).setId(Integer.toString(i)).setSource(FIELD, docs[i]));
            }
            final BulkResponse response = bulk.get();
            assertFalse(response.buildFailureMessage(), response.hasFailures());
            client().admin().indices().prepareFlush(INDEX).get();
            client().admin().indices().prepareRefresh(INDEX).get();
        }

        final IndexShard shard = indexService.getShard(0);
        final Map<String, byte[]> before = assertDataBlindAndCollectQuantized(shard, NUM_SEGMENTS);
        final double recallBefore = recall(docs, queries, spaceType);

        client().admin().indices().prepareForceMerge(INDEX).setMaxNumSegments(1).get();
        client().admin().indices().prepareRefresh(INDEX).get();

        final Map<String, byte[]> after = assertDataBlindAndCollectQuantized(shard, 1);
        final double recallAfter = recall(docs, queries, spaceType);
        logger.info("space={} dim={} recall@{} before merge={} after merge={}", spaceType, dim, K, recallBefore, recallAfter);

        // With a zero centroid there is nothing to recompute on merge, so quantized bytes must be identical.
        assertEquals(NUM_DOCS, after.size());
        for (Map.Entry<String, byte[]> e : before.entrySet()) {
            assertArrayEquals("quantized bytes changed by merge for doc " + e.getKey(), e.getValue(), after.get(e.getKey()));
        }
        assertTrue("recall before merge too low: " + recallBefore, recallBefore >= 0.6);
        assertTrue("recall after merge too low: " + recallAfter, recallAfter >= 0.6);
    }

    /** Asserts every segment was written DATA_BLIND_WITH_FLOATS and returns quantized bytes keyed by _id. */
    private Map<String, byte[]> assertDataBlindAndCollectQuantized(final IndexShard shard, final int expectedSegments) throws Exception {
        final Map<String, byte[]> quantizedById = new HashMap<>();
        try (Engine.Searcher searcher = shard.acquireSearcher("test")) {
            assertEquals(expectedSegments, searcher.getIndexReader().leaves().size());
            for (LeafReaderContext ctx : searcher.getIndexReader().leaves()) {
                final LeafReader leaf = FilterLeafReader.unwrap(ctx.reader());
                KnnVectorsReader reader = ((SegmentReader) leaf).getVectorReader();
                reader = ((PerFieldKnnVectorsFormat.FieldsReader) reader).getFieldReader(FIELD);
                reader = ((Faiss1040ScalarQuantizedKnnVectorsReader) reader).getFlatVectorsReader();
                final Lucene104ScalarQuantizedVectorsReader sqReader =
                    (Lucene104ScalarQuantizedVectorsReader) ((Faiss1040ScalarQuantizedFlatVectorsReader) reader)
                        .getDelegateFlatVectorsReader();
                assertEquals(Lucene104ScalarQuantizedVectorsFormat.Mode.DATA_BLIND_WITH_FLOATS, sqReader.getMode(FIELD));

                final QuantizedByteVectorValues qvv = sqReader.getQuantizedVectorValues(FIELD);
                final var storedFields = leaf.storedFields();
                for (int ord = 0; ord < qvv.size(); ord++) {
                    final int doc = qvv.ordToDoc(ord);
                    final org.apache.lucene.util.BytesRef idBytes = storedFields.document(doc).getBinaryValue("_id");
                    final String id = org.opensearch.index.mapper.Uid.decodeId(idBytes.bytes, idBytes.offset, idBytes.length);
                    quantizedById.put(id, qvv.vectorValue(ord).clone());
                }
            }
        }
        return quantizedById;
    }

    private double recall(final float[][] docs, final float[][] queries, final String spaceType) {
        int hits = 0;
        for (float[] query : queries) {
            final SearchResponse response = client().prepareSearch(INDEX)
                .setQuery(new KNNQueryBuilder(FIELD, query, K))
                .setSize(K)
                .setFetchSource(false)
                .get();
            final Set<Integer> found = new HashSet<>();
            for (SearchHit hit : response.getHits().getHits()) {
                found.add(Integer.parseInt(hit.getId()));
            }
            for (int id : groundTruth(docs, query, spaceType)) {
                if (found.contains(id)) {
                    hits++;
                }
            }
        }
        return (double) hits / (queries.length * K);
    }

    private static int[] groundTruth(final float[][] docs, final float[] query, final String spaceType) {
        final Integer[] ids = new Integer[docs.length];
        final double[] scores = new double[docs.length];
        for (int i = 0; i < docs.length; i++) {
            ids[i] = i;
            double s = 0;
            for (int d = 0; d < query.length; d++) {
                s += "l2".equals(spaceType) ? -(docs[i][d] - query[d]) * (docs[i][d] - query[d]) : docs[i][d] * query[d];
            }
            scores[i] = s;
        }
        Arrays.sort(ids, (a, b) -> Double.compare(scores[b], scores[a]));
        return Arrays.stream(ids, 0, K).mapToInt(Integer::intValue).toArray();
    }

    private static float[][] randomVectors(final Random random, final int n, final int dim, final String spaceType) {
        // Offset by a shared non-zero mean so centering would matter; data-blind quantizes against zero.
        final float[] mean = new float[dim];
        final Random meanRandom = new Random(7);
        for (int d = 0; d < dim; d++) {
            mean[d] = meanRandom.nextFloat() * 4 - 2;
        }
        final float[][] vectors = new float[n][dim];
        for (int i = 0; i < n; i++) {
            double norm = 0;
            for (int d = 0; d < dim; d++) {
                vectors[i][d] = (float) random.nextGaussian() + mean[d];
                norm += vectors[i][d] * vectors[i][d];
            }
            if ("cosinesimil".equals(spaceType)) {
                for (int d = 0; d < dim; d++) {
                    vectors[i][d] /= (float) Math.sqrt(norm);
                }
            }
        }
        return vectors;
    }
}
