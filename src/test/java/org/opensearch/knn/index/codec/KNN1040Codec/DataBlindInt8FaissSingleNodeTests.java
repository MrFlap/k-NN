/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.util.BytesRef;
import org.opensearch.action.bulk.BulkRequestBuilder;
import org.opensearch.action.bulk.BulkResponse;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.IndexService;
import org.opensearch.index.engine.Engine;
import org.opensearch.index.mapper.Uid;
import org.opensearch.index.shard.IndexShard;
import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.knn.index.mapper.RandomOrthogonalVectorTransformer;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.search.SearchHit;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * End to end through a Faiss HNSW field at {@code compression_level: 4x}: vectors are rotated, stored as data-blind int8
 * codes, searched with memory-optimized search, and merged without changing the codes (no centroid, so re-encoding the
 * merged floats reproduces each segment's bytes).
 */
public class DataBlindInt8FaissSingleNodeTests extends KNNSingleNodeTestCase {

    private static final String INDEX = "int8";
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
            .field("compression_level", "4x")
            .startObject("method")
            .field("name", "hnsw")
            .field("engine", "faiss")
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        client().admin().indices().preparePutMapping(INDEX).setSource(mapping).get();

        final Random random = new Random(42);
        final float[][] docs = randomVectors(random, NUM_DOCS, dim);
        final float[][] queries = randomVectors(random, NUM_QUERIES, dim);

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
        final Map<String, byte[]> before = assertInt8AndCollectCodes(shard, NUM_SEGMENTS);
        final double recallBefore = recall(docs, queries, spaceType);

        client().admin().indices().prepareForceMerge(INDEX).setMaxNumSegments(1).get();
        client().admin().indices().prepareRefresh(INDEX).get();

        final Map<String, byte[]> after = assertInt8AndCollectCodes(shard, 1);
        final double recallAfter = recall(docs, queries, spaceType);
        logger.info("space={} dim={} recall@{} before merge={} after merge={}", spaceType, dim, K, recallBefore, recallAfter);

        assertEquals(NUM_DOCS, after.size());
        for (Map.Entry<String, byte[]> e : before.entrySet()) {
            assertArrayEquals("int8 codes changed by merge for doc " + e.getKey(), e.getValue(), after.get(e.getKey()));
        }
        assertTrue("recall before merge too low: " + recallBefore, recallBefore >= 0.9);
        assertTrue("recall after merge too low: " + recallAfter, recallAfter >= 0.9);
    }

    /** Asserts every segment stores rotated data-blind int8 codes and returns them keyed by _id. */
    private Map<String, byte[]> assertInt8AndCollectCodes(final IndexShard shard, final int expectedSegments) throws Exception {
        final Map<String, byte[]> codesById = new HashMap<>();
        try (Engine.Searcher searcher = shard.acquireSearcher("test")) {
            assertEquals(expectedSegments, searcher.getIndexReader().leaves().size());
            for (LeafReaderContext ctx : searcher.getIndexReader().leaves()) {
                final LeafReader leaf = FilterLeafReader.unwrap(ctx.reader());
                final FieldInfo fieldInfo = leaf.getFieldInfos().fieldInfo(FIELD);
                assertTrue("vectors are rotated", RandomOrthogonalVectorTransformer.isAppliedTo(fieldInfo.attributes()));

                final PerFieldKnnVectorsFormat.FieldsReader perField = (PerFieldKnnVectorsFormat.FieldsReader) ((SegmentReader) leaf)
                    .getVectorReader();
                final Faiss1040ScalarQuantizedKnnVectorsReader reader = (Faiss1040ScalarQuantizedKnnVectorsReader) perField.getFieldReader(
                    FIELD
                );
                final ScalarQuantizedFloatVectorValues values = (ScalarQuantizedFloatVectorValues) reader.getFlatVectorsReader()
                    .getFloatVectorValues(FIELD);
                final DataBlindInt8VectorValues codes = (DataBlindInt8VectorValues) values.getQuantizedVectorValues();

                final var storedFields = leaf.storedFields();
                for (int ord = 0; ord < codes.size(); ord++) {
                    final BytesRef idBytes = storedFields.document(values.getFloatVectorValues().ordToDoc(ord)).getBinaryValue("_id");
                    codesById.put(Uid.decodeId(idBytes.bytes, idBytes.offset, idBytes.length), codes.vectorValue(ord).clone());
                }
            }
        }
        return codesById;
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
            double dot = 0;
            double l2 = 0;
            double norm = 0;
            for (int d = 0; d < query.length; d++) {
                dot += docs[i][d] * query[d];
                l2 += (docs[i][d] - query[d]) * (docs[i][d] - query[d]);
                norm += docs[i][d] * docs[i][d];
            }
            scores[i] = switch (spaceType) {
                case "l2" -> -l2;
                case "cosinesimil" -> dot / Math.sqrt(norm);
                default -> dot;
            };
        }
        Arrays.sort(ids, (a, b) -> Double.compare(scores[b], scores[a]));
        return Arrays.stream(ids, 0, K).mapToInt(Integer::intValue).toArray();
    }

    private static float[][] randomVectors(final Random random, final int n, final int dim) {
        // A shared non-zero mean, so a centroid would matter; data-blind quantizes against zero.
        final float[] mean = new float[dim];
        final Random meanRandom = new Random(7);
        for (int d = 0; d < dim; d++) {
            mean[d] = meanRandom.nextFloat() * 4 - 2;
        }
        final float[][] vectors = new float[n][dim];
        for (float[] v : vectors) {
            for (int d = 0; d < dim; d++) {
                v[d] = (float) random.nextGaussian() + mean[d];
            }
        }
        return vectors;
    }
}
