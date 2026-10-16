/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.codec.KNN1040Codec.DataBlindInt8Quantizer;
import org.opensearch.knn.index.codec.KNN1040Codec.Faiss1040DataBlindInt8KnnVectorsFormat;
import org.opensearch.knn.index.warmup.WarmableReader;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;

import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.stream.IntStream;

/**
 * End to end through {@link Faiss1040DataBlindInt8KnnVectorsFormat}: writes float vectors of a {@code bits=8} SQ field,
 * which stores data-blind int8 codes and builds the native Faiss HNSW graph from them, then reads the segment back.
 */
public class Faiss1040DataBlindInt8FormatTests extends KNNTestCase {

    static final int DIMENSIONS = 24;
    static final int NUM_DOCS = 300;
    static final String FIELD_NAME = "target_field";

    @SneakyThrows
    public void testWriteThenRead_whenL2_thenInt8CodesAndGraphAreRead() {
        assertWriteThenRead("l2", VectorSimilarityFunction.EUCLIDEAN);
    }

    @SneakyThrows
    public void testWriteThenRead_whenInnerProduct_thenInt8CodesAndGraphAreRead() {
        assertWriteThenRead("innerproduct", VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT);
    }

    @SneakyThrows
    public void testSearch_whenL2_thenHitsCarryInt8ScoresAndMatchBruteForce() {
        assertSearch("l2", VectorSimilarityFunction.EUCLIDEAN, 1);
    }

    @SneakyThrows
    public void testSearch_whenInnerProduct_thenHitsCarryInt8ScoresAndMatchBruteForce() {
        assertSearch("innerproduct", VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT, 1);
    }

    @SneakyThrows
    public void testSearch_whenSparseDocs_thenOrdinalsAreMappedToDocs() {
        assertSearch("l2", VectorSimilarityFunction.EUCLIDEAN, 3);
    }

    private void assertSearch(final String spaceType, final VectorSimilarityFunction similarity, final int docStride) throws Exception {
        final int k = 10;
        try (Directory directory = new MMapDirectory(createTempDir("faiss1040_int8_search"))) {
            final Random random = new Random(7);
            final float[][] vectors = randomVectors(random, NUM_DOCS, DIMENSIONS);
            final SegmentReadState readState = write(directory, spaceType, similarity, vectors, docStride);
            try (KnnVectorsReader reader = new Faiss1040DataBlindInt8KnnVectorsFormat().fieldsReader(readState)) {
                final float[] query = randomVectors(random, 1, DIMENSIONS)[0];
                final TopKnnCollector collector = new TopKnnCollector(k, Integer.MAX_VALUE, KnnSearchStrategy.Hnsw.DEFAULT);
                reader.search(FIELD_NAME, query, collector, AcceptDocs.fromLiveDocs(null, NUM_DOCS * docStride));
                final TopDocs topDocs = collector.topDocs();
                assertEquals(k, topDocs.scoreDocs.length);

                final float[] int8Scores = new float[NUM_DOCS];
                for (int ord = 0; ord < NUM_DOCS; ord++) {
                    int8Scores[ord] = int8Score(query, vectors[ord], similarity);
                }
                final Set<Integer> bruteForceTopK = new HashSet<>();
                IntStream.range(0, NUM_DOCS)
                    .boxed()
                    .sorted((a, b) -> Float.compare(int8Scores[b], int8Scores[a]))
                    .limit(k)
                    .forEach(ord -> bruteForceTopK.add(ord * docStride));

                int matched = 0;
                for (ScoreDoc hit : topDocs.scoreDocs) {
                    assertEquals("doc " + hit.doc + " has no vector", 0, hit.doc % docStride);
                    assertEquals("score of doc " + hit.doc, int8Scores[hit.doc / docStride], hit.score, 0f);
                    matched += bruteForceTopK.contains(hit.doc) ? 1 : 0;
                }
                assertTrue("recall@" + k + " was " + matched, matched >= 9);
            }
        }
    }

    /** The Java int8 score of {@code doc} against {@code query}, as the searcher computes it. */
    static float int8Score(final float[] query, final float[] doc, final VectorSimilarityFunction similarity) {
        final DataBlindInt8Quantizer quantizer = new DataBlindInt8Quantizer(query.length);
        final byte[] queryCode = new byte[query.length];
        final byte[] docCode = new byte[query.length];
        final DataBlindInt8Quantizer.QueryTerms terms = quantizer.queryTerms(quantizer.encode(query, queryCode));
        final DataBlindInt8Quantizer.Int8Code d = quantizer.encode(doc, docCode);
        final long unsignedDot = DataBlindInt8Quantizer.unsignedDot(queryCode, docCode, 0, query.length);
        return DataBlindInt8Quantizer.score(unsignedDot, terms, d.scale(), d.sum(), d.sqNorm(), similarity);
    }

    private void assertWriteThenRead(final String spaceType, final VectorSimilarityFunction similarity) throws Exception {
        try (Directory rawDirectory = new MMapDirectory(createTempDir("faiss1040_int8"))) {
            final float[][] vectors = randomVectors(new Random(42), NUM_DOCS, DIMENSIONS);
            final SegmentReadState readState = write(rawDirectory, spaceType, similarity, vectors, 1);
            final FaissMemoryOptimizedSearcherTests.ReadTrackingDirectory spyDirectory =
                new FaissMemoryOptimizedSearcherTests.ReadTrackingDirectory(rawDirectory);
            final SegmentReadState spyReadState = new SegmentReadState(
                spyDirectory,
                readState.segmentInfo,
                readState.fieldInfos,
                IOContext.DEFAULT
            );

            try (KnnVectorsReader reader = new Faiss1040DataBlindInt8KnnVectorsFormat().fieldsReader(spyReadState)) {
                assertEquals(NUM_DOCS, reader.getFloatVectorValues(FIELD_NAME).size());
                spyDirectory.resetReadFlags();
                ((WarmableReader) reader).warmUp(FIELD_NAME);
                assertTrue("Warmup should read the .faiss graph", spyDirectory.wasExtensionRead(".faiss"));
                assertTrue("Warmup should read the int8 codes", spyDirectory.wasExtensionRead(".vdbi"));
                assertTrue("Warmup should read the float vectors", spyDirectory.wasExtensionRead(".vec"));
            }
        }
    }

    /** Flushes {@code vectors} as one segment, vector {@code i} on doc {@code i * docStride}, and returns a read state. */
    static SegmentReadState write(
        final Directory directory,
        final String spaceType,
        final VectorSimilarityFunction similarity,
        final float[][] vectors,
        final int docStride
    ) throws Exception {
        final int maxDoc = vectors.length * docStride;
        final FieldInfo fieldInfo = fieldInfo(spaceType, similarity);
        final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });
        final SegmentInfo segInfo = new SegmentInfo(
            directory,
            Version.LATEST,
            Version.LATEST,
            "_0",
            maxDoc,
            false,
            false,
            null,
            Collections.emptyMap(),
            StringHelper.randomId(),
            Collections.emptyMap(),
            null
        );
        final SegmentWriteState writeState = new SegmentWriteState(
            InfoStream.NO_OUTPUT,
            directory,
            segInfo,
            fieldInfos,
            null,
            IOContext.DEFAULT
        );
        try (KnnVectorsWriter writer = new Faiss1040DataBlindInt8KnnVectorsFormat().fieldsWriter(writeState)) {
            @SuppressWarnings("unchecked")
            final KnnFieldVectorsWriter<float[]> fw = (KnnFieldVectorsWriter<float[]>) writer.addField(fieldInfo);
            for (int ord = 0; ord < vectors.length; ord++) {
                fw.addValue(ord * docStride, vectors[ord]);
            }
            writer.flush(maxDoc, null);
            writer.finish();
        }
        final Set<String> files = new HashSet<>();
        for (String file : directory.listAll()) {
            if (file.startsWith("_")) {
                files.add(file);
            }
        }
        segInfo.setFiles(files);
        return new SegmentReadState(directory, segInfo, fieldInfos, IOContext.DEFAULT);
    }

    private static FieldInfo fieldInfo(final String spaceType, final VectorSimilarityFunction similarity) {
        final String parameters = "{\"index_description\":\"BHNSW16,Flat\",\"spaceType\":\""
            + spaceType
            + "\",\"name\":\"hnsw\",\"data_type\":\"float\",\"parameters\":{\"ef_search\":256,\"ef_construction\":256,\"m\":16,"
            + "\"encoder\":{\"name\":\"sq\",\"bits\":8}}}";
        return new FieldInfo(
            FIELD_NAME,
            0,
            false,
            false,
            false,
            IndexOptions.NONE,
            DocValuesType.NONE,
            DocValuesSkipIndexType.NONE,
            -1,
            Map.of(
                KNNConstants.PARAMETERS,
                parameters,
                KNNConstants.VECTOR_DATA_TYPE_FIELD,
                "float",
                KNNConstants.KNN_ENGINE,
                "faiss",
                KNNConstants.SQ_CONFIG,
                "bits=8",
                KNNVectorFieldMapper.KNN_FIELD,
                "true",
                KNNConstants.SPACE_TYPE,
                spaceType
            ),
            0,
            0,
            0,
            DIMENSIONS,
            VectorEncoding.FLOAT32,
            similarity,
            false,
            false
        );
    }

    static float[][] randomVectors(final Random random, final int n, final int dim) {
        final float[][] vectors = new float[n][dim];
        for (float[] v : vectors) {
            for (int d = 0; d < dim; d++) {
                v[d] = (float) random.nextGaussian();
            }
        }
        return vectors;
    }
}
