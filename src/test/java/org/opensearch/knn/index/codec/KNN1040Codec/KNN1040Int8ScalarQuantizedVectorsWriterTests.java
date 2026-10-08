/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer.QuantizationResult;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.util.UnitTestCodec;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import static org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding.UNSIGNED_BYTE;

/**
 * Tests for {@link KNN1040Int8ScalarQuantizedVectorsWriter}: the files it writes must be readable by the stock Lucene104 reader and
 * score like the exact similarity function, through flush, merge and index-sorted segments.
 */
public class KNN1040Int8ScalarQuantizedVectorsWriterTests extends KNNTestCase {

    private static final String FIELD_NAME = "vector";
    private static final int DIMENSION = 64;
    private static final int NUM_VECTORS = 100;

    private static final class WrittenSegment {
        final SegmentReadState readState;
        final float[][] vectors;

        WrittenSegment(final SegmentReadState readState, final float[][] vectors) {
            this.readState = readState;
            this.vectors = vectors;
        }
    }

    private float[] randomVector(final VectorSimilarityFunction similarity) {
        final float[] vector = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            vector[i] = (float) random().nextGaussian();
        }
        if (similarity == VectorSimilarityFunction.DOT_PRODUCT) {
            VectorUtil.l2normalize(vector);
        }
        return vector;
    }

    @SneakyThrows
    private WrittenSegment writeSegment(final MMapDirectory dir, final VectorSimilarityFunction similarity) {
        final FieldInfo fieldInfo = new FieldInfo(
            FIELD_NAME,
            0,
            false,
            false,
            false,
            IndexOptions.NONE,
            DocValuesType.NONE,
            DocValuesSkipIndexType.NONE,
            -1,
            Map.of(),
            0,
            0,
            0,
            DIMENSION,
            VectorEncoding.FLOAT32,
            similarity,
            false,
            false
        );
        final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });
        final SegmentInfo segmentInfo = new SegmentInfo(
            dir,
            Version.LATEST,
            Version.LATEST,
            "_0",
            NUM_VECTORS,
            false,
            false,
            null,
            Collections.emptyMap(),
            StringHelper.randomId(),
            new HashMap<>(),
            null
        );
        final SegmentWriteState writeState = new SegmentWriteState(
            InfoStream.NO_OUTPUT,
            dir,
            segmentInfo,
            fieldInfos,
            null,
            IOContext.DEFAULT
        );

        final float[][] vectors = new float[NUM_VECTORS][];
        final KNN1040ScalarQuantizedVectorsFormat format = new KNN1040ScalarQuantizedVectorsFormat(UNSIGNED_BYTE);
        try (FlatVectorsWriter writer = format.fieldsWriter(writeState)) {
            @SuppressWarnings("unchecked")
            final FlatFieldVectorsWriter<float[]> fieldWriter = (FlatFieldVectorsWriter<float[]>) writer.addField(fieldInfo);
            for (int i = 0; i < NUM_VECTORS; i++) {
                vectors[i] = randomVector(similarity);
                // The writer keeps the instance it is given, so hand it a copy and keep the original for comparison.
                fieldWriter.addValue(i, vectors[i].clone());
            }
            writer.flush(NUM_VECTORS, null);
            writer.finish();
        }
        return new WrittenSegment(new SegmentReadState(dir, segmentInfo, fieldInfos, IOContext.DEFAULT), vectors);
    }

    public void testFlush_whenReadWithStockReader_thenScoresMatchExactSimilarity() throws Exception {
        // Tolerances are absolute and per similarity, and sized at roughly 5 standard deviations of the int8 error for
        // random gaussian 64-d vectors (a dot product between two of them is off by about 0.08 from the quantization of
        // both sides): euclidean scores are 1 / (1 + d²) so they are tiny, cosine and dot product are in [0, 1], and max
        // inner product is a shifted dot product.
        final Map<VectorSimilarityFunction, Float> tolerances = Map.of(
            VectorSimilarityFunction.EUCLIDEAN,
            1e-4f,
            VectorSimilarityFunction.COSINE,
            5e-3f,
            VectorSimilarityFunction.DOT_PRODUCT,
            5e-3f,
            VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT,
            0.5f
        );
        for (final Map.Entry<VectorSimilarityFunction, Float> entry : tolerances.entrySet()) {
            final VectorSimilarityFunction similarity = entry.getKey();
            try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
                final WrittenSegment segment = writeSegment(dir, similarity);
                final KNN1040ScalarQuantizedVectorsFormat format = new KNN1040ScalarQuantizedVectorsFormat(UNSIGNED_BYTE);
                try (FlatVectorsReader reader = format.fieldsReader(segment.readState)) {
                    for (int q = 0; q < 5; q++) {
                        final float[] query = randomVector(similarity);
                        final RandomVectorScorer scorer = reader.getRandomVectorScorer(FIELD_NAME, query.clone());
                        for (int ord = 0; ord < NUM_VECTORS; ord++) {
                            assertEquals(
                                similarity.name() + " ord=" + ord,
                                similarity.compare(query, segment.vectors[ord]),
                                scorer.score(ord),
                                entry.getValue()
                            );
                        }
                    }
                }
            }
        }
    }

    public void testFlush_thenStoredCodesAndCentroidMatchTheSimpleQuantizer() throws Exception {
        for (final VectorSimilarityFunction similarity : new VectorSimilarityFunction[] {
            VectorSimilarityFunction.EUCLIDEAN,
            VectorSimilarityFunction.COSINE }) {
            try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
                final WrittenSegment segment = writeSegment(dir, similarity);
                final KNN1040ScalarQuantizedVectorsFormat format = new KNN1040ScalarQuantizedVectorsFormat(UNSIGNED_BYTE);
                try (FlatVectorsReader reader = format.fieldsReader(segment.readState)) {
                    final FloatVectorValues floatValues = reader.getFloatVectorValues(FIELD_NAME);
                    final QuantizedByteVectorValues quantized = KNN1040ScalarQuantizedUtils.extractQuantizedByteVectorValues(floatValues);

                    assertEquals(UNSIGNED_BYTE, quantized.getScalarEncoding());
                    assertEquals(0f, quantized.getCentroidDP(), 0f);
                    for (final float c : quantized.getCentroid()) {
                        assertEquals(0f, c, 0f);
                    }

                    final Int8ScalarQuantizer quantizer = new Int8ScalarQuantizer(similarity);
                    for (int ord = 0; ord < NUM_VECTORS; ord++) {
                        final float[] expectedInput = segment.vectors[ord].clone();
                        if (similarity == VectorSimilarityFunction.COSINE) {
                            VectorUtil.l2normalize(expectedInput);
                        }
                        final byte[] expectedCodes = new byte[DIMENSION];
                        final QuantizationResult expected = quantizer.scalarQuantize(
                            expectedInput,
                            expectedCodes,
                            Int8ScalarQuantizer.BITS,
                            null
                        );

                        assertArrayEquals(similarity.name() + " ord=" + ord, expectedCodes, quantized.vectorValue(ord).clone());
                        assertEquals(expected, quantized.getCorrectiveTerms(ord));
                    }
                }
            }
        }
    }

    // Only UNSIGNED_BYTE is redirected to the int8 writer; every other encoding keeps going through Lucene's writer.
    public void testFieldsWriter_whenEncoding_thenOnlyUnsignedByteUsesInt8Writer() throws Exception {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[0]);
            for (final ScalarEncoding encoding : ScalarEncoding.values()) {
                final SegmentInfo segmentInfo = new SegmentInfo(
                    dir,
                    Version.LATEST,
                    Version.LATEST,
                    "_" + encoding.name().toLowerCase(java.util.Locale.ROOT),
                    1,
                    false,
                    false,
                    null,
                    Collections.emptyMap(),
                    StringHelper.randomId(),
                    new HashMap<>(),
                    null
                );
                final SegmentWriteState writeState = new SegmentWriteState(
                    InfoStream.NO_OUTPUT,
                    dir,
                    segmentInfo,
                    fieldInfos,
                    null,
                    IOContext.DEFAULT
                );
                try (FlatVectorsWriter writer = new KNN1040ScalarQuantizedVectorsFormat(encoding).fieldsWriter(writeState)) {
                    assertEquals(encoding.name(), encoding == UNSIGNED_BYTE, writer instanceof KNN1040Int8ScalarQuantizedVectorsWriter);
                }
            }
        }
    }

    // ---------------------------------------------------------------------------------------------------------------
    // End to end: HNSW over int8 codes through flush, merge and index-sorted segments.
    // ---------------------------------------------------------------------------------------------------------------

    public void testHnsw_whenFlushedSegments_thenRecallIsHigh() throws Exception {
        assertRecall(VectorSimilarityFunction.EUCLIDEAN, false, false);
    }

    public void testHnsw_whenMerged_thenRecallIsHigh() throws Exception {
        assertRecall(VectorSimilarityFunction.EUCLIDEAN, true, false);
    }

    public void testHnsw_whenMergedCosine_thenRecallIsHigh() throws Exception {
        assertRecall(VectorSimilarityFunction.COSINE, true, false);
    }

    public void testHnsw_whenIndexSortedAndMerged_thenRecallIsHigh() throws Exception {
        assertRecall(VectorSimilarityFunction.EUCLIDEAN, true, true);
    }

    @SneakyThrows
    private void assertRecall(final VectorSimilarityFunction similarity, final boolean merge, final boolean sorted) {
        final int dimension = 32;
        final int numDocs = 1200;
        final int k = 10;
        final int numQueries = 20;
        final Random random = random();

        final float[][] vectors = new float[numDocs][dimension];
        for (final float[] vector : vectors) {
            for (int d = 0; d < dimension; d++) {
                vector[d] = (float) random.nextGaussian();
            }
        }

        // A tiny segments threshold of 0 forces a real graph even for the small flushed segments.
        final Codec codec = new UnitTestCodec(() -> new KNN1040HnswScalarQuantizedVectorsFormat(UNSIGNED_BYTE, 16, 100, 1, null, 0));
        final IndexWriterConfig iwc = new IndexWriterConfig().setCodec(codec);
        if (sorted) {
            iwc.setIndexSort(new Sort(new SortField("sort_key", SortField.Type.LONG)));
        }

        try (Directory dir = newDirectory(); IndexWriter writer = new IndexWriter(dir, iwc)) {
            for (int i = 0; i < numDocs; i++) {
                final Document doc = new Document();
                doc.add(new KnnFloatVectorField(FIELD_NAME, vectors[i], similarity));
                doc.add(new StoredField("id", i));
                doc.add(new NumericDocValuesField("sort_key", random.nextInt(1_000_000)));
                writer.addDocument(doc);
                if ((i + 1) % 400 == 0) {
                    writer.commit();
                }
            }
            if (merge) {
                writer.forceMerge(1);
            }
            writer.commit();

            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                if (merge) {
                    assertEquals(1, reader.leaves().size());
                }
                final IndexSearcher searcher = new IndexSearcher(reader);
                double recall = 0;
                for (int q = 0; q < numQueries; q++) {
                    final float[] query = vectors[random.nextInt(numDocs)].clone();
                    for (int d = 0; d < dimension; d++) {
                        query[d] += (float) random.nextGaussian() * 0.1f;
                    }
                    final Set<Integer> expected = exactTopK(vectors, query, similarity, k);
                    final TopDocs topDocs = searcher.search(new KnnFloatVectorQuery(FIELD_NAME, query, k), k);
                    int hits = 0;
                    for (final ScoreDoc scoreDoc : topDocs.scoreDocs) {
                        final int id = searcher.storedFields().document(scoreDoc.doc).getField("id").numericValue().intValue();
                        if (expected.contains(id)) {
                            hits++;
                        }
                    }
                    recall += (double) hits / k;
                }
                recall /= numQueries;
                assertTrue(
                    "recall@" + k + " was " + recall + " (similarity=" + similarity + ", merge=" + merge + ", sorted=" + sorted + ")",
                    recall >= 0.8
                );
            }
        }
    }

    private static Set<Integer> exactTopK(
        final float[][] vectors,
        final float[] query,
        final VectorSimilarityFunction similarity,
        final int k
    ) {
        final List<Integer> ids = new ArrayList<>();
        final float[] scores = new float[vectors.length];
        for (int i = 0; i < vectors.length; i++) {
            ids.add(i);
            scores[i] = similarity.compare(query, vectors[i]);
        }
        ids.sort((a, b) -> Float.compare(scores[b], scores[a]));
        return new HashSet<>(ids.subList(0, k));
    }
}
