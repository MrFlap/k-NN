/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.util.UnitTestCodec;

import java.util.Random;

/**
 * Writes {@link KNN1040DataBlindInt8VectorsFormat} segments and checks that every stored record is exactly the encoding
 * of the float vector at the same ordinal, after flush, after an index-sorted flush and after a merge.
 */
public class KNN1040DataBlindInt8VectorsFormatTests extends KNNTestCase {

    private static final String FIELD = "v";
    private static final int DIMENSION = 13;
    private static final int NUM_DOCS = 60;

    @SneakyThrows
    public void testFlush_whenSparseFieldInSeveralSegments_thenCodesMatchFloats() {
        try (Directory dir = newDirectory()) {
            index(dir, newIndexWriterConfig(), false);
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                assertTrue(reader.leaves().size() > 1);
                assertCodesMatchFloats(reader);
            }
        }
    }

    @SneakyThrows
    public void testFlush_whenIndexSorted_thenCodesMatchFloats() {
        try (Directory dir = newDirectory()) {
            index(dir, newIndexWriterConfig().setIndexSort(new Sort(new SortField("sort", SortField.Type.LONG, true))), false);
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                assertCodesMatchFloats(reader);
            }
        }
    }

    @SneakyThrows
    public void testMerge_whenForceMerged_thenCodesMatchFloats() {
        try (Directory dir = newDirectory()) {
            index(dir, newIndexWriterConfig(), true);
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                assertEquals(1, reader.leaves().size());
                assertCodesMatchFloats(reader);
            }
        }
    }

    @SneakyThrows
    public void testRandomVectorScorer_whenScored_thenMatchesQuantizerFormula() {
        try (Directory dir = newDirectory()) {
            index(dir, newIndexWriterConfig(), true);
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                final FlatVectorsReader flatReader = flatReader(reader.leaves().get(0));
                final FloatVectorValues floats = ((ScalarQuantizedFloatVectorValues) flatReader.getFloatVectorValues(FIELD))
                    .getFloatVectorValues();
                final float[] query = floats.copy().vectorValue(0).clone();
                final RandomVectorScorer scorer = flatReader.getRandomVectorScorer(FIELD, query);
                for (int ord = 0; ord < floats.size(); ord++) {
                    final float[] doc = floats.vectorValue(ord);
                    assertEquals(DataBlindInt8QuantizerTests.score(query, doc, VectorSimilarityFunction.EUCLIDEAN), scorer.score(ord), 0f);
                    assertEquals(floats.ordToDoc(ord), scorer.ordToDoc(ord));
                }
            }
        }
    }

    private void index(final Directory dir, final IndexWriterConfig iwc, final boolean forceMerge) throws Exception {
        iwc.setCodec(new UnitTestCodec(KNN1040DataBlindInt8VectorsFormat::new));
        if (forceMerge == false) {
            iwc.setMergePolicy(NoMergePolicy.INSTANCE);
        }
        final Random random = new Random(42);
        try (IndexWriter writer = new IndexWriter(dir, iwc)) {
            for (int i = 0; i < NUM_DOCS; i++) {
                final Document doc = new Document();
                doc.add(new NumericDocValuesField("sort", random.nextInt(1000)));
                // Every third doc has no vector, so ord != doc.
                if (i % 3 != 0) {
                    doc.add(
                        new KnnFloatVectorField(
                            FIELD,
                            DataBlindInt8QuantizerTests.gaussian(random, DIMENSION),
                            VectorSimilarityFunction.EUCLIDEAN
                        )
                    );
                }
                writer.addDocument(doc);
                if (i % 20 == 19) {
                    writer.flush();
                }
            }
            if (forceMerge) {
                writer.forceMerge(1);
            }
        }
    }

    private static void assertCodesMatchFloats(final DirectoryReader reader) throws Exception {
        final DataBlindInt8Quantizer quantizer = new DataBlindInt8Quantizer(DIMENSION);
        final byte[] expected = new byte[DIMENSION];
        for (LeafReaderContext leaf : reader.leaves()) {
            final ScalarQuantizedFloatVectorValues values = (ScalarQuantizedFloatVectorValues) flatReader(leaf).getFloatVectorValues(FIELD);
            final DataBlindInt8VectorValues codes = (DataBlindInt8VectorValues) values.getQuantizedVectorValues();
            assertEquals(values.size(), codes.size());
            for (int ord = 0; ord < values.size(); ord++) {
                final DataBlindInt8Quantizer.Int8Code c = quantizer.encode(values.vectorValue(ord), expected);
                assertArrayEquals("ord " + ord, expected, codes.vectorValue(ord));
                assertEquals(c.scale(), codes.scale(ord), 0f);
                assertEquals(c.sum(), codes.sum(ord));
                assertEquals(c.sqNorm(), codes.sqNorm(ord), 0f);
            }
        }
    }

    private static FlatVectorsReader flatReader(final LeafReaderContext leaf) {
        final KnnVectorsReader perField = Lucene.segmentReader(leaf.reader()).getVectorReader();
        return (FlatVectorsReader) ((PerFieldKnnVectorsFormat.FieldsReader) perField).getFieldReader(FIELD);
    }
}
