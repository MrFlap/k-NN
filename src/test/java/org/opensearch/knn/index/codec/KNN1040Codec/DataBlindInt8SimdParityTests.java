/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.VectorUtil;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.util.UnitTestCodec;
import org.opensearch.knn.memoryoptsearch.MemorySegmentAddressExtractorUtil;

import java.util.Random;

/**
 * The native data-blind int8 kernel must reproduce {@link DataBlindInt8RandomVectorScorer} bit for bit, through bulk and
 * single scoring, for every similarity and for dimensions that exercise each ISA path's tail handling.
 */
public class DataBlindInt8SimdParityTests extends KNNTestCase {

    private static final String FIELD = "v";
    private static final int NUM_VECTORS = 75; // not a multiple of the kernel's 8 rows
    private static final int[] DIMENSIONS = { 1, 7, 16, 33, 63, 64, 65, 100, 128, 129, 768 };
    private static final VectorSimilarityFunction[] SIMILARITIES = {
        VectorSimilarityFunction.EUCLIDEAN,
        VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT,
        VectorSimilarityFunction.COSINE };

    @SneakyThrows
    public void testNativeScores_whenAnyDimensionAndSimilarity_thenBitIdenticalToJava() {
        for (int dimension : DIMENSIONS) {
            for (VectorSimilarityFunction similarity : SIMILARITIES) {
                assertParity(dimension, similarity);
            }
        }
    }

    private void assertParity(final int dimension, final VectorSimilarityFunction similarity) throws Exception {
        final Random random = new Random(31L * dimension + similarity.ordinal());
        final float[][] vectors = vectors(random, dimension, similarity);
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final IndexWriterConfig iwc = new IndexWriterConfig().setCodec(new UnitTestCodec(KNN1040DataBlindInt8VectorsFormat::new));
            try (IndexWriter writer = new IndexWriter(dir, iwc)) {
                for (float[] vector : vectors) {
                    final Document doc = new Document();
                    doc.add(new KnnFloatVectorField(FIELD, vector, similarity));
                    writer.addDocument(doc);
                }
                writer.forceMerge(1);
            }
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                final ScalarQuantizedFloatVectorValues values =
                    (ScalarQuantizedFloatVectorValues) ((PerFieldKnnVectorsFormat.FieldsReader) Lucene.segmentReader(
                        reader.leaves().get(0).reader()
                    ).getVectorReader()).getFieldReader(FIELD).getFloatVectorValues(FIELD);
                final DataBlindInt8VectorValues codes = (DataBlindInt8VectorValues) values.getQuantizedVectorValues();
                final IndexInput slice = codes.getSlice();
                final long[] addressAndSize = MemorySegmentAddressExtractorUtil.tryExtractAddressAndSize(slice, 0, slice.length());
                assertNotNull("int8 records must be memory-mapped", addressAndSize);

                final float[] query = vectors(random, dimension, similarity)[0];
                final DataBlindInt8RandomVectorScorer java = new DataBlindInt8RandomVectorScorer(
                    values.getFloatVectorValues(),
                    codes.copy(),
                    query,
                    similarity
                );
                final DataBlindInt8BulkSimdRandomVectorScorer simd = new DataBlindInt8BulkSimdRandomVectorScorer(
                    values.getFloatVectorValues(),
                    codes,
                    query,
                    addressAndSize,
                    similarity
                );

                // Bulk, in a shuffled order and a batch size that leaves a tail.
                final int[] ords = new int[NUM_VECTORS];
                for (int i = 0; i < NUM_VECTORS; i++) {
                    ords[i] = (i * 31) % NUM_VECTORS;
                }
                final float[] simdScores = new float[NUM_VECTORS];
                float expectedMax = Float.NEGATIVE_INFINITY;
                final float max = simd.bulkScore(ords, simdScores, NUM_VECTORS);
                for (int i = 0; i < NUM_VECTORS; i++) {
                    final float expected = java.score(ords[i]);
                    expectedMax = Math.max(expectedMax, expected);
                    assertEquals(
                        "dim=" + dimension + " sim=" + similarity + " ord=" + ords[i],
                        Float.floatToIntBits(expected),
                        Float.floatToIntBits(simdScores[i])
                    );
                    assertEquals(Float.floatToIntBits(expected), Float.floatToIntBits(simd.score(ords[i])));
                }
                assertEquals(Float.floatToIntBits(expectedMax), Float.floatToIntBits(max));
            }
        }
    }

    /** Gaussian vectors plus a zero vector and a large-magnitude outlier; unit length for COSINE. */
    private static float[][] vectors(final Random random, final int dimension, final VectorSimilarityFunction similarity) {
        final float[][] vectors = new float[NUM_VECTORS][dimension];
        for (int i = 1; i < NUM_VECTORS; i++) {
            for (int d = 0; d < dimension; d++) {
                vectors[i][d] = (float) random.nextGaussian() * (i == NUM_VECTORS - 1 ? 1000f : 1f);
            }
            if (similarity == VectorSimilarityFunction.COSINE) {
                VectorUtil.l2normalize(vectors[i]);
            }
        }
        if (similarity == VectorSimilarityFunction.COSINE) {
            // A zero vector cannot be indexed under COSINE.
            vectors[0] = vectors[1].clone();
        }
        return vectors;
    }
}
