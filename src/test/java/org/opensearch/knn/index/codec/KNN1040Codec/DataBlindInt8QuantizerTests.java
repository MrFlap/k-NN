/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.knn.KNNTestCase;

import java.util.Random;

/**
 * Encode then score, checked against the full-precision similarity it estimates. Ported from ClusterANN's
 * {@code DataBlindInt8QuantizerTests}.
 */
public class DataBlindInt8QuantizerTests extends KNNTestCase {

    public void testScore_whenMaximumInnerProduct_thenEstimatesTrueInnerProduct() {
        for (int dim : new int[] { 8, 16, 128, 768, 1024 }) {
            final Random random = new Random(dim * 7L);
            final float[] query = gaussian(random, dim);
            final float[] doc = gaussian(random, dim);

            final float est = score(query, doc, VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT);

            // Invert the MIP transform: est = dot + 1 (dot >= 0) or 1 / (1 - dot) (dot < 0).
            final double recoveredDot = est >= 1f ? (est - 1.0) : (1.0 - 1.0 / est);
            assertEquals("dim " + dim, dot(query, doc), recoveredDot, 0.05 * Math.sqrt(dim));
        }
    }

    public void testScore_whenEuclidean_thenEstimatesTrueSquaredDistance() {
        for (int dim : new int[] { 8, 128, 768 }) {
            final Random random = new Random(dim * 11L);
            final float[] query = gaussian(random, dim);
            final float[] doc = gaussian(random, dim);

            final float est = score(query, doc, VectorSimilarityFunction.EUCLIDEAN);

            // est = 1 / (1 + d²)
            final double trueD2 = squaredDistance(query, doc);
            assertEquals("dim " + dim, trueD2, 1.0 / est - 1.0, 0.01 * trueD2);
        }
    }

    public void testScore_whenIdenticalVectors_thenSelfDistanceIsNearZero() {
        final float[] v = gaussian(new Random(1L), 256);
        assertEquals(1f, score(v, v.clone(), VectorSimilarityFunction.EUCLIDEAN), 1e-3f);
    }

    public void testEncode_whenZeroVector_thenOffsetCodesWithUnitScaleAndZeroSum() {
        final int dim = 32;
        final byte[] code = new byte[dim];
        final DataBlindInt8Quantizer.Int8Code c = new DataBlindInt8Quantizer(dim).encode(new float[dim], code);

        assertEquals(1f, c.scale(), 0f);
        assertEquals(0, c.sum());
        for (byte b : code) {
            assertEquals((byte) DataBlindInt8Quantizer.OFFSET, b);
        }
    }

    public void testEncode_whenLargestCoordinate_thenItMapsToTheGridEdge() {
        final byte[] code = new byte[3];
        final DataBlindInt8Quantizer.Int8Code c = new DataBlindInt8Quantizer(3).encode(new float[] { -2f, 1f, 0f }, code);

        assertEquals(2f / 127, c.scale(), 0f);
        assertEquals(1, code[0] & 0xFF);       // -127 + 128
        assertEquals(128 + 64, code[1] & 0xFF); // round(63.5) = 64
        assertEquals(128, code[2] & 0xFF);
        assertEquals(-127 + 64, c.sum());
        assertEquals(5f, c.sqNorm(), 0f);
    }

    public void testUnsignedDot_whenAnyLengthAndOffset_thenMatchesNaiveLoop() {
        final Random random = new Random(3L);
        for (int dim = 1; dim < 40; dim++) {
            final byte[] query = new byte[dim];
            final byte[] docs = new byte[dim + 5];
            random.nextBytes(query);
            random.nextBytes(docs);
            long expected = 0;
            for (int i = 0; i < dim; i++) {
                expected += (long) (query[i] & 0xFF) * (docs[5 + i] & 0xFF);
            }
            assertEquals("dim " + dim, expected, DataBlindInt8Quantizer.unsignedDot(query, docs, 5, dim));
        }
    }

    public void testScore_whenDotProduct_thenThrows() {
        final DataBlindInt8Quantizer.QueryTerms terms = new DataBlindInt8Quantizer.QueryTerms(1f, 1f, 0L);
        expectThrows(
            IllegalStateException.class,
            () -> DataBlindInt8Quantizer.score(0L, terms, 1f, 0, 1f, VectorSimilarityFunction.DOT_PRODUCT)
        );
    }

    static float score(final float[] query, final float[] doc, final VectorSimilarityFunction sim) {
        final int dim = query.length;
        final DataBlindInt8Quantizer quantizer = new DataBlindInt8Quantizer(dim);
        final byte[] qCode = new byte[dim];
        final byte[] dCode = new byte[dim];
        final DataBlindInt8Quantizer.QueryTerms terms = quantizer.queryTerms(quantizer.encode(query, qCode));
        final DataBlindInt8Quantizer.Int8Code d = quantizer.encode(doc, dCode);
        final long unsignedDot = DataBlindInt8Quantizer.unsignedDot(qCode, dCode, 0, dim);
        return DataBlindInt8Quantizer.score(unsignedDot, terms, d.scale(), d.sum(), d.sqNorm(), sim);
    }

    static float[] gaussian(final Random random, final int dim) {
        final float[] v = new float[dim];
        for (int i = 0; i < dim; i++) {
            v[i] = (float) random.nextGaussian();
        }
        return v;
    }

    private static double dot(final float[] a, final float[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += (double) a[i] * b[i];
        }
        return s;
    }

    private static double squaredDistance(final float[] a, final float[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            final double d = a[i] - b[i];
            s += d * d;
        }
        return s;
    }
}
