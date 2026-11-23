/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec.rabitq;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.VectorUtil;

/**
 * A query's encodings and estimator terms against one segment's centroid: the ClusterANN POC's
 * {@code TieredRaBitQCluster.Context} plus its per-cluster terms, with the segment as the one cluster.
 *
 * <p>The query is encoded to 4 bits for the coarse tier, so {@code <top bits, q>} is one 4-plane popcount over the
 * packed bit plane, and to 8 bits for the refine tier, so {@code <code, q>} is one byte dot product. Euclidean fields
 * subtract the stored {@code <top bits, c>} on the coarse tier and encode {@code q - c} itself for the refine tier.
 */
final class TieredRaBitQQuery {

    final byte[] planes;
    final float lo;
    final float delta;
    /** 8-bit refine query: {@code q} for inner product, {@code q - c} for Euclidean. */
    final byte[] q8;
    final float delta8;
    private final boolean l2;
    private final boolean cosine;
    private final float sumQp;
    private final float qcResNormSq;
    private final float qc;
    private static final float COARSE_HALF = ExtendedRaBitQ.half(TieredRaBitQRecords.COARSE_BITS);
    private static final float REFINE_HALF = ExtendedRaBitQ.half(TieredRaBitQRecords.REFINE_BITS);

    /** @param q the query, already normalized for cosine fields */
    TieredRaBitQQuery(final float[] q, final float[] c, final VectorSimilarityFunction similarity) {
        final int d = q.length;
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        float maxAbs = 0f;
        double sum = 0;
        for (final float v : q) {
            min = Math.min(min, v);
            max = Math.max(max, v);
            maxAbs = Math.max(maxAbs, Math.abs(v));
            sum += v;
        }
        final float delta = max > min ? (max - min) / 15f : 1f;
        final int bytes = TieredRaBitQRecords.bitBytes(d);
        final byte[] planes = new byte[4 * bytes];
        final float delta8 = maxAbs > 0 ? maxAbs / 127f : 1f;
        final byte[] q8 = new byte[d];
        for (int i = 0; i < d; i++) {
            final int u = Math.min(15, Math.max(0, Math.round((q[i] - min) / delta)));
            for (int j = 0; j < 4; j++) {
                if (((u >>> j) & 1) != 0) {
                    planes[j * bytes + (i >>> 3)] |= (byte) (1 << (i & 7));
                }
            }
            q8[i] = (byte) Math.min(127, Math.max(-127, Math.round(q[i] / delta8)));
        }
        this.planes = planes;
        this.lo = min;
        this.delta = delta;
        this.l2 = similarity == VectorSimilarityFunction.EUCLIDEAN;
        this.cosine = similarity == VectorSimilarityFunction.COSINE;

        // Cell terms: Euclidean estimates <o, q - c>, inner product <o, q>.
        final float normSq = VectorUtil.dotProduct(q, q);
        this.qc = VectorUtil.dotProduct(q, c);
        float sumC = 0;
        if (l2) {
            for (final float v : c) {
                sumC += v;
            }
        }
        this.sumQp = l2 ? (float) sum - sumC : (float) sum;
        this.qcResNormSq = l2 ? Math.max(0f, normSq - 2 * qc + VectorUtil.dotProduct(c, c)) : 0f;
        // The refine scores are the final scores, so a Euclidean field encodes q - c itself: rounding q alone would
        // scale the error with |q| rather than |q - c|.
        if (l2) {
            this.q8 = new byte[d];
            this.delta8 = encodeResidual(q, c, this.q8);
        } else {
            this.q8 = q8;
            this.delta8 = delta8;
        }
    }

    /** Coarse estimate from the top bit of the code; {@code dots} is the 4-plane popcount. */
    float coarse(final int dots, final int pop, final float resNorm, final float gC, final float oC, final float bC) {
        float codeDot = lo * pop + delta * dots;
        if (l2) {
            codeDot -= bC;
        }
        final float g = gC <= 0 ? 1f : gC;
        final float od = oC == 0 ? 1f : oC;
        return similarity((codeDot - COARSE_HALF * sumQp) / g / od, resNorm);
    }

    /** Refine estimate from all bits of the code; {@code dot} is {@code <code, q8>}. */
    float refine(final int dot, final float resNorm, final float gR, final float oR) {
        final float g = gR <= 0 ? 1f : gR;
        final float od = oR == 0 ? 1f : oR;
        return similarity((delta8 * dot - REFINE_HALF * sumQp) / g / od, resNorm);
    }

    /** Rounds {@code q - c} to signed 8-bit codes in {@code out}; returns the step. */
    private static float encodeResidual(final float[] q, final float[] c, final byte[] out) {
        float maxAbs = 0f;
        for (int d = 0; d < out.length; d++) {
            maxAbs = Math.max(maxAbs, Math.abs(q[d] - c[d]));
        }
        final float delta = maxAbs > 0 ? maxAbs / 127f : 1f;
        final float inv = 1f / delta;
        for (int d = 0; d < out.length; d++) {
            out[d] = (byte) Math.min(127, Math.max(-127, Math.round((q[d] - c[d]) * inv)));
        }
        return delta;
    }

    /** The prototype's estimate-to-similarity mapping, given the estimated {@code <o, q'>}. */
    private float similarity(final float est, final float resNorm) {
        if (l2) {
            final float d2 = Math.max(0f, qcResNormSq - 2f * resNorm * est + resNorm * resNorm);
            return 1f / (1f + d2);
        }
        final float dot = qc + resNorm * est;
        if (cosine) {
            return Math.max(0f, (1f + dot) / 2f);
        }
        return dot >= 0 ? dot + 1f : 1f / (1f - dot);
    }
}
