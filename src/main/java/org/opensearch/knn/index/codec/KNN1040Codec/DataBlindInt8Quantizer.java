/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.index.VectorSimilarityFunction;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;

/**
 * Data-blind absolute int8 quantization, ported from ClusterANN's {@code DataBlindInt8Quantizer} and
 * {@code Int8DotProduct} (SearchServicesKnnVectorFormats). Queries and documents are encoded the same way, so one
 * scoring formula serves graph build, Java search and native SIMD search.
 *
 * <p><b>Data-blind.</b> Nothing is trained and no centroid is used: a vector is quantized absolutely, in the rotated
 * space the random orthogonal transform already produced. The only per-vector quantity is {@code maxAbs/127}.
 *
 * <p><b>Encoding.</b> {@code scale = maxAbs/127}; each coordinate is {@code q = clip(round(v[d]/scale), -127, 127)},
 * stored as the unsigned byte {@code q + 128}. Three corrections travel with the code: {@code scale}, {@code Σq} and
 * {@code ‖v‖²}.
 *
 * <p><b>Scoring.</b> With the query encoded the same way:
 * <pre>
 *   signedDot = Σ(qᵤ·dᵤ) − 128·Σq_query − 16384·dim − 128·Σq_doc
 *   dot       = signedDot · qScale · dScale
 * </pre>
 * then the metric transform, which yields Lucene-range scores.
 */
public final class DataBlindInt8Quantizer {

    /** Unsigned-byte offset: a signed code {@code q ∈ [-127,127]} is stored as {@code q + 128 ∈ [1,255]}. */
    public static final int OFFSET = 128;

    private static final int MAX_SIGNED = 127;
    private static final long OFFSET_SQ = (long) OFFSET * OFFSET;
    private static final VarHandle LONG_LE = MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);

    private final int dimension;

    public DataBlindInt8Quantizer(final int dimension) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("dimension must be positive, got " + dimension);
        }
        this.dimension = dimension;
    }

    /**
     * The corrections stored alongside one vector's code.
     *
     * @param scale  {@code maxAbs/127}
     * @param sum    {@code Σ q} over signed codes
     * @param sqNorm {@code ‖v‖²}
     */
    public record Int8Code(float scale, int sum, float sqNorm) {
    }

    /**
     * The query-only terms of the score, computed once per query.
     *
     * @param scale          the query's {@code maxAbs/127}
     * @param sqNorm         the query's {@code ‖q‖²}
     * @param offsetResidual {@code 128·Σq_query + 16384·dim}
     */
    public record QueryTerms(float scale, float sqNorm, long offsetResidual) {
    }

    public QueryTerms queryTerms(final Int8Code query) {
        final long offsetResidual = (long) OFFSET * query.sum() + OFFSET_SQ * dimension;
        return new QueryTerms(query.scale(), query.sqNorm(), offsetResidual);
    }

    /**
     * Quantizes {@code vector} into {@code codeOut} (one unsigned byte per dimension) and returns its corrections. A zero
     * vector is encoded at the offset with {@code scale = 1} and {@code sum = 0}, which scoring handles without a special
     * case.
     */
    public Int8Code encode(final float[] vector, final byte[] codeOut) {
        float maxAbs = 0f;
        double sqNorm = 0;
        for (int d = 0; d < dimension; d++) {
            final float x = vector[d];
            final float a = Math.abs(x);
            if (a > maxAbs) {
                maxAbs = a;
            }
            sqNorm += (double) x * x;
        }
        if (maxAbs == 0f) {
            for (int d = 0; d < dimension; d++) {
                codeOut[d] = (byte) OFFSET;
            }
            return new Int8Code(1f, 0, (float) sqNorm);
        }
        final float scale = maxAbs / MAX_SIGNED;
        final float inv = MAX_SIGNED / maxAbs;
        int sum = 0;
        for (int d = 0; d < dimension; d++) {
            int q = Math.round(vector[d] * inv);
            if (q > MAX_SIGNED) {
                q = MAX_SIGNED;
            } else if (q < -MAX_SIGNED) {
                q = -MAX_SIGNED;
            }
            sum += q;
            codeOut[d] = (byte) (q + OFFSET);
        }
        return new Int8Code(scale, sum, (float) sqNorm);
    }

    /**
     * Similarity of a query against one document code, in Lucene's score range for {@code sim}.
     *
     * @param unsignedDot {@code Σ(qᵤ·dᵤ)} from {@link #unsignedDot}
     * @param query       the query's precomputed terms
     * @param docScale    the document's {@code scale}
     * @param docSum      the document's {@code Σq}
     * @param docSqNorm   the document's {@code ‖v‖²}
     * @param sim         the field's similarity function
     */
    public static float score(
        final long unsignedDot,
        final QueryTerms query,
        final float docScale,
        final int docSum,
        final float docSqNorm,
        final VectorSimilarityFunction sim
    ) {
        final long signedDot = unsignedDot - query.offsetResidual() - (long) OFFSET * docSum;
        final double dot = (double) signedDot * query.scale() * docScale;
        switch (sim) {
            case EUCLIDEAN: {
                double s = query.sqNorm() + docSqNorm - 2.0 * dot;
                if (s < 0) {
                    s = 0;
                }
                return (float) (1.0 / (1.0 + s));
            }
            case MAXIMUM_INNER_PRODUCT:
                return dot >= 0 ? (float) (dot + 1.0) : (float) (1.0 / (1.0 - dot));
            case COSINE:
                return (float) Math.max(0.0, (1.0 + dot) / 2.0);
            default:
                throw new IllegalStateException("Data-blind int8 does not support " + sim);
        }
    }

    /** {@code Σᵢ (query[i]&0xFF)·(docs[offset+i]&0xFF)} for {@code i} in {@code [0, dim)}. */
    public static long unsignedDot(final byte[] query, final byte[] docs, final int offset, final int dim) {
        long sum = 0;
        int i = 0;
        // Eight bytes at a time through a little-endian VarHandle, so the JIT can vectorize the widening multiply.
        for (final int bound = dim & -Long.BYTES; i < bound; i += Long.BYTES) {
            final long q = (long) LONG_LE.get(query, i);
            final long d = (long) LONG_LE.get(docs, offset + i);
            for (int lane = 0; lane < Long.BYTES; lane++) {
                final int shift = lane << 3;
                sum += ((q >>> shift) & 0xFF) * ((d >>> shift) & 0xFF);
            }
        }
        for (; i < dim; i++) {
            sum += (long) (query[i] & 0xFF) * (docs[offset + i] & 0xFF);
        }
        return sum;
    }
}
