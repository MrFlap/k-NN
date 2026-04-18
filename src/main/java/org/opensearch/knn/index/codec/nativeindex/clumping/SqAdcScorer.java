/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.clumping;

import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene104.QuantizedByteVectorValues;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

/**
 * Java reference implementation of the 4-bit query ⊗ 1-bit data ADC score used by the native SIMD
 * path (see {@code default_simd_similarity_function.cpp}). This is the same formula used by the
 * Faiss SQ search context and produces scores on the same scale.
 *
 * <p>The class is used at build time by {@link ClumpingIndexBuildStrategy} to pick the top-N
 * candidate markers for a hidden vector without having to spin up the native SIMD context.
 * Correctness — not throughput — is the goal here, since this path only runs during index build.
 *
 * <h2>Score formula</h2>
 * <pre>
 *   qcDist  = sum over bit planes p in [0..3] of popcount(plane_p AND data) &lt;&lt; p
 *   raw     = ax*ay*dim + ay*lx*x1 + ax*ly*y1 + lx*ly*qcDist
 *   IP:   score = raw + queryAdditional + dataAdditional - centroidDp           (then ipToMaxIp)
 *   L2:   score = max(0, queryAdditional + dataAdditional - 2 * raw)            (then 1/(1+x))
 * </pre>
 * where {@code ay, ly, queryAdditional, y1} come from the 4-bit quantized query and
 * {@code ax, lx, dataAdditional, x1} from the per-vector data corrections. {@code FOUR_BIT_SCALE}
 * is applied to the query's {@code (upper-lower)} exactly as in the native code.
 */
final class SqAdcScorer {

    private static final float FOUR_BIT_SCALE = 1.0f / 15.0f;

    private SqAdcScorer() {}

    /**
     * Quantizes a fresh query vector into 4 bit-planes ready for ADC scoring against 1-bit data
     * codes. The four bit planes are laid out contiguously, {@code binaryCodeBytes} each — this
     * is exactly the {@code targetQuantized} layout produced by
     * {@code KNN1040ScalarQuantizedVectorScorer} before it's handed to the SIMD context.
     *
     * <p>NOTE: {@code queryCopy} is mutated by {@code OptimizedScalarQuantizer.scalarQuantize},
     * matching the behavior expected by callers in the SIMD path.
     */
    static QuantizedQuery quantizeQuery(
        QuantizedByteVectorValues qbvv,
        float[] queryCopy
    ) throws java.io.IOException {
        final Lucene104ScalarQuantizedVectorsFormat.ScalarEncoding encoding = qbvv.getScalarEncoding();
        final OptimizedScalarQuantizer quantizer = qbvv.getQuantizer();
        final int dim = qbvv.dimension();
        // Discretized dimension count (one entry per dim) used to size the scratch buffer
        // that scalarQuantize writes into — see Lucene104ScalarQuantizedVectorScorer for the
        // canonical sizing. This is NOT the per-plane byte count.
        final int discretizedDim = encoding.getDiscreteDimensions(dim);
        // Per-bit-plane byte count, matching the native SIMD path's binaryCodeBytes
        // (see default_simd_similarity_function.cpp: binaryCodeBytes = (dim + 7) / 8).
        final int binaryCodeBytes = (dim + 7) / 8;

        final byte[] scratch = new byte[discretizedDim];
        final byte[] targetQuantized = new byte[encoding.getQueryPackedLength(discretizedDim)];

        final OptimizedScalarQuantizer.QuantizationResult q = quantizer.scalarQuantize(
            queryCopy,
            scratch,
            encoding.getQueryBits(),
            qbvv.getCentroid()
        );

        OptimizedScalarQuantizer.transposeHalfByte(scratch, targetQuantized);

        return new QuantizedQuery(
            targetQuantized,
            binaryCodeBytes,
            q.lowerInterval(),
            q.upperInterval(),
            q.additionalCorrection(),
            q.quantizedComponentSum()
        );
    }

    /**
     * Scores a single 1-bit data code + corrections against the quantized query using the
     * IP (inner product) ADC formula. The centroid dot-product is provided separately because it
     * is a segment-level scalar, not per-vector.
     */
    static float scoreIp(QuantizedQuery q, byte[] dataCode, SqVectorEntry corrections, int dimension, float centroidDp) {
        final int qcDist = int4BitDotProduct(q.planes, dataCode, q.binaryCodeBytes);
        final float ay = q.lowerInterval;
        final float ly = (q.upperInterval - q.lowerInterval) * FOUR_BIT_SCALE;
        final float y1 = q.quantizedComponentSum;

        final float ax = corrections.lowerInterval;
        final float lx = corrections.upperInterval - corrections.lowerInterval;
        final float x1 = corrections.quantizedComponentSum;

        float raw = ax * ay * dimension
            + ay * lx * x1
            + ax * ly * y1
            + lx * ly * qcDist;
        return raw + q.additionalCorrection + corrections.additionalCorrection - centroidDp;
    }

    /**
     * Scores a single 1-bit data code + corrections against the quantized query using the
     * L2 ADC formula.
     */
    static float scoreL2(QuantizedQuery q, byte[] dataCode, SqVectorEntry corrections, int dimension) {
        final int qcDist = int4BitDotProduct(q.planes, dataCode, q.binaryCodeBytes);
        final float ay = q.lowerInterval;
        final float ly = (q.upperInterval - q.lowerInterval) * FOUR_BIT_SCALE;
        final float y1 = q.quantizedComponentSum;

        final float ax = corrections.lowerInterval;
        final float lx = corrections.upperInterval - corrections.lowerInterval;
        final float x1 = corrections.quantizedComponentSum;

        float raw = ax * ay * dimension
            + ay * lx * x1
            + ax * ly * y1
            + lx * ly * qcDist;
        return Math.max(0f, q.additionalCorrection + corrections.additionalCorrection - 2f * raw);
    }

    /**
     * Core int4BitDotProduct. See the native reference in
     * {@code default_simd_similarity_function.cpp}. The Java port uses 64-bit longs for the
     * main word loop and falls back to bytes for the tail.
     */
    private static int int4BitDotProduct(byte[] q, byte[] d, int binaryCodeBytes) {
        int result = 0;
        final int words = binaryCodeBytes >>> 3;
        final int remainStart = words << 3;

        for (int bitPlane = 0; bitPlane < 4; bitPlane++) {
            int sub = 0;
            final int planeOffset = bitPlane * binaryCodeBytes;

            for (int w = 0; w < words; w++) {
                int off = w * 8;
                long qWord = readLongLE(q, planeOffset + off);
                long dWord = readLongLE(d, off);
                sub += Long.bitCount(qWord & dWord);
            }

            for (int r = remainStart; r < binaryCodeBytes; r++) {
                sub += Integer.bitCount((q[planeOffset + r] & d[r]) & 0xff);
            }

            result += sub << bitPlane;
        }
        return result;
    }

    private static long readLongLE(byte[] buf, int off) {
        return ((long) (buf[off] & 0xff))
            | ((long) (buf[off + 1] & 0xff) << 8)
            | ((long) (buf[off + 2] & 0xff) << 16)
            | ((long) (buf[off + 3] & 0xff) << 24)
            | ((long) (buf[off + 4] & 0xff) << 32)
            | ((long) (buf[off + 5] & 0xff) << 40)
            | ((long) (buf[off + 6] & 0xff) << 48)
            | ((long) (buf[off + 7] & 0xff) << 56);
    }

    /**
     * Carrier for a query vector that has already been 4-bit quantized and transposed into bit
     * planes. Constructed once per search (or per hidden-vector assignment during build) and
     * reused across all scoring calls.
     */
    static final class QuantizedQuery {
        final byte[] planes;          // 4 bit planes × binaryCodeBytes
        final int binaryCodeBytes;
        final float lowerInterval;
        final float upperInterval;
        final float additionalCorrection;
        final int quantizedComponentSum;

        QuantizedQuery(byte[] planes, int binaryCodeBytes, float l, float u, float add, int qcs) {
            this.planes = planes;
            this.binaryCodeBytes = binaryCodeBytes;
            this.lowerInterval = l;
            this.upperInterval = u;
            this.additionalCorrection = add;
            this.quantizedComponentSum = qcs;
        }
    }
}
