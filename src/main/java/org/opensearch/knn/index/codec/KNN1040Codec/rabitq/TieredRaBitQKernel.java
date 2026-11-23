/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec.rabitq;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.util.VectorUtil;
import org.opensearch.knn.jni.SimdVectorComputeService;

import java.util.Random;

/**
 * The integer inner products of a tiered RaBitQ scan, as in the ClusterANN POC's {@code TieredRaBitQKernel}: the
 * native SIMD kernels ({@code rabitq_tiered_kernel.cpp}) when they load and agree with the Java loops, else the Java
 * loops ({@link VectorUtil#int4BitDotProduct} and {@link VectorUtil#dotProduct}).
 */
@Log4j2
public final class TieredRaBitQKernel {

    /** Kill switch for the native kernels; false scores with the Java SIMD loops. */
    static final boolean USE_NATIVE = Boolean.parseBoolean(System.getProperty("knn.rabitq.native", "true")) && nativeAgrees();

    private TieredRaBitQKernel() {}

    /** {@code out[i] = sum over j < 4 of 2^j * popcount(planes[j] & bits[i])}, {@code n} rows of {@code bitBytes}. */
    public static void coarseDots(final byte[] planes, final byte[] bits, final int bitBytes, final int n, final int[] out) {
        if (USE_NATIVE) {
            SimdVectorComputeService.rabitqCoarseDots(planes, bits, bitBytes, n, out);
        } else {
            javaCoarseDots(planes, bits, bitBytes, n, out);
        }
    }

    /** {@code out[i] = sum over d < dim of recs[i * stride + d] * q[d]}; codes in [0, 127], q in [-127, 127]. */
    public static void refineDots(final byte[] recs, final int stride, final int m, final int dim, final byte[] q, final int[] out) {
        if (USE_NATIVE) {
            SimdVectorComputeService.rabitqRefineDots(recs, stride, m, dim, q, out);
        } else {
            javaRefineDots(recs, stride, m, dim, q, out);
        }
    }

    static void javaCoarseDots(final byte[] planes, final byte[] bits, final int bitBytes, final int n, final int[] out) {
        final byte[] plane = new byte[bitBytes];
        for (int i = 0; i < n; i++) {
            System.arraycopy(bits, i * bitBytes, plane, 0, bitBytes);
            out[i] = (int) VectorUtil.int4BitDotProduct(planes, plane);
        }
    }

    static void javaRefineDots(final byte[] recs, final int stride, final int m, final int dim, final byte[] q, final int[] out) {
        final byte[] code = new byte[dim];
        for (int r = 0; r < m; r++) {
            System.arraycopy(recs, r * stride, code, 0, dim);
            out[r] = VectorUtil.dotProduct(code, q);
        }
    }

    /** Loads the native kernels and checks them against the Java loops on an input that hits every SIMD tail. */
    private static boolean nativeAgrees() {
        try {
            final Random random = new Random(42);
            final int dim = 1000;
            final int bitBytes = (dim + 7) >>> 3;
            final int n = 5;
            final byte[] planes = new byte[4 * bitBytes];
            final byte[] bits = new byte[n * bitBytes];
            random.nextBytes(planes);
            random.nextBytes(bits);
            final int[] a = new int[n];
            final int[] b = new int[n];
            SimdVectorComputeService.rabitqCoarseDots(planes, bits, bitBytes, n, a);
            javaCoarseDots(planes, bits, bitBytes, n, b);
            final int stride = dim + 8;
            final byte[] recs = new byte[n * stride];
            final byte[] q = new byte[dim];
            for (int i = 0; i < recs.length; i++) {
                recs[i] = (byte) random.nextInt(128);
            }
            for (int d = 0; d < dim; d++) {
                q[d] = (byte) (random.nextInt(255) - 127);
            }
            final int[] c = new int[n];
            final int[] e = new int[n];
            SimdVectorComputeService.rabitqRefineDots(recs, stride, n, dim, q, c);
            javaRefineDots(recs, stride, n, dim, q, e);
            if (java.util.Arrays.equals(a, b) && java.util.Arrays.equals(c, e)) {
                log.info("RaBitQ tiered native kernels enabled");
                return true;
            }
            log.warn("RaBitQ tiered native kernels disagree with Java; using Java loops");
        } catch (Throwable t) {
            log.warn("RaBitQ tiered native kernels unavailable; using Java loops", t);
        }
        return false;
    }
}
