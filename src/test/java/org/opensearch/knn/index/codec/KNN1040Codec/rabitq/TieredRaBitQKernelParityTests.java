/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec.rabitq;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.jni.SimdVectorComputeService;

import java.util.Random;

/**
 * The native tiered RaBitQ kernels return exactly the integers the Java loops compute, across dimensions that hit every
 * SIMD tail (ported from ESKNN's BulkSimdRaBitQKernelParityTests).
 */
public class TieredRaBitQKernelParityTests extends KNNTestCase {

    private static final int[] DIMS = { 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 384, 768, 1000, 1024, 1536 };

    public void testCoarseDots_matchJavaExactly() {
        assumeTrue("native SIMD library not loadable in this JVM", TieredRaBitQKernel.USE_NATIVE);
        final Random random = new Random(random().nextLong());
        for (int dim : DIMS) {
            final int bitBytes = (dim + 7) >>> 3;
            for (int n : new int[] { 1, 3, 257 }) {
                final byte[] planes = new byte[4 * bitBytes];
                final byte[] bits = new byte[n * bitBytes];
                random.nextBytes(planes);
                random.nextBytes(bits);
                final int[] out = new int[n + 1];
                out[n] = 12345;
                SimdVectorComputeService.rabitqCoarseDots(planes, bits, bitBytes, n, out);
                final int[] expected = new int[n];
                TieredRaBitQKernel.javaCoarseDots(planes, bits, bitBytes, n, expected);
                for (int i = 0; i < n; i++) {
                    assertEquals("dim=" + dim + " n=" + n + " i=" + i, expected[i], out[i]);
                }
                assertEquals("must not write past n", 12345, out[n]);
            }
        }
    }

    public void testRefineDots_matchJavaExactly() {
        assumeTrue("native SIMD library not loadable in this JVM", TieredRaBitQKernel.USE_NATIVE);
        final Random random = new Random(random().nextLong());
        for (int dim : DIMS) {
            final int stride = dim + 8;
            for (int m : new int[] { 1, 3, 101 }) {
                final byte[] recs = new byte[m * stride];
                for (int i = 0; i < recs.length; i++) {
                    recs[i] = (byte) random.nextInt(128);
                }
                final byte[] q = new byte[dim];
                for (int d = 0; d < dim; d++) {
                    q[d] = (byte) (random.nextInt(255) - 127);
                }
                // Extremes: the largest magnitudes the format writes, which would saturate a 16-bit pairwise sum.
                if (dim > 1) {
                    for (int d = 0; d < dim; d++) {
                        recs[d] = 127;
                        q[d] = (byte) (d % 2 == 0 ? 127 : -127);
                    }
                }
                final int[] out = new int[m + 1];
                out[m] = 12345;
                SimdVectorComputeService.rabitqRefineDots(recs, stride, m, dim, q, out);
                for (int i = 0; i < m; i++) {
                    int expected = 0;
                    for (int d = 0; d < dim; d++) {
                        expected += recs[i * stride + d] * q[d];
                    }
                    assertEquals("dim=" + dim + " m=" + m + " i=" + i, expected, out[i]);
                }
                assertEquals("must not write past m", 12345, out[m]);
            }
        }
    }

    public void testJavaCoarseMatchesDefinition() {
        final Random random = new Random(random().nextLong());
        final int bitBytes = 13;
        final byte[] planes = new byte[4 * bitBytes];
        final byte[] bits = new byte[2 * bitBytes];
        random.nextBytes(planes);
        random.nextBytes(bits);
        final int[] out = new int[2];
        TieredRaBitQKernel.javaCoarseDots(planes, bits, bitBytes, 2, out);
        for (int i = 0; i < 2; i++) {
            int expected = 0;
            for (int j = 0; j < 4; j++) {
                for (int b = 0; b < bitBytes; b++) {
                    expected += Integer.bitCount((planes[j * bitBytes + b] & bits[i * bitBytes + b]) & 0xFF) << j;
                }
            }
            assertEquals(expected, out[i]);
        }
    }

    public void testRejectsShortArrays() {
        assumeTrue("native SIMD library not loadable in this JVM", TieredRaBitQKernel.USE_NATIVE);
        expectThrows(Exception.class, () -> SimdVectorComputeService.rabitqCoarseDots(new byte[4], new byte[1], 2, 1, new int[1]));
        expectThrows(Exception.class, () -> SimdVectorComputeService.rabitqRefineDots(new byte[8], 8, 2, 8, new byte[8], new int[2]));
        expectThrows(Exception.class, () -> SimdVectorComputeService.rabitqRefineDots(new byte[8], 4, 1, 8, new byte[8], new int[1]));
    }
}
