/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.misc.index;

import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.bpreorder.BinaryToFloatVectorValues;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.bpreorder.QuantizedBipartiteReorderStrategy;
import org.junit.Test;

import java.io.IOException;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

/**
 * Tests for quantized BP reordering via BinaryToFloatVectorValues adapter.
 */
public class BpByteVectorReordererTests extends LuceneTestCase {

    private static ByteVectorValues fromBytes(byte[][] vectors) {
        return new ByteVectorValues() {
            @Override
            public byte[] vectorValue(int ord) { return vectors[ord]; }

            @Override
            public int dimension() { return vectors[0].length; }

            @Override
            public int size() { return vectors.length; }

            @Override
            public ByteVectorValues copy() { return this; }
        };
    }

    @Test
    public void testValidPermutation() throws IOException {
        Random rng = new Random(42);
        byte[][] vectors = new byte[100][16];
        for (int i = 0; i < 100; i++) {
            rng.nextBytes(vectors[i]);
        }

        QuantizedBipartiteReorderStrategy strategy = new QuantizedBipartiteReorderStrategy();
        int[] permutation = strategy.computePermutationFromQuantized(fromBytes(vectors), 1);

        assertEquals(100, permutation.length);
        Set<Integer> seen = new HashSet<>();
        for (int i = 0; i < 100; i++) {
            assertTrue("ord out of range: " + permutation[i], permutation[i] >= 0 && permutation[i] < 100);
            assertTrue("duplicate ord: " + permutation[i], seen.add(permutation[i]));
        }
    }

    @Test
    public void testClusteringQuality() throws IOException {
        // Create 200 packed binary vectors in 2 well-separated clusters.
        // Cluster A (even indices): all bits 0 → all bytes 0x00
        // Cluster B (odd indices): all bits 1 → all bytes 0xFF
        // These are maximally separated in Hamming space.
        int n = 200;
        int codeSize = 32;
        byte[][] vectors = new byte[n][codeSize];
        Random rng = new Random(42);
        for (int i = 0; i < n; i++) {
            if (i % 2 == 0) {
                // Cluster A: mostly 0 bits with small noise
                for (int d = 0; d < codeSize; d++) {
                    vectors[i][d] = (byte) (rng.nextInt(4)); // low bits only
                }
            } else {
                // Cluster B: mostly 1 bits with small noise
                for (int d = 0; d < codeSize; d++) {
                    vectors[i][d] = (byte) (0xFC | rng.nextInt(4)); // high bits set
                }
            }
        }

        QuantizedBipartiteReorderStrategy strategy = new QuantizedBipartiteReorderStrategy();
        int[] permutation = strategy.computePermutationFromQuantized(fromBytes(vectors), 1);

        // After reordering, same-cluster vectors should be adjacent.
        int sameCluster = 0;
        for (int i = 0; i < n - 1; i++) {
            int oldA = permutation[i];
            int oldB = permutation[i + 1];
            if ((oldA % 2) == (oldB % 2)) {
                sameCluster++;
            }
        }
        float adjacency = (float) sameCluster / (n - 1);
        assertTrue("Expected >80% same-cluster adjacency, got " + (adjacency * 100) + "%",
            adjacency > 0.80f);
    }

    @Test
    public void testMultiThreaded() throws IOException {
        Random rng = new Random(99);
        byte[][] vectors = new byte[500][24];
        for (int i = 0; i < 500; i++) {
            rng.nextBytes(vectors[i]);
        }

        // Thread-safe: each copy() returns the same instance, but vectorValue returns the
        // backing array directly (safe because BP calls copy() per thread and each thread
        // reads different ords)
        QuantizedBipartiteReorderStrategy strategy = new QuantizedBipartiteReorderStrategy();
        int[] permutation = strategy.computePermutationFromQuantized(fromBytes(vectors), 4);

        assertEquals(500, permutation.length);
        Set<Integer> seen = new HashSet<>();
        for (int i = 0; i < 500; i++) {
            assertTrue(seen.add(permutation[i]));
        }
    }

    @Test
    public void testBinaryToFloatDecoding() throws IOException {
        // Verify the adapter correctly decodes packed bits to {+1, -1}
        byte[][] vectors = new byte[][] { { (byte) 0b10101010, (byte) 0b11110000 } };
        BinaryToFloatVectorValues decoded = new BinaryToFloatVectorValues(fromBytes(vectors));

        assertEquals(16, decoded.dimension()); // 2 bytes * 8 bits
        assertEquals(1, decoded.size());

        float[] result = decoded.vectorValue(0);
        // byte 0 = 0b10101010: bits 1,3,5,7 are set
        assertEquals(-1.0f, result[0], 0); // bit 0 = 0
        assertEquals(1.0f, result[1], 0);  // bit 1 = 1
        assertEquals(-1.0f, result[2], 0); // bit 2 = 0
        assertEquals(1.0f, result[3], 0);  // bit 3 = 1
        assertEquals(-1.0f, result[4], 0); // bit 4 = 0
        assertEquals(1.0f, result[5], 0);  // bit 5 = 1
        assertEquals(-1.0f, result[6], 0); // bit 6 = 0
        assertEquals(1.0f, result[7], 0);  // bit 7 = 1

        // byte 1 = 0b11110000: bits 4,5,6,7 are set
        assertEquals(-1.0f, result[8], 0);  // bit 0 = 0
        assertEquals(-1.0f, result[9], 0);  // bit 1 = 0
        assertEquals(-1.0f, result[10], 0); // bit 2 = 0
        assertEquals(-1.0f, result[11], 0); // bit 3 = 0
        assertEquals(1.0f, result[12], 0);  // bit 4 = 1
        assertEquals(1.0f, result[13], 0);  // bit 5 = 1
        assertEquals(1.0f, result[14], 0);  // bit 6 = 1
        assertEquals(1.0f, result[15], 0);  // bit 7 = 1
    }

    @Test
    public void testSmallInput() throws IOException {
        byte[][] vectors = new byte[][] {
            { 0x00, 0x00 },
            { (byte) 0xFF, (byte) 0xFF }
        };

        QuantizedBipartiteReorderStrategy strategy = new QuantizedBipartiteReorderStrategy();
        int[] permutation = strategy.computePermutationFromQuantized(fromBytes(vectors), 1);

        assertEquals(2, permutation.length);
        Set<Integer> seen = new HashSet<>();
        for (int i = 0; i < 2; i++) {
            assertTrue(seen.add(permutation[i]));
        }
    }

    @Test
    public void testFallbackToFloatPath() throws IOException {
        // Verify the fallback computePermutation (float path) still works
        float[][] floatVectors = new float[100][8];
        Random rng = new Random(42);
        for (int i = 0; i < 100; i++) {
            for (int d = 0; d < 8; d++) {
                floatVectors[i][d] = rng.nextFloat();
            }
        }

        org.apache.lucene.index.FloatVectorValues fvv = org.apache.lucene.index.FloatVectorValues.fromFloats(
            java.util.Arrays.stream(floatVectors).map(v -> v).collect(java.util.stream.Collectors.toList()),
            8
        );

        QuantizedBipartiteReorderStrategy strategy = new QuantizedBipartiteReorderStrategy();
        int[] permutation = strategy.computePermutation(fvv, 1, VectorSimilarityFunction.EUCLIDEAN);

        assertEquals(100, permutation.length);
        Set<Integer> seen = new HashSet<>();
        for (int i = 0; i < 100; i++) {
            assertTrue(seen.add(permutation[i]));
        }
    }
}
