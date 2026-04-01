/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.misc.index;

import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.misc.index.BpByteVectorReorderer;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.junit.Test;

import java.io.IOException;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ForkJoinPool;

/**
 * Tests for {@link BpByteVectorReorderer} — BP reordering on byte (quantized) vectors.
 */
public class BpByteVectorReordererTests extends LuceneTestCase {

    /**
     * Create a simple in-memory ByteVectorValues for testing.
     */
    private static ByteVectorValues fromBytes(byte[][] vectors) {
        return new ByteVectorValues() {
            @Override
            public byte[] vectorValue(int ord) {
                return vectors[ord];
            }

            @Override
            public int dimension() {
                return vectors[0].length;
            }

            @Override
            public int size() {
                return vectors.length;
            }

            @Override
            public ByteVectorValues copy() {
                return this; // safe for single-threaded tests
            }
        };
    }

    @Test
    public void testValidPermutation() throws IOException {
        // 100 random byte vectors of dimension 16
        Random rng = new Random(42);
        byte[][] vectors = new byte[100][16];
        for (int i = 0; i < 100; i++) {
            rng.nextBytes(vectors[i]);
        }

        BpByteVectorReorderer reorderer = new BpByteVectorReorderer();
        reorderer.setMinPartitionSize(2);
        Sorter.DocMap map = reorderer.computeValueMap(fromBytes(vectors), null);

        // Verify it's a valid bijection
        assertEquals(100, map.size());
        Set<Integer> seen = new HashSet<>();
        for (int i = 0; i < 100; i++) {
            int oldOrd = map.newToOld(i);
            assertTrue("ord out of range: " + oldOrd, oldOrd >= 0 && oldOrd < 100);
            assertTrue("duplicate ord: " + oldOrd, seen.add(oldOrd));
        }

        // Verify round-trip
        for (int i = 0; i < 100; i++) {
            assertEquals(i, map.oldToNew(map.newToOld(i)));
        }
    }

    @Test
    public void testClusteringQuality() throws IOException {
        // Create 200 vectors in 2 well-separated clusters
        // Cluster A: bytes near 0, Cluster B: bytes near 200
        // Interleave them: A, B, A, B, ...
        int n = 200;
        int dim = 32;
        byte[][] vectors = new byte[n][dim];
        Random rng = new Random(123);
        for (int i = 0; i < n; i++) {
            byte base = (i % 2 == 0) ? (byte) 10 : (byte) 200;
            for (int d = 0; d < dim; d++) {
                vectors[i][d] = (byte) (base + rng.nextInt(20) - 10);
            }
        }

        BpByteVectorReorderer reorderer = new BpByteVectorReorderer();
        reorderer.setMinPartitionSize(2);
        Sorter.DocMap map = reorderer.computeValueMap(fromBytes(vectors), null);

        // After reordering, same-cluster vectors should be adjacent.
        // Check that >80% of consecutive pairs are in the same cluster.
        int sameCluster = 0;
        for (int i = 0; i < n - 1; i++) {
            int oldA = map.newToOld(i);
            int oldB = map.newToOld(i + 1);
            if ((oldA % 2) == (oldB % 2)) {
                sameCluster++;
            }
        }
        float adjacency = (float) sameCluster / (n - 1);
        assertTrue("Expected >80% same-cluster adjacency, got " + (adjacency * 100) + "%",
            adjacency > 0.80f);
    }

    @Test
    public void testWithForkJoinPool() throws IOException {
        Random rng = new Random(99);
        byte[][] vectors = new byte[500][24];
        for (int i = 0; i < 500; i++) {
            rng.nextBytes(vectors[i]);
        }

        // Need a thread-safe copy() for multi-threaded execution
        ByteVectorValues threadSafeValues = new ByteVectorValues() {
            @Override
            public byte[] vectorValue(int ord) {
                // Return a copy to avoid cross-thread buffer sharing
                return vectors[ord].clone();
            }

            @Override
            public int dimension() { return 24; }

            @Override
            public int size() { return 500; }

            @Override
            public ByteVectorValues copy() {
                return this; // each call to vectorValue returns a fresh copy
            }
        };

        BpByteVectorReorderer reorderer = new BpByteVectorReorderer();
        reorderer.setMinPartitionSize(2);
        ForkJoinPool pool = new ForkJoinPool(4);
        try {
            TaskExecutor executor = new TaskExecutor(pool);
            Sorter.DocMap map = reorderer.computeValueMap(threadSafeValues, executor);

            assertEquals(500, map.size());
            Set<Integer> seen = new HashSet<>();
            for (int i = 0; i < 500; i++) {
                assertTrue(seen.add(map.newToOld(i)));
            }
        } finally {
            pool.shutdown();
        }
    }

    @Test
    public void testSmallInput() throws IOException {
        // 2 vectors — minimum for BP
        byte[][] vectors = new byte[][] {
            {0, 0, 0, 0},
            {(byte) 255, (byte) 255, (byte) 255, (byte) 255}
        };

        BpByteVectorReorderer reorderer = new BpByteVectorReorderer();
        reorderer.setMinPartitionSize(1);
        Sorter.DocMap map = reorderer.computeValueMap(fromBytes(vectors), null);

        assertEquals(2, map.size());
        Set<Integer> seen = new HashSet<>();
        for (int i = 0; i < 2; i++) {
            assertTrue(seen.add(map.newToOld(i)));
        }
    }
}
