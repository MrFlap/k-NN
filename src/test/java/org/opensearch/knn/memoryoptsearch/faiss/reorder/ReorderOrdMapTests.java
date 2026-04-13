/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.opensearch.test.OpenSearchTestCase;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class ReorderOrdMapTests extends OpenSearchTestCase {

    public void testBasicPermutation() {
        // Original: v0, v1, v2 → Reordered: v2, v0, v1
        int[] newOrd2Old = { 2, 0, 1 };
        ReorderOrdMap map = new ReorderOrdMap(newOrd2Old);

        assertArrayEquals(new int[] { 2, 0, 1 }, map.newOrd2Old);
        // oldOrd2New: v0→pos1, v1→pos2, v2→pos0
        assertArrayEquals(new int[] { 1, 2, 0 }, map.oldOrd2New);
    }

    public void testIdentityPermutation() {
        int[] newOrd2Old = { 0, 1, 2, 3 };
        ReorderOrdMap map = new ReorderOrdMap(newOrd2Old);

        assertArrayEquals(new int[] { 0, 1, 2, 3 }, map.oldOrd2New);
    }

    public void testReversePermutation() {
        int[] newOrd2Old = { 3, 2, 1, 0 };
        ReorderOrdMap map = new ReorderOrdMap(newOrd2Old);

        assertArrayEquals(new int[] { 3, 2, 1, 0 }, map.oldOrd2New);
    }

    public void testRoundTrip() {
        // For any valid permutation, oldOrd2New[newOrd2Old[i]] == i
        int[] newOrd2Old = { 4, 2, 0, 3, 1 };
        ReorderOrdMap map = new ReorderOrdMap(newOrd2Old);

        for (int i = 0; i < newOrd2Old.length; i++) {
            assertEquals("roundtrip newOrd→oldOrd→newOrd failed at " + i, i, map.oldOrd2New[map.newOrd2Old[i]]);
            assertEquals("roundtrip oldOrd→newOrd→oldOrd failed at " + i, i, map.newOrd2Old[map.oldOrd2New[i]]);
        }
    }

    public void testSingleElement() {
        int[] newOrd2Old = { 0 };
        ReorderOrdMap map = new ReorderOrdMap(newOrd2Old);

        assertArrayEquals(new int[] { 0 }, map.oldOrd2New);
    }

    public void testPermutationCoversAllOrds() {
        int n = 100;
        // Create a random-ish permutation: reverse
        int[] newOrd2Old = new int[n];
        for (int i = 0; i < n; i++) {
            newOrd2Old[i] = n - 1 - i;
        }
        ReorderOrdMap map = new ReorderOrdMap(newOrd2Old);

        Set<Integer> seenNew = new HashSet<>();
        Set<Integer> seenOld = new HashSet<>();
        for (int i = 0; i < n; i++) {
            seenNew.add(map.oldOrd2New[i]);
            seenOld.add(map.newOrd2Old[i]);
        }
        assertEquals(n, seenNew.size());
        assertEquals(n, seenOld.size());
    }
}
