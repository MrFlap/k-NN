/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

/**
 * Example:
 * Original vectors : v0, v1, v2
 * After reordering : v2, v0, v1
 * <p>
 * Then newOrd2Old = [2, 0, 1], oldOrd2New = [1, 2, 0]
 * `newOrd2Old` answers where is the location of a vector in the original order.
 * For example, newOrd2Old[1] == 0 explains that 1st vector was 0th in the original order.
 * `oldOrd2New` answers where a vector was repositioned in the new order.
 * For example, oldOrd2New[2] == 0 explains that 2nd vector was moved to 0th in the new order.
 */
public class ReorderOrdMap {
    public final int[] oldOrd2New;
    public final int[] newOrd2Old;

    public ReorderOrdMap(final int[] newOrd2Old) {
        this.newOrd2Old = newOrd2Old;
        final int numVectors = newOrd2Old.length;
        this.oldOrd2New = new int[numVectors];

        // Invert: if newOrd2Old[i] = j, then oldOrd2New[j] = i
        for (int newOrd = 0; newOrd < numVectors; newOrd++) {
            oldOrd2New[newOrd2Old[newOrd]] = newOrd;
        }
    }
}
