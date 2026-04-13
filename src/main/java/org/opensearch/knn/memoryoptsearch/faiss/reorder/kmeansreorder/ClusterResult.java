/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder.kmeansreorder;

/**
 * Result of a merge-aware k-means operation: the reorder permutation and the
 * cluster summary for the segment (needed for future merges).
 */
public record ClusterResult(int[] permutation, ClusterSummary summary) {}
