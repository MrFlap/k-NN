/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.query.memoryoptsearch.MemoryOptimizedKNNWeight;

import java.lang.reflect.Field;

public class MemoryOptimizedKNNWeightTests extends KNNTestCase {

    @SneakyThrows
    public void testAcornFilteredSearchThresholdIsZero() {
        Field field = MemoryOptimizedKNNWeight.class.getDeclaredField("DEFAULT_HNSW_SEARCH_STRATEGY");
        field.setAccessible(true);
        KnnSearchStrategy.Hnsw strategy = (KnnSearchStrategy.Hnsw) field.get(null);

        assertEquals(0, strategy.filteredSearchThreshold());
        assertFalse(strategy.useFilteredSearch(0.5f));
        assertFalse(strategy.useFilteredSearch(1.0f));
        assertFalse(strategy.useFilteredSearch(0.0f));
    }
}
