/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import lombok.RequiredArgsConstructor;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.memoryoptsearch.faiss.FaissHNSWIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;

import java.io.IOException;

@RequiredArgsConstructor
public class FaissHNSWIndexReorderer extends FaissIndexReorderTransformer {
    private final String indexType;

    @Override
    protected void doTransform(FaissIndex index, IndexInput indexInput, IndexOutput indexOutput, ReorderOrdMap reorderOrdMap)
        throws IOException {
        final FaissHNSWIndex actualIndex = (FaissHNSWIndex) index;

        // Write index type
        writeIndexType(indexType, indexOutput);

        // Copy common header
        copyCommonHeader(actualIndex, indexOutput);

        // HNSW reordering
        FaissHnswReorderer.transform(actualIndex.getFaissHnsw(), indexInput, indexOutput, reorderOrdMap);

        // Transform flat vectors storage
        transform(actualIndex.getFlatVectors(), indexInput, indexOutput, reorderOrdMap);
    }
}
