/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import lombok.RequiredArgsConstructor;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissBinaryHnswIndex;

import java.io.IOException;

@RequiredArgsConstructor
public class FaissBinaryHnswIndexReorderer extends FaissIndexReorderTransformer {
    private final String indexType;

    @Override
    protected void doTransform(FaissIndex index, IndexInput indexInput, IndexOutput indexOutput, ReorderOrdMap reorderOrdMap)
        throws IOException {
        final FaissBinaryHnswIndex actualIndex = (FaissBinaryHnswIndex) index;

        // Write index type
        writeIndexType(indexType, indexOutput);

        // Binary header
        copyBinaryCommonHeader(actualIndex, indexOutput);

        // HNSW reordering
        FaissHnswReorderer.transform(actualIndex.getFaissHnsw(), indexInput, indexOutput, reorderOrdMap);

        // Transform flat vectors
        transform(actualIndex.getStorage(), indexInput, indexOutput, reorderOrdMap);
    }
}
