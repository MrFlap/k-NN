/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import lombok.RequiredArgsConstructor;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIdMapIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;

import java.io.IOException;

@RequiredArgsConstructor
public class FaissIdMapIndexReorderer extends FaissIndexReorderTransformer {
    private final String indexType;

    @Override
    protected void doTransform(FaissIndex index, IndexInput indexInput, IndexOutput indexOutput, ReorderOrdMap reorderOrdMap)
        throws IOException {
        final FaissIdMapIndex actualIndex = (FaissIdMapIndex) index;

        // Write index type
        writeIndexType(indexType, indexOutput);

        // Copy header - use correct format based on index type
        if (indexType.equals(FaissIdMapIndex.IXMP)) {
            copyCommonHeader(actualIndex, indexOutput);
        } else {
            copyBinaryCommonHeader(actualIndex, indexOutput);
        }

        // Transform nested index
        transform(actualIndex.getNestedIndex(), indexInput, indexOutput, reorderOrdMap);

        // Transform id map
        indexOutput.writeLong(actualIndex.getTotalNumberOfVectors());
        final int[] ordToDocs = actualIndex.getOrdToDocs();
        for (final int oldOrd : reorderOrdMap.newOrd2Old) {
            final int docId;
            if (ordToDocs == null) {
                docId = oldOrd;
            } else {
                docId = ordToDocs[oldOrd];
            }
            indexOutput.writeLong(docId);
        }
    }
}
