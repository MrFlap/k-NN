/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import lombok.RequiredArgsConstructor;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndexFloatFlat;

import java.io.IOException;

@RequiredArgsConstructor
public class FaissIndexFloatFlatReorderer extends FaissIndexReorderTransformer {
    private final String indexType;

    @Override
    protected void doTransform(FaissIndex index, IndexInput indexInput, IndexOutput indexOutput, ReorderOrdMap reorderOrdMap)
        throws IOException {
        final FaissIndexFloatFlat actualIndex = (FaissIndexFloatFlat) index;

        // Write index type
        writeIndexType(indexType, indexOutput);

        // Copy common header
        copyCommonHeader(actualIndex, indexOutput);

        // Reorder vectors
        final int numVectors = actualIndex.getTotalNumberOfVectors();
        final int dimension = actualIndex.getDimension();
        final FloatVectorValues vectorValues = actualIndex.getFloatValues(indexInput);

        // Write vector count
        indexOutput.writeLong(numVectors);

        // Write reordered vectors
        for (int newOrd = 0; newOrd < numVectors; newOrd++) {
            final int oldOrd = reorderOrdMap.newOrd2Old[newOrd];
            final float[] vector = vectorValues.vectorValue(oldOrd);
            for (float v : vector) {
                indexOutput.writeInt(Float.floatToIntBits(v));
            }
        }
    }
}
