/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissBinaryIndex;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

public abstract class FaissIndexReorderTransformer {
    public static void transform(
        final FaissIndex index,
        IndexInput indexInput,
        final IndexOutput indexOutput,
        final ReorderOrdMap reorderOrdMap
    ) throws IOException {
        final FaissIndexReorderTransformer transformer = IndexTypeToFaissIndexReordererMapping.get(index.getIndexType());
        transformer.doTransform(index, indexInput, indexOutput, reorderOrdMap);
    }

    protected void writeIndexType(final String indexType, final IndexOutput indexOuptut) throws IOException {
        final byte[] indexTypeBytes = indexType.getBytes(StandardCharsets.UTF_8);
        indexOuptut.writeBytes(indexTypeBytes, 0, indexTypeBytes.length);
    }

    protected void copyBinaryCommonHeader(final FaissBinaryIndex binaryIndex, IndexOutput indexOutput) throws IOException {
        // Dimension + Code size + #vectors
        indexOutput.writeInt(binaryIndex.getDimension());
        indexOutput.writeInt(binaryIndex.getCodeSize());
        indexOutput.writeLong(binaryIndex.getTotalNumberOfVectors());

        // Copy trained field (deprecated)
        indexOutput.writeByte(binaryIndex.getIsTrainedDeprecatedField());

        // Copy the original metric type
        indexOutput.writeInt(binaryIndex.getOriginalMetricType());
    }

    protected void copyCommonHeader(final FaissIndex index, IndexOutput indexOutput) throws IOException {
        // Dimension
        indexOutput.writeInt(index.getDimension());
        // Total number of vectors
        indexOutput.writeLong(index.getTotalNumberOfVectors());
        // 2 dummy deprecated fields
        indexOutput.writeLong(0);
        indexOutput.writeLong(0);
        // isTrained field (deprecated, always 1)
        indexOutput.writeByte((byte) 1);
        // Metric type: 0 = INNER_PRODUCT, 1 = L2
        int metricType = index.getSpaceType() == org.opensearch.knn.index.SpaceType.INNER_PRODUCT ? 0 : 1;
        indexOutput.writeInt(metricType);
    }

    protected abstract void doTransform(FaissIndex index, IndexInput indexInput, IndexOutput indexOutput, ReorderOrdMap reorderOrdMap)
        throws IOException;
}
