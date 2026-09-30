/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.util.BytesRef;
import org.opensearch.common.CheckedSupplier;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.mapper.RandomOrthogonalVectorTransformer;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapperUtil;

import java.io.IOException;

public abstract class AbstractPerFieldDerivedVectorTransformer implements PerFieldDerivedVectorTransformer {
    /**
     * Utility method for formatting the vector values based on the vector data type.
     *
     * @param fieldInfo fieldinfo for the vector field
     * @param vectorSupplier supplies vector (without clone)
     * @param vectorCloneSupplier supplies clone of vector.
     * @return vector formatted based on the vector data type. Typically, this will be a float[] or int[].
     * @throws IOException if unable to deserialize stored vector
     */
    protected Object formatVector(
        FieldInfo fieldInfo,
        CheckedSupplier<Object, IOException> vectorSupplier,
        CheckedSupplier<Object, IOException> vectorCloneSupplier
    ) throws IOException {
        Object vectorValue = vectorSupplier.get();
        Object formatted;
        // If the vector value is a byte[], we must deserialize
        if (vectorValue instanceof byte[]) {
            BytesRef vectorBytesRef = new BytesRef((byte[]) vectorValue);
            VectorDataType vectorDataType = FieldInfoExtractor.extractVectorDataType(fieldInfo);
            formatted = KNNVectorFieldMapperUtil.deserializeStoredVector(vectorBytesRef, vectorDataType);
        } else {
            formatted = vectorCloneSupplier.get();
        }
        // Source holds the vector as provided, so undo the random orthogonal transform applied at index time.
        if (formatted instanceof float[] floats) {
            return RandomOrthogonalVectorTransformer.inverseIfApplied(fieldInfo.attributes(), floats);
        }
        return formatted;
    }
}
