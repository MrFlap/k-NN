/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.model;

import lombok.Builder;
import lombok.ToString;
import lombok.Value;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.store.IndexOutputWithBuffer;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

import java.util.Map;
import java.util.function.Supplier;

@Value
@Builder(toBuilder = true)
@ToString
public class BuildIndexParams {
    String fieldName;
    KNNEngine knnEngine;
    IndexOutputWithBuffer indexOutputWithBuffer;
    VectorDataType vectorDataType;
    Map<String, Object> parameters;
    /**
     * An optional quantization state that contains required information for quantization
     */
    @Nullable
    QuantizationState quantizationState;
    Supplier<KNNVectorValues<?>> knnVectorValuesSupplier;
    int totalLiveDocs;
    SegmentWriteState segmentWriteState;
    boolean isFlush;
    /**
     * The merge state for merge operations. This is null for flush operations.
     * Used by ClumpingIndexBuildStrategy to read hidden vectors from source segments.
     */
    @Nullable
    MergeState mergeState;
}
