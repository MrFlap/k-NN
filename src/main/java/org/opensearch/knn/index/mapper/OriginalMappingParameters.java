/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import lombok.Getter;
import lombok.Setter;
import org.opensearch.core.common.Strings;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNMethodContext;

/**
 * Utility class to store the original mapping parameters for a KNNVectorFieldMapper. These parameters need to be
 * kept around for when a {@link KNNVectorFieldMapper} is built from merge
 */
@Getter
public final class OriginalMappingParameters {
    private final VectorDataType vectorDataType;
    private final int dimension;
    private final KNNMethodContext knnMethodContext;

    // To support our legacy field mapping, on parsing, if index.knn=true and no method is
    // passed, we build a KNNMethodContext using the space type, ef_construction and m that are set in the index
    // settings. However, for fieldmappers for merging, we need to be able to initialize one field mapper from
    // another (see
    // https://github.com/opensearch-project/OpenSearch/blob/2.16.0/server/src/main/java/org/opensearch/index/mapper/ParametrizedFieldMapper.java#L98).
    // The problem is that in this case, the settings are set to empty so we cannot properly resolve the KNNMethodContext.
    // (see
    // https://github.com/opensearch-project/OpenSearch/blob/2.16.0/server/src/main/java/org/opensearch/index/mapper/ParametrizedFieldMapper.java#L130).
    // While we could override the KNNMethodContext parameter initializer to set the knnMethodContext based on the
    // constructed KNNMethodContext from the other field mapper, this can result in merge conflict/serialization
    // exceptions. See
    // (https://github.com/opensearch-project/OpenSearch/blob/2.16.0/server/src/main/java/org/opensearch/index/mapper/ParametrizedFieldMapper.java#L322-L324).
    // So, what we do is pass in a "resolvedKNNMethodContext" to ensure we track this resolveKnnMethodContext.
    // A similar approach was taken for https://github.com/opendistro-for-elasticsearch/k-NN/issues/288
    //
    // In almost all cases except when dealing with the mapping, the resolved context should be used
    @Setter
    private KNNMethodContext resolvedKnnMethodContext;
    private final String mode;
    private final String compressionLevel;
    private final int clumpingFactor;
    private final String modelId;
    private final String topLevelSpaceType;
    private final String topLevelEngine;

    /**
     * Full constructor with all parameters including clumping factor.
     */
    public OriginalMappingParameters(
        VectorDataType vectorDataType,
        int dimension,
        KNNMethodContext knnMethodContext,
        String mode,
        String compressionLevel,
        int clumpingFactor,
        String modelId,
        String topLevelSpaceType,
        String topLevelEngine
    ) {
        this.vectorDataType = vectorDataType;
        this.dimension = dimension;
        this.knnMethodContext = knnMethodContext;
        this.resolvedKnnMethodContext = knnMethodContext;
        this.mode = mode;
        this.compressionLevel = compressionLevel;
        this.clumpingFactor = clumpingFactor;
        this.modelId = modelId;
        this.topLevelSpaceType = topLevelSpaceType;
        this.topLevelEngine = topLevelEngine;
    }

    /**
     * Backward-compatible constructor without clumping factor (defaults to 1 - no clumping).
     */
    public OriginalMappingParameters(
        VectorDataType vectorDataType,
        int dimension,
        KNNMethodContext knnMethodContext,
        String mode,
        String compressionLevel,
        String modelId,
        String topLevelSpaceType,
        String topLevelEngine
    ) {
        this(vectorDataType, dimension, knnMethodContext, mode, compressionLevel, 1, modelId, topLevelSpaceType, topLevelEngine);
    }

    /**
     * Initialize the parameters from the builder
     *
     * @param builder The builder to initialize from
     */
    public OriginalMappingParameters(KNNVectorFieldMapper.Builder builder) {
        this.vectorDataType = builder.vectorDataType.get();
        this.knnMethodContext = builder.knnMethodContext.get();
        this.resolvedKnnMethodContext = builder.knnMethodContext.get();
        this.dimension = builder.dimension.get();
        this.mode = builder.mode.get();
        this.compressionLevel = builder.compressionLevel.get();
        this.clumpingFactor = builder.clumpingFactor.get();
        this.modelId = builder.modelId.get();
        this.topLevelSpaceType = builder.topLevelSpaceType.get();
        this.topLevelEngine = builder.topLevelEngine.get();
    }

    /**
     * Determine if the mapping used the legacy mechanism to setup the index. The legacy mechanism is used if
     * the index is created only by specifying the dimension. If this is the case, the constructed parameters
     * need to be collected from the index settings
     *
     * @return true if the mapping used the legacy mechanism, false otherwise
     */
    public boolean isLegacyMapping() {
        if (knnMethodContext != null) {
            return false;
        }

        if (modelId != null) {
            return false;
        }

        return Strings.isEmpty(mode) && Strings.isEmpty(compressionLevel);
    }

    /**
     * Check if clumping is enabled for this field mapping.
     *
     * @return true if clumping factor is greater than 1
     */
    public boolean isClumpingEnabled() {
        return clumpingFactor > 1;
    }
}
