/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.Setter;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.lang3.StringUtils;
import org.apache.lucene.index.FieldInfo;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.codec.nativeindex.remote.RemoteIndexBuildStrategy;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.repositories.RepositoriesService;

import java.io.IOException;
import java.util.function.Supplier;

import static org.opensearch.knn.common.FieldInfoExtractor.extractKNNEngine;
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.index.KNNSettings.isKNNRemoteVectorBuildEnabled;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.initializeVectorValues;

/**
 * Creates the {@link NativeIndexBuildStrategy}
 */
@Log4j2
public final class NativeIndexBuildStrategyFactory {

    private final Supplier<RepositoriesService> repositoriesServiceSupplier;
    private final IndexSettings indexSettings;
    @Setter
    private KNNLibraryIndexingContext knnLibraryIndexingContext;

    public NativeIndexBuildStrategyFactory() {
        this(null, null);
    }

    public NativeIndexBuildStrategyFactory(Supplier<RepositoriesService> repositoriesServiceSupplier, IndexSettings indexSettings) {
        this.repositoriesServiceSupplier = repositoriesServiceSupplier;
        this.indexSettings = indexSettings;
    }

    /**
     * @param fieldInfo         Field related attributes/info
     * @param totalLiveDocs     Number of documents with the vector field. This values comes from {@link org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsWriter#flush}
     *                          and {@link org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsWriter#mergeOneField}
     * @param knnVectorValues   An instance of {@link KNNVectorValues} which is used to evaluate the size threshold KNN_REMOTE_VECTOR_BUILD_THRESHOLD
     * @param indexInfo  An instance of {@link BuildIndexParams} containing relevant index info
     * @return The {@link NativeIndexBuildStrategy} to be used. Intended to be used by {@link NativeIndexWriter}
     * @throws IOException
     */
    public NativeIndexBuildStrategy getBuildStrategy(
        final FieldInfo fieldInfo,
        final int totalLiveDocs,
        final KNNVectorValues<?> knnVectorValues,
        BuildIndexParams indexInfo
    ) throws IOException {
        final KNNEngine knnEngine = extractKNNEngine(fieldInfo);
        boolean isTemplate = fieldInfo.attributes().containsKey(MODEL_ID);
        boolean iterative = !isTemplate && KNNEngine.FAISS == knnEngine;

        NativeIndexBuildStrategy strategy = iterative
            ? MemOptimizedNativeIndexBuildStrategy.getInstance()
            : DefaultIndexBuildStrategy.getInstance();

        initializeVectorValues(knnVectorValues);
        long vectorBlobLength = ((long) knnVectorValues.bytesPerVector()) * totalLiveDocs;

        if (isKNNRemoteVectorBuildEnabled()
            && repositoriesServiceSupplier != null
            && indexSettings != null
            && knnEngine.supportsRemoteIndexBuild(knnLibraryIndexingContext)
            && RemoteIndexBuildStrategy.shouldBuildIndexRemotely(indexSettings, vectorBlobLength)) {
            strategy = new RemoteIndexBuildStrategy(repositoriesServiceSupplier, strategy, indexSettings, knnLibraryIndexingContext);
        }

        // Check if clumping is enabled and wrap the strategy with ClumpingIndexBuildStrategy
        Integer clumpingFactor = extractClumpingFactor(fieldInfo);
        if (clumpingFactor != null) {
            log.debug("Clumping enabled for field {} with factor {}", fieldInfo.name, clumpingFactor);
            strategy = new ClumpingIndexBuildStrategy(strategy, clumpingFactor);
        }

        return strategy;
    }

    /**
     * Extracts the clumping factor from field attributes.
     *
     * @param fieldInfo The field info containing attributes
     * @return The clumping factor if clumping is enabled, null otherwise
     */
    private Integer extractClumpingFactor(final FieldInfo fieldInfo) {
        String clumpingFactorString = fieldInfo.getAttribute(KNNConstants.CLUMPING_PARAMETER);
        
        // Debug logging to help diagnose clumping issues
        if (log.isDebugEnabled()) {
            log.debug("Field {} attributes: {}", fieldInfo.name, fieldInfo.attributes());
        }
        
        if (StringUtils.isEmpty(clumpingFactorString)) {
            log.debug("No clumping factor found for field {}", fieldInfo.name);
            return null;
        }
        try {
            int factor = Integer.parseInt(clumpingFactorString);
            log.info("Clumping factor {} extracted for field {}", factor, fieldInfo.name);
            return factor;
        } catch (NumberFormatException e) {
            log.warn("Invalid clumping factor value '{}' for field {}, ignoring clumping", clumpingFactorString, fieldInfo.name);
            return null;
        }
    }
}
