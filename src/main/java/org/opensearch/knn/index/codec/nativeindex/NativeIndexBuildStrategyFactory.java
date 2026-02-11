/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.Setter;
import org.apache.lucene.index.FieldInfo;
import org.opensearch.index.IndexSettings;
<<<<<<< HEAD
import org.opensearch.knn.common.FieldInfoExtractor;
=======
import org.opensearch.knn.index.codec.nativeindex.clumping.ClumpingIndexBuildStrategy;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
>>>>>>> 27da7db6 (Clumping init)
import org.opensearch.knn.index.codec.nativeindex.remote.RemoteIndexBuildStrategy;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.faiss.FaissSQEncoder;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.repositories.RepositoriesService;

import java.io.IOException;
import java.util.function.Supplier;

import static org.opensearch.knn.common.FieldInfoExtractor.extractKNNEngine;
<<<<<<< HEAD
import static org.opensearch.knn.common.KNNConstants.MIN_DOCS_FOR_REMOTE_INDEX_BUILD;
=======
import static org.opensearch.knn.common.KNNConstants.CLUMPING_FACTOR;
>>>>>>> 27da7db6 (Clumping init)
import static org.opensearch.knn.common.KNNConstants.MODEL_ID;
import static org.opensearch.knn.index.KNNSettings.isKNNRemoteVectorBuildEnabled;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.initializeVectorValues;

/**
 * Creates the {@link NativeIndexBuildStrategy}
 */
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
     * @return The {@link NativeIndexBuildStrategy} to be used. Intended to be used by {@link NativeIndexWriter}
     * @throws IOException
     */
    public NativeIndexBuildStrategy getBuildStrategy(
        final FieldInfo fieldInfo,
        final int totalLiveDocs,
        final KNNVectorValues<?> knnVectorValues
    ) throws IOException {
        final KNNEngine knnEngine = extractKNNEngine(fieldInfo);
        final boolean isTemplate = fieldInfo.attributes().containsKey(MODEL_ID);
        final boolean iterative = !isTemplate && KNNEngine.FAISS == knnEngine;
        final boolean isFaissSQOneBitField = FieldInfoExtractor.isSQField(fieldInfo)
            && FieldInfoExtractor.extractSQConfig(fieldInfo).getBits() == FaissSQEncoder.Bits.ONE.getValue();

        // Determine build strategy
        final NativeIndexBuildStrategy strategy;
        if (isFaissSQOneBitField) {
            strategy = MemOptimizedScalarQuantizedIndexBuildStrategy.getInstance();
        } else if (iterative) {
            strategy = MemOptimizedNativeIndexBuildStrategy.getInstance();
        } else {
            strategy = DefaultIndexBuildStrategy.getInstance();
        }

        initializeVectorValues(knnVectorValues);
        long vectorBlobLength = ((long) knnVectorValues.bytesPerVector()) * totalLiveDocs;

        if (totalLiveDocs > MIN_DOCS_FOR_REMOTE_INDEX_BUILD
            && isKNNRemoteVectorBuildEnabled()
            && knnEngine.supportsRemoteIndexBuild(knnLibraryIndexingContext)
            && RemoteIndexBuildStrategy.shouldBuildIndexRemotely(indexSettings, vectorBlobLength)) {
            strategy = new RemoteIndexBuildStrategy(repositoriesServiceSupplier, strategy, indexSettings, knnLibraryIndexingContext);
        }

        // Wrap with clumping if configured
        String clumpingFactorStr = fieldInfo.attributes().get(CLUMPING_FACTOR);
        if (clumpingFactorStr != null) {
            int clumpingFactor = Integer.parseInt(clumpingFactorStr);
            if (clumpingFactor >= 2) {
                strategy = new ClumpingIndexBuildStrategy(strategy, clumpingFactor);
            }
        }

        return strategy;
    }
}
