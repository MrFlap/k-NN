/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.KNN1040Codec.Faiss1040ScalarQuantizedKnnVectorsFormat;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.engine.CodecFormatResolver;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.VectorReorderStrategy;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.bpreorder.BipartiteReorderStrategy;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.kmeansreorder.KMeansReorderStrategy;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.kmeansreorder.MergeAwareKMeansReorderStrategy;

import java.util.Map;
import java.util.Optional;

/**
 * {@link CodecFormatResolver} implementation for native engines (FAISS, NMSLIB).
 * Encapsulates the {@link NativeEngines990KnnVectorsFormat} creation logic including
 * {@code approximateThreshold} lookup from index settings.
 *
 * <p>Placed in the {@code faiss} package alongside {@link FaissMethodResolver} because
 * NMSLIB is deprecated and no new NMSLIB indices are created.</p>
 */
public class FaissCodecFormatResolver implements CodecFormatResolver {

    private final Optional<MapperService> mapperService;
    private final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory;

    public FaissCodecFormatResolver(
        Optional<MapperService> mapperService,
        NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory
    ) {
        this.mapperService = mapperService;
        this.nativeIndexBuildStrategyFactory = nativeIndexBuildStrategyFactory;
    }

    /**
     * Resolves the format for a specific field. Returns {@link Faiss1040ScalarQuantizedKnnVectorsFormat} when
     * the encoder is sq with bits=1, otherwise falls back to the default native format.
     */
    @Override
    public KnnVectorsFormat resolve(
        String field,
        KNNMethodContext methodContext,
        Map<String, Object> params,
        int defaultMaxConnections,
        int defaultBeamWidth
    ) {
        if (isSQOneBitEncoder(params)) {
            return new Faiss1040ScalarQuantizedKnnVectorsFormat(getReorderStrategy());
        }
        return resolve();
    }

    @Override
    public KnnVectorsFormat resolve() {
        final int approximateThreshold = getApproximateThresholdValue();
        final VectorReorderStrategy reorderStrategy = getReorderStrategy();
        final boolean replacementFree = isReplacementFree();
        return new NativeEngines990KnnVectorsFormat(approximateThreshold, nativeIndexBuildStrategyFactory, reorderStrategy, replacementFree);
    }

    /**
     * Retrieves the approximate threshold value from index settings.
     * Falls back to the default value when the setting is not explicitly configured.
     */
    private int getApproximateThresholdValue() {
        final IndexSettings indexSettings = mapperService.get().getIndexSettings();
        final Integer approximateThresholdValue = indexSettings.getValue(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_SETTING);
        return approximateThresholdValue != null
            ? approximateThresholdValue
            : KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE;
    }

    private static boolean isSQOneBitEncoder(Map<String, Object> params) {
        return FaissSQEncoder.isSQOneBit(params);
    }

    private VectorReorderStrategy getReorderStrategy() {
        if (mapperService.isEmpty()) return null;
        final IndexSettings indexSettings = mapperService.get().getIndexSettings();
        final String strategy = indexSettings.getValue(KNNSettings.INDEX_KNN_ADVANCED_REORDER_STRATEGY_SETTING);
        switch (strategy) {
            case "kmeans":
                final int numClusters = indexSettings.getValue(KNNSettings.INDEX_KNN_ADVANCED_REORDER_KMEANS_NUM_CLUSTERS_SETTING);
                return new KMeansReorderStrategy(numClusters, 25);
            case "kmeans_merge_aware":
                final int numClustersMerge = indexSettings.getValue(KNNSettings.INDEX_KNN_ADVANCED_REORDER_KMEANS_NUM_CLUSTERS_SETTING);
                return new MergeAwareKMeansReorderStrategy(numClustersMerge, 25);
            case "none":
                return null;
            default:
                return new BipartiteReorderStrategy();
        }
    }

    private boolean isReplacementFree() {
        if (mapperService.isEmpty()) return false;
        return "replacement_free".equals(
            mapperService.get().getIndexSettings().getValue(KNNSettings.INDEX_KNN_ADVANCED_REORDER_IMPLEMENTATION_SETTING)
        );
    }
}
