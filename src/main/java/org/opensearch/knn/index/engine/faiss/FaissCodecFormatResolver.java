/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.KNN1040Codec.Faiss1040ScalarQuantizedKnnVectorsFormat;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.codec.KNN1040Codec.FaissPadRotateKnnVectorsFormat;
import org.opensearch.knn.index.engine.CodecFormatResolver;
import org.opensearch.knn.index.engine.KNNMethodContext;

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
     * Resolves the format for a specific field.
     * <ul>
     *   <li>sq(bits=1) with {@code pad_rotate_mode=true} (x8) -> {@link FaissPadRotateKnnVectorsFormat}</li>
     *   <li>sq(bits=1) without pad-rotate (x32) -> {@link Faiss1040ScalarQuantizedKnnVectorsFormat}</li>
     *   <li>anything else -> default native format</li>
     * </ul>
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
            if (isSQPadRotateMode(params)) {
                return new FaissPadRotateKnnVectorsFormat(nativeIndexBuildStrategyFactory);
            }
            return new Faiss1040ScalarQuantizedKnnVectorsFormat(nativeIndexBuildStrategyFactory);
        }
        return resolve();
    }

    @Override
    public KnnVectorsFormat resolve() {
        final int approximateThreshold = getApproximateThresholdValue();
        return new NativeEngines990KnnVectorsFormat(approximateThreshold, nativeIndexBuildStrategyFactory);
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

    /**
     * Returns true when the encoder context on {@code params} is sq(bits=1) with the
     * {@link KNNConstants#FAISS_SQ_PAD_ROTATE} marker set to {@code true}.
     */
    private static boolean isSQPadRotateMode(Map<String, Object> params) {
        if (params == null) {
            return false;
        }
        Object encoderObj = params.get(KNNConstants.METHOD_ENCODER_PARAMETER);
        if (encoderObj instanceof org.opensearch.knn.index.engine.MethodComponentContext == false) {
            return false;
        }
        org.opensearch.knn.index.engine.MethodComponentContext encoderCtx = (org.opensearch.knn.index.engine.MethodComponentContext) encoderObj;
        return FaissSQEncoder.isPadRotateModeEnabled(encoderCtx);
    }
}
