/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import org.opensearch.Version;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.TrainingConfigValidationInput;
import org.opensearch.knn.index.engine.TrainingConfigValidationOutput;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.io.IOException;
import java.util.Map;
import java.util.function.Function;

import static org.opensearch.knn.common.KNNConstants.BUILD_ALGORITHM_HASHPRUNE;
import static org.opensearch.knn.common.KNNConstants.BUILD_ALGORITHM_HNSW;
import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_PQ;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_BUILD_ALGORITHM;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.SQ_BITS;

public class FaissHNSWMethodTests extends KNNTestCase {

    public void testValidate_whenHalfFloatWithNoEncoder_thenAccepted() throws IOException {
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .dimension(10)
            .vectorDataType(VectorDataType.HALF_FLOAT)
            .build();

        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .endObject();
        KNNMethodContext knnMethodContext = KNNMethodContext.parse(xContentBuilderToMap(xContentBuilder));

        assertNull(new FaissHNSWMethod().validate(knnMethodContext, knnMethodConfigContext));
    }

    public void testValidate_whenHalfFloatWithSqBits16Encoder_thenRejected() {
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .dimension(10)
            .vectorDataType(VectorDataType.HALF_FLOAT)
            .build();

        KNNMethodContext knnMethodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(
                METHOD_HNSW,
                Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 16)))
            )
        );

        assertNotNull(new FaissHNSWMethod().validate(knnMethodContext, knnMethodConfigContext));
    }

    public void testValidate_whenHalfFloatWithSqBits1Encoder_thenAccepted() {
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .dimension(10)
            .vectorDataType(VectorDataType.HALF_FLOAT)
            .build();

        KNNMethodContext knnMethodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(
                METHOD_HNSW,
                Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 1)))
            )
        );

        assertNull(new FaissHNSWMethod().validate(knnMethodContext, knnMethodConfigContext));
    }

    public void testSupportedEncoders_containsFlatSqPqAndQFrame() {
        Map<String, Encoder> encoders = FaissHNSWMethod.SUPPORTED_ENCODERS;
        assertTrue(encoders.containsKey(ENCODER_FLAT));
        assertTrue(encoders.containsKey(ENCODER_SQ));
        assertTrue(encoders.containsKey(ENCODER_PQ));
        assertTrue(encoders.containsKey(QFrameBitEncoder.NAME));
        assertEquals(4, encoders.size());
    }

    public void testSupportedEncoders_sqEncoderIsFaissSQEncoder() {
        Encoder sqEncoder = FaissHNSWMethod.SUPPORTED_ENCODERS.get(ENCODER_SQ);
        assertNotNull(sqEncoder);
        assertTrue(sqEncoder instanceof FaissSQEncoder);
    }

    public void testSupportedEncoders_flatEncoderCompressionIsX1() {
        Encoder flatEncoder = FaissHNSWMethod.SUPPORTED_ENCODERS.get(ENCODER_FLAT);
        assertEquals(CompressionLevel.x1, flatEncoder.calculateCompressionLevel(null, null));
    }

    public void testSupportedEncoders_sqEncoderBits1CompressionIsX32() {
        Encoder sqEncoder = FaissHNSWMethod.SUPPORTED_ENCODERS.get(ENCODER_SQ);
        MethodComponentContext mcc = new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 1));
        assertEquals(CompressionLevel.x32, sqEncoder.calculateCompressionLevel(mcc, null));
    }

    public void testSupportedEncoders_sqEncoderBits16CompressionIsX2() {
        Encoder sqEncoder = FaissHNSWMethod.SUPPORTED_ENCODERS.get(ENCODER_SQ);
        MethodComponentContext mcc = new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 16));
        assertEquals(CompressionLevel.x2, sqEncoder.calculateCompressionLevel(mcc, null));
    }

    public void testTrainingConfigValidation_whenHNSWWithSQBits1_thenValid() {
        FaissHNSWMethod method = new FaissHNSWMethod();
        KNNMethodConfigContext configContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .compressionLevel(CompressionLevel.x32)
            .build();

        KNNMethodContext methodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            org.opensearch.knn.index.SpaceType.L2,
            new MethodComponentContext(
                METHOD_HNSW,
                Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 1)))
            )
        );

        KNNLibraryIndexingContext indexingContext = method.getKNNLibraryIndexingContext(methodContext, configContext);
        Function<TrainingConfigValidationInput, TrainingConfigValidationOutput> validationSetup = indexingContext
            .getTrainingConfigValidationSetup();

        TrainingConfigValidationOutput output = validationSetup.apply(
            TrainingConfigValidationInput.builder().knnMethodContext(methodContext).knnMethodConfigContext(configContext).build()
        );
        assertNull(output.getValid());
    }

    public void testTrainingConfigValidation_whenHNSWWithSQBits1AndX2Compression_thenInvalid() {
        FaissHNSWMethod method = new FaissHNSWMethod();
        KNNMethodConfigContext configContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .compressionLevel(CompressionLevel.x2)
            .build();

        KNNMethodContext methodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            org.opensearch.knn.index.SpaceType.L2,
            new MethodComponentContext(
                METHOD_HNSW,
                Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 1)))
            )
        );

        KNNLibraryIndexingContext indexingContext = method.getKNNLibraryIndexingContext(methodContext, configContext);
        Function<TrainingConfigValidationInput, TrainingConfigValidationOutput> validationSetup = indexingContext
            .getTrainingConfigValidationSetup();

        TrainingConfigValidationOutput output = validationSetup.apply(
            TrainingConfigValidationInput.builder().knnMethodContext(methodContext).knnMethodConfigContext(configContext).build()
        );
        assertNotNull(output.getValid());
        assertFalse(output.getValid());
        assertTrue(output.getErrorMessage().contains("incompatible"));
    }

    public void testTrainingConfigValidation_whenHNSWWithNoEncoder_thenValid() {
        FaissHNSWMethod method = new FaissHNSWMethod();
        KNNMethodConfigContext configContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .build();

        KNNMethodContext methodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            org.opensearch.knn.index.SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, Map.of())
        );

        KNNLibraryIndexingContext indexingContext = method.getKNNLibraryIndexingContext(methodContext, configContext);
        Function<TrainingConfigValidationInput, TrainingConfigValidationOutput> validationSetup = indexingContext
            .getTrainingConfigValidationSetup();

        TrainingConfigValidationOutput output = validationSetup.apply(
            TrainingConfigValidationInput.builder().knnMethodContext(methodContext).knnMethodConfigContext(configContext).build()
        );
        assertNull(output.getValid());
    }

    public void testDefaultEncoder_isFlatEncoder() {
        FaissHNSWMethod method = new FaissHNSWMethod();
        KNNMethodConfigContext configContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .build();

        KNNMethodContext methodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            org.opensearch.knn.index.SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, Map.of())
        );

        Map<String, Object> libraryParams = method.getKNNLibraryIndexingContext(methodContext, configContext).getLibraryParameters();
        @SuppressWarnings("unchecked")
        Map<String, Object> innerParams = (Map<String, Object>) libraryParams.get(PARAMETERS);
        @SuppressWarnings("unchecked")
        Map<String, Object> encoderParams = (Map<String, Object>) innerParams.get(METHOD_ENCODER_PARAMETER);
        assertEquals(ENCODER_FLAT, encoderParams.get(NAME));
    }

    public void testValidate_whenHashPruneWithSupportedConfiguration_thenAccepted() {
        for (SpaceType spaceType : new SpaceType[] { SpaceType.L2, SpaceType.INNER_PRODUCT, SpaceType.COSINESIMIL }) {
            assertNull(validateBuildAlgorithm(BUILD_ALGORITHM_HASHPRUNE, spaceType, VectorDataType.FLOAT, null, Mode.NOT_CONFIGURED));
            assertNull(
                validateBuildAlgorithm(
                    BUILD_ALGORITHM_HASHPRUNE,
                    spaceType,
                    VectorDataType.FLOAT,
                    new MethodComponentContext(ENCODER_FLAT, Map.of()),
                    Mode.IN_MEMORY
                )
            );
        }
        for (int bits : new int[] { 1, 2, 4 }) {
            assertNull(
                validateBuildAlgorithm(
                    BUILD_ALGORITHM_HASHPRUNE,
                    SpaceType.COSINESIMIL,
                    VectorDataType.FLOAT,
                    new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, bits)),
                    Mode.ON_DISK
                )
            );
        }
        assertNull(validateBuildAlgorithm(BUILD_ALGORITHM_HNSW, SpaceType.L2, VectorDataType.BYTE, null, Mode.NOT_CONFIGURED));
    }

    public void testValidate_whenHashPruneWithUnsupportedConfiguration_thenRejected() {
        assertNotNull(validateBuildAlgorithm(BUILD_ALGORITHM_HASHPRUNE, SpaceType.L2, VectorDataType.BYTE, null, Mode.NOT_CONFIGURED));
        assertNotNull(
            validateBuildAlgorithm(BUILD_ALGORITHM_HASHPRUNE, SpaceType.L2, VectorDataType.HALF_FLOAT, null, Mode.NOT_CONFIGURED)
        );
        assertNotNull(
            validateBuildAlgorithm(
                BUILD_ALGORITHM_HASHPRUNE,
                SpaceType.L2,
                VectorDataType.FLOAT,
                new MethodComponentContext(ENCODER_SQ, Map.of(SQ_BITS, 16)),
                Mode.NOT_CONFIGURED
            )
        );
        assertNotNull(
            validateBuildAlgorithm(
                BUILD_ALGORITHM_HASHPRUNE,
                SpaceType.L2,
                VectorDataType.FLOAT,
                new MethodComponentContext(ENCODER_PQ, Map.of()),
                Mode.NOT_CONFIGURED
            )
        );
    }

    public void testValidate_whenUnknownBuildAlgorithm_thenRejected() {
        assertNotNull(validateBuildAlgorithm("unknown", SpaceType.L2, VectorDataType.FLOAT, null, Mode.NOT_CONFIGURED));
    }

    @SuppressWarnings("unchecked")
    public void testLibraryParameters_passBuildAlgorithmOnlyWhenSet() {
        FaissHNSWMethod method = new FaissHNSWMethod();
        KNNMethodConfigContext configContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .build();

        KNNMethodContext unset = new KNNMethodContext(KNNEngine.FAISS, SpaceType.L2, new MethodComponentContext(METHOD_HNSW, Map.of()));
        Map<String, Object> unsetParams = (Map<String, Object>) method.getKNNLibraryIndexingContext(unset, configContext)
            .getLibraryParameters()
            .get(PARAMETERS);
        assertFalse(unsetParams.containsKey(METHOD_PARAMETER_BUILD_ALGORITHM));

        KNNMethodContext hashPrune = new KNNMethodContext(
            KNNEngine.FAISS,
            SpaceType.L2,
            new MethodComponentContext(METHOD_HNSW, Map.of(METHOD_PARAMETER_BUILD_ALGORITHM, BUILD_ALGORITHM_HASHPRUNE))
        );
        Map<String, Object> hashPruneParams = (Map<String, Object>) method.getKNNLibraryIndexingContext(hashPrune, configContext)
            .getLibraryParameters()
            .get(PARAMETERS);
        assertEquals(BUILD_ALGORITHM_HASHPRUNE, hashPruneParams.get(METHOD_PARAMETER_BUILD_ALGORITHM));
    }

    private static org.opensearch.common.ValidationException validateBuildAlgorithm(
        String buildAlgorithm,
        SpaceType spaceType,
        VectorDataType vectorDataType,
        MethodComponentContext encoder,
        Mode mode
    ) {
        Map<String, Object> parameters = encoder == null
            ? Map.of(METHOD_PARAMETER_BUILD_ALGORITHM, buildAlgorithm)
            : Map.of(METHOD_PARAMETER_BUILD_ALGORITHM, buildAlgorithm, METHOD_ENCODER_PARAMETER, encoder);
        KNNMethodContext knnMethodContext = new KNNMethodContext(
            KNNEngine.FAISS,
            spaceType,
            new MethodComponentContext(METHOD_HNSW, parameters)
        );
        KNNMethodConfigContext knnMethodConfigContext = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .dimension(16)
            .vectorDataType(vectorDataType)
            .mode(mode)
            .build();
        return new FaissHNSWMethod().validate(knnMethodContext, knnMethodConfigContext);
    }
}
