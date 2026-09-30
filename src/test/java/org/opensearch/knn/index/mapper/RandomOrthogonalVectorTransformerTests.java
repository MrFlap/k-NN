/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.opensearch.Version;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;

import java.util.Arrays;
import java.util.Collections;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_FLAT;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.RANDOM_ORTHOGONAL_TRANSFORM;
import static org.opensearch.knn.common.KNNConstants.RANDOM_ORTHOGONAL_TRANSFORM_FWHH_V1;

public class RandomOrthogonalVectorTransformerTests extends KNNTestCase {

    public void testIsSupported() {
        assertTrue(RandomOrthogonalVectorTransformer.isSupported(context(KNNEngine.FAISS, SpaceType.L2, ENCODER_SQ), VectorDataType.FLOAT));
        assertTrue(
            RandomOrthogonalVectorTransformer.isSupported(
                context(KNNEngine.LUCENE, SpaceType.COSINESIMIL, ENCODER_SQ),
                VectorDataType.FLOAT
            )
        );
        assertTrue(
            RandomOrthogonalVectorTransformer.isSupported(
                context(KNNEngine.FAISS, SpaceType.INNER_PRODUCT, ENCODER_SQ),
                VectorDataType.FLOAT
            )
        );

        assertFalse(RandomOrthogonalVectorTransformer.isSupported(null, VectorDataType.FLOAT));
        assertFalse(RandomOrthogonalVectorTransformer.isSupported(context(KNNEngine.FAISS, SpaceType.L2, ENCODER_SQ), VectorDataType.BYTE));
        assertFalse(
            RandomOrthogonalVectorTransformer.isSupported(context(KNNEngine.FAISS, SpaceType.L2, ENCODER_SQ), VectorDataType.HALF_FLOAT)
        );
        assertFalse(
            RandomOrthogonalVectorTransformer.isSupported(context(KNNEngine.FAISS, SpaceType.L2, ENCODER_FLAT), VectorDataType.FLOAT)
        );
        assertFalse(RandomOrthogonalVectorTransformer.isSupported(context(KNNEngine.FAISS, SpaceType.L2, null), VectorDataType.FLOAT));
        assertFalse(
            RandomOrthogonalVectorTransformer.isSupported(context(KNNEngine.FAISS, SpaceType.L1, ENCODER_SQ), VectorDataType.FLOAT)
        );
        assertFalse(
            RandomOrthogonalVectorTransformer.isSupported(context(KNNEngine.NMSLIB, SpaceType.L2, ENCODER_SQ), VectorDataType.FLOAT)
        );
    }

    public void testIsEnabled() {
        final KNNMethodContext sq = context(KNNEngine.FAISS, SpaceType.L2, ENCODER_SQ);
        final Version current = Version.CURRENT;

        assertTrue(RandomOrthogonalVectorTransformer.isEnabled(null, sq, VectorDataType.FLOAT, current));
        assertTrue(RandomOrthogonalVectorTransformer.isEnabled(true, sq, VectorDataType.FLOAT, current));
        assertFalse(RandomOrthogonalVectorTransformer.isEnabled(false, sq, VectorDataType.FLOAT, current));
        assertFalse(
            RandomOrthogonalVectorTransformer.isEnabled(null, context(KNNEngine.FAISS, SpaceType.L2, null), VectorDataType.FLOAT, current)
        );

        // Indices created before the feature are never transformed, whatever the mapping says.
        assertFalse(RandomOrthogonalVectorTransformer.isEnabled(null, sq, VectorDataType.FLOAT, Version.V_3_8_0));
        assertFalse(RandomOrthogonalVectorTransformer.isEnabled(true, sq, VectorDataType.FLOAT, Version.V_3_8_0));
        assertFalse(RandomOrthogonalVectorTransformer.isEnabled(null, sq, VectorDataType.FLOAT, null));
    }

    public void testIsAppliedTo() {
        assertFalse(RandomOrthogonalVectorTransformer.isAppliedTo(null));
        assertFalse(RandomOrthogonalVectorTransformer.isAppliedTo(Collections.emptyMap()));
        assertTrue(RandomOrthogonalVectorTransformer.isAppliedTo(Map.of(RANDOM_ORTHOGONAL_TRANSFORM, RANDOM_ORTHOGONAL_TRANSFORM_FWHH_V1)));
        expectThrows(
            IllegalStateException.class,
            () -> RandomOrthogonalVectorTransformer.isAppliedTo(Map.of(RANDOM_ORTHOGONAL_TRANSFORM, "unknown"))
        );
    }

    public void testInverseIfApplied_restoresOriginalWithoutModifyingInput() {
        final int d = 1000;
        final float[] original = randomVector(d);
        final float[] transformed = RandomOrthogonalVectorTransformer.forDimension(d).transform(original, false);
        final float[] stored = Arrays.copyOf(transformed, d);

        final float[] restored = RandomOrthogonalVectorTransformer.inverseIfApplied(
            Map.of(RANDOM_ORTHOGONAL_TRANSFORM, RANDOM_ORTHOGONAL_TRANSFORM_FWHH_V1),
            stored
        );
        assertNotSame(stored, restored);
        assertArrayEquals(transformed, stored, 0f);
        assertArrayEquals(original, restored, 1e-5f);

        assertSame(stored, RandomOrthogonalVectorTransformer.inverseIfApplied(Collections.emptyMap(), stored));
    }

    public void testForDimension_isShared() {
        assertSame(RandomOrthogonalVectorTransformer.forDimension(128), RandomOrthogonalVectorTransformer.forDimension(128));
        assertNotSame(RandomOrthogonalVectorTransformer.forDimension(128), RandomOrthogonalVectorTransformer.forDimension(129));
    }

    public void testWithRandomOrthogonalTransform_composesAfterBaseWithoutModifyingInput() {
        final int d = 100;
        final float[] input = randomVector(d);
        final float[] original = Arrays.copyOf(input, d);
        final RandomOrthogonalVectorTransformer orthogonal = RandomOrthogonalVectorTransformer.forDimension(d);

        // The no-op base returns the caller's array, which must not be transformed in place.
        final VectorTransformer noopComposed = VectorTransformerFactory.withRandomOrthogonalTransform(
            VectorTransformerFactory.NOOP_VECTOR_TRANSFORMER,
            d
        );
        assertArrayEquals(orthogonal.transform(original, false), noopComposed.transform(input, false), 0f);
        assertArrayEquals(original, input, 0f);

        final VectorTransformer normalizeComposed = VectorTransformerFactory.withRandomOrthogonalTransform(
            new NormalizeVectorTransformer(),
            d
        );
        final float[] expected = orthogonal.transform(new NormalizeVectorTransformer().transform(original, false), true);
        assertArrayEquals(expected, normalizeComposed.transform(input, false), 1e-6f);
        assertArrayEquals(original, input, 0f);

        final float[] inPlace = normalizeComposed.transform(input, true);
        assertSame(input, inPlace);
        assertArrayEquals(expected, inPlace, 1e-6f);
    }

    private static KNNMethodContext context(KNNEngine engine, SpaceType spaceType, String encoder) {
        final Map<String, Object> parameters = encoder == null
            ? Collections.emptyMap()
            : Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(encoder, Collections.emptyMap()));
        return new KNNMethodContext(engine, spaceType, new MethodComponentContext(METHOD_HNSW, parameters));
    }

    private static float[] randomVector(int d) {
        final float[] vector = new float[d];
        for (int i = 0; i < d; i++) {
            vector[i] = randomFloat() * 2 - 1;
        }
        return vector;
    }
}
