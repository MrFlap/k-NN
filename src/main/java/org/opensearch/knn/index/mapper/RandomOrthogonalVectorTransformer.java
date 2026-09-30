/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.opensearch.Version;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;

import java.util.Arrays;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.RANDOM_ORTHOGONAL_TRANSFORM;
import static org.opensearch.knn.common.KNNConstants.RANDOM_ORTHOGONAL_TRANSFORM_FWHH_V1;
import static org.opensearch.knn.common.KNNConstants.RANDOM_ORTHOGONAL_TRANSFORM_MIN_VERSION;

/**
 * Applies a seeded random orthogonal transform to vectors, so L2 distances, inner products and cosine similarities are
 * preserved as long as indexed and query vectors are transformed with the same seed.
 * <p>
 * Fields using the transform carry the {@code random_orthogonal_transform} field attribute, which readers of stored vectors
 * (derived source, script doc values, docvalue_fields) use to undo it via {@link #inverseIfApplied(Map, float[])}.
 */
public class RandomOrthogonalVectorTransformer implements VectorTransformer {
    // Seed for RANDOM_ORTHOGONAL_TRANSFORM_FWHH_V1. Changing it changes the transform of existing indices.
    static final long DEFAULT_SEED = 42L;

    // Orthogonal transforms only preserve these similarities.
    private static final Set<SpaceType> SUPPORTED_SPACE_TYPES = Set.of(SpaceType.L2, SpaceType.INNER_PRODUCT, SpaceType.COSINESIMIL);

    private static final Map<Integer, RandomOrthogonalVectorTransformer> BY_DIMENSION = new ConcurrentHashMap<>();

    private final FastWalshHadamardHouseholder orthogonalTransform;

    RandomOrthogonalVectorTransformer(int d) {
        this(d, DEFAULT_SEED);
    }

    RandomOrthogonalVectorTransformer(int d, long seed) {
        this(new FastWalshHadamardHouseholder(d, seed));
    }

    RandomOrthogonalVectorTransformer(FastWalshHadamardHouseholder transform) {
        this.orthogonalTransform = transform;
    }

    /**
     * Returns the shared transformer for the given dimension, using the default seed.
     *
     * @param dimension vector dimension
     * @return transformer for {@code dimension}
     */
    public static RandomOrthogonalVectorTransformer forDimension(final int dimension) {
        return BY_DIMENSION.computeIfAbsent(dimension, RandomOrthogonalVectorTransformer::new);
    }

    @Override
    public float[] transform(final float[] vector, final boolean inplaceUpdate) {
        if (vector == null) {
            throw new IllegalArgumentException("Input vector cannot be null");
        }
        final float[] target = inplaceUpdate ? vector : Arrays.copyOf(vector, vector.length);
        orthogonalTransform.apply(target);
        return target;
    }

    /**
     * Undoes {@link #transform(float[], boolean)}.
     *
     * @param vector transformed vector
     * @param inplaceUpdate whether to update {@code vector} in place or return a new array
     * @return the original vector, up to floating point round-off
     */
    public float[] inverseTransform(final float[] vector, final boolean inplaceUpdate) {
        if (vector == null) {
            throw new IllegalArgumentException("Input vector cannot be null");
        }
        final float[] target = inplaceUpdate ? vector : Arrays.copyOf(vector, vector.length);
        orthogonalTransform.applyInverse(target);
        return target;
    }

    /**
     * Transforming byte vectors is not supported since the result is not representable as bytes.
     *
     * @throws UnsupportedOperationException always
     */
    @Override
    public void transform(byte[] vector) {
        throw new UnsupportedOperationException("Byte array orthogonal transform is not supported");
    }

    /**
     * Whether the transform can be applied to a field: float vectors encoded with the Faiss or Lucene {@code sq} encoder, using a
     * space type the transform preserves.
     *
     * @param resolvedKnnMethodContext resolved method context of the field, may be null
     * @param vectorDataType vector data type of the field
     * @return true if the transform is supported
     */
    public static boolean isSupported(final KNNMethodContext resolvedKnnMethodContext, final VectorDataType vectorDataType) {
        if (resolvedKnnMethodContext == null || vectorDataType != VectorDataType.FLOAT) {
            return false;
        }
        final KNNEngine engine = resolvedKnnMethodContext.getKnnEngine();
        if (engine != KNNEngine.FAISS && engine != KNNEngine.LUCENE) {
            return false;
        }
        if (SUPPORTED_SPACE_TYPES.contains(resolvedKnnMethodContext.getSpaceType()) == false) {
            return false;
        }
        final MethodComponentContext methodComponentContext = resolvedKnnMethodContext.getMethodComponentContext();
        if (methodComponentContext == null || methodComponentContext.getParameters() == null) {
            return false;
        }
        final Object encoder = methodComponentContext.getParameters().get(METHOD_ENCODER_PARAMETER);
        return encoder instanceof MethodComponentContext encoderContext && ENCODER_SQ.equals(encoderContext.getName());
    }

    /**
     * Resolves whether a field uses the transform. Indices created before
     * {@link org.opensearch.knn.common.KNNConstants#RANDOM_ORTHOGONAL_TRANSFORM_MIN_VERSION} never do. Otherwise, supported fields
     * use it unless it is explicitly disabled.
     *
     * @param configured value of the {@code random_orthogonal_transform} mapping parameter, or null when not set
     * @param resolvedKnnMethodContext resolved method context of the field, may be null
     * @param vectorDataType vector data type of the field
     * @param indexCreatedVersion version the index was created on
     * @return true if the field uses the transform
     */
    public static boolean isEnabled(
        final Boolean configured,
        final KNNMethodContext resolvedKnnMethodContext,
        final VectorDataType vectorDataType,
        final Version indexCreatedVersion
    ) {
        if (indexCreatedVersion == null || indexCreatedVersion.before(RANDOM_ORTHOGONAL_TRANSFORM_MIN_VERSION)) {
            return false;
        }
        if (isSupported(resolvedKnnMethodContext, vectorDataType) == false) {
            return false;
        }
        return configured == null || configured;
    }

    /**
     * Whether the given field attributes mark the field's stored vectors as transformed.
     *
     * @param fieldAttributes attributes of the field's {@link org.apache.lucene.index.FieldInfo} or
     *                        {@link org.apache.lucene.document.FieldType}, may be null
     * @return true if the stored vectors are transformed
     * @throws IllegalStateException if the attribute names an unknown transform
     */
    public static boolean isAppliedTo(final Map<String, String> fieldAttributes) {
        final String value = fieldAttributes == null ? null : fieldAttributes.get(RANDOM_ORTHOGONAL_TRANSFORM);
        if (value == null) {
            return false;
        }
        if (RANDOM_ORTHOGONAL_TRANSFORM_FWHH_V1.equals(value)) {
            return true;
        }
        throw new IllegalStateException(String.format(Locale.ROOT, "Unsupported %s [%s]", RANDOM_ORTHOGONAL_TRANSFORM, value));
    }

    /**
     * Undoes the transform if the field attributes mark the field as transformed. Stored vectors may be buffers owned by the
     * reader, so {@code vector} itself is never modified.
     *
     * @param fieldAttributes attributes of the field, may be null
     * @param vector stored vector
     * @return a new array with the transform undone if it was applied, otherwise {@code vector}
     */
    public static float[] inverseIfApplied(final Map<String, String> fieldAttributes, final float[] vector) {
        if (vector != null && isAppliedTo(fieldAttributes)) {
            return forDimension(vector.length).inverseTransform(vector, false);
        }
        return vector;
    }
}
