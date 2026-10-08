/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

/**
 * Simple symmetric int8 scalar quantizer: one scale per vector, no centroid, no interval optimization.
 *
 * <p>Each vector is mapped to signed codes {@code q_i = round(x_i / scale)} with {@code scale = max|x_i| / 127}, so
 * {@code q_i ∈ [-127, 127]} and the sign of every component is preserved. This is a good fit for vectors that went through
 * the random orthogonal transform, whose components are close to symmetric and similarly distributed.
 *
 * <p>The codes are stored in Lucene's existing {@code UNSIGNED_BYTE} layout as the offset-binary value {@code code = q + 128}, so
 * the stock reader and scorer arithmetic keep working unchanged. Lucene dequantizes a code as
 * {@code lower + code * (upper - lower) / 255}, which equals {@code scale * q} for
 * <pre>
 *   lower = -128 * scale,  upper = 127 * scale   (step = scale)
 * </pre>
 * With a zero centroid (which is how fields written with this quantizer are recorded) the remaining corrective terms are
 * {@code quantizedComponentSum = Σ code_i} and {@code additionalCorrection = ||x||²} for euclidean and {@code 0} (the dot
 * product with the centroid) otherwise.
 *
 * <p>Extends {@link OptimizedScalarQuantizer} only so it can be handed to code that expects one; it overrides
 * {@link #scalarQuantize} and ignores the centroid argument.
 */
public final class Int8ScalarQuantizer extends OptimizedScalarQuantizer {

    /** Document bit width this quantizer produces. */
    public static final byte BITS = 8;

    // Largest magnitude of a signed code. -128 is left unused so the range is symmetric.
    static final int MAX_CODE = 127;
    // Offset that turns a signed code in [-127, 127] into the stored unsigned code in [1, 255].
    static final int CODE_OFFSET = 128;

    private final VectorSimilarityFunction similarityFunction;

    public Int8ScalarQuantizer(final VectorSimilarityFunction similarityFunction) {
        super(similarityFunction);
        this.similarityFunction = similarityFunction;
    }

    /**
     * Quantizes {@code vector} into {@code destination}. {@code vector} is not modified and {@code centroid} is ignored.
     *
     * @param vector vector to quantize
     * @param destination output codes, at least {@code vector.length} long
     * @param bits must be {@link #BITS}
     * @param centroid ignored, the centroid of an int8 field is always zero
     * @return corrective terms in Lucene's {@code UNSIGNED_BYTE} convention
     */
    @Override
    public QuantizationResult scalarQuantize(final float[] vector, final byte[] destination, final byte bits, final float[] centroid) {
        if (bits != BITS) {
            throw new IllegalArgumentException("Int8ScalarQuantizer only supports " + BITS + " bits, got " + bits);
        }
        if (destination.length < vector.length) {
            throw new IllegalArgumentException(
                "destination length [" + destination.length + "] is smaller than vector length [" + vector.length + "]"
            );
        }

        float maxAbs = 0f;
        float norm2 = 0f;
        for (final float v : vector) {
            maxAbs = Math.max(maxAbs, Math.abs(v));
            norm2 += v * v;
        }

        // An all-zero vector quantizes to all-zero signed codes whatever the scale is, so any positive scale will do.
        final float scale = maxAbs == 0f ? 1f : maxAbs / MAX_CODE;
        final float inverseScale = 1f / scale;

        int componentSum = 0;
        for (int i = 0; i < vector.length; i++) {
            final int signedCode = Math.max(-MAX_CODE, Math.min(MAX_CODE, Math.round(vector[i] * inverseScale)));
            final int code = signedCode + CODE_OFFSET;
            destination[i] = (byte) code;
            componentSum += code;
        }
        // Positions beyond the vector length are not used by the 8-bit layout (no padding), but keep them as zero-valued codes.
        for (int i = vector.length; i < destination.length; i++) {
            destination[i] = (byte) CODE_OFFSET;
            componentSum += CODE_OFFSET;
        }

        return new QuantizationResult(
            -CODE_OFFSET * scale,
            MAX_CODE * scale,
            similarityFunction == VectorSimilarityFunction.EUCLIDEAN ? norm2 : 0f,
            componentSum
        );
    }
}
