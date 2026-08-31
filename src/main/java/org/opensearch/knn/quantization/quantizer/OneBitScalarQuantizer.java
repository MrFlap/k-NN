/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.quantization.quantizer;

import org.opensearch.knn.index.engine.faiss.QFrameBitEncoder;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;
import org.opensearch.knn.quantization.models.quantizationOutput.QuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationParams.ScalarQuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
import org.opensearch.knn.quantization.models.requests.TrainingRequest;
import org.opensearch.knn.quantization.sampler.Sampler;
import org.opensearch.knn.quantization.sampler.SamplerType;
import org.opensearch.knn.quantization.sampler.SamplingFactory;

import java.io.IOException;

import static org.opensearch.knn.common.KNNConstants.ADC_CORRECTION_FACTOR;

/**
 * OneBitScalarQuantizer is responsible for quantizing vectors using a single bit per dimension.
 * It computes the mean of each dimension during training and then uses these means as thresholds
 * for quantizing the vectors.
 */
public class OneBitScalarQuantizer implements Quantizer<float[], byte[]> {
    private final int samplingSize; // Sampling size for training
    private final boolean shouldUseRandomRotation;
    private static final boolean IS_TRAINING_REQUIRED = true;
    private final Sampler sampler; // Sampler for training
    // Currently Lucene has sampling size as
    // 25000 for segment level training , Keeping same
    // to having consistent, Will revisit
    // if this requires change
    private static final int DEFAULT_SAMPLE_SIZE = 25000;

    /**
     * Constructs a OneBitScalarQuantizer with a default sampling size of 25000.
     */
    public OneBitScalarQuantizer() {
        this(DEFAULT_SAMPLE_SIZE, QFrameBitEncoder.DEFAULT_ENABLE_RANDOM_ROTATION, SamplingFactory.getSampler(SamplerType.RESERVOIR));
    }

    /**
     * Constructs a OneBitScalarQuantizer with a specified sampling size.
     *
     * @param samplingSize the number of samples to use for training.
     */
    public OneBitScalarQuantizer(final int samplingSize, final boolean shouldUseRandomRotation, final Sampler sampler) {
        this.samplingSize = samplingSize;
        this.shouldUseRandomRotation = shouldUseRandomRotation;
        this.sampler = sampler;
    }

    public OneBitScalarQuantizer(final boolean shouldUseRandomRotation) {
        this(DEFAULT_SAMPLE_SIZE, shouldUseRandomRotation, SamplingFactory.getSampler(SamplerType.RESERVOIR));
    }

    public OneBitScalarQuantizer(final int samplingSize, final Sampler sampler) {
        this.samplingSize = samplingSize;
        this.shouldUseRandomRotation = QFrameBitEncoder.DEFAULT_ENABLE_RANDOM_ROTATION;
        this.sampler = sampler;
    }

    /**
     * Trains the quantizer by calculating the mean of each dimension from the sampled vectors.
     * These means are used as thresholds in the quantization process.
     *
     * @param trainingRequest the request containing the data and parameters for training.
     * @return a OneBitScalarQuantizationState containing the calculated means.
     */
    @Override
    public QuantizationState train(final TrainingRequest<float[]> trainingRequest) throws IOException {
        int[] sampledDocIds = sampler.sample(trainingRequest.getTotalNumberOfVectors(), samplingSize);
        return QuantizerHelper.calculateQuantizationState(
            trainingRequest,
            sampledDocIds,
            ScalarQuantizationParams.builder()
                .sqType(ScalarQuantizationType.ONE_BIT)
                .enableRandomRotation(this.shouldUseRandomRotation)
                .build()
        );
    }

    /**
     * Quantizes the provided vector using the given quantization state.
     * It compares each dimension of the vector against the corresponding mean (threshold) to determine the quantized value.
     *
     * @param vector the vector to quantize.
     * @param state  the quantization state containing the means for each dimension.
     * @param output the QuantizationOutput object to store the quantized representation of the vector.
     */
    @Override
    public void quantize(float[] vector, final QuantizationState state, final QuantizationOutput<byte[]> output) {
        if (vector == null) {
            throw new IllegalArgumentException("Vector to quantize must not be null.");
        }
        validateState(state);
        int vectorLength = vector.length;
        OneBitScalarQuantizationState binaryState = (OneBitScalarQuantizationState) state;
        float[] thresholds = binaryState.getMeanThresholds();
        if (thresholds == null || thresholds.length != vectorLength) {
            throw new IllegalArgumentException("Thresholds must not be null and must match the dimension of the vector.");
        }
        float[][] rotationMatrix = binaryState.getRotationMatrix();
        if (rotationMatrix != null) {
            vector = RandomGaussianRotation.applyRotation(vector, rotationMatrix);
        }
        output.prepareQuantizedVector(vectorLength);
        BitPacker.quantizeAndPackBits(vector, thresholds, output.getQuantizedVector());
    }

    /**
     * Transform vector with ADC. ADC allows us to score full-precision query vectors against binary document vectors.
     * <p>
     * With exact ADC enabled (the default), the transform encodes the <em>exact</em> distance to the implied
     * reconstruction of each document. See {@link #transformVectorWithExactADC} for the derivation. The returned
     * offset must be added to the raw kernel output before score translation, otherwise scores from different
     * segments are not comparable.
     * <p>
     * With exact ADC disabled, the legacy transform is used:
     * q_d = (q_d - x_d) / (y_d - x_d) where x_d is the mean of all document entries quantized to 0 (the below threshold mean)
     * and y_d is the mean of all document entries quantized to 1 (the above threshold mean). The legacy transform drops
     * the per-dimension (y_d - x_d)^2 weighting, so its scores carry a segment-dependent scale and offset.
     *
     * @param vector array of floats, modified in-place.
     * @param state The {@link QuantizationState} containing the state of the trained quantizer.
     * @param spaceType spaceType (l2 or innerproduct). Used to identify whether an additional correction term should be applied.
     * @param exactAdc selects the exact transform over the legacy one.
     * @return the segment-level additive offset to apply to the raw kernel output; zero for the legacy transform.
     */
    @Override
    public float transformWithADC(float[] vector, final QuantizationState state, final SpaceType spaceType, final boolean exactAdc) {
        validateState(state);
        OneBitScalarQuantizationState binaryState = (OneBitScalarQuantizationState) state;

        float[][] rotationMatrix = binaryState.getRotationMatrix();

        float[] rotatedVector = vector.clone();

        if (rotationMatrix != null) {
            rotatedVector = RandomGaussianRotation.applyRotation(vector, rotationMatrix);
        }

        final float distanceOffset;
        if (exactAdc) {
            distanceOffset = transformVectorWithExactADC(rotatedVector, binaryState, spaceType);
        } else if (shouldDoADCCorrection(spaceType)) {
            transformVectorWithADCCorrection(rotatedVector, binaryState);
            distanceOffset = 0f;
        } else {
            transformVectorWithADCNoCorrection(rotatedVector, binaryState);
            distanceOffset = 0f;
        }

        System.arraycopy(rotatedVector, 0, vector, 0, vector.length);
        return distanceOffset;
    }

    /**
     * Encodes the exact distance to the ADC reconstruction into the existing float[] query channel.
     * <p>
     * A document's bits b imply the reconstruction r_d = x_d + b_d * delta_d, with delta_d = y_d - x_d. Because
     * b_d is in {0, 1}, both metrics are exactly <em>linear</em> in the bits:
     * <pre>
     *   ||q - r||^2 = SUM (q_d - x_d)^2 + SUM b_d * [delta_d^2 - 2 * delta_d * (q_d - x_d)]
     *   &lt;q, r&gt;      = SUM q_d * x_d    + SUM b_d * [q_d * delta_d]
     * </pre>
     * The ADC kernels already evaluate {@code kernelConstant + SUM b_d * coeff_d}, deriving both terms from the
     * transformed query t: the L2 kernel uses {@code coeff_d = 1 - 2 * t_d} with {@code kernelConstant = SUM t_d^2},
     * and the inner-product kernel uses {@code coeff_d = t_d} with a zero constant. So we solve for the t that
     * reproduces the exact coefficients, and return the difference between the true constant and the constant the
     * kernel will compute for itself.
     * <p>
     * Relationship to the legacy transform: for L2 the legacy correction factor of 2 already produced these exact
     * coefficients. Substituting {@code t_d = delta_d^2 * (q'_d - 0.5) + 0.5} into {@code 1 - 2 * t_d} gives
     * {@code delta_d^2 * (1 - 2 * q'_d)}, which is what the exact derivation yields. So for L2 the only thing this
     * changes is the constant: the kernel contributes {@code SUM t_d^2}, which has no physical meaning, where the
     * true distance needs {@code SUM (q_d - x_d)^2}. That difference is invisible when ranking inside one segment
     * and is precisely the error that breaks cross-segment merging. Inner product and cosine differ in the
     * coefficients too - the legacy transform divides by {@code delta_d} where the exact one multiplies by it - so
     * for those spaces this also changes ordering within a segment.
     * <p>
     * Note on conditioning: for L2, {@code t_d} is centred near 0.5, so {@code SUM t_d^2} grows with the dimension
     * (~D/4) while the true constant is small. The kernel therefore adds a large value that this offset subtracts
     * back out, costing float precision in the part that carries the signal. Inner product and cosine are
     * unaffected (zero kernel constant). Passing coefficients and the constant to the kernel explicitly would
     * remove the cancellation for L2; that requires a JNI signature change and is left as a follow-up.
     *
     * @param vector rotated query, overwritten in place with the encoded query.
     * @param state quantization state supplying the below/above threshold means.
     * @param spaceType space type selecting the metric.
     * @return the additive offset for the raw kernel output.
     */
    private float transformVectorWithExactADC(final float[] vector, final OneBitScalarQuantizationState state, final SpaceType spaceType) {
        final float[] belowThresholdMeans = state.getBelowThresholdMeans();
        final float[] aboveThresholdMeans = state.getAboveThresholdMeans();

        if (SpaceType.L2.equals(spaceType)) {
            double trueConstant = 0.0;
            double kernelConstant = 0.0;
            for (int i = 0; i < vector.length; i++) {
                final float below = belowThresholdMeans[i];
                final float delta = aboveThresholdMeans[i] - below;
                final float centered = vector[i] - below;
                final float coefficient = delta * delta - 2f * delta * centered;
                // The L2 kernel derives coeff_d as (1 - 2 * t_d), so invert for t_d.
                final float encoded = (1f - coefficient) * 0.5f;
                vector[i] = encoded;
                trueConstant += (double) centered * centered;
                kernelConstant += (double) encoded * encoded;
            }
            return (float) (trueConstant - kernelConstant);
        }

        // INNER_PRODUCT and COSINESIMIL both reach the kernel as a raw inner product.
        double trueConstant = 0.0;
        for (int i = 0; i < vector.length; i++) {
            final float below = belowThresholdMeans[i];
            final float delta = aboveThresholdMeans[i] - below;
            trueConstant += (double) vector[i] * below;
            // The inner-product kernel uses coeff_d = t_d directly, and contributes no constant of its own.
            vector[i] = vector[i] * delta;
        }
        return (float) trueConstant;
    }

    private boolean shouldDoADCCorrection(SpaceType spaceType) {
        // Note that correction will not work for cosine similarity since these vectors are normalized and correction will break
        // normalization.
        // A normalization-aware correction term may be added in the future so we can support inner product spaces.
        return SpaceType.L2.equals(spaceType);
    }

    private void transformVectorWithADCNoCorrection(float[] vector, final OneBitScalarQuantizationState binaryState) {
        for (int i = 0; i < vector.length; ++i) {
            float aboveThreshold = binaryState.getAboveThresholdMeans()[i];
            float belowThreshold = binaryState.getBelowThresholdMeans()[i];

            vector[i] = (vector[i] - belowThreshold) / (aboveThreshold - belowThreshold);
        }
    }

    private void transformVectorWithADCCorrection(float[] vector, final OneBitScalarQuantizationState binaryState) {
        for (int i = 0; i < vector.length; i++) {
            float aboveThreshold = binaryState.getAboveThresholdMeans()[i];
            float belowThreshold = binaryState.getBelowThresholdMeans()[i];
            double correction = Math.pow(aboveThreshold - belowThreshold, ADC_CORRECTION_FACTOR);
            vector[i] = (vector[i] - belowThreshold) / (aboveThreshold - belowThreshold);
            vector[i] = (float) correction * (vector[i] - 0.5f) + 0.5f;
        }
    }

    /**
     * Validates the quantization state to ensure it is of the expected type.
     *
     * @param state the quantization state to validate.
     * @throws IllegalArgumentException if the state is not an instance of OneBitScalarQuantizationState.
     */
    private void validateState(final QuantizationState state) {
        if (!(state instanceof OneBitScalarQuantizationState)) {
            throw new IllegalArgumentException("Quantization state must be of type OneBitScalarQuantizationState.");
        }
    }
}
