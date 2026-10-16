/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.jni.SimdVectorComputeService;
import org.opensearch.knn.jni.SimdVectorComputeService.SimilarityFunctionType;

/**
 * Scores a float query against memory-mapped data-blind int8 codes with the native SIMD kernel ported from ClusterANN.
 * Scores are bit-identical to {@link DataBlindInt8RandomVectorScorer}. The native search context is per thread, so like
 * the 1-bit SQ scorer this one must score on the thread that created it.
 */
class DataBlindInt8BulkSimdRandomVectorScorer extends RandomVectorScorer.AbstractRandomVectorScorer {

    private final FloatVectorValues docMappedValues;

    /**
     * @param docMappedValues the field's float values, which own the ord-to-doc mapping and accepted ords
     * @param codes           the int8 codes sharing those ordinals; {@link #values()}, so prefetching reads them
     * @param addressAndSize  the mmap regions of {@code codes}, from {@code MemorySegmentAddressExtractorUtil}
     */
    DataBlindInt8BulkSimdRandomVectorScorer(
        final FloatVectorValues docMappedValues,
        final DataBlindInt8VectorValues codes,
        final float[] query,
        final long[] addressAndSize,
        final VectorSimilarityFunction similarityFunction
    ) {
        super(codes);
        this.docMappedValues = docMappedValues;
        final int dimension = codes.dimension();
        final DataBlindInt8Quantizer quantizer = new DataBlindInt8Quantizer(dimension);
        final byte[] queryCode = new byte[dimension];
        final DataBlindInt8Quantizer.Int8Code terms = quantizer.encode(query, queryCode);
        SimdVectorComputeService.saveInt8SearchContext(
            queryCode,
            terms.scale(),
            terms.sum(),
            terms.sqNorm(),
            addressAndSize,
            nativeType(similarityFunction).ordinal(),
            dimension
        );
    }

    private static SimilarityFunctionType nativeType(final VectorSimilarityFunction similarityFunction) {
        return switch (similarityFunction) {
            case EUCLIDEAN -> SimilarityFunctionType.DB_INT8_L2;
            case MAXIMUM_INNER_PRODUCT -> SimilarityFunctionType.DB_INT8_IP;
            case COSINE -> SimilarityFunctionType.DB_INT8_COSINE;
            default -> throw new IllegalStateException("Data-blind int8 does not support " + similarityFunction);
        };
    }

    @Override
    public int ordToDoc(final int ord) {
        return docMappedValues.ordToDoc(ord);
    }

    @Override
    public Bits getAcceptOrds(final Bits acceptDocs) {
        return docMappedValues.getAcceptOrds(acceptDocs);
    }

    @Override
    public float bulkScore(final int[] internalVectorIds, final float[] scores, final int numVectors) {
        return SimdVectorComputeService.scoreSimilarityInBulk(internalVectorIds, scores, numVectors);
    }

    @Override
    public float score(final int internalVectorId) {
        return SimdVectorComputeService.scoreSimilarity(internalVectorId);
    }
}
