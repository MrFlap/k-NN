/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

import java.io.IOException;

/**
 * Scores a float query against data-blind int8 codes in Java: the query is encoded once with
 * {@link DataBlindInt8Quantizer}, then each candidate costs one unsigned byte dot and the shared epilogue
 * {@link DataBlindInt8Quantizer#score}.
 */
public class DataBlindInt8RandomVectorScorer extends RandomVectorScorer.AbstractRandomVectorScorer {

    private final FloatVectorValues docMappedValues;
    private final DataBlindInt8VectorValues codes;
    private final VectorSimilarityFunction similarityFunction;
    private final byte[] queryCode;
    private final DataBlindInt8Quantizer.QueryTerms queryTerms;

    /**
     * @param docMappedValues the field's float values, which own the ord-to-doc mapping and accepted ords
     * @param codes           the int8 codes sharing those ordinals; {@link #values()}, so prefetching reads them
     * @param query           the query, already in the field's (rotated) space
     */
    public DataBlindInt8RandomVectorScorer(
        final FloatVectorValues docMappedValues,
        final DataBlindInt8VectorValues codes,
        final float[] query,
        final VectorSimilarityFunction similarityFunction
    ) {
        super(codes);
        this.docMappedValues = docMappedValues;
        this.codes = codes;
        this.similarityFunction = similarityFunction;
        final DataBlindInt8Quantizer quantizer = new DataBlindInt8Quantizer(codes.dimension());
        this.queryCode = new byte[codes.dimension()];
        this.queryTerms = quantizer.queryTerms(quantizer.encode(query, queryCode));
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
    public float score(final int ord) throws IOException {
        final long unsignedDot = DataBlindInt8Quantizer.unsignedDot(queryCode, codes.vectorValue(ord), 0, queryCode.length);
        return DataBlindInt8Quantizer.score(
            unsignedDot,
            queryTerms,
            codes.scale(ord),
            codes.sum(ord),
            codes.sqNorm(ord),
            similarityFunction
        );
    }
}
