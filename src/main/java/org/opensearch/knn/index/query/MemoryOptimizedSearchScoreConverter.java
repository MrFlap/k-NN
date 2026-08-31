/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.search.ScoreDoc;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;

/**
 * Utility class for converting between Faiss and Lucene score representations
 * in memory-optimized search.
 *
 * <p>Memory-optimized search runs Lucene on top of a Faiss index. It leverages
 * Lucene’s efficient algorithms and Lucene’s {@code Directory} architecture for efficient loading to
 * produce the same results as when memory optimization is disabled.</p>
 * With the same query, results are expected to be identical regardless of
 * whether memory optimization is enabled.
 *
 * <p>However, unlike {@link KNNEngine},
 * the input here is a Faiss score, which must be converted to Lucene’s
 * scoring range.</p>
 *
 * <p>For example, Faiss uses inner product while Lucene uses
 * maximum inner product. When converting distances, this class maps
 * the Faiss score into the maximum inner product range so Lucene can
 * interpret it correctly during search.</p>
 *
 * <p>Conversely, it also converts Lucene scores back into Faiss scores so that
 * the same query produces consistent results across both implementations.
 *
 * <p>Note that this should be used only when memory_optimized_search is enabled.
 *
 */
public final class MemoryOptimizedSearchScoreConverter {
    /**
     * Convert Faiss distance to Lucene score.
     *
     * @param distance Faiss distance
     * @param spaceType Space type being used.
     * @return Converted value to be used during Lucene search algorithm.
     */
    public static float distanceToRadialThreshold(final float distance, final SpaceType spaceType) {
        switch (spaceType) {
            case INNER_PRODUCT:
                // Faiss distance for IP is -dot. Negate to get raw dot product for Lucene.
                return KNNEngine.LUCENE.distanceToRadialThreshold(-distance, spaceType);
            case COSINESIMIL:
                // For cosine similarity, `distance = 1 - inner_product_value`.
                // therefore, we should extract it then convert it to max_inner_product_value
                final float innerProductValue = KNNEngine.FAISS.distanceToRadialThreshold(distance, SpaceType.COSINESIMIL);

                // Convert inner product value to max inner product value.
                return SpaceType.INNER_PRODUCT.scoreTranslation(-innerProductValue);
            default:
                return KNNEngine.LUCENE.distanceToRadialThreshold(distance, spaceType);
        }
    }

    /**
     * Convert Faiss score to Lucene radial threshold.
     *
     * @param score Faiss score
     * @param spaceType Space type that's being used
     * @return Converted radial threshold for Lucene
     */
    public static float scoreToRadialThreshold(final float score, final SpaceType spaceType) {
        if (spaceType != SpaceType.COSINESIMIL) {
            return KNNEngine.LUCENE.scoreToRadialThreshold(score, spaceType);
        }

        // Since `score = (2 - (1 - inner_product_value)) / 2 = (1 + inner_product_value) / 2`,
        // we should extract it then convert it to max inner product value.
        final float innerProductValue = KNNEngine.FAISS.scoreToRadialThreshold(score, SpaceType.COSINESIMIL);

        // Convert inner product value to max inner product value.
        return SpaceType.INNER_PRODUCT.scoreTranslation(-innerProductValue);
    }

    /**
     * Adds a constant offset to each score in raw distance/similarity space.
     *
     * <p>Memory-optimized search returns scores already mapped into Lucene's range, but exact ADC needs to restore a
     * per-segment constant that lives in raw space. This inverts the Lucene mapping, applies the offset, and maps
     * back. Both mappings are strictly monotone, so this preserves the ordering within the segment while making the
     * scores comparable against other segments.</p>
     *
     * <p>Must be called before {@link #convertToCosineScore}, while cosine scores are still in
     * MAXIMUM_INNER_PRODUCT format.</p>
     *
     * @param scoreDocs results to adjust in place
     * @param spaceType space type the scores were produced for
     * @param rawScoreOffset offset to add in raw distance (L2) or inner product (IP/cosine) space
     */
    public static void applyRawScoreOffset(final ScoreDoc[] scoreDocs, final SpaceType spaceType, final float rawScoreOffset) {
        switch (spaceType) {
            case L2 -> {
                for (final ScoreDoc scoreDoc : scoreDocs) {
                    // score = 1 / (1 + distance)
                    final float distance = 1f / scoreDoc.score - 1f;
                    scoreDoc.score = SpaceType.L2.scoreTranslation(Math.max(distance + rawScoreOffset, 0f));
                }
            }
            case INNER_PRODUCT, COSINESIMIL -> {
                for (final ScoreDoc scoreDoc : scoreDocs) {
                    // Reverse MAXIMUM_INNER_PRODUCT score translation to recover the raw inner product value.
                    final float innerProductValue = scoreDoc.score >= 1 ? scoreDoc.score - 1 : 1 - 1 / scoreDoc.score;
                    scoreDoc.score = SpaceType.INNER_PRODUCT.scoreTranslation(-1 * (innerProductValue + rawScoreOffset));
                }
            }
            default -> throw new UnsupportedOperationException("Raw score offset is not supported for space type " + spaceType);
        }
    }

    /**
     * This method converts Lucene's max inner product score to Faiss cosine score to ensure user
     * to get the same results with the same query.
     *
     * @param scoreDocs Results from internal search before returning.
     */
    public static void convertToCosineScore(final ScoreDoc[] scoreDocs) {
        for (final ScoreDoc scoreDoc : scoreDocs) {
            scoreDoc.score = convertInnerProductScoreToCosineScore(scoreDoc.score);
        }
    }

    /**
     * Converts a single Lucene MAXIMUM_INNER_PRODUCT score to a Faiss cosine similarity score.
     *
     * <p>MAXIMUM_INNER_PRODUCT maps negative inner product values to (0, 1] and positive values
     * to (1, +inf). This method reverses that mapping to recover the raw inner product value,
     * then transforms it into the cosine similarity score range.</p>
     *
     * @param ipScore the MAXIMUM_INNER_PRODUCT-format score
     * @return the equivalent cosine similarity score
     */
    public static float convertInnerProductScoreToCosineScore(final float ipScore) {
        // Reverse MAXIMUM_INNER_PRODUCT score translation to recover the raw inner product value.
        final float innerProductValue = ipScore >= 1 ? ipScore - 1 : 1 - 1 / ipScore;
        // Transform to cosine similarity score range.
        return KNNEngine.FAISS.score(innerProductValue, SpaceType.COSINESIMIL);
    }
}
