/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clumping;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.util.Bits;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Expands marker vector search results to include their associated hidden vectors.
 * 
 * This is the core component of clumping-based search that:
 * 1. Takes the initial k-NN search results (marker vectors only)
 * 2. Retrieves hidden vectors associated with each marker
 * 3. Scores all vectors (markers + hidden) against the query
 * 4. Returns the top-k results from the combined set
 */
@Log4j2
public class ClumpingExpander {

    private final ClumpingContext clumpingContext;

    public ClumpingExpander(ClumpingContext clumpingContext) {
        this.clumpingContext = clumpingContext;
    }

    /**
     * Expands marker vector results to include hidden vectors and returns top-k.
     * 
     * @param leafReaderContext the leaf reader context
     * @param markerResults the initial search results containing only marker vectors
     * @param queryVector the query vector for scoring
     * @param spaceType the space type for distance calculation
     * @param vectorDataType the vector data type
     * @param fieldName the field name
     * @param k the final number of results to return
     * @param liveDocs live docs bits (null if all docs are live)
     * @return expanded and re-ranked top-k results
     * @throws IOException if reading hidden vectors fails
     */
    public TopDocs expandAndRescore(
        LeafReaderContext leafReaderContext,
        TopDocs markerResults,
        float[] queryVector,
        SpaceType spaceType,
        VectorDataType vectorDataType,
        String fieldName,
        int k,
        Bits liveDocs
    ) throws IOException {
        if (!clumpingContext.isEffectivelyEnabled()) {
            return markerResults;
        }

        if (markerResults.scoreDocs.length == 0) {
            return markerResults;
        }

        SegmentReader reader = Lucene.segmentReader(leafReaderContext.reader());
        String segmentName = reader.getSegmentName();

        // Create store to read hidden vectors
        try (ClumpingVectorStore store = new ClumpingVectorStore(
            reader.directory(),
            segmentName,
            fieldName
        )) {
            // Collect marker doc IDs from results
            Set<Integer> markerDocIds = new HashSet<>();
            for (ScoreDoc scoreDoc : markerResults.scoreDocs) {
                markerDocIds.add(scoreDoc.doc);
            }

            // Read hidden vectors for these markers
            Map<Integer, float[]> hiddenVectors = store.readHiddenVectorsForMarkers(markerDocIds);

            if (hiddenVectors.isEmpty()) {
                log.debug("No hidden vectors found for {} markers in segment {}", markerDocIds.size(), segmentName);
                return markerResults;
            }

            // Read the mapping to know which hidden docs belong to which markers
            HiddenVectorMapping mapping = store.readHiddenVectorMapping();

            // Build combined results: markers + their hidden vectors
            List<ScoreDoc> combinedResults = new ArrayList<>();

            // Add marker results (already scored)
            combinedResults.addAll(Arrays.asList(markerResults.scoreDocs));

            // Score and add hidden vectors
            for (ScoreDoc markerScoreDoc : markerResults.scoreDocs) {
                List<Integer> hiddenDocIds = mapping.getHiddenDocsForMarker(markerScoreDoc.doc);
                
                for (int hiddenDocId : hiddenDocIds) {
                    // Check if doc is live
                    if (liveDocs != null && !liveDocs.get(hiddenDocId)) {
                        continue;
                    }

                    float[] hiddenVector = hiddenVectors.get(hiddenDocId);
                    if (hiddenVector != null) {
                        float score = computeScore(queryVector, hiddenVector, spaceType, vectorDataType);
                        combinedResults.add(new ScoreDoc(hiddenDocId, score));
                    }
                }
            }

            // Sort by score descending and take top-k
            combinedResults.sort(Comparator.comparing((ScoreDoc sd) -> sd.score).reversed());
            
            int resultCount = Math.min(k, combinedResults.size());
            ScoreDoc[] topK = combinedResults.subList(0, resultCount).toArray(new ScoreDoc[0]);

            log.debug(
                "Clumping expansion: {} markers -> {} total candidates -> {} results in segment {}",
                markerResults.scoreDocs.length,
                combinedResults.size(),
                resultCount,
                segmentName
            );

            return new TopDocs(new TotalHits(resultCount, TotalHits.Relation.EQUAL_TO), topK);
        }
    }

    /**
     * Computes the score between query and candidate vectors.
     * Uses the space type's scoring function.
     */
    private float computeScore(float[] queryVector, float[] candidateVector, SpaceType spaceType, VectorDataType vectorDataType) {
        // Compute raw distance/similarity based on space type
        float rawScore;
        
        switch (spaceType) {
            case L2:
                rawScore = computeL2Distance(queryVector, candidateVector);
                // Convert L2 distance to score: 1 / (1 + distance)
                return 1.0f / (1.0f + rawScore);
                
            case COSINESIMIL:
                rawScore = computeCosineSimilarity(queryVector, candidateVector);
                // Cosine similarity is already in [-1, 1], normalize to [0, 1]
                return (1.0f + rawScore) / 2.0f;
                
            case INNER_PRODUCT:
                rawScore = computeInnerProduct(queryVector, candidateVector);
                // Inner product can be negative, use sigmoid-like transformation
                if (rawScore >= 0) {
                    return 1.0f / (1.0f + rawScore) + 1.0f;
                } else {
                    return 1.0f / (1.0f - rawScore);
                }
                
            case L1:
                rawScore = computeL1Distance(queryVector, candidateVector);
                return 1.0f / (1.0f + rawScore);
                
            case LINF:
                rawScore = computeLinfDistance(queryVector, candidateVector);
                return 1.0f / (1.0f + rawScore);
                
            case HAMMING:
                rawScore = computeHammingDistance(queryVector, candidateVector);
                return 1.0f / (1.0f + rawScore);
                
            default:
                throw new IllegalArgumentException("Unsupported space type: " + spaceType);
        }
    }

    private float computeL2Distance(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return (float) Math.sqrt(sum);
    }

    private float computeCosineSimilarity(float[] a, float[] b) {
        float dotProduct = 0;
        float normA = 0;
        float normB = 0;
        
        for (int i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        
        if (normA == 0 || normB == 0) {
            return 0;
        }
        
        return dotProduct / (float) (Math.sqrt(normA) * Math.sqrt(normB));
    }

    private float computeInnerProduct(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    private float computeL1Distance(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.abs(a[i] - b[i]);
        }
        return sum;
    }

    private float computeLinfDistance(float[] a, float[] b) {
        float max = 0;
        for (int i = 0; i < a.length; i++) {
            max = Math.max(max, Math.abs(a[i] - b[i]));
        }
        return max;
    }

    private float computeHammingDistance(float[] a, float[] b) {
        int distance = 0;
        for (int i = 0; i < a.length; i++) {
            if ((int) a[i] != (int) b[i]) {
                distance++;
            }
        }
        return distance;
    }
}
