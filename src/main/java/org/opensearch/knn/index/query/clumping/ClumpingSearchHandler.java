/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clumping;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.HitQueue;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.util.Bits;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.query.PerLeafResult;
import org.opensearch.knn.indices.ModelDao;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Handles clumping-based search by expanding marker vector results to include hidden vectors.
 * 
 * This handler is invoked after the initial k-NN search on marker vectors completes.
 * It retrieves hidden vectors associated with each marker, scores them against the query,
 * and returns the combined top-k results.
 */
@Log4j2
public class ClumpingSearchHandler {

    private final ClumpingContext clumpingContext;
    private final ModelDao modelDao;

    public ClumpingSearchHandler(ClumpingContext clumpingContext, ModelDao modelDao) {
        this.clumpingContext = clumpingContext;
        this.modelDao = modelDao;
    }

    /**
     * Checks if clumping expansion is needed for the given context.
     * 
     * @return true if clumping is enabled and should be applied
     */
    public boolean isClumpingEnabled() {
        return clumpingContext != null && clumpingContext.isEffectivelyEnabled();
    }

    /**
     * Expands marker vector results to include hidden vectors and returns top-k.
     * 
     * @param leafReaderContext the leaf reader context
     * @param perLeafResult the initial search result containing marker vectors
     * @param queryVector the query vector for scoring
     * @param fieldName the field name
     * @param k the final number of results to return
     * @return expanded PerLeafResult with combined marker and hidden vector results
     * @throws IOException if reading hidden vectors fails
     */
    public PerLeafResult expandClumpedResults(
        LeafReaderContext leafReaderContext,
        PerLeafResult perLeafResult,
        float[] queryVector,
        String fieldName,
        int k
    ) throws IOException {
        if (!isClumpingEnabled()) {
            return perLeafResult;
        }

        TopDocs markerResults = perLeafResult.getResult();
        if (markerResults.scoreDocs.length == 0) {
            return perLeafResult;
        }

        SegmentReader reader = Lucene.segmentReader(leafReaderContext.reader());
        String segmentName = reader.getSegmentName();

        // Get field info for space type and vector data type
        FieldInfo fieldInfo = FieldInfoExtractor.getFieldInfo(reader, fieldName);
        if (fieldInfo == null) {
            log.warn("Field info not found for field {} in segment {}", fieldName, segmentName);
            return perLeafResult;
        }

        SpaceType spaceType = FieldInfoExtractor.getSpaceType(modelDao, fieldInfo);
        VectorDataType vectorDataType = FieldInfoExtractor.extractVectorDataType(fieldInfo);

        // Collect marker doc IDs from results
        Set<Integer> markerDocIds = new HashSet<>();
        for (ScoreDoc scoreDoc : markerResults.scoreDocs) {
            markerDocIds.add(scoreDoc.doc);
        }

        // Create store to read hidden vectors
        try (ClumpingVectorStore store = new ClumpingVectorStore(
            reader.directory(),
            segmentName,
            fieldName
        )) {
            // Read hidden vector mapping
            HiddenVectorMapping mapping = store.readHiddenVectorMapping();
            if (mapping.isEmpty()) {
                log.debug("No hidden vector mapping found for segment {}", segmentName);
                return perLeafResult;
            }

            // Read hidden vectors for these markers
            Map<Integer, float[]> hiddenVectors = store.readHiddenVectorsForMarkers(markerDocIds);
            if (hiddenVectors.isEmpty()) {
                log.debug("No hidden vectors found for {} markers in segment {}", markerDocIds.size(), segmentName);
                return perLeafResult;
            }

            // Get live docs for filtering
            Bits liveDocs = leafReaderContext.reader().getLiveDocs();

            // Build combined results using a min-heap for efficiency
            HitQueue queue = new HitQueue(k, true);
            ScoreDoc topDoc = queue.top();

            // Add marker results
            for (ScoreDoc markerScoreDoc : markerResults.scoreDocs) {
                if (markerScoreDoc.score > topDoc.score) {
                    topDoc.score = markerScoreDoc.score;
                    topDoc.doc = markerScoreDoc.doc;
                    topDoc = queue.updateTop();
                }
            }

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
                        if (score > topDoc.score) {
                            topDoc.score = score;
                            topDoc.doc = hiddenDocId;
                            topDoc = queue.updateTop();
                        }
                    }
                }
            }

            // Remove negative scores (from initialization)
            while (queue.size() > 0 && queue.top().score < 0) {
                queue.pop();
            }

            // Extract results in descending score order
            ScoreDoc[] topK = new ScoreDoc[queue.size()];
            for (int i = topK.length - 1; i >= 0; i--) {
                topK[i] = queue.pop();
            }

            log.debug(
                "Clumping expansion: {} markers -> {} hidden candidates -> {} results in segment {}",
                markerResults.scoreDocs.length,
                hiddenVectors.size(),
                topK.length,
                segmentName
            );

            TopDocs expandedResults = new TopDocs(new TotalHits(topK.length, TotalHits.Relation.EQUAL_TO), topK);
            
            return new PerLeafResult(
                perLeafResult.getFilterBits(),
                perLeafResult.getFilterBitsCardinality(),
                expandedResults,
                PerLeafResult.SearchMode.EXACT_SEARCH // Mark as exact since we rescored
            );
        }
    }

    /**
     * Expands clumped results for multiple leaf contexts.
     * 
     * @param leafReaderContexts the leaf reader contexts
     * @param perLeafResults the initial search results
     * @param queryVector the query vector
     * @param fieldName the field name
     * @param k the final k
     * @return list of expanded results
     * @throws IOException if expansion fails
     */
    public List<PerLeafResult> expandClumpedResults(
        List<LeafReaderContext> leafReaderContexts,
        List<PerLeafResult> perLeafResults,
        float[] queryVector,
        String fieldName,
        int k
    ) throws IOException {
        if (!isClumpingEnabled()) {
            return perLeafResults;
        }

        List<PerLeafResult> expandedResults = new ArrayList<>(perLeafResults.size());
        
        for (int i = 0; i < perLeafResults.size(); i++) {
            PerLeafResult expanded = expandClumpedResults(
                leafReaderContexts.get(i),
                perLeafResults.get(i),
                queryVector,
                fieldName,
                k
            );
            expandedResults.add(expanded);
        }

        return expandedResults;
    }

    /**
     * Computes the score between query and candidate vectors.
     */
    private float computeScore(float[] queryVector, float[] candidateVector, SpaceType spaceType, VectorDataType vectorDataType) {
        float rawScore;
        
        switch (spaceType) {
            case L2:
                rawScore = computeL2SquaredDistance(queryVector, candidateVector);
                // Convert L2 squared distance to score: 1 / (1 + distance)
                return 1.0f / (1.0f + rawScore);
                
            case COSINESIMIL:
                rawScore = computeCosineSimilarity(queryVector, candidateVector);
                // Cosine similarity is already in [-1, 1], normalize to [0, 2] then shift
                return (1.0f + rawScore) / 2.0f;
                
            case INNER_PRODUCT:
                rawScore = computeInnerProduct(queryVector, candidateVector);
                // Inner product scoring similar to Lucene
                if (rawScore >= 0) {
                    return 1.0f + rawScore;
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

    private float computeL2SquaredDistance(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
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
