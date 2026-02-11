/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clumping;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.store.Directory;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80CompoundDirectory;
import org.opensearch.knn.index.codec.nativeindex.clumping.ClumpFileReader;
import org.opensearch.knn.index.query.KNNWeight;
import org.opensearch.knn.index.query.PerLeafResult;
import org.opensearch.knn.indices.ModelDao;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Expands clumped search results by reading hidden vectors directly from the .clump
 * sidecar file and scoring them inline against the query vector.
 * <p>
 * The .clump file (v2 format) stores all vector data alongside the marker-to-hidden
 * mapping, so expansion reads vectors sequentially from the clump file without any
 * random access into Lucene's vector storage.
 */
@Log4j2
public final class ClumpingExpander {

    private ClumpingExpander() {}

    /**
     * Expands per-leaf results by reading hidden vectors from the .clump file,
     * scoring them directly against the query vector, and merging into results.
     * <p>
     * No ExactSearcher or Lucene vector storage access is needed — all vector data
     * is read sequentially from the clump file.
     *
     * @param perLeafResults     The per-leaf results from ANN search (containing marker vectors)
     * @param leafReaderContexts The leaf reader contexts
     * @param knnWeight          The KNN weight (unused in v2, kept for API compatibility)
     * @param fieldName          The vector field name
     * @param queryVector        The float query vector (may be null for byte vectors)
     * @param byteQueryVector    The byte query vector (may be null for float vectors)
     * @param k                  The final k to return
     * @return Expanded per-leaf results including hidden vectors
     */
    public static List<PerLeafResult> expandClumpedResults(
        List<PerLeafResult> perLeafResults,
        List<LeafReaderContext> leafReaderContexts,
        KNNWeight knnWeight,
        String fieldName,
        float[] queryVector,
        byte[] byteQueryVector,
        int k
    ) throws IOException {
        List<PerLeafResult> expandedResults = new ArrayList<>(perLeafResults.size());

        for (int i = 0; i < perLeafResults.size(); i++) {
            PerLeafResult leafResult = perLeafResults.get(i);
            LeafReaderContext leafCtx = leafReaderContexts.get(i);

            if (leafResult.getResult().scoreDocs.length == 0) {
                expandedResults.add(leafResult);
                continue;
            }

            SegmentReader reader = Lucene.segmentReader(leafCtx.reader());
            String segmentName = reader.getSegmentName();

            // Get the outer directory where .clumpc files live.
            // When segments are compounded, reader.directory() is a KNN80CompoundDirectory
            // whose listAll() only shows files inside the .cfs. The .clumpc sidecar lives
            // in the outer directory, so we need to unwrap it.
            Directory clumpDirectory = reader.directory();
            if (clumpDirectory instanceof KNN80CompoundDirectory) {
                clumpDirectory = ((KNN80CompoundDirectory) clumpDirectory).getDir();
            }

            // Check if a clump file exists for this segment and field
            boolean hasClumpFile;
            try {
                hasClumpFile = ClumpFileReader.clumpFileExists(clumpDirectory, segmentName, fieldName);
            } catch (IOException e) {
                log.warn("Error checking for clump file, skipping expansion for segment {}", segmentName, e);
                expandedResults.add(leafResult);
                continue;
            }

            if (hasClumpFile == false) {
                expandedResults.add(leafResult);
                continue;
            }

            // Get the similarity function for scoring hidden vectors
            KNNVectorSimilarityFunction similarityFunction = getSimilarityFunction(reader, fieldName);
            if (similarityFunction == null) {
                log.warn("Could not determine similarity function for field {}, skipping expansion", fieldName);
                expandedResults.add(leafResult);
                continue;
            }

            // Get marker doc IDs from the search results
            int[] markerDocIds = Arrays.stream(leafResult.getResult().scoreDocs)
                .mapToInt(sd -> sd.doc)
                .toArray();

            // Read hidden vectors from .clump file and score them inline
            List<ScoreDoc> scoredHidden;
            try {
                scoredHidden = ClumpFileReader.getHiddenVectorsScored(
                    clumpDirectory,
                    segmentName,
                    fieldName,
                    markerDocIds,
                    queryVector,
                    byteQueryVector,
                    similarityFunction
                );
            } catch (IOException e) {
                log.warn("Error reading clump file, skipping expansion for segment {}", segmentName, e);
                expandedResults.add(leafResult);
                continue;
            }

            if (scoredHidden.isEmpty()) {
                expandedResults.add(leafResult);
                continue;
            }

            // Remove any hidden docs that are already in the marker results
            Set<Integer> existingDocIds = new HashSet<>();
            for (ScoreDoc sd : leafResult.getResult().scoreDocs) {
                existingDocIds.add(sd.doc);
            }
            scoredHidden.removeIf(sd -> existingDocIds.contains(sd.doc));

            if (scoredHidden.isEmpty()) {
                expandedResults.add(leafResult);
                continue;
            }

            // Merge marker results with scored hidden results
            ScoreDoc[] markerDocs = leafResult.getResult().scoreDocs;
            ScoreDoc[] merged = new ScoreDoc[markerDocs.length + scoredHidden.size()];
            System.arraycopy(markerDocs, 0, merged, 0, markerDocs.length);
            for (int j = 0; j < scoredHidden.size(); j++) {
                merged[markerDocs.length + j] = scoredHidden.get(j);
            }

            TotalHits totalHits = new TotalHits(merged.length, TotalHits.Relation.EQUAL_TO);
            TopDocs mergedTopDocs = new TopDocs(totalHits, merged);

            expandedResults.add(new PerLeafResult(
                leafResult.getFilterBits(),
                leafResult.getFilterBitsCardinality(),
                mergedTopDocs,
                leafResult.getSearchMode()
            ));

            log.debug(
                "Clumping expansion: segment={}, markers={}, hidden={}, merged={}",
                segmentName,
                markerDocIds.length,
                scoredHidden.size(),
                merged.length
            );
        }

        return expandedResults;
    }

    /**
     * Extracts the KNNVectorSimilarityFunction for the given field from the segment reader.
     */
    private static KNNVectorSimilarityFunction getSimilarityFunction(SegmentReader reader, String fieldName) {
        FieldInfo fieldInfo = FieldInfoExtractor.getFieldInfo(reader, fieldName);
        if (fieldInfo == null) {
            return null;
        }
        try {
            ModelDao modelDao = ModelDao.OpenSearchKNNModelDao.getInstance();
            SpaceType spaceType = FieldInfoExtractor.getSpaceType(modelDao, fieldInfo);
            return spaceType.getKnnVectorSimilarityFunction();
        } catch (Exception e) {
            log.warn("Failed to get similarity function for field {}: {}", fieldName, e.getMessage());
            return null;
        }
    }
}
