/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clumping;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.store.Directory;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80CompoundDirectory;
import org.opensearch.knn.index.codec.nativeindex.clumping.ClumpFileReader;
import org.opensearch.knn.index.query.PerLeafResult;
import org.opensearch.knn.index.query.KNNWeight;
import org.opensearch.knn.index.query.exactsearch.ExactSearcher;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Expands clumped search results by looking up hidden vectors associated with marker vectors.
 * <p>
 * After an ANN search returns marker vectors, this class reads the .clump sidecar file to find
 * hidden vectors associated with each marker, performs exact scoring on those hidden vectors,
 * and merges them into the result set.
 */
@Log4j2
public final class ClumpingExpander {

    private ClumpingExpander() {}

    /**
     * Expands per-leaf results by adding hidden vectors associated with marker vectors.
     * For each leaf, reads the .clump file, finds hidden doc IDs for the returned markers,
     * scores them via exact search, and merges into the result set.
     *
     * @param perLeafResults     The per-leaf results from ANN search (containing marker vectors)
     * @param leafReaderContexts The leaf reader contexts
     * @param knnWeight          The KNN weight for exact scoring
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

            // Get marker doc IDs from the search results
            int[] markerDocIds = Arrays.stream(leafResult.getResult().scoreDocs)
                .mapToInt(sd -> sd.doc)
                .toArray();

            // Look up hidden doc IDs
            List<Integer> hiddenDocIds;
            try {
                hiddenDocIds = ClumpFileReader.getHiddenDocIds(
                    clumpDirectory,
                    segmentName,
                    fieldName,
                    markerDocIds
                );
            } catch (IOException e) {
                log.warn("Error reading clump file, skipping expansion for segment {}", segmentName, e);
                expandedResults.add(leafResult);
                continue;
            }

            if (hiddenDocIds.isEmpty()) {
                expandedResults.add(leafResult);
                continue;
            }

            // Score hidden vectors via exact search
            Set<Integer> hiddenDocIdSet = new HashSet<>(hiddenDocIds);
            // Remove any that are already in the results
            for (ScoreDoc sd : leafResult.getResult().scoreDocs) {
                hiddenDocIdSet.remove(sd.doc);
            }

            if (hiddenDocIdSet.isEmpty()) {
                expandedResults.add(leafResult);
                continue;
            }

            // Build a DocIdSetIterator for the hidden docs
            DocIdSetIterator hiddenDocsIter = buildDocIdSetIterator(hiddenDocIdSet);

            ExactSearcher.ExactSearcherContext exactCtx = ExactSearcher.ExactSearcherContext.builder()
                .matchedDocsIterator(hiddenDocsIter)
                .numberOfMatchedDocs(hiddenDocIdSet.size())
                .useQuantizedVectorsForSearch(false)
                .k(hiddenDocIdSet.size()) // score all hidden vectors
                .field(fieldName)
                .floatQueryVector(queryVector)
                .byteQueryVector(byteQueryVector)
                .build();

            TopDocs hiddenResults = knnWeight.exactSearch(leafCtx, exactCtx);

            // Merge marker results with hidden results
            ScoreDoc[] merged = mergeResults(leafResult.getResult().scoreDocs, hiddenResults.scoreDocs);
            TotalHits totalHits = new TotalHits(merged.length, TotalHits.Relation.EQUAL_TO);
            TopDocs mergedTopDocs = new TopDocs(totalHits, merged);

            expandedResults.add(new PerLeafResult(
                leafResult.getFilterBits(),
                leafResult.getFilterBitsCardinality(),
                mergedTopDocs,
                leafResult.getSearchMode()
            ));

            log.info(
                "Clumping expansion: segment={}, markers={}, hidden={}, merged={}",
                segmentName,
                markerDocIds.length,
                hiddenDocIds.size(),
                merged.length
            );
        }

        return expandedResults;
    }

    private static ScoreDoc[] mergeResults(ScoreDoc[] markers, ScoreDoc[] hidden) {
        ScoreDoc[] merged = new ScoreDoc[markers.length + hidden.length];
        System.arraycopy(markers, 0, merged, 0, markers.length);
        System.arraycopy(hidden, 0, merged, markers.length, hidden.length);
        return merged;
    }

    private static DocIdSetIterator buildDocIdSetIterator(Set<Integer> docIds) {
        int[] sorted = docIds.stream().mapToInt(Integer::intValue).sorted().toArray();
        return new DocIdSetIterator() {
            int idx = -1;
            int doc = -1;

            @Override
            public int docID() {
                return doc;
            }

            @Override
            public int nextDoc() {
                idx++;
                if (idx >= sorted.length) {
                    doc = NO_MORE_DOCS;
                } else {
                    doc = sorted[idx];
                }
                return doc;
            }

            @Override
            public int advance(int target) {
                while (nextDoc() < target) {
                    if (doc == NO_MORE_DOCS) return NO_MORE_DOCS;
                }
                return doc;
            }

            @Override
            public long cost() {
                return sorted.length;
            }
        };
    }
}
