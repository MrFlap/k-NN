/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clumping;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TaskExecutor;
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
import java.util.concurrent.Callable;

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
     * Expands per-leaf results in parallel using the provided {@link TaskExecutor}
     * for cross-segment parallelism. Within each leaf, marker expansion is further
     * parallelized via a parallel stream in {@link ClumpFileReader}.
     *
     * @param perLeafResults     The per-leaf results from ANN search (containing marker vectors)
     * @param leafReaderContexts The leaf reader contexts
     * @param knnWeight          The KNN weight (unused in v2, kept for API compatibility)
     * @param fieldName          The vector field name
     * @param queryVector        The float query vector (may be null for byte vectors)
     * @param byteQueryVector    The byte query vector (may be null for float vectors)
     * @param k                  The final k to return
     * @param taskExecutor       The Lucene task executor from the IndexSearcher for parallel execution
     * @return Expanded per-leaf results including hidden vectors
     */
    public static List<PerLeafResult> expandClumpedResults(
        List<PerLeafResult> perLeafResults,
        List<LeafReaderContext> leafReaderContexts,
        KNNWeight knnWeight,
        String fieldName,
        float[] queryVector,
        byte[] byteQueryVector,
        int k,
        TaskExecutor taskExecutor
    ) throws IOException {
        List<Callable<PerLeafResult>> tasks = new ArrayList<>(perLeafResults.size());
        for (int i = 0; i < perLeafResults.size(); i++) {
            final PerLeafResult leafResult = perLeafResults.get(i);
            final LeafReaderContext leafCtx = leafReaderContexts.get(i);
            tasks.add(() -> expandLeaf(leafResult, leafCtx, fieldName, queryVector, byteQueryVector, k));
        }
        return taskExecutor.invokeAll(tasks);
    }

    /**
     * Expands a single leaf's marker results by reading and scoring hidden vectors
     * from the .clump sidecar file.
     */
    private static PerLeafResult expandLeaf(
        PerLeafResult leafResult,
        LeafReaderContext leafCtx,
        String fieldName,
        float[] queryVector,
        byte[] byteQueryVector,
        int k
    ) throws IOException {
        if (leafResult.getResult().scoreDocs.length == 0) {
            return leafResult;
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
            return leafResult;
        }

        if (hasClumpFile == false) {
            log.warn("[CLUMPING DEBUG] No clump file found for segment={}, field={}", segmentName, fieldName);
            return leafResult;
        }

        // Get the similarity function for scoring hidden vectors
        KNNVectorSimilarityFunction similarityFunction = getSimilarityFunction(reader, fieldName);
        if (similarityFunction == null) {
            log.warn("Could not determine similarity function for field {}, skipping expansion", fieldName);
            return leafResult;
        }

        // Get marker doc IDs from the search results
        int[] markerDocIds = Arrays.stream(leafResult.getResult().scoreDocs)
            .mapToInt(sd -> sd.doc)
            .toArray();

        log.warn(
            "[CLUMPING DEBUG] segment={}, field={}, numMarkers={}, markerDocIds(first5)={}",
            segmentName, fieldName, markerDocIds.length,
            Arrays.toString(Arrays.copyOf(markerDocIds, Math.min(5, markerDocIds.length)))
        );

        // For SQ_1BIT clump files, obtain FloatVectorValues for the fp32 rescore step.
        FloatVectorValues floatVectorValues = null;
        boolean isSq = false;
        try {
            isSq = ClumpFileReader.isSqClumpFile(clumpDirectory, segmentName, fieldName);
        } catch (IOException e) {
            log.warn("Error checking SQ clump type for segment {}", segmentName, e);
        }
        if (isSq) {
            try {
                floatVectorValues = reader.getFloatVectorValues(fieldName);
            } catch (IOException e) {
                log.warn("Could not obtain FloatVectorValues for SQ rescore, segment {}", segmentName, e);
            }
        }

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
                similarityFunction,
                floatVectorValues,
                k
            );
        } catch (IOException e) {
            log.warn("Error reading clump file, skipping expansion for segment {}", segmentName, e);
            return leafResult;
        }

        log.warn("[CLUMPING DEBUG] segment={}, scoredHidden.size()={}", segmentName, scoredHidden.size());

        if (scoredHidden.isEmpty()) {
            return leafResult;
        }

        // Re-score the markers using the same similarity function as the hidden vectors.
        // The ANN search scores (ADC-based for SQ, HNSW-based for others) are on a different
        // scale than the FP16 L2 scores used for hidden vectors. Re-scoring markers ensures
        // all scores are comparable so reduceToTopK selects the true nearest neighbors.
        ScoreDoc[] rescored;
        try {
            rescored = ClumpFileReader.rescoreMarkers(
                clumpDirectory,
                segmentName,
                fieldName,
                leafResult.getResult().scoreDocs,
                queryVector,
                byteQueryVector,
                similarityFunction,
                floatVectorValues
            );
        } catch (IOException e) {
            log.warn("Error rescoring markers from clump file, using original scores", e);
            rescored = leafResult.getResult().scoreDocs;
        }

        // Assertion: every marker docID returned by ANN search must exist in the clump file's
        // marker table. If not, the ANN search is returning wrong docIDs (e.g. ordinals instead
        // of Lucene docIDs), which means the id_map translation is broken.
        // Always log at WARN level so this shows up without debug logging enabled.
        for (ScoreDoc sd : leafResult.getResult().scoreDocs) {
            boolean isMarker = ClumpFileReader.isMarkerDocId(clumpDirectory, segmentName, fieldName, sd.doc);
            if (!isMarker) {
                log.warn(
                    "[CLUMPING ASSERTION FAILED] ANN result docId={} is NOT a marker in the clump file "
                        + "for segment={}, field={}. This indicates ordinal/docID confusion in the search path.",
                    sd.doc, segmentName, fieldName
                );
            }
        }
        log.warn(
            "[CLUMPING DEBUG] segment={}, field={}, numMarkers={}, markerDocIds(first5)={}",
            segmentName, fieldName, markerDocIds.length,
            Arrays.toString(Arrays.copyOf(markerDocIds, Math.min(5, markerDocIds.length)))
        );

        // Remove any hidden docs that are already in the marker results
        Set<Integer> existingDocIds = new HashSet<>();
        for (ScoreDoc sd : leafResult.getResult().scoreDocs) {
            existingDocIds.add(sd.doc);
        }
        scoredHidden.removeIf(sd -> existingDocIds.contains(sd.doc));

        if (scoredHidden.isEmpty() && rescored == leafResult.getResult().scoreDocs) {
            return leafResult;
        }

        // Merge re-scored markers with scored hidden results
        ScoreDoc[] merged = new ScoreDoc[rescored.length + scoredHidden.size()];
        System.arraycopy(rescored, 0, merged, 0, rescored.length);
        for (int j = 0; j < scoredHidden.size(); j++) {
            merged[rescored.length + j] = scoredHidden.get(j);
        }

        TotalHits totalHits = new TotalHits(merged.length, TotalHits.Relation.EQUAL_TO);
        TopDocs mergedTopDocs = new TopDocs(totalHits, merged);

        log.debug(
            "Clumping expansion: segment={}, markers={}, hidden={}, merged={}",
            segmentName,
            rescored.length,
            scoredHidden.size(),
            merged.length
        );

        return new PerLeafResult(
            leafResult.getFilterBits(),
            leafResult.getFilterBitsCardinality(),
            mergedTopDocs,
            leafResult.getSearchMode()
        );
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
