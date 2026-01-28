/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import com.google.common.annotations.VisibleForTesting;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.HitQueue;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopDocsCollector;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.Bits;
import org.opensearch.common.Nullable;
import org.opensearch.common.StopWatch;
import org.opensearch.common.lucene.Lucene;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.codec.clumping.ClumpingFileReader;
import org.opensearch.knn.index.codec.clumping.HiddenVectorEntry;
import org.opensearch.knn.index.query.clumping.ClumpingContext;
import org.opensearch.knn.index.query.common.QueryUtils;
import org.opensearch.knn.index.query.iterators.GroupedNestedDocIdSetIterator;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Callable;

/**
 * A query that wraps an inner k-NN query and performs clumping-based search.
 * 
 * Clumping is an optimization technique that reduces the main k-NN index size by only
 * indexing a subset of vectors (marker vectors) while storing the remaining vectors
 * (hidden vectors) in a separate file on disk. During search, this query:
 * 
 * <ol>
 *   <li>Executes the inner query to search marker vectors</li>
 *   <li>Expands results to include hidden vectors associated with each marker</li>
 *   <li>Rescores all candidates (markers + hidden) using exact distance calculation</li>
 *   <li>Returns the top-k results after rescoring</li>
 * </ol>
 * 
 * <h2>Interaction with Oversampling (Rescore)</h2>
 * 
 * Requirement 9.3: When both oversampling and clumping are enabled, oversampling is applied
 * to the marker search phase before clumping expansion. The order of operations is:
 * 
 * <ol>
 *   <li>Oversample marker search: The inner query retrieves k * oversample_factor marker results</li>
 *   <li>Clumping expansion: Hidden vectors associated with marker results are retrieved</li>
 *   <li>Rescore: All candidates (markers + hidden) are rescored using exact distance</li>
 * </ol>
 * 
 * When only clumping is enabled (no oversampling), the clumping expansion factor is applied
 * to the marker search to ensure sufficient candidates after expansion.
 * 
 * This class follows the pattern established by {@link RescoreKNNVectorQuery}.
 * 
 * @see ClumpingContext for clumping configuration
 * @see ClumpingFileReader for reading hidden vectors
 */
@Log4j2
public class ClumpingKNNVectorQuery extends Query {

    private final Query innerQuery;
    private final String field;
    private final int k;
    private final float[] queryVector;
    private final int shardId;
    private final ClumpingContext clumpingContext;
    private final SpaceType spaceType;
    private final Query filterQuery;
    private final BitSetProducer parentFilter;

    /**
     * Constructs a new ClumpingKNNVectorQuery.
     *
     * @param innerQuery      The inner query to execute for marker search
     * @param field           The field name containing the vector data
     * @param k               The number of nearest neighbors to return
     * @param queryVector     The query vector to compare against document vectors
     * @param shardId         The shard ID (for logging purposes)
     * @param clumpingContext The clumping configuration
     * @param spaceType       The space type for distance calculation
     */
    public ClumpingKNNVectorQuery(
        Query innerQuery,
        String field,
        int k,
        float[] queryVector,
        int shardId,
        ClumpingContext clumpingContext,
        SpaceType spaceType
    ) {
        this(innerQuery, field, k, queryVector, shardId, clumpingContext, spaceType, null, null);
    }

    /**
     * Constructs a new ClumpingKNNVectorQuery with filter support.
     *
     * @param innerQuery      The inner query to execute for marker search (filter is already applied to this)
     * @param field           The field name containing the vector data
     * @param k               The number of nearest neighbors to return
     * @param queryVector     The query vector to compare against document vectors
     * @param shardId         The shard ID (for logging purposes)
     * @param clumpingContext The clumping configuration
     * @param spaceType       The space type for distance calculation
     * @param filterQuery     The filter query to apply during hidden vector expansion (may be null)
     */
    public ClumpingKNNVectorQuery(
        Query innerQuery,
        String field,
        int k,
        float[] queryVector,
        int shardId,
        ClumpingContext clumpingContext,
        SpaceType spaceType,
        Query filterQuery
    ) {
        this(innerQuery, field, k, queryVector, shardId, clumpingContext, spaceType, filterQuery, null);
    }

    /**
     * Constructs a new ClumpingKNNVectorQuery with filter and nested field support.
     *
     * @param innerQuery      The inner query to execute for marker search (filter is already applied to this)
     * @param field           The field name containing the vector data
     * @param k               The number of nearest neighbors to return
     * @param queryVector     The query vector to compare against document vectors
     * @param shardId         The shard ID (for logging purposes)
     * @param clumpingContext The clumping configuration
     * @param spaceType       The space type for distance calculation
     * @param filterQuery     The filter query to apply during hidden vector expansion (may be null)
     * @param parentFilter    The parent filter for nested documents (may be null for non-nested fields)
     */
    public ClumpingKNNVectorQuery(
        Query innerQuery,
        String field,
        int k,
        float[] queryVector,
        int shardId,
        ClumpingContext clumpingContext,
        SpaceType spaceType,
        Query filterQuery,
        BitSetProducer parentFilter
    ) {
        this.innerQuery = innerQuery;
        this.field = field;
        this.k = k;
        this.queryVector = queryVector;
        this.shardId = shardId;
        this.clumpingContext = clumpingContext;
        this.spaceType = spaceType;
        this.filterQuery = filterQuery;
        this.parentFilter = parentFilter;
    }

    @VisibleForTesting
    Query getInnerQuery() {
        return innerQuery;
    }

    @VisibleForTesting
    String getField() {
        return field;
    }

    @VisibleForTesting
    int getK() {
        return k;
    }

    @VisibleForTesting
    float[] getQueryVector() {
        return queryVector;
    }

    @VisibleForTesting
    ClumpingContext getClumpingContext() {
        return clumpingContext;
    }

    @VisibleForTesting
    Query getFilterQuery() {
        return filterQuery;
    }

    @VisibleForTesting
    BitSetProducer getParentFilter() {
        return parentFilter;
    }

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {
        // 1. Rewrite and create weight for inner query (marker search)
        final Query rewrittenInnerQuery = searcher.rewrite(innerQuery);
        final Weight weight = searcher.createWeight(rewrittenInnerQuery, scoreMode, boost);

        // 2. Create filter weight if filter query is present (for hidden vector expansion)
        // Requirement 9.1: Apply filter to both marker search and hidden vector expansion
        final Weight filterWeight;
        if (filterQuery != null) {
            final Query rewrittenFilterQuery = searcher.rewrite(filterQuery);
            filterWeight = searcher.createWeight(rewrittenFilterQuery, ScoreMode.COMPLETE_NO_SCORES, 1.0f);
        } else {
            filterWeight = null;
        }

        // 3. Execute clumping search (marker search + expansion + rescore) per leaf
        final StopWatch stopWatch = startStopWatch();
        final TopDocs[] perLeafResults = doClumpingSearch(searcher, weight, filterWeight);
        stopStopWatchAndLog(stopWatch, "clumping_search");

        // 4. Merge results from all leaves and return top-k
        final TopDocs topK = TopDocs.merge(k, perLeafResults);

        if (topK.scoreDocs.length == 0) {
            return new MatchNoDocsQuery().createWeight(searcher, scoreMode, boost);
        }

        return QueryUtils.getInstance()
            .createDocAndScoreQuery(searcher.getIndexReader(), topK)
            .createWeight(searcher, scoreMode, boost);
    }

    /**
     * Executes clumping search across all leaves in parallel.
     * 
     * For each leaf:
     * 1. Get marker results from the inner query
     * 2. Expand to include hidden vectors
     * 3. Rescore all candidates
     *
     * @param searcher     The index searcher
     * @param weight       The weight from the inner query
     * @param filterWeight The weight from the filter query (may be null)
     * @return Array of TopDocs, one per leaf
     * @throws IOException if an I/O error occurs
     */
    private TopDocs[] doClumpingSearch(IndexSearcher searcher, Weight weight, Weight filterWeight) throws IOException {
        List<LeafReaderContext> leaves = searcher.getIndexReader().leaves();
        List<Callable<TopDocs>> tasks = new ArrayList<>(leaves.size());

        for (LeafReaderContext leaf : leaves) {
            tasks.add(() -> searchLeaf(weight, filterWeight, leaf));
        }

        return searcher.getTaskExecutor().invokeAll(tasks).toArray(TopDocs[]::new);
    }

    /**
     * Processes a single leaf: marker search, expansion, and rescore.
     *
     * @param weight       The weight from the inner query
     * @param filterWeight The weight from the filter query (may be null)
     * @param leaf         The leaf reader context
     * @return TopDocs containing rescored results for this leaf
     * @throws IOException if an I/O error occurs
     */
    private TopDocs searchLeaf(Weight weight, Weight filterWeight, LeafReaderContext leaf) throws IOException {
        // 1. Get marker results for this leaf
        final StopWatch markerSearchStopWatch = startStopWatch();
        Scorer scorer = weight.scorer(leaf);
        if (scorer == null) {
            return TopDocsCollector.EMPTY_TOPDOCS;
        }

        // 2. Collect marker doc IDs and their scores
        List<Integer> markerDocIds = collectMarkerDocIds(scorer);
        stopStopWatchAndLog(markerSearchStopWatch, "marker_search");
        
        if (markerDocIds.isEmpty()) {
            return TopDocsCollector.EMPTY_TOPDOCS;
        }

        // 3. Create filter bits for hidden vector expansion
        // Requirement 9.1: Apply filter during hidden vector expansion
        Bits filterBits = createFilterBits(leaf, filterWeight);

        // 4. Expand to include hidden vectors (applying filter)
        final StopWatch expansionStopWatch = startStopWatch();
        Map<Integer, float[]> candidates = expandWithHiddenVectors(leaf, markerDocIds, filterBits);
        stopStopWatchAndLog(expansionStopWatch, "hidden_vector_expansion");

        // 5. Requirement 9.2: Handle nested fields - expand to include sibling nested documents
        // When a parent filter is present, we need to retrieve all sibling nested documents
        // for the candidate documents to maintain parent-child relationships
        if (parentFilter != null) {
            candidates = expandWithNestedSiblings(leaf, candidates, filterBits);
        }

        // 6. Rescore all candidates
        return rescoreCandidates(candidates, leaf);
    }

    /**
     * Creates filter bits from the filter weight for a given leaf.
     * 
     * The returned Bits can be used to check if a document passes the filter.
     * If filterWeight is null, returns null (meaning no filtering).
     *
     * @param leaf         The leaf reader context
     * @param filterWeight The filter weight (may be null)
     * @return Bits for filter checking, or null if no filter
     * @throws IOException if an I/O error occurs
     */
    private Bits createFilterBits(LeafReaderContext leaf, Weight filterWeight) throws IOException {
        if (filterWeight == null) {
            return null;
        }
        return QueryUtils.getInstance().createBits(leaf, filterWeight);
    }

    /**
     * Collects marker document IDs from the scorer.
     *
     * @param scorer The scorer from the inner query
     * @return List of marker document IDs
     * @throws IOException if an I/O error occurs
     */
    private List<Integer> collectMarkerDocIds(Scorer scorer) throws IOException {
        List<Integer> markerDocIds = new ArrayList<>();
        DocIdSetIterator iterator = scorer.iterator();

        int docId;
        while ((docId = iterator.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
            markerDocIds.add(docId);
        }

        return markerDocIds;
    }

    /**
     * Expands marker results to include hidden vectors.
     * 
     * This method performs two key operations:
     * 1. Loads marker vectors from the index (so they can be rescored)
     * 2. Reads the clumping file to retrieve hidden vectors associated with each marker
     * 
     * The returned map contains both marker vectors and their associated hidden vectors,
     * all of which will be rescored to determine the final top-k results.
     * 
     * If the clumping file is missing or corrupted, the method falls back to marker-only
     * search by loading just the marker vectors from the index.
     * 
     * Requirement 9.1: When a filter is provided, hidden vectors that don't pass the filter
     * are excluded from the candidate set.
     *
     * @param leaf         The leaf reader context
     * @param markerDocIds List of marker document IDs from the first-pass search
     * @param filterBits   Bits for filter checking (null if no filter)
     * @return Map of document ID to vector for all candidates (markers + hidden)
     * @throws IOException if an I/O error occurs
     */
    @VisibleForTesting
    Map<Integer, float[]> expandWithHiddenVectors(LeafReaderContext leaf, List<Integer> markerDocIds, Bits filterBits) throws IOException {
        Map<Integer, float[]> candidates = new HashMap<>();

        // Get the SegmentReader to access directory and segment name
        final SegmentReader segmentReader = Lucene.segmentReader(leaf.reader());
        final Directory directory = segmentReader.directory();
        final String segmentName = segmentReader.getSegmentName();

        // First, load marker vectors from the index (they need to be rescored too)
        // Requirement 7.4: Candidate set should contain both marker vectors and their associated hidden vectors
        // Note: Markers already passed the filter in the inner query, so no need to filter them again
        loadMarkerVectors(leaf, markerDocIds, candidates);

        // Requirement 12.1: If the clumping file is corrupted or missing, fall back to marker-only search
        // Wrap all clumping file operations in try-catch to handle any errors gracefully
        try {
            // Check if clumping file exists for this segment
            if (!ClumpingFileReader.exists(directory, segmentName, field)) {
                log.debug("No clumping file found for segment={}, field={}. Using marker-only search.", segmentName, field);
                // Fall back to marker-only search - we already loaded marker vectors above
                return candidates;
            }

            // Load hidden vectors from the clumping file
            // Requirement 7.1: Read clumping file to retrieve hidden vectors associated with each marker result
            try (ClumpingFileReader clumpingReader = ClumpingFileReader.open(directory, segmentName, field)) {
                // Requirement 12.3: Handle per-marker expansion failures
                // Process each marker individually to ensure partial failures don't affect successful expansions
                int hiddenVectorCount = 0;
                int filteredOutCount = 0;
                int failedMarkerCount = 0;

                for (Integer markerDocId : markerDocIds) {
                    try {
                        // Get hidden vectors for this specific marker
                        List<HiddenVectorEntry> hiddenVectors = clumpingReader.getHiddenVectors(markerDocId);

                        // Requirement 7.2: If a marker has no associated hidden vectors, continue with just the marker
                        // (The marker is already in candidates from loadMarkerVectors)
                        for (HiddenVectorEntry hiddenVector : hiddenVectors) {
                            int hiddenDocId = hiddenVector.getDocId();

                            // Requirement 9.1: Apply filter during hidden vector expansion
                            // Ensure filtered documents don't appear in results
                            if (filterBits != null && !passesFilter(filterBits, hiddenDocId)) {
                                filteredOutCount++;
                                continue;
                            }

                            // Convert leaf-local doc ID to global doc ID
                            int globalDocId = leaf.docBase + hiddenDocId;
                            candidates.put(globalDocId, hiddenVector.getVector());
                            hiddenVectorCount++;
                        }
                    } catch (IOException e) {
                        // Requirement 12.3: If hidden vector expansion fails for a specific marker,
                        // continue with remaining markers and log a warning
                        failedMarkerCount++;
                        log.warn(
                            "Failed to read hidden vectors for marker docId={} in segment={}, field={}. Error: {}. Continuing with other markers.",
                            markerDocId,
                            segmentName,
                            field,
                            e.getMessage()
                        );
                        // Continue with other markers - the marker itself is already in candidates
                    }
                }

                if (failedMarkerCount > 0) {
                    log.warn(
                        "Hidden vector expansion failed for {} out of {} markers in segment={}. Results may be less accurate.",
                        failedMarkerCount,
                        markerDocIds.size(),
                        segmentName
                    );
                }

                if (filteredOutCount > 0) {
                    log.debug(
                        "Filtered out {} hidden vectors during expansion for segment={}",
                        filteredOutCount,
                        segmentName
                    );
                }

                log.debug(
                    "Expanded {} markers to {} total candidates ({} hidden vectors, {} failed markers) for segment={}",
                    markerDocIds.size(),
                    candidates.size(),
                    hiddenVectorCount,
                    failedMarkerCount,
                    segmentName
                );
            }
        } catch (IOException e) {
            // Requirement 12.1: Fall back to marker-only search on error (corrupted or unreadable clumping file)
            log.warn(
                "Failed to read clumping file for segment={}, field={}. Error: {}. Falling back to marker-only search.",
                segmentName,
                field,
                e.getMessage(),
                e
            );
            // Marker vectors are already loaded, so we can continue with marker-only search
        } catch (Exception e) {
            // Requirement 12.1: Catch any unexpected exceptions to ensure graceful degradation
            log.warn(
                "Unexpected error while reading clumping file for segment={}, field={}. Error: {}. Falling back to marker-only search.",
                segmentName,
                field,
                e.getMessage(),
                e
            );
            // Marker vectors are already loaded, so we can continue with marker-only search
        }

        return candidates;
    }

    /**
     * Backward-compatible version of expandWithHiddenVectors without filter support.
     * 
     * @param leaf         The leaf reader context
     * @param markerDocIds List of marker document IDs from the first-pass search
     * @return Map of document ID to vector for all candidates (markers + hidden)
     * @throws IOException if an I/O error occurs
     */
    @VisibleForTesting
    Map<Integer, float[]> expandWithHiddenVectors(LeafReaderContext leaf, List<Integer> markerDocIds) throws IOException {
        return expandWithHiddenVectors(leaf, markerDocIds, null);
    }

    /**
     * Checks if a document passes the filter.
     * 
     * @param filterBits The filter bits
     * @param docId      The document ID to check
     * @return true if the document passes the filter, false otherwise
     */
    private boolean passesFilter(Bits filterBits, int docId) {
        // Handle edge cases for Bits implementations
        if (filterBits instanceof Bits.MatchAllBits) {
            return true;
        }
        if (filterBits instanceof Bits.MatchNoBits) {
            return false;
        }
        // Check if docId is within bounds and passes the filter
        if (docId < 0 || docId >= filterBits.length()) {
            return false;
        }
        return filterBits.get(docId);
    }

    /**
     * Expands the candidate set to include sibling nested documents.
     * 
     * Requirement 9.2: When clumping is used with nested fields, parent-child document
     * relationships must be correctly maintained in the results.
     * 
     * This method retrieves all sibling nested documents for the candidate documents
     * that belong to the same parent document. This ensures that when a nested document
     * is found as a candidate (either as a marker or hidden vector), all its sibling
     * nested documents are also included in the candidate set for rescoring.
     * 
     * The method uses the parent filter (BitSetProducer) to identify parent documents
     * and then retrieves all nested documents belonging to those parents.
     *
     * @param leaf       The leaf reader context
     * @param candidates The current candidate map (doc ID -> vector)
     * @param filterBits Bits for filter checking (null if no filter)
     * @return Updated candidate map including sibling nested documents
     * @throws IOException if an I/O error occurs
     */
    @VisibleForTesting
    Map<Integer, float[]> expandWithNestedSiblings(LeafReaderContext leaf, Map<Integer, float[]> candidates, Bits filterBits)
        throws IOException {
        if (candidates.isEmpty()) {
            return candidates;
        }

        final StopWatch stopWatch = startStopWatch();

        // Get the parent BitSet for this leaf
        BitSet parentBitSet = parentFilter.getBitSet(leaf);
        if (parentBitSet == null) {
            log.debug("No parent BitSet found for leaf, skipping nested expansion");
            return candidates;
        }

        // Collect leaf-local doc IDs from candidates (need to convert from global to local)
        java.util.Set<Integer> localDocIds = new java.util.HashSet<>();
        for (Integer globalDocId : candidates.keySet()) {
            int localDocId = globalDocId - leaf.docBase;
            if (localDocId >= 0) {
                localDocIds.add(localDocId);
            }
        }

        if (localDocIds.isEmpty()) {
            return candidates;
        }

        // Create filter bits for nested expansion - use MatchAllBits if no filter
        Bits effectiveFilterBits = filterBits != null ? filterBits : new Bits.MatchAllBits(leaf.reader().maxDoc());

        // Get all sibling nested documents using GroupedNestedDocIdSetIterator
        // This iterator returns all nested documents belonging to the same parent
        // as the input document IDs
        DocIdSetIterator siblingIterator = new GroupedNestedDocIdSetIterator(parentBitSet, localDocIds, effectiveFilterBits);

        // Load vectors for all sibling nested documents
        final SegmentReader segmentReader = Lucene.segmentReader(leaf.reader());
        final FieldInfo fieldInfo = FieldInfoExtractor.getFieldInfo(segmentReader, field);

        if (fieldInfo == null) {
            log.warn(
                "Field info not found for field={} in segment={}. Cannot expand nested siblings.",
                field,
                segmentReader.getSegmentName()
            );
            return candidates;
        }

        // Get vector values for the field
        final KNNVectorValues<float[]> vectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, segmentReader);

        // Create a new map to hold the expanded candidates
        Map<Integer, float[]> expandedCandidates = new HashMap<>(candidates);
        int siblingCount = 0;

        // Iterate through all sibling nested documents
        int siblingDocId;
        while ((siblingDocId = siblingIterator.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
            int globalSiblingDocId = leaf.docBase + siblingDocId;

            // Skip if already in candidates
            if (expandedCandidates.containsKey(globalSiblingDocId)) {
                continue;
            }

            // Load the vector for this sibling document
            try {
                int advancedDocId = vectorValues.advance(siblingDocId);
                if (advancedDocId == siblingDocId) {
                    float[] vector = vectorValues.conditionalCloneVector();
                    expandedCandidates.put(globalSiblingDocId, vector);
                    siblingCount++;
                }
            } catch (IOException e) {
                log.debug("Failed to load vector for sibling nested doc ID {}: {}", siblingDocId, e.getMessage());
                // Continue with other siblings
            }
        }

        stopStopWatchAndLog(stopWatch, "nested_expansion");

        log.debug(
            "Expanded {} candidates to {} total candidates ({} sibling nested docs) for segment={}",
            candidates.size(),
            expandedCandidates.size(),
            siblingCount,
            segmentReader.getSegmentName()
        );

        return expandedCandidates;
    }

    /**
     * Loads marker vectors from the index into the candidates map.
     * 
     * This method retrieves the actual vector data for each marker document ID
     * from the Lucene index, so they can be rescored along with hidden vectors.
     *
     * @param leaf         The leaf reader context
     * @param markerDocIds List of marker document IDs to load
     * @param candidates   Map to populate with marker doc ID -> vector
     * @throws IOException if an I/O error occurs
     */
    private void loadMarkerVectors(LeafReaderContext leaf, List<Integer> markerDocIds, Map<Integer, float[]> candidates)
        throws IOException {
        final SegmentReader segmentReader = Lucene.segmentReader(leaf.reader());
        final FieldInfo fieldInfo = FieldInfoExtractor.getFieldInfo(segmentReader, field);

        if (fieldInfo == null) {
            log.warn("Field info not found for field={} in segment={}. Cannot load marker vectors.", field, segmentReader.getSegmentName());
            return;
        }

        // Get vector values for the field
        final KNNVectorValues<float[]> vectorValues = KNNVectorValuesFactory.getVectorValues(fieldInfo, segmentReader);

        // Sort marker doc IDs to enable efficient sequential access
        List<Integer> sortedMarkerDocIds = new ArrayList<>(markerDocIds);
        sortedMarkerDocIds.sort(Integer::compareTo);

        // Load each marker vector
        for (int markerDocId : sortedMarkerDocIds) {
            try {
                // Advance to the marker doc ID
                int advancedDocId = vectorValues.advance(markerDocId);
                if (advancedDocId == markerDocId) {
                    // Clone the vector since the underlying array may be reused
                    float[] vector = vectorValues.conditionalCloneVector();
                    // Convert leaf-local doc ID to global doc ID
                    int globalDocId = leaf.docBase + markerDocId;
                    candidates.put(globalDocId, vector);
                } else {
                    log.debug("Marker doc ID {} not found in vector values (advanced to {})", markerDocId, advancedDocId);
                }
            } catch (IOException e) {
                log.warn("Failed to load marker vector for doc ID {}: {}", markerDocId, e.getMessage());
                // Continue with other markers
            }
        }

        log.debug("Loaded {} marker vectors for segment={}", candidates.size(), segmentReader.getSegmentName());
    }

    /**
     * Rescores all candidates using exact distance calculation.
     * 
     * This method computes the exact distance between the query vector and each
     * candidate vector (both markers and hidden vectors), then returns the top-k 
     * results sorted by score.
     * 
     * The implementation follows the ExactSearcher pattern, using a min-heap (HitQueue)
     * for efficient top-k selection. This ensures that hidden vectors scoring higher
     * than their associated markers are correctly ranked above the markers.
     * 
     * Requirements:
     * - 8.1: Compute exact distances between query vector and all candidate vectors
     * - 8.3: Sort all candidates by score and return top-k results
     * - 8.4: Hidden vectors scoring higher than markers are correctly ranked
     *
     * @param candidates Map of document ID to vector for all candidates (markers + hidden)
     * @param leaf       The leaf reader context
     * @return TopDocs containing the top-k rescored results
     * @throws IOException if an I/O error occurs
     */
    @VisibleForTesting
    TopDocs rescoreCandidates(Map<Integer, float[]> candidates, LeafReaderContext leaf) throws IOException {
        if (candidates.isEmpty()) {
            return TopDocsCollector.EMPTY_TOPDOCS;
        }

        final StopWatch stopWatch = startStopWatch();

        // Use HitQueue (min-heap) for efficient top-k selection, following ExactSearcher pattern
        // Initialize with MAX DocID and Score as -INF
        final HitQueue queue = new HitQueue(k, true);
        ScoreDoc topDoc = queue.top();

        // Score all candidates using exact distance calculation
        // Requirement 8.1: Compute exact distances for all candidates (markers + hidden)
        for (Map.Entry<Integer, float[]> entry : candidates.entrySet()) {
            int docId = entry.getKey();
            float[] candidateVector = entry.getValue();

            // Calculate score using the space type's scoring function
            float score = calculateScore(queryVector, candidateVector);

            // Only update the heap if this score is better than the worst score in the heap
            // Requirement 8.4: Hidden vectors scoring higher than markers will be ranked correctly
            if (score > topDoc.score) {
                topDoc.score = score;
                topDoc.doc = docId;
                // Update the heap - this brings the doc with worst score to the top
                topDoc = queue.updateTop();
            }
        }

        // Remove any placeholder entries with negative infinity scores
        // This handles the case where candidates.size() < k
        while (queue.size() > 0 && queue.top().score == Float.NEGATIVE_INFINITY) {
            queue.pop();
        }

        // Extract results in descending score order
        // Requirement 8.3: Sort all candidates by score and return top-k
        ScoreDoc[] topScoreDocs = new ScoreDoc[queue.size()];
        for (int i = topScoreDocs.length - 1; i >= 0; i--) {
            topScoreDocs[i] = queue.pop();
        }

        stopStopWatchAndLog(stopWatch, "rescore");

        return new TopDocs(new TotalHits(topScoreDocs.length, TotalHits.Relation.EQUAL_TO), topScoreDocs);
    }

    /**
     * Calculates the score between the query vector and a candidate vector.
     * 
     * The score is calculated using the space type's vector similarity function.
     *
     * @param queryVector     The query vector
     * @param candidateVector The candidate vector
     * @return The similarity score
     */
    private float calculateScore(float[] queryVector, float[] candidateVector) {
        // Use the space type's scoring function
        return spaceType.getKnnVectorSimilarityFunction().compare(queryVector, candidateVector);
    }

    /**
     * Starts a stopwatch for timing if debug logging is enabled.
     *
     * @return A started StopWatch, or null if debug logging is disabled
     */
    private StopWatch startStopWatch() {
        if (log.isDebugEnabled()) {
            return new StopWatch().start();
        }
        return null;
    }

    /**
     * Stops the stopwatch and logs the elapsed time.
     *
     * @param stopWatch The stopwatch to stop (may be null)
     * @param phase     The name of the phase being timed
     */
    private void stopStopWatchAndLog(@Nullable final StopWatch stopWatch, String phase) {
        if (log.isDebugEnabled() && stopWatch != null) {
            stopWatch.stop();
            log.debug(
                "[{}] shard: [{}], field: [{}], phase: [{}], time in nanos: [{}]",
                this.getClass().getSimpleName(),
                shardId,
                field,
                phase,
                stopWatch.totalTime().nanos()
            );
        }
    }

    @Override
    public String toString(String field) {
        return this.getClass().getSimpleName()
            + "[innerQuery="
            + innerQuery
            + ", field="
            + this.field
            + ", k="
            + k
            + ", queryVector="
            + Arrays.toString(queryVector)
            + ", shardId="
            + shardId
            + ", clumpingContext="
            + clumpingContext
            + ", filterQuery="
            + filterQuery
            + ", parentFilter="
            + parentFilter
            + "]";
    }

    @Override
    public void visit(QueryVisitor visitor) {
        visitor.visitLeaf(this);
    }

    @Override
    public boolean equals(Object obj) {
        if (!sameClassAs(obj)) {
            return false;
        }
        ClumpingKNNVectorQuery other = (ClumpingKNNVectorQuery) obj;
        return Objects.equals(innerQuery, other.innerQuery)
            && Objects.equals(field, other.field)
            && k == other.k
            && Arrays.equals(queryVector, other.queryVector)
            && shardId == other.shardId
            && Objects.equals(clumpingContext, other.clumpingContext)
            && Objects.equals(spaceType, other.spaceType)
            && Objects.equals(filterQuery, other.filterQuery)
            && Objects.equals(parentFilter, other.parentFilter);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(innerQuery, field, k, shardId, clumpingContext, spaceType, filterQuery, parentFilter);
        result = 31 * result + Arrays.hashCode(queryVector);
        return result;
    }
}
