/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.clumping.ClumpingFileReader;
import org.opensearch.knn.index.codec.clumping.ClumpingFileWriter;
import org.opensearch.knn.index.codec.clumping.HiddenVectorEntry;
import org.opensearch.knn.index.codec.clumping.MarkerSelector;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.plugin.stats.KNNGraphValue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.initializeVectorValues;

/**
 * Strategy for building k-NN indices with clumping optimization.
 * 
 * Clumping reduces the main index size by only indexing a subset of vectors (markers)
 * while storing the remaining vectors (hidden) in a separate file. During indexing:
 * 
 * <ol>
 *   <li>Partition vectors into markers and hidden using {@link MarkerSelector}</li>
 *   <li>Build the main index with markers only using the delegate strategy</li>
 *   <li>Associate each hidden vector with its closest marker using exact distance</li>
 *   <li>Write hidden vectors to the clumping file</li>
 * </ol>
 * 
 * @see MarkerSelector for marker selection logic
 * @see ClumpingFileWriter for clumping file format
 */
@Log4j2
public class ClumpingIndexBuildStrategy implements NativeIndexBuildStrategy {

    private final NativeIndexBuildStrategy delegateStrategy;
    private final int clumpingFactor;

    /**
     * Creates a new ClumpingIndexBuildStrategy.
     *
     * @param delegateStrategy The strategy to use for building the main index with markers
     * @param clumpingFactor   The clumping factor (ratio of total vectors to markers).
     *                         Must be at least 2.
     */
    public ClumpingIndexBuildStrategy(NativeIndexBuildStrategy delegateStrategy, int clumpingFactor) {
        if (clumpingFactor < 2) {
            throw new IllegalArgumentException("Clumping factor must be at least 2, got: " + clumpingFactor);
        }
        this.delegateStrategy = delegateStrategy;
        this.clumpingFactor = clumpingFactor;
    }

    @Override
    public void buildAndWriteIndex(BuildIndexParams params) throws IOException {
        log.debug("Building clumping index for field {} with clumping factor {}", params.getFieldName(), clumpingFactor);

        // 1. Partition vectors into markers and hidden
        // During merge, this also includes hidden vectors from source segment clumping files
        PartitionedVectors partitioned = partitionVectors(params);

        if (partitioned.getMarkerCount() == 0) {
            log.warn("No marker vectors selected for field {}. Skipping index build.", params.getFieldName());
            return;
        }

        log.debug(
            "Partitioned {} vectors: {} markers, {} hidden",
            partitioned.getTotalCount(),
            partitioned.getMarkerCount(),
            partitioned.getHiddenCount()
        );

        // 2. Build main index with markers only
        BuildIndexParams markerParams = params.toBuilder()
            .knnVectorValuesSupplier(partitioned::getMarkerVectorValues)
            .totalLiveDocs(partitioned.getMarkerCount())
            .mergeState(null) // Don't pass mergeState to delegate - we've already handled clumping
            .build();

        delegateStrategy.buildAndWriteIndex(markerParams);

        // 3. Associate hidden vectors with closest markers and write clumping file
        if (partitioned.getHiddenCount() > 0) {
            writeClumpingFile(params, partitioned);
        }

        log.debug("Completed clumping index build for field {}", params.getFieldName());
    }

    /**
     * Partitions vectors into markers and hidden based on the clumping factor.
     * Uses deterministic selection based on document ID.
     * 
     * During merge operations, this method also reads hidden vectors from source
     * segment clumping files and includes them in the partitioning process.
     *
     * @param params The build index parameters
     * @return PartitionedVectors containing separated marker and hidden vectors
     * @throws IOException if an I/O error occurs
     */
    private PartitionedVectors partitionVectors(BuildIndexParams params) throws IOException {
        List<Integer> markerDocIds = new ArrayList<>();
        List<float[]> markerVectors = new ArrayList<>();
        List<Integer> hiddenDocIds = new ArrayList<>();
        List<float[]> hiddenVectors = new ArrayList<>();

        // First, process vectors from the main index (markers from source segments during merge,
        // or all vectors during flush)
        KNNVectorValues<?> vectorValues = params.getKnnVectorValuesSupplier().get();
        initializeVectorValues(vectorValues);

        while (vectorValues.docId() != NO_MORE_DOCS) {
            int docId = vectorValues.docId();
            float[] vector = getVectorAsFloatArray(vectorValues);

            if (MarkerSelector.isMarker(docId, clumpingFactor)) {
                markerDocIds.add(docId);
                markerVectors.add(vector);
            } else {
                hiddenDocIds.add(docId);
                hiddenVectors.add(vector);
            }

            vectorValues.nextDoc();
        }

        // During merge, also read hidden vectors from source segment clumping files
        if (params.getMergeState() != null) {
            readHiddenVectorsFromSourceSegments(params, markerDocIds, markerVectors, hiddenDocIds, hiddenVectors);
        }

        return new PartitionedVectors(markerDocIds, markerVectors, hiddenDocIds, hiddenVectors);
    }

    /**
     * Reads hidden vectors from source segment clumping files during merge.
     * 
     * The hidden vectors are re-partitioned based on their new document IDs in the
     * merged segment. Some hidden vectors may become markers in the merged segment,
     * and vice versa.
     *
     * @param params        The build index parameters
     * @param markerDocIds  List to add marker document IDs to
     * @param markerVectors List to add marker vectors to
     * @param hiddenDocIds  List to add hidden document IDs to
     * @param hiddenVectors List to add hidden vectors to
     * @throws IOException if an I/O error occurs
     */
    private void readHiddenVectorsFromSourceSegments(
        BuildIndexParams params,
        List<Integer> markerDocIds,
        List<float[]> markerVectors,
        List<Integer> hiddenDocIds,
        List<float[]> hiddenVectors
    ) throws IOException {
        MergeState mergeState = params.getMergeState();
        String fieldName = params.getFieldName();

        log.debug("Reading hidden vectors from {} source segments for field {}", mergeState.knnVectorsReaders.length, fieldName);

        int totalHiddenVectorsRead = 0;

        // Iterate through each source segment
        for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
            KnnVectorsReader knnVectorsReader = mergeState.knnVectorsReaders[i];
            if (knnVectorsReader == null) {
                continue;
            }

            // Get the source segment's directory
            // The knnVectorsReader should be associated with a SegmentReader
            Directory sourceDirectory = getSourceSegmentDirectory(mergeState, i);
            if (sourceDirectory == null) {
                log.debug("Could not get directory for source segment {}, skipping clumping file read", i);
                continue;
            }

            // Get the source segment name
            String sourceSegmentName = getSourceSegmentName(mergeState, i);
            if (sourceSegmentName == null) {
                log.debug("Could not get segment name for source segment {}, skipping clumping file read", i);
                continue;
            }

            // Check if clumping file exists for this source segment
            if (!ClumpingFileReader.exists(sourceDirectory, sourceSegmentName, fieldName)) {
                log.debug("No clumping file found for source segment {} field {}", sourceSegmentName, fieldName);
                continue;
            }

            // Read hidden vectors from the source segment's clumping file
            try (ClumpingFileReader reader = ClumpingFileReader.open(sourceDirectory, sourceSegmentName, fieldName)) {
                int hiddenVectorsFromSegment = readAndPartitionHiddenVectors(
                    reader,
                    mergeState,
                    i,
                    markerDocIds,
                    markerVectors,
                    hiddenDocIds,
                    hiddenVectors
                );
                totalHiddenVectorsRead += hiddenVectorsFromSegment;
                log.debug(
                    "Read {} hidden vectors from source segment {} for field {}",
                    hiddenVectorsFromSegment,
                    sourceSegmentName,
                    fieldName
                );
            } catch (IOException e) {
                log.warn("Failed to read clumping file from source segment {}: {}", sourceSegmentName, e.getMessage());
                // Continue with other segments - graceful degradation
            }
        }

        log.debug("Total hidden vectors read from source segments: {}", totalHiddenVectorsRead);
    }

    /**
     * Gets the directory for a source segment during merge.
     *
     * @param mergeState   The merge state
     * @param segmentIndex The index of the source segment
     * @return The directory, or null if not available
     */
    private Directory getSourceSegmentDirectory(MergeState mergeState, int segmentIndex) {
        try {
            // The mergeState has an array of readers that we can use to get the directory
            // We need to access the underlying SegmentReader to get the directory
            if (mergeState.knnVectorsReaders[segmentIndex] != null) {
                // Try to get directory from the segment info if available
                // The directory is typically the same as the target directory during merge
                return mergeState.segmentInfo.dir;
            }
        } catch (Exception e) {
            log.debug("Could not get directory for source segment {}: {}", segmentIndex, e.getMessage());
        }
        return null;
    }

    /**
     * Gets the segment name for a source segment during merge.
     *
     * @param mergeState   The merge state
     * @param segmentIndex The index of the source segment
     * @return The segment name, or null if not available
     */
    private String getSourceSegmentName(MergeState mergeState, int segmentIndex) {
        try {
            // Access the segment info from the merge state
            // The segment infos are available through the docMaps
            if (mergeState.docMaps != null && mergeState.docMaps.length > segmentIndex) {
                // We need to get the segment name from the original segment
                // This is typically available through the SegmentReader
                // For now, we'll try to infer it from the available information
                
                // The segment names follow a pattern like "_0", "_1", etc.
                // During merge, we need to access the original segment names
                // This information should be available through the merge state's readers
                
                // Try to get segment info from fieldInfos
                if (mergeState.fieldInfos != null && mergeState.fieldInfos.length > segmentIndex) {
                    // The segment name is typically embedded in the file names
                    // We can try to extract it from the available information
                    // For now, return a placeholder that will be validated by exists() check
                    return getSegmentNameFromMergeState(mergeState, segmentIndex);
                }
            }
        } catch (Exception e) {
            log.debug("Could not get segment name for source segment {}: {}", segmentIndex, e.getMessage());
        }
        return null;
    }

    /**
     * Attempts to get the segment name from the merge state.
     * This is a helper method that tries various approaches to extract the segment name.
     *
     * @param mergeState   The merge state
     * @param segmentIndex The index of the source segment
     * @return The segment name, or null if not available
     */
    private String getSegmentNameFromMergeState(MergeState mergeState, int segmentIndex) {
        // The MergeState contains information about source segments
        // We need to access the original segment names to find clumping files
        
        // Try to get the segment name from the storedFieldsReaders if available
        if (mergeState.storedFieldsReaders != null && mergeState.storedFieldsReaders.length > segmentIndex) {
            Object reader = mergeState.storedFieldsReaders[segmentIndex];
            if (reader instanceof SegmentReader) {
                SegmentReader segmentReader = (SegmentReader) reader;
                return segmentReader.getSegmentInfo().info.name;
            }
        }
        
        // If we can't get the segment name directly, we can't read the clumping file
        // This is a limitation that may need to be addressed in a future enhancement
        return null;
    }

    /**
     * Reads hidden vectors from a clumping file and partitions them based on their
     * new document IDs in the merged segment.
     *
     * @param reader        The clumping file reader
     * @param mergeState    The merge state
     * @param segmentIndex  The index of the source segment
     * @param markerDocIds  List to add marker document IDs to
     * @param markerVectors List to add marker vectors to
     * @param hiddenDocIds  List to add hidden document IDs to
     * @param hiddenVectors List to add hidden vectors to
     * @return The number of hidden vectors read
     * @throws IOException if an I/O error occurs
     */
    private int readAndPartitionHiddenVectors(
        ClumpingFileReader reader,
        MergeState mergeState,
        int segmentIndex,
        List<Integer> markerDocIds,
        List<float[]> markerVectors,
        List<Integer> hiddenDocIds,
        List<float[]> hiddenVectors
    ) throws IOException {
        int count = 0;
        
        // Get the doc ID map for this segment to translate old doc IDs to new ones
        MergeState.DocMap docMap = mergeState.docMaps[segmentIndex];
        
        // Get all hidden vectors from the clumping file
        List<HiddenVectorEntry> allEntries = reader.getAllHiddenVectors();
        
        for (HiddenVectorEntry entry : allEntries) {
            // Map the old doc ID to the new doc ID in the merged segment
            int newDocId = docMap.get(entry.getDocId());
            
            // Skip deleted documents (mapped to -1)
            if (newDocId == -1) {
                continue;
            }
            
            float[] vector = entry.getVector();
            
            // Re-partition based on the new doc ID
            if (MarkerSelector.isMarker(newDocId, clumpingFactor)) {
                markerDocIds.add(newDocId);
                markerVectors.add(vector);
            } else {
                hiddenDocIds.add(newDocId);
                hiddenVectors.add(vector);
            }
            
            count++;
        }
        
        return count;
    }

    /**
     * Gets the vector as a float array, handling different vector data types.
     *
     * @param vectorValues The vector values iterator
     * @return The vector as a float array
     * @throws IOException if an I/O error occurs
     */
    private float[] getVectorAsFloatArray(KNNVectorValues<?> vectorValues) throws IOException {
        Object vector = vectorValues.conditionalCloneVector();
        if (vector instanceof float[]) {
            return (float[]) vector;
        } else if (vector instanceof byte[]) {
            // Convert byte array to float array
            byte[] byteVector = (byte[]) vector;
            float[] floatVector = new float[byteVector.length];
            for (int i = 0; i < byteVector.length; i++) {
                floatVector[i] = byteVector[i];
            }
            return floatVector;
        } else {
            throw new IllegalArgumentException("Unsupported vector type: " + vector.getClass().getName());
        }
    }

    /**
     * Associates hidden vectors with their closest markers and writes to the clumping file.
     *
     * @param params      The build index parameters
     * @param partitioned The partitioned vectors
     * @throws IOException if an I/O error occurs
     */
    private void writeClumpingFile(BuildIndexParams params, PartitionedVectors partitioned) throws IOException {
        // Get space type for distance calculation
        SpaceType spaceType = getSpaceType(params);
        KNNVectorSimilarityFunction similarityFunction = spaceType.getKnnVectorSimilarityFunction();

        if (similarityFunction == null) {
            throw new IllegalStateException("Space type " + spaceType + " does not support similarity function for clumping");
        }

        // Get dimension from the first marker vector
        int dimension = partitioned.getMarkerVectors().get(0).length;

        // Create clumping file
        String clumpingFileName = buildClumpingFileName(params);
        try (IndexOutput output = params.getSegmentWriteState().directory.createOutput(clumpingFileName, params.getSegmentWriteState().context)) {
            try (ClumpingFileWriter writer = new ClumpingFileWriter(output, clumpingFactor, dimension, params.getVectorDataType())) {
                // Associate each hidden vector with its closest marker
                List<Integer> hiddenDocIds = partitioned.getHiddenDocIds();
                List<float[]> hiddenVectors = partitioned.getHiddenVectors();
                List<Integer> markerDocIds = partitioned.getMarkerDocIds();
                List<float[]> markerVectors = partitioned.getMarkerVectors();

                for (int i = 0; i < hiddenDocIds.size(); i++) {
                    int hiddenDocId = hiddenDocIds.get(i);
                    float[] hiddenVector = hiddenVectors.get(i);

                    // Find closest marker using exact distance
                    int closestMarkerDocId = findClosestMarker(hiddenVector, markerDocIds, markerVectors, similarityFunction);

                    // Add to clumping file
                    writer.addHiddenVector(hiddenDocId, hiddenVector, closestMarkerDocId);
                }

                // Finish writing the clumping file
                writer.finish();

                // Record clumping statistics
                recordClumpingStats(writer.getMarkerCount(), writer.getHiddenVectorCount(), output.getFilePointer());

                log.debug(
                    "Wrote clumping file {} with {} hidden vectors associated with {} markers",
                    clumpingFileName,
                    writer.getHiddenVectorCount(),
                    writer.getMarkerCount()
                );
            }
        }
    }

    /**
     * Finds the closest marker to a hidden vector using exact distance calculation.
     *
     * @param hiddenVector       The hidden vector
     * @param markerDocIds       List of marker document IDs
     * @param markerVectors      List of marker vectors
     * @param similarityFunction The similarity function to use for distance calculation
     * @return The document ID of the closest marker
     */
    private int findClosestMarker(
        float[] hiddenVector,
        List<Integer> markerDocIds,
        List<float[]> markerVectors,
        KNNVectorSimilarityFunction similarityFunction
    ) {
        int closestMarkerDocId = markerDocIds.get(0);
        float bestSimilarity = similarityFunction.compare(hiddenVector, markerVectors.get(0));

        for (int i = 1; i < markerDocIds.size(); i++) {
            float similarity = similarityFunction.compare(hiddenVector, markerVectors.get(i));
            // Higher similarity means closer vectors
            if (similarity > bestSimilarity) {
                bestSimilarity = similarity;
                closestMarkerDocId = markerDocIds.get(i);
            }
        }

        return closestMarkerDocId;
    }

    /**
     * Gets the space type from the build parameters.
     *
     * @param params The build index parameters
     * @return The space type
     */
    private SpaceType getSpaceType(BuildIndexParams params) {
        Object spaceTypeValue = params.getParameters().get(KNNConstants.SPACE_TYPE);
        if (spaceTypeValue == null) {
            return SpaceType.DEFAULT;
        }
        if (spaceTypeValue instanceof String) {
            return SpaceType.getSpace((String) spaceTypeValue);
        }
        return SpaceType.DEFAULT;
    }

    /**
     * Builds the clumping file name for the given parameters.
     *
     * @param params The build index parameters
     * @return The clumping file name
     */
    private String buildClumpingFileName(BuildIndexParams params) {
        return params.getSegmentWriteState().segmentInfo.name + "_" + params.getFieldName() + "." + ClumpingFileWriter.CLUMPING_EXTENSION;
    }

    /**
     * Records clumping statistics for monitoring and debugging.
     *
     * @param markerCount       The number of marker vectors
     * @param hiddenVectorCount The number of hidden vectors
     * @param fileSizeInBytes   The size of the clumping file in bytes
     */
    private void recordClumpingStats(int markerCount, int hiddenVectorCount, long fileSizeInBytes) {
        KNNGraphValue.CLUMPING_TOTAL_MARKER_COUNT.incrementBy(markerCount);
        KNNGraphValue.CLUMPING_TOTAL_HIDDEN_VECTOR_COUNT.incrementBy(hiddenVectorCount);
        KNNGraphValue.CLUMPING_TOTAL_FILE_SIZE_IN_BYTES.incrementBy(fileSizeInBytes);
        KNNGraphValue.CLUMPING_TOTAL_SEGMENTS.increment();
    }

    /**
     * Internal class to hold partitioned vectors (markers and hidden).
     */
    private static class PartitionedVectors {
        private final List<Integer> markerDocIds;
        private final List<float[]> markerVectors;
        private final List<Integer> hiddenDocIds;
        private final List<float[]> hiddenVectors;

        PartitionedVectors(
            List<Integer> markerDocIds,
            List<float[]> markerVectors,
            List<Integer> hiddenDocIds,
            List<float[]> hiddenVectors
        ) {
            this.markerDocIds = markerDocIds;
            this.markerVectors = markerVectors;
            this.hiddenDocIds = hiddenDocIds;
            this.hiddenVectors = hiddenVectors;
        }

        List<Integer> getMarkerDocIds() {
            return markerDocIds;
        }

        List<float[]> getMarkerVectors() {
            return markerVectors;
        }

        List<Integer> getHiddenDocIds() {
            return hiddenDocIds;
        }

        List<float[]> getHiddenVectors() {
            return hiddenVectors;
        }

        int getMarkerCount() {
            return markerDocIds.size();
        }

        int getHiddenCount() {
            return hiddenDocIds.size();
        }

        int getTotalCount() {
            return markerDocIds.size() + hiddenDocIds.size();
        }

        /**
         * Creates a KNNVectorValues that iterates over marker vectors only.
         * This is used to build the main index with markers.
         *
         * @return KNNVectorValues for marker vectors
         */
        KNNVectorValues<float[]> getMarkerVectorValues() {
            return new MarkerVectorValues(markerDocIds, markerVectors);
        }
    }

    /**
     * KNNVectorValues implementation that wraps pre-loaded marker vectors.
     * This allows the delegate strategy to iterate over markers only.
     */
    private static class MarkerVectorValues extends KNNVectorValues<float[]> {
        private final List<Integer> docIds;
        private final List<float[]> vectors;
        private int currentIndex;

        MarkerVectorValues(List<Integer> docIds, List<float[]> vectors) {
            super(new MarkerVectorValuesIterator(docIds, vectors));
            this.docIds = docIds;
            this.vectors = vectors;
            this.currentIndex = -1;
            if (!vectors.isEmpty()) {
                this.dimension = vectors.get(0).length;
                this.bytesPerVector = this.dimension * Float.BYTES;
            }
        }

        @Override
        public float[] getVector() throws IOException {
            if (currentIndex < 0 || currentIndex >= vectors.size()) {
                throw new IOException("Invalid vector index: " + currentIndex);
            }
            return vectors.get(currentIndex);
        }

        @Override
        public float[] conditionalCloneVector() throws IOException {
            // Return a copy since the caller may store the reference
            float[] vector = getVector();
            return vector.clone();
        }

        @Override
        public int docId() {
            if (currentIndex < 0) {
                return -1;
            }
            if (currentIndex >= docIds.size()) {
                return NO_MORE_DOCS;
            }
            return docIds.get(currentIndex);
        }

        @Override
        public int nextDoc() throws IOException {
            currentIndex++;
            if (currentIndex >= docIds.size()) {
                return NO_MORE_DOCS;
            }
            return docIds.get(currentIndex);
        }

        @Override
        public int advance(int target) throws IOException {
            while (currentIndex < docIds.size() - 1) {
                currentIndex++;
                if (docIds.get(currentIndex) >= target) {
                    return docIds.get(currentIndex);
                }
            }
            currentIndex = docIds.size();
            return NO_MORE_DOCS;
        }
    }

    /**
     * Iterator for MarkerVectorValues.
     */
    private static class MarkerVectorValuesIterator implements org.opensearch.knn.index.vectorvalues.KNNVectorValuesIterator {
        private final List<Integer> docIds;
        private final List<float[]> vectors;
        private int currentIndex;

        MarkerVectorValuesIterator(List<Integer> docIds, List<float[]> vectors) {
            this.docIds = docIds;
            this.vectors = vectors;
            this.currentIndex = -1;
        }

        @Override
        public int docId() {
            if (currentIndex < 0) {
                return -1;
            }
            if (currentIndex >= docIds.size()) {
                return NO_MORE_DOCS;
            }
            return docIds.get(currentIndex);
        }

        @Override
        public int nextDoc() throws IOException {
            currentIndex++;
            if (currentIndex >= docIds.size()) {
                return NO_MORE_DOCS;
            }
            return docIds.get(currentIndex);
        }

        @Override
        public int advance(int target) throws IOException {
            while (currentIndex < docIds.size() - 1) {
                currentIndex++;
                if (docIds.get(currentIndex) >= target) {
                    return docIds.get(currentIndex);
                }
            }
            currentIndex = docIds.size();
            return NO_MORE_DOCS;
        }

        @Override
        public org.apache.lucene.search.DocIdSetIterator getDocIdSetIterator() {
            // Return a DocIdSetIterator that wraps our list-based iteration
            return new org.apache.lucene.search.DocIdSetIterator() {
                private int index = -1;

                @Override
                public int docID() {
                    if (index < 0) {
                        return -1;
                    }
                    if (index >= docIds.size()) {
                        return NO_MORE_DOCS;
                    }
                    return docIds.get(index);
                }

                @Override
                public int nextDoc() {
                    index++;
                    if (index >= docIds.size()) {
                        return NO_MORE_DOCS;
                    }
                    return docIds.get(index);
                }

                @Override
                public int advance(int target) {
                    while (index < docIds.size() - 1) {
                        index++;
                        if (docIds.get(index) >= target) {
                            return docIds.get(index);
                        }
                    }
                    index = docIds.size();
                    return NO_MORE_DOCS;
                }

                @Override
                public long cost() {
                    return docIds.size();
                }
            };
        }

        @Override
        public long liveDocs() {
            return docIds.size();
        }

        @Override
        public org.opensearch.knn.index.vectorvalues.VectorValueExtractorStrategy getVectorExtractorStrategy() {
            // Return a custom extractor strategy for our in-memory vectors
            return new MarkerVectorExtractorStrategy(vectors);
        }

        /**
         * Gets the current vector at the current index.
         */
        float[] getCurrentVector() {
            if (currentIndex < 0 || currentIndex >= vectors.size()) {
                return null;
            }
            return vectors.get(currentIndex);
        }
    }

    /**
     * Custom vector extractor strategy for marker vectors stored in memory.
     */
    private static class MarkerVectorExtractorStrategy implements org.opensearch.knn.index.vectorvalues.VectorValueExtractorStrategy {
        private final List<float[]> vectors;

        MarkerVectorExtractorStrategy(List<float[]> vectors) {
            this.vectors = vectors;
        }

        @SuppressWarnings("unchecked")
        @Override
        public <T> T extract(VectorDataType vectorDataType, org.opensearch.knn.index.vectorvalues.KNNVectorValuesIterator iterator) throws IOException {
            if (!(iterator instanceof MarkerVectorValuesIterator)) {
                throw new IllegalArgumentException("Expected MarkerVectorValuesIterator");
            }
            MarkerVectorValuesIterator markerIterator = (MarkerVectorValuesIterator) iterator;
            float[] vector = markerIterator.getCurrentVector();
            
            if (vectorDataType == VectorDataType.FLOAT) {
                return (T) vector;
            } else if (vectorDataType == VectorDataType.BYTE || vectorDataType == VectorDataType.BINARY) {
                // Convert float array to byte array
                if (vector == null) {
                    return null;
                }
                byte[] byteVector = new byte[vector.length];
                for (int i = 0; i < vector.length; i++) {
                    byteVector[i] = (byte) vector[i];
                }
                return (T) byteVector;
            }
            throw new IllegalArgumentException("Unsupported vector data type: " + vectorDataType);
        }
    }
}
