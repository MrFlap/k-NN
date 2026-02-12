/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.clumping;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Reads a .clump sidecar file (v2 format with inline vectors) to expand marker
 * vector results. Reads hidden vectors sequentially from the clump file and scores
 * them directly against the query vector, avoiding random access into Lucene's
 * vector storage.
 * <p>
 * See {@link ClumpFileFormat} for the binary layout.
 */
@Log4j2
public final class ClumpFileReader {

    private ClumpFileReader() {}

    /**
     * Checks whether a clump file exists for the given segment and field.
     */
    public static boolean clumpFileExists(Directory directory, String segmentName, String fieldName) throws IOException {
        String clumpFileName = resolveClumpFileName(directory, segmentName, fieldName);
        return clumpFileName != null;
    }

    /**
     * Resolves the actual clump file name, checking for both the compound (.clumpc)
     * and original (.clump) variants. Returns null if neither exists.
     */
    private static String resolveClumpFileName(Directory directory, String segmentName, String fieldName) throws IOException {
        String baseName = ClumpFileWriter.buildClumpFileName(segmentName, fieldName);
        String compoundName = baseName + KNNConstants.COMPOUND_EXTENSION;
        for (String file : directory.listAll()) {
            if (file.equals(compoundName)) {
                return compoundName;
            }
            if (file.equals(baseName)) {
                return baseName;
            }
        }
        return null;
    }

    /**
     * Expands marker doc IDs by reading their hidden vectors from the .clump file,
     * scoring each hidden vector against the query vector, and returning scored results.
     * <p>
     * The read phase is parallelized: each matched marker's raw hidden entry bytes are
     * read concurrently via cloned {@link IndexInput} handles on a parallel stream.
     * The scoring phase runs sequentially over the pre-read byte buffers to avoid
     * thread-safety concerns with the similarity function and to keep CPU work on the
     * calling thread.
     *
     * @param directory          The segment directory
     * @param segmentName        The segment name
     * @param fieldName          The vector field name
     * @param markerDocIds       The marker doc IDs from the ANN search results
     * @param floatQueryVector   The float query vector (null for byte vectors)
     * @param byteQueryVector    The byte query vector (null for float vectors)
     * @param similarityFunction The similarity function for scoring
     * @return List of ScoreDoc for hidden vectors. Does NOT include the markers themselves.
     */
    public static List<ScoreDoc> getHiddenVectorsScored(
        Directory directory,
        String segmentName,
        String fieldName,
        int[] markerDocIds,
        float[] floatQueryVector,
        byte[] byteQueryVector,
        KNNVectorSimilarityFunction similarityFunction
    ) throws IOException {
        String clumpFileName = resolveClumpFileName(directory, segmentName, fieldName);
        if (clumpFileName == null) {
            return Collections.emptyList();
        }

        try (IndexInput input = directory.openInput(clumpFileName, IOContext.DEFAULT)) {
            // Read header
            int numMarkers = input.readInt();
            if (numMarkers == 0) {
                return Collections.emptyList();
            }
            int dimension = input.readInt();
            byte vectorDataType = input.readByte();
            int vectorSize = ClumpFileFormat.vectorBytes(dimension, vectorDataType);

            // Read the marker table into arrays for binary search
            int[] allMarkerDocIds = new int[numMarkers];
            int[] allNumHidden = new int[numMarkers];
            long[] allClumpDataOffsets = new long[numMarkers];

            for (int i = 0; i < numMarkers; i++) {
                allMarkerDocIds[i] = input.readInt();
                allNumHidden[i] = input.readInt();
                allClumpDataOffsets[i] = input.readLong();
            }

            // Resolve which marker table indices match the query markers
            int[] matchedIndices = Arrays.stream(markerDocIds)
                .map(docId -> Arrays.binarySearch(allMarkerDocIds, docId))
                .filter(idx -> idx >= 0 && allNumHidden[idx] > 0)
                .toArray();

            if (matchedIndices.length == 0) {
                return Collections.emptyList();
            }

            // Sort matched indices by file offset for better I/O locality
            Arrays.sort(matchedIndices);

            boolean isFloat = (vectorDataType == ClumpFileFormat.VECTOR_TYPE_FLOAT);
            boolean isFp16 = (vectorDataType == ClumpFileFormat.VECTOR_TYPE_FP16);
            int hiddenEntrySize = Integer.BYTES + vectorSize;

            // Phase 1: READ — parallel. Each marker's hidden entries are bulk-read into
            // a byte[] via a cloned IndexInput (independent seek cursor, shared file handle).
            byte[][] markerBulkData = new byte[matchedIndices.length][];
            Arrays.stream(matchedIndices).parallel().forEach(markerIndex -> {
                int idx = Arrays.binarySearch(matchedIndices, markerIndex);
                int numHidden = allNumHidden[markerIndex];
                int bytesToRead = numHidden * hiddenEntrySize;
                byte[] buf = new byte[bytesToRead];
                try (IndexInput clonedInput = input.clone()) {
                    clonedInput.seek(allClumpDataOffsets[markerIndex] + vectorSize); // skip marker vector
                    clonedInput.readBytes(buf, 0, bytesToRead);
                } catch (IOException e) {
                    log.warn("Error reading hidden entries for marker index {}", markerIndex, e);
                }
                markerBulkData[idx] = buf;
            });

            // Phase 2: SCORE — sequential over the pre-read byte buffers.
            List<ScoreDoc> scoredHidden = new ArrayList<>();
            for (int m = 0; m < matchedIndices.length; m++) {
                int markerIndex = matchedIndices[m];
                byte[] bulk = markerBulkData[m];
                if (bulk == null) {
                    continue;
                }
                int numHidden = allNumHidden[markerIndex];
                java.nio.ByteBuffer bb = java.nio.ByteBuffer.wrap(bulk).order(java.nio.ByteOrder.LITTLE_ENDIAN);

                if (isFloat) {
                    float[] reusableVector = new float[dimension];
                    for (int j = 0; j < numHidden; j++) {
                        int hiddenDocId = bb.getInt();
                        for (int d = 0; d < dimension; d++) {
                            reusableVector[d] = bb.getFloat();
                        }
                        float score = similarityFunction.compare(floatQueryVector, reusableVector);
                        scoredHidden.add(new ScoreDoc(hiddenDocId, score));
                    }
                } else if (isFp16) {
                    // FP16: read 2 bytes per dimension, decode to float via Float.float16ToFloat
                    float[] reusableVector = new float[dimension];
                    for (int j = 0; j < numHidden; j++) {
                        int hiddenDocId = bb.getInt();
                        for (int d = 0; d < dimension; d++) {
                            reusableVector[d] = Float.float16ToFloat(bb.getShort());
                        }
                        float score = similarityFunction.compare(floatQueryVector, reusableVector);
                        scoredHidden.add(new ScoreDoc(hiddenDocId, score));
                    }
                } else {
                    byte[] reusableVector = new byte[dimension];
                    for (int j = 0; j < numHidden; j++) {
                        int hiddenDocId = bb.getInt();
                        bb.get(reusableVector);
                        float score = similarityFunction.compare(byteQueryVector, reusableVector);
                        scoredHidden.add(new ScoreDoc(hiddenDocId, score));
                    }
                }
            }

            return scoredHidden;
        }
    }

    /**
     * Returns just the hidden doc IDs (without scoring) for backward compatibility
     * or cases where only doc IDs are needed.
     */
    public static List<Integer> getHiddenDocIds(
        Directory directory,
        String segmentName,
        String fieldName,
        int[] markerDocIds
    ) throws IOException {
        String clumpFileName = resolveClumpFileName(directory, segmentName, fieldName);
        if (clumpFileName == null) {
            return Collections.emptyList();
        }

        try (IndexInput input = directory.openInput(clumpFileName, IOContext.DEFAULT)) {
            int numMarkers = input.readInt();
            if (numMarkers == 0) {
                return Collections.emptyList();
            }
            int dimension = input.readInt();
            byte vectorDataType = input.readByte();
            int vectorSize = ClumpFileFormat.vectorBytes(dimension, vectorDataType);

            int[] allMarkerDocIds = new int[numMarkers];
            int[] allNumHidden = new int[numMarkers];
            long[] allClumpDataOffsets = new long[numMarkers];

            for (int i = 0; i < numMarkers; i++) {
                allMarkerDocIds[i] = input.readInt();
                allNumHidden[i] = input.readInt();
                allClumpDataOffsets[i] = input.readLong();
            }

            List<Integer> hiddenDocIds = new ArrayList<>();
            for (int queryMarkerDocId : markerDocIds) {
                int markerIndex = Arrays.binarySearch(allMarkerDocIds, queryMarkerDocId);
                if (markerIndex < 0) {
                    continue;
                }

                int numHidden = allNumHidden[markerIndex];
                if (numHidden == 0) {
                    continue;
                }

                // Seek past marker vector, then read hidden doc IDs (skipping vector data)
                long clumpOffset = allClumpDataOffsets[markerIndex];
                input.seek(clumpOffset + vectorSize);

                for (int j = 0; j < numHidden; j++) {
                    hiddenDocIds.add(input.readInt());
                    // Skip the vector data
                    input.seek(input.getFilePointer() + vectorSize);
                }
            }

            return hiddenDocIds;
        }
    }

}
