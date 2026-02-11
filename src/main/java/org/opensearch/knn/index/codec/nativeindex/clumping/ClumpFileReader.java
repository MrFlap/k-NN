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
     * All vector data is read sequentially from the clump file — no random access into
     * Lucene's vector storage is needed.
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

            // For each query marker, find its index, seek to its clump data, read + score
            List<ScoreDoc> scoredHidden = new ArrayList<>();
            boolean isFloat = (vectorDataType == ClumpFileFormat.VECTOR_TYPE_FLOAT);

            for (int queryMarkerDocId : markerDocIds) {
                int markerIndex = Arrays.binarySearch(allMarkerDocIds, queryMarkerDocId);
                if (markerIndex < 0) {
                    continue;
                }

                int numHidden = allNumHidden[markerIndex];
                if (numHidden == 0) {
                    continue;
                }

                // Seek to clump data: skip marker vector, then read hidden entries sequentially
                long clumpOffset = allClumpDataOffsets[markerIndex];
                input.seek(clumpOffset + vectorSize); // skip marker vector

                for (int j = 0; j < numHidden; j++) {
                    int hiddenDocId = input.readInt();
                    float score;
                    if (isFloat) {
                        float[] hiddenVector = readFloatVector(input, dimension);
                        score = similarityFunction.compare(floatQueryVector, hiddenVector);
                    } else {
                        byte[] hiddenVector = readByteVector(input, dimension);
                        score = similarityFunction.compare(byteQueryVector, hiddenVector);
                    }
                    scoredHidden.add(new ScoreDoc(hiddenDocId, score));
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

    private static float[] readFloatVector(IndexInput input, int dimension) throws IOException {
        float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            vector[i] = Float.intBitsToFloat(input.readInt());
        }
        return vector;
    }

    private static byte[] readByteVector(IndexInput input, int dimension) throws IOException {
        byte[] vector = new byte[dimension];
        input.readBytes(vector, 0, dimension);
        return vector;
    }
}
