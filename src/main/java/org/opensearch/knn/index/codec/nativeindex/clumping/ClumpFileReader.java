/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.clumping;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;

import org.opensearch.knn.common.KNNConstants;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Reads a .clump sidecar file to expand marker vector results to include hidden vectors.
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
     * Resolves the actual clump file name, checking for both the compound (.clumpc) and original (.clump) variants.
     * Returns null if neither exists.
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
     * Expands a set of marker doc IDs to include their associated hidden doc IDs.
     *
     * @param directory     The segment directory
     * @param segmentName   The segment name
     * @param fieldName     The vector field name
     * @param markerDocIds  The marker doc IDs from the ANN search results
     * @return List of hidden doc IDs associated with the given markers. Does NOT include the markers themselves.
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

            // Read offset table
            int[] offsets = new int[numMarkers + 1];
            for (int i = 0; i <= numMarkers; i++) {
                offsets[i] = input.readInt();
            }

            int totalHiddenVectors = offsets[numMarkers];

            // Calculate where sections start
            long adjDataStart = ClumpFileFormat.adjDataStart(numMarkers);
            long markerTableStart = ClumpFileFormat.markerDocIdTableStart(numMarkers, totalHiddenVectors);

            // Read marker doc ID table
            int[] allMarkerDocIds = new int[numMarkers];
            input.seek(markerTableStart);
            for (int i = 0; i < numMarkers; i++) {
                allMarkerDocIds[i] = input.readInt();
            }

            // For each query marker doc ID, find its index and read hidden vectors
            List<Integer> hiddenDocIds = new ArrayList<>();
            for (int queryMarkerDocId : markerDocIds) {
                int markerIndex = Arrays.binarySearch(allMarkerDocIds, queryMarkerDocId);
                if (markerIndex < 0) {
                    // Not a marker vector in this clump file, skip
                    continue;
                }

                int startOffset = offsets[markerIndex];
                int endOffset = offsets[markerIndex + 1];
                int numHidden = endOffset - startOffset;

                if (numHidden > 0) {
                    input.seek(adjDataStart + (long) startOffset * Integer.BYTES);
                    for (int j = 0; j < numHidden; j++) {
                        hiddenDocIds.add(input.readInt());
                    }
                }
            }

            return hiddenDocIds;
        }
    }
}
