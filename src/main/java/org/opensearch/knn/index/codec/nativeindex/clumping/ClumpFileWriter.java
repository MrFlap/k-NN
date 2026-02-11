/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.clumping;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.IndexOutput;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Writes a .clump sidecar file that maps marker vectors to their associated hidden vectors.
 * See {@link ClumpFileFormat} for the binary layout.
 */
@Log4j2
public final class ClumpFileWriter {

    private ClumpFileWriter() {}

    /**
     * Writes the clump file for a given field.
     *
     * @param state             Segment write state for creating output files
     * @param fieldName         The vector field name
     * @param markerDocIds      Ordered list of marker doc IDs (sorted ascending)
     * @param markerToHidden    Map from marker doc ID to list of hidden doc IDs associated with it
     */
    public static void writeClumpFile(
        SegmentWriteState state,
        String fieldName,
        List<Integer> markerDocIds,
        Map<Integer, List<Integer>> markerToHidden
    ) throws IOException {
        String clumpFileName = buildClumpFileName(state.segmentInfo.name, fieldName);
        int numMarkers = markerDocIds.size();

        try (IndexOutput output = state.directory.createOutput(clumpFileName, state.context)) {
            // Header: numMarkers
            output.writeInt(numMarkers);

            // Build offset table
            int runningOffset = 0;
            int[] offsets = new int[numMarkers + 1];
            for (int i = 0; i < numMarkers; i++) {
                offsets[i] = runningOffset;
                List<Integer> hidden = markerToHidden.get(markerDocIds.get(i));
                runningOffset += (hidden != null ? hidden.size() : 0);
            }
            offsets[numMarkers] = runningOffset;

            // Write offset table
            for (int offset : offsets) {
                output.writeInt(offset);
            }

            // Write adjacency data (hidden doc IDs)
            for (int i = 0; i < numMarkers; i++) {
                List<Integer> hidden = markerToHidden.get(markerDocIds.get(i));
                if (hidden != null) {
                    for (int hiddenDocId : hidden) {
                        output.writeInt(hiddenDocId);
                    }
                }
            }

            // Write marker doc ID table
            for (int markerDocId : markerDocIds) {
                output.writeInt(markerDocId);
            }

            CodecUtil.writeFooter(output);
            log.debug("Wrote clump file {} with {} markers and {} total hidden vectors", clumpFileName, numMarkers, runningOffset);
        }
    }

    public static String buildClumpFileName(String segmentName, String fieldName) {
        return segmentName + "_" + fieldName + ClumpFileFormat.CLUMP_FILE_EXTENSION;
    }
}
