/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder.kmeansreorder;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.IndexOutput;

import java.io.IOException;

/**
 * Writes a {@link ClusterSummary} to a .kcs sidecar file.
 * Uses a simple header (no segment ID) so it can be read from any segment during merge.
 */
public class ClusterSummaryWriter {

    public static final String KCS_EXTENSION = ".kcs";
    public static final String CODEC_NAME = "KNNClusterSummary";
    public static final int VERSION_CURRENT = 0;

    public static String writeAndGetFileName(SegmentWriteState state, FieldInfo fieldInfo, ClusterSummary summary)
        throws IOException {
        String fileName = state.segmentInfo.name + "_" + fieldInfo.name + KCS_EXTENSION;
        try (IndexOutput out = state.directory.createOutput(fileName, state.context)) {
            CodecUtil.writeHeader(out, CODEC_NAME, VERSION_CURRENT);
            out.writeString(fieldInfo.name);
            out.writeInt(summary.k);
            out.writeInt(summary.dimension);
            out.writeInt(summary.metricType);
            out.writeInt(summary.numVectors);
            for (int i = 0; i < summary.k; i++) {
                out.writeInt(summary.counts[i]);
                writeFloatLE(out, summary.sumOfSquares[i]);
                for (int d = 0; d < summary.dimension; d++) {
                    writeFloatLE(out, summary.centroids[i][d]);
                }
                for (int d = 0; d < summary.dimension; d++) {
                    writeFloatLE(out, summary.linearSums[i][d]);
                }
            }
            CodecUtil.writeFooter(out);
        }
        // Register in segment files if the file set is available (merge path).
        // During flush, the TrackingDirectoryWrapper handles file tracking.
        try {
            state.segmentInfo.addFile(fileName);
        } catch (Exception e) {
            // setFiles may be null during flush — that's OK, TrackingDirectoryWrapper tracks it
        }
        return fileName;
    }

    private static void writeFloatLE(IndexOutput out, float value) throws IOException {
        out.writeInt(Float.floatToIntBits(value));
    }
}
