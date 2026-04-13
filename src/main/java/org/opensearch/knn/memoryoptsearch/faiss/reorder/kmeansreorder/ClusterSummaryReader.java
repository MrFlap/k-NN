/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder.kmeansreorder;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;

/**
 * Reads a {@link ClusterSummary} from a .kcs sidecar file.
 */
public class ClusterSummaryReader {

    public static ClusterSummary read(Directory dir, String fileName, String expectedField) throws IOException {
        try (ChecksumIndexInput in = dir.openChecksumInput(fileName)) {
            CodecUtil.checkHeader(in, ClusterSummaryWriter.CODEC_NAME, ClusterSummaryWriter.VERSION_CURRENT, ClusterSummaryWriter.VERSION_CURRENT);

            String fieldName = in.readString();
            if (!fieldName.equals(expectedField)) {
                throw new IOException("Expected field [" + expectedField + "] but found [" + fieldName + "] in " + fileName);
            }

            int k = in.readInt();
            int dimension = in.readInt();
            int metricType = in.readInt();
            int numVectors = in.readInt();

            float[][] centroids = new float[k][dimension];
            float[][] linearSums = new float[k][dimension];
            int[] counts = new int[k];
            float[] sumOfSquares = new float[k];

            for (int i = 0; i < k; i++) {
                counts[i] = in.readInt();
                sumOfSquares[i] = readFloatLE(in);
                for (int d = 0; d < dimension; d++) {
                    centroids[i][d] = readFloatLE(in);
                }
                for (int d = 0; d < dimension; d++) {
                    linearSums[i][d] = readFloatLE(in);
                }
            }

            CodecUtil.checkFooter(in);
            return new ClusterSummary(k, dimension, metricType, numVectors, centroids, linearSums, counts, sumOfSquares);
        }
    }

    private static float readFloatLE(IndexInput in) throws IOException {
        return Float.intBitsToFloat(in.readInt());
    }
}
