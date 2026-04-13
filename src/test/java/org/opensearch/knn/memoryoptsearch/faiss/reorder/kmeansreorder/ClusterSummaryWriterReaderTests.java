/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder.kmeansreorder;

import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.Collections;
import java.util.HashSet;

public class ClusterSummaryWriterReaderTests extends OpenSearchTestCase {

    public void testRoundTrip() throws IOException {
        int k = 3;
        int dim = 4;
        int metricType = KMeansClusterer.METRIC_L2;
        int numVectors = 100;

        float[][] centroids = new float[k][dim];
        float[][] linearSums = new float[k][dim];
        int[] counts = new int[k];
        float[] sumOfSquares = new float[k];

        for (int i = 0; i < k; i++) {
            counts[i] = 30 + i * 5;
            sumOfSquares[i] = i * 10.5f;
            for (int d = 0; d < dim; d++) {
                centroids[i][d] = i * 10.0f + d;
                linearSums[i][d] = centroids[i][d] * counts[i];
            }
        }

        ClusterSummary original = new ClusterSummary(k, dim, metricType, numVectors, centroids, linearSums, counts, sumOfSquares);

        // Write
        Directory dir = new ByteBuffersDirectory();
        SegmentWriteState state = createSegmentWriteState(dir, "test_field", dim);
        FieldInfo fieldInfo = state.fieldInfos.fieldInfo("test_field");
        String fileName = ClusterSummaryWriter.writeAndGetFileName(state, fieldInfo, original);

        // Read
        ClusterSummary loaded = ClusterSummaryReader.read(dir, fileName, "test_field");

        // Verify
        assertEquals(k, loaded.k);
        assertEquals(dim, loaded.dimension);
        assertEquals(metricType, loaded.metricType);
        assertEquals(numVectors, loaded.numVectors);

        for (int i = 0; i < k; i++) {
            assertEquals(counts[i], loaded.counts[i]);
            assertEquals(sumOfSquares[i], loaded.sumOfSquares[i], 0.001f);
            for (int d = 0; d < dim; d++) {
                assertEquals(centroids[i][d], loaded.centroids[i][d], 0.001f);
                assertEquals(linearSums[i][d], loaded.linearSums[i][d], 0.001f);
            }
        }
    }

    public void testWrongFieldNameThrows() throws IOException {
        ClusterSummary summary = new ClusterSummary(1, 2, 0, 10,
            new float[][] {{1, 2}}, new float[][] {{10, 20}}, new int[] {10}, new float[] {5f});

        Directory dir = new ByteBuffersDirectory();
        SegmentWriteState state = createSegmentWriteState(dir, "field_a", 2);
        FieldInfo fieldInfo = state.fieldInfos.fieldInfo("field_a");
        String fileName = ClusterSummaryWriter.writeAndGetFileName(state, fieldInfo, summary);

        expectThrows(IOException.class, () -> ClusterSummaryReader.read(dir, fileName, "field_b"));
    }

    private SegmentWriteState createSegmentWriteState(Directory dir, String fieldName, int dim) {
        FieldInfo fi = new FieldInfo(
            fieldName, 0, false, false, false,
            IndexOptions.NONE, DocValuesType.NONE, DocValuesSkipIndexType.NONE, -1,
            Collections.emptyMap(), 0, 0, 0,
            dim, VectorEncoding.FLOAT32, VectorSimilarityFunction.EUCLIDEAN,
            false, false
        );
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fi });
        SegmentInfo segInfo = new SegmentInfo(
            dir, Version.LATEST, Version.LATEST, "_0", 100, false,
            false, org.apache.lucene.codecs.Codec.getDefault(),
            Collections.emptyMap(), StringHelper.randomId(), Collections.emptyMap(), null
        );
        segInfo.setFiles(new HashSet<>());
        return new SegmentWriteState(null, dir, segInfo, fieldInfos, null, IOContext.DEFAULT);
    }
}
