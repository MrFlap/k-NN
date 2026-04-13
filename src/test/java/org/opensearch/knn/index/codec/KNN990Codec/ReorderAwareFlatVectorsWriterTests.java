/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.Collections;
import java.util.HashSet;

public class ReorderAwareFlatVectorsWriterTests extends OpenSearchTestCase {

    private SegmentWriteState createWriteState(Directory dir) {
        SegmentInfo segInfo = new SegmentInfo(
            dir, Version.LATEST, Version.LATEST, "_0", 0, false,
            false, org.apache.lucene.codecs.Codec.getDefault(),
            Collections.emptyMap(), StringHelper.randomId(), Collections.emptyMap(), null
        );
        segInfo.setFiles(new HashSet<>());
        FieldInfos fieldInfos = new FieldInfos(new org.apache.lucene.index.FieldInfo[0]);
        return new SegmentWriteState(null, dir, segInfo, fieldInfos, null, IOContext.DEFAULT);
    }

    public void testGetMetaOutputAndGetVectorDataOutputNonNull() throws IOException {
        try (Directory dir = new ByteBuffersDirectory()) {
            SegmentWriteState state = createWriteState(dir);
            try (ReorderAwareFlatVectorsWriter writer =
                     new ReorderAwareFlatVectorsWriter(state, FlatVectorScorerUtil.getLucene99FlatVectorsScorer())) {
                assertNotNull(writer.getMetaOutput());
                assertNotNull(writer.getVectorDataOutput());
            }
        }
    }

    public void testFinishAndCloseProducesValidFiles() throws IOException {
        try (Directory dir = new ByteBuffersDirectory()) {
            SegmentWriteState state = createWriteState(dir);
            try (ReorderAwareFlatVectorsWriter writer =
                     new ReorderAwareFlatVectorsWriter(state, FlatVectorScorerUtil.getLucene99FlatVectorsScorer())) {
                writer.finish();
            }
            // After finish + close, the meta and vec files should exist and have non-zero length
            String[] files = dir.listAll();
            boolean foundMeta = false;
            boolean foundVec = false;
            for (String file : files) {
                if (file.endsWith("." + ReorderAwareFlatVectorsWriter.META_EXTENSION)) {
                    foundMeta = true;
                    assertTrue(dir.fileLength(file) > 0);
                }
                if (file.endsWith("." + ReorderAwareFlatVectorsWriter.VECTOR_DATA_EXTENSION)) {
                    foundVec = true;
                    assertTrue(dir.fileLength(file) > 0);
                }
            }
            assertTrue("Meta file should exist", foundMeta);
            assertTrue("Vector data file should exist", foundVec);
        }
    }
}
