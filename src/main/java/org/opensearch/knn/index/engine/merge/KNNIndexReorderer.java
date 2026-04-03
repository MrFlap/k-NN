/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.merge;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.misc.index.BpVectorReorderer;
import org.apache.lucene.misc.index.IndexReorderer;
import org.apache.lucene.store.Directory;

import java.io.IOException;
import java.util.concurrent.Executor;

/**
 * An {@link IndexReorderer} that discovers the first float vector field in a segment
 * and delegates to {@link BpVectorReorderer} for bipartite graph partitioning reorder.
 * <p>
 * This allows using {@link BPReorderingMergePolicy} without knowing the vector field
 * name at merge policy creation time.
 */
@Log4j2
public class KNNIndexReorderer implements IndexReorderer {

    @Override
    public Sorter.DocMap computeDocMap(CodecReader reader, Directory tempDir, Executor executor) throws IOException {
        String vectorField = findFirstFloatVectorField(reader.getFieldInfos());
        if (vectorField == null) {
            log.debug("No float vector field found in segment {}, skipping reorder", reader);
            return null;
        }

        log.debug("Reordering segment by vector field [{}] with {} docs", vectorField, reader.numDocs());
        BpVectorReorderer reorderer = new BpVectorReorderer(vectorField);
        return reorderer.computeDocMap(reader, tempDir, executor);
    }

    private static String findFirstFloatVectorField(FieldInfos fieldInfos) {
        for (FieldInfo fi : fieldInfos) {
            if (fi.hasVectorValues() && fi.getVectorEncoding() == VectorEncoding.FLOAT32) {
                return fi.name;
            }
        }
        return null;
    }
}
