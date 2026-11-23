/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec.rabitq;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsWriter;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutorService;

import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.DEFAULT_BEAM_WIDTH;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.DEFAULT_MAX_CONN;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.DEFAULT_NUM_MERGE_WORKER;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.HNSW_GRAPH_THRESHOLD;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.MAXIMUM_BEAM_WIDTH;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.MAXIMUM_MAX_CONN;

/**
 * Lucene HNSW over {@link KNN1040RaBitQFlatVectorsFormat}: the graph is built from raw floats, traversed with the
 * 1-bit RaBitQ coarse estimate, and the traversal's {@code k} (= ef_search) results are rescored with the 7-bit
 * refine estimate, as the ClusterANN tiered RaBitQ scan rescores its coarse shortlist.
 */
public class KNN1040HnswRaBitQVectorsFormat extends KnnVectorsFormat {

    static final String FORMAT_NAME = "KNN1040HnswRaBitQVectorsFormat";
    private static final KNN1040RaBitQFlatVectorsFormat FLAT_FORMAT = new KNN1040RaBitQFlatVectorsFormat();

    private final int maxConn;
    private final int beamWidth;
    private final int numMergeWorkers;
    private final TaskExecutor mergeExec;
    private final int tinySegmentsThreshold;

    public KNN1040HnswRaBitQVectorsFormat() {
        this(DEFAULT_MAX_CONN, DEFAULT_BEAM_WIDTH, DEFAULT_NUM_MERGE_WORKER, null, HNSW_GRAPH_THRESHOLD);
    }

    public KNN1040HnswRaBitQVectorsFormat(
        int maxConn,
        int beamWidth,
        int numMergeWorkers,
        ExecutorService mergeExec,
        int tinySegmentsThreshold
    ) {
        super(FORMAT_NAME);
        if (maxConn <= 0 || maxConn > MAXIMUM_MAX_CONN) {
            throw new IllegalArgumentException(
                "maxConn must be positive and less than or equal to " + MAXIMUM_MAX_CONN + "; maxConn=" + maxConn
            );
        }
        if (beamWidth <= 0 || beamWidth > MAXIMUM_BEAM_WIDTH) {
            throw new IllegalArgumentException(
                "beamWidth must be positive and less than or equal to " + MAXIMUM_BEAM_WIDTH + "; beamWidth=" + beamWidth
            );
        }
        if (numMergeWorkers == 1 && mergeExec != null) {
            throw new IllegalArgumentException("No executor service is needed as we'll use single thread to merge");
        }
        this.maxConn = maxConn;
        this.beamWidth = beamWidth;
        this.numMergeWorkers = numMergeWorkers;
        this.mergeExec = mergeExec != null ? new TaskExecutor(mergeExec) : null;
        this.tinySegmentsThreshold = tinySegmentsThreshold;
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new Lucene99HnswVectorsWriter(
            state,
            maxConn,
            beamWidth,
            FLAT_FORMAT,
            FLAT_FORMAT.fieldsWriter(state),
            numMergeWorkers,
            mergeExec,
            tinySegmentsThreshold
        );
    }

    @Override
    public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        final KNN1040RaBitQFlatVectorsReader flat = FLAT_FORMAT.fieldsReader(state);
        return new Reader(new Lucene99HnswVectorsReader(state, flat), flat);
    }

    @Override
    public int getMaxDimensions(String fieldName) {
        return KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE);
    }

    @Override
    public String toString() {
        return String.format(
            Locale.ROOT,
            "%s(maxConn=%d, beamWidth=%d, tinySegmentsThreshold=%d, flatVectorFormat=%s)",
            getClass().getSimpleName(),
            maxConn,
            beamWidth,
            tinySegmentsThreshold,
            FLAT_FORMAT
        );
    }

    /** Traverses with the coarse tier into a private collector, then hands the refined scores to the caller's. */
    static final class Reader extends KnnVectorsReader implements HnswGraphProvider {
        private final Lucene99HnswVectorsReader hnsw;
        private final KNN1040RaBitQFlatVectorsReader flat;

        Reader(Lucene99HnswVectorsReader hnsw, KNN1040RaBitQFlatVectorsReader flat) {
            this.hnsw = hnsw;
            this.flat = flat;
        }

        @Override
        public void search(String field, float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
            final TopKnnCollector coarse = new TopKnnCollector(
                knnCollector.k(),
                (int) Math.min(Integer.MAX_VALUE, knnCollector.visitLimit()),
                knnCollector.getSearchStrategy()
            );
            hnsw.search(field, target, coarse, acceptDocs);
            // Carries the visit count so a filtered search past its visit limit still falls back to exact search.
            knnCollector.incVisitedCount((int) Math.min(Integer.MAX_VALUE, coarse.visitedCount()));
            flat.refine(field, target, coarse.topDocs().scoreDocs, knnCollector);
        }

        @Override
        public void search(String field, byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
            hnsw.search(field, target, knnCollector, acceptDocs);
        }

        @Override
        public HnswGraph getGraph(String field) throws IOException {
            return hnsw.getGraph(field);
        }

        @Override
        public FloatVectorValues getFloatVectorValues(String field) throws IOException {
            return hnsw.getFloatVectorValues(field);
        }

        @Override
        public ByteVectorValues getByteVectorValues(String field) throws IOException {
            return hnsw.getByteVectorValues(field);
        }

        @Override
        public void checkIntegrity() throws IOException {
            hnsw.checkIntegrity();
        }

        @Override
        public KnnVectorsReader getMergeInstance() throws IOException {
            return hnsw.getMergeInstance();
        }

        @Override
        public void finishMerge() throws IOException {
            hnsw.finishMerge();
        }

        @Override
        public Map<String, Long> getOffHeapByteSize(FieldInfo fieldInfo) {
            return hnsw.getOffHeapByteSize(fieldInfo);
        }

        @Override
        public void close() throws IOException {
            hnsw.close();
        }
    }
}
