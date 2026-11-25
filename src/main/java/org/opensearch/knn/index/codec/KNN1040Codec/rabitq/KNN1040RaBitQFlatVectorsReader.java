/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec.rabitq;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.index.codec.KNN1040Codec.rabitq.KNN1040RaBitQFlatVectorsFormat.COARSE_CODEC_NAME;
import static org.opensearch.knn.index.codec.KNN1040Codec.rabitq.KNN1040RaBitQFlatVectorsFormat.COARSE_EXTENSION;
import static org.opensearch.knn.index.codec.KNN1040Codec.rabitq.KNN1040RaBitQFlatVectorsFormat.META_CODEC_NAME;
import static org.opensearch.knn.index.codec.KNN1040Codec.rabitq.KNN1040RaBitQFlatVectorsFormat.META_EXTENSION;
import static org.opensearch.knn.index.codec.KNN1040Codec.rabitq.KNN1040RaBitQFlatVectorsFormat.REFINE_CODEC_NAME;
import static org.opensearch.knn.index.codec.KNN1040Codec.rabitq.KNN1040RaBitQFlatVectorsFormat.REFINE_EXTENSION;
import static org.opensearch.knn.index.codec.KNN1040Codec.rabitq.KNN1040RaBitQFlatVectorsFormat.VERSION_CURRENT;
import static org.opensearch.knn.index.codec.KNN1040Codec.rabitq.KNN1040RaBitQFlatVectorsFormat.VERSION_ROTATION;
import static org.opensearch.knn.index.codec.KNN1040Codec.rabitq.KNN1040RaBitQFlatVectorsFormat.VERSION_START;

/**
 * Reads raw floats through the Lucene99 flat reader and scores with the tiered RaBitQ records: {@link
 * #getRandomVectorScorer} returns the coarse (1-bit code x 4-bit query) scorer HNSW traverses with, and {@link #refine}
 * rescores a traversal's results from all 7 bits (7-bit code x 8-bit query). Graph build and merge score raw floats
 * ({@link #getFlatVectorScorer}).
 */
public final class KNN1040RaBitQFlatVectorsReader extends FlatVectorsReader {

    private static final long SHALLOW_SIZE = RamUsageEstimator.shallowSizeOfInstance(KNN1040RaBitQFlatVectorsReader.class);
    private static final VarHandle FLOAT_LE = MethodHandles.byteArrayViewVarHandle(float[].class, ByteOrder.LITTLE_ENDIAN);
    private static final VarHandle INT_LE = MethodHandles.byteArrayViewVarHandle(int[].class, ByteOrder.LITTLE_ENDIAN);

    private final FlatVectorsReader raw;
    private final IndexInput coarse;
    private final IndexInput refine;
    private final Map<String, Entry> fields = new HashMap<>();

    private record Entry(int dimension, int size, VectorSimilarityFunction similarity, HadamardRotation rotation, float[] centroid,
        long coarseOffset, long coarseLength, long refineOffset, long refineLength) {
    }

    KNN1040RaBitQFlatVectorsReader(SegmentReadState state, FlatVectorsReader raw) throws IOException {
        this.raw = raw;
        IndexInput coarseIn = null;
        IndexInput refineIn = null;
        boolean success = false;
        try {
            readMeta(state);
            coarseIn = openData(state, COARSE_EXTENSION, COARSE_CODEC_NAME);
            refineIn = openData(state, REFINE_EXTENSION, REFINE_CODEC_NAME);
            success = true;
        } finally {
            if (success == false) {
                IOUtils.closeWhileHandlingException(coarseIn, refineIn, raw);
            }
        }
        this.coarse = coarseIn;
        this.refine = refineIn;
    }

    private void readMeta(SegmentReadState state) throws IOException {
        final String name = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, META_EXTENSION);
        try (ChecksumIndexInput meta = state.directory.openChecksumInput(name)) {
            Throwable priorE = null;
            try {
                final int version = CodecUtil.checkIndexHeader(
                    meta,
                    META_CODEC_NAME,
                    VERSION_START,
                    VERSION_CURRENT,
                    state.segmentInfo.getId(),
                    state.segmentSuffix
                );
                for (int number = meta.readInt(); number != -1; number = meta.readInt()) {
                    final FieldInfo info = state.fieldInfos.fieldInfo(number);
                    if (info == null) {
                        throw new CorruptIndexException("Invalid field number: " + number, meta);
                    }
                    final int dimension = meta.readVInt();
                    final int size = meta.readInt();
                    final HadamardRotation rotation = version >= VERSION_ROTATION
                        ? HadamardRotation.create(dimension, meta.readLong())
                        : null;
                    final float[] centroid = new float[dimension];
                    for (int d = 0; d < dimension; d++) {
                        centroid[d] = Float.intBitsToFloat(meta.readInt());
                    }
                    fields.put(
                        info.name,
                        new Entry(
                            dimension,
                            size,
                            info.getVectorSimilarityFunction(),
                            rotation,
                            centroid,
                            meta.readVLong(),
                            meta.readVLong(),
                            meta.readVLong(),
                            meta.readVLong()
                        )
                    );
                }
            } catch (Throwable t) {
                priorE = t;
            } finally {
                CodecUtil.checkFooter(meta, priorE);
            }
        }
    }

    private static IndexInput openData(SegmentReadState state, String extension, String codec) throws IOException {
        final String name = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, extension);
        final IndexInput in = state.directory.openInput(name, state.context);
        boolean success = false;
        try {
            CodecUtil.checkIndexHeader(in, codec, VERSION_START, VERSION_CURRENT, state.segmentInfo.getId(), state.segmentSuffix);
            CodecUtil.retrieveChecksum(in);
            success = true;
            return in;
        } finally {
            if (success == false) {
                IOUtils.closeWhileHandlingException(in);
            }
        }
    }

    private TieredRaBitQQuery query(Entry entry, float[] target) {
        float[] q = entry.similarity() == VectorSimilarityFunction.COSINE
            ? KNN1040RaBitQFlatVectorsWriter.normalized(target, new float[target.length])
            : target;
        if (entry.rotation() != null) {
            final float[] rotated = new float[q.length];
            entry.rotation().rotate(q, rotated);
            q = rotated;
        }
        return new TieredRaBitQQuery(q, entry.centroid(), entry.similarity());
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(String field, float[] target) throws IOException {
        final Entry entry = fields.get(field);
        final FloatVectorValues values = raw.getFloatVectorValues(field);
        if (entry == null || values == null) {
            return raw.getRandomVectorScorer(field, target);
        }
        return new CoarseScorer(
            values,
            coarse.slice("rbq-coarse", entry.coarseOffset(), entry.coarseLength()),
            query(entry, target),
            entry
        );
    }

    /**
     * Rescores {@code hits} (doc ids from a traversal of this segment) from all bits of their codes and collects the
     * rescored estimates into {@code collector}.
     */
    void refine(String field, float[] target, ScoreDoc[] hits, KnnCollector collector) throws IOException {
        final Entry entry = fields.get(field);
        if (hits.length == 0) {
            return;
        }
        if (entry == null) {
            for (ScoreDoc hit : hits) {
                collector.collect(hit.doc, hit.score);
            }
            return;
        }
        final TieredRaBitQQuery query = query(entry, target);
        final int dimension = entry.dimension();
        final int coarseBytes = TieredRaBitQRecords.coarseBytes(dimension);
        final int refineBytes = TieredRaBitQRecords.refineBytes(dimension);
        final int bitBytes = TieredRaBitQRecords.bitBytes(dimension);

        // Doc ids back to ordinals: walk the raw values' iterator over the hits in doc order.
        final ScoreDoc[] byDoc = hits.clone();
        Arrays.sort(byDoc, Comparator.comparingInt(h -> h.doc));
        final KnnVectorValues.DocIndexIterator iterator = raw.getFloatVectorValues(field).iterator();
        final int m = byDoc.length;
        final byte[] recs = new byte[m * refineBytes];
        final float[] resNorm = new float[m];
        final IndexInput refineIn = refine.slice("rbq-refine", entry.refineOffset(), entry.refineLength());
        final IndexInput coarseIn = coarse.slice("rbq-coarse", entry.coarseOffset(), entry.coarseLength());
        for (int r = 0; r < m; r++) {
            final int doc = byDoc[r].doc;
            if (iterator.docID() != doc && iterator.advance(doc) != doc) {
                throw new IllegalStateException("doc " + doc + " has no vector in field " + field);
            }
            final long ord = iterator.index();
            refineIn.seek(ord * refineBytes);
            refineIn.readBytes(recs, r * refineBytes, refineBytes);
            coarseIn.seek(ord * coarseBytes + bitBytes);
            resNorm[r] = Float.intBitsToFloat(coarseIn.readInt());
        }
        final int[] dots = new int[m];
        TieredRaBitQKernel.refineDots(recs, refineBytes, m, dimension, query.q8, dots);
        for (int r = 0; r < m; r++) {
            final int cp = r * refineBytes + dimension;
            final float gR = (float) FLOAT_LE.get(recs, cp);
            final float oR = (float) FLOAT_LE.get(recs, cp + Float.BYTES);
            collector.collect(byDoc[r].doc, query.refine(dots[r], resNorm[r], gR, oR));
        }
    }

    /** HNSW traversal scorer: the coarse estimate from the top bit of each code. */
    private static final class CoarseScorer extends RandomVectorScorer.AbstractRandomVectorScorer {
        private final IndexInput in;
        private final TieredRaBitQQuery query;
        private final int bitBytes;
        private final int rowBytes;
        private final byte[] row;
        private final byte[] plane;
        private byte[] rows = new byte[0];
        private byte[] bits = new byte[0];
        private int[] dots = new int[0];

        CoarseScorer(FloatVectorValues values, IndexInput in, TieredRaBitQQuery query, Entry entry) {
            super(values);
            this.in = in;
            this.query = query;
            this.bitBytes = TieredRaBitQRecords.bitBytes(entry.dimension());
            this.rowBytes = TieredRaBitQRecords.coarseBytes(entry.dimension());
            this.row = new byte[rowBytes];
            this.plane = new byte[bitBytes];
        }

        @Override
        public float score(int node) throws IOException {
            in.seek((long) node * rowBytes);
            in.readBytes(row, 0, rowBytes);
            System.arraycopy(row, 0, plane, 0, bitBytes);
            return score(row, 0, (int) VectorUtil.int4BitDotProduct(query.planes, plane));
        }

        /** One native call per neighbor list, as the POC makes one per cluster. */
        @Override
        public float bulkScore(int[] nodes, float[] scores, int numNodes) throws IOException {
            if (rows.length < numNodes * rowBytes) {
                rows = new byte[numNodes * rowBytes];
                bits = new byte[numNodes * bitBytes];
                dots = new int[numNodes];
            }
            for (int i = 0; i < numNodes; i++) {
                in.seek((long) nodes[i] * rowBytes);
                in.readBytes(rows, i * rowBytes, rowBytes);
                System.arraycopy(rows, i * rowBytes, bits, i * bitBytes, bitBytes);
            }
            TieredRaBitQKernel.coarseDots(query.planes, bits, bitBytes, numNodes, dots);
            float max = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < numNodes; i++) {
                scores[i] = score(rows, i * rowBytes, dots[i]);
                max = Math.max(max, scores[i]);
            }
            return max;
        }

        private float score(byte[] buf, int offset, int dot) {
            final int p = offset + bitBytes;
            return query.coarse(
                dot,
                (int) INT_LE.get(buf, p + 4 * Float.BYTES),
                (float) FLOAT_LE.get(buf, p),
                (float) FLOAT_LE.get(buf, p + Float.BYTES),
                (float) FLOAT_LE.get(buf, p + 2 * Float.BYTES),
                (float) FLOAT_LE.get(buf, p + 3 * Float.BYTES)
            );
        }
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(String field, byte[] target) throws IOException {
        return raw.getRandomVectorScorer(field, target);
    }

    @Override
    public FlatVectorsScorer getFlatVectorScorer(String field) throws IOException {
        return raw.getFlatVectorScorer(field);
    }

    @Override
    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
        return raw.getFloatVectorValues(field);
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) throws IOException {
        return raw.getByteVectorValues(field);
    }

    @Override
    public void checkIntegrity() throws IOException {
        raw.checkIntegrity();
        CodecUtil.checksumEntireFile(coarse);
        CodecUtil.checksumEntireFile(refine);
    }

    @Override
    public Map<String, Long> getOffHeapByteSize(FieldInfo fieldInfo) {
        final Entry entry = fields.get(fieldInfo.name);
        final Map<String, Long> rawSizes = raw.getOffHeapByteSize(fieldInfo);
        if (entry == null) {
            return rawSizes;
        }
        return KnnVectorsReader.mergeOffHeapByteSizeMaps(
            rawSizes,
            Map.of(COARSE_EXTENSION, entry.coarseLength(), REFINE_EXTENSION, entry.refineLength())
        );
    }

    @Override
    public long ramBytesUsed() {
        long centroids = 0;
        for (Entry entry : fields.values()) {
            centroids += RamUsageEstimator.sizeOf(entry.centroid());
        }
        return SHALLOW_SIZE + raw.ramBytesUsed() + centroids;
    }

    @Override
    public void finishMerge() throws IOException {
        raw.finishMerge();
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(raw, coarse, refine);
    }
}
