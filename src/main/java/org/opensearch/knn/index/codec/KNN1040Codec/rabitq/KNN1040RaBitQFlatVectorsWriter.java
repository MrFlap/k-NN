/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec.rabitq;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOConsumer;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.opensearch.knn.index.codec.KNN1040Codec.rabitq.KNN1040RaBitQFlatVectorsFormat.COARSE_CODEC_NAME;
import static org.opensearch.knn.index.codec.KNN1040Codec.rabitq.KNN1040RaBitQFlatVectorsFormat.COARSE_EXTENSION;
import static org.opensearch.knn.index.codec.KNN1040Codec.rabitq.KNN1040RaBitQFlatVectorsFormat.META_CODEC_NAME;
import static org.opensearch.knn.index.codec.KNN1040Codec.rabitq.KNN1040RaBitQFlatVectorsFormat.META_EXTENSION;
import static org.opensearch.knn.index.codec.KNN1040Codec.rabitq.KNN1040RaBitQFlatVectorsFormat.REFINE_CODEC_NAME;
import static org.opensearch.knn.index.codec.KNN1040Codec.rabitq.KNN1040RaBitQFlatVectorsFormat.REFINE_EXTENSION;
import static org.opensearch.knn.index.codec.KNN1040Codec.rabitq.KNN1040RaBitQFlatVectorsFormat.VERSION_CURRENT;

/**
 * Writes raw floats through the Lucene99 flat writer, then each float field's centroid and tiered RaBitQ records in
 * ordinal order (insertion order on flush, sorted order for a sorted segment, merged order on merge), matching the raw
 * vectors' ordinals.
 */
public final class KNN1040RaBitQFlatVectorsWriter extends FlatVectorsWriter {

    private static final long SHALLOW_RAM_BYTES_USED = RamUsageEstimator.shallowSizeOfInstance(KNN1040RaBitQFlatVectorsWriter.class);

    private final FlatVectorsWriter raw;
    private final IndexOutput meta;
    private final IndexOutput coarse;
    private final IndexOutput refine;
    private final List<Field> fields = new ArrayList<>();
    private boolean finished;

    private record Field(FieldInfo info, FlatFieldVectorsWriter<?> writer) {
    }

    /** One pass over a field's vectors in ordinal order; called twice (centroid, then encode). */
    private interface VectorPass {
        void forEach(IOConsumer<float[]> consumer) throws IOException;
    }

    KNN1040RaBitQFlatVectorsWriter(SegmentWriteState state, FlatVectorsWriter raw) throws IOException {
        super(raw.getFlatVectorScorer());
        this.raw = raw;
        boolean success = false;
        try {
            meta = open(state, META_EXTENSION, META_CODEC_NAME);
            coarse = open(state, COARSE_EXTENSION, COARSE_CODEC_NAME);
            refine = open(state, REFINE_EXTENSION, REFINE_CODEC_NAME);
            success = true;
        } finally {
            if (success == false) {
                IOUtils.closeWhileHandlingException(this);
            }
        }
    }

    private static IndexOutput open(SegmentWriteState state, String extension, String codec) throws IOException {
        final String name = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, extension);
        final IndexOutput out = state.directory.createOutput(name, state.context);
        CodecUtil.writeIndexHeader(out, codec, VERSION_CURRENT, state.segmentInfo.getId(), state.segmentSuffix);
        return out;
    }

    @Override
    public FlatFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
        if (fieldInfo.getVectorEncoding() != VectorEncoding.FLOAT32) {
            throw new IllegalArgumentException(
                "RaBitQ format only supports FLOAT32 encoding, got ["
                    + fieldInfo.getVectorEncoding()
                    + "] for field ["
                    + fieldInfo.name
                    + "]"
            );
        }
        final FlatFieldVectorsWriter<?> writer = raw.addField(fieldInfo);
        fields.add(new Field(fieldInfo, writer));
        return writer;
    }

    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
        for (Field field : fields) {
            @SuppressWarnings("unchecked")
            final List<float[]> vectors = (List<float[]>) field.writer().getVectors();
            final int[] newToOld;
            if (sortMap == null) {
                newToOld = null;
            } else {
                newToOld = new int[field.writer().getDocsWithFieldSet().cardinality()];
                mapOldOrdToNewOrd(field.writer().getDocsWithFieldSet(), sortMap, null, newToOld, null);
            }
            writeField(field.info(), consumer -> {
                for (int ord = 0; ord < vectors.size(); ord++) {
                    consumer.accept(vectors.get(newToOld == null ? ord : newToOld[ord]));
                }
            });
        }
        raw.flush(maxDoc, sortMap);
    }

    @Override
    public void mergeOneFlatVectorField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        raw.mergeOneFlatVectorField(fieldInfo, mergeState);
        writeField(fieldInfo, consumer -> {
            final FloatVectorValues values = MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
            final KnnVectorValues.DocIndexIterator iterator = values.iterator();
            int count = 0;
            for (int doc = iterator.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = iterator.nextDoc()) {
                if ((count++ & 1023) == 0) {
                    mergeState.checkAborted(); // encoding is the costly part of this merge
                }
                consumer.accept(values.vectorValue(iterator.index()));
            }
        });
    }

    private void writeField(FieldInfo fieldInfo, VectorPass vectors) throws IOException {
        final int dimension = fieldInfo.getVectorDimension();
        final boolean cosine = fieldInfo.getVectorSimilarityFunction() == VectorSimilarityFunction.COSINE;
        final float[] scratch = new float[dimension];
        final float[] rotated = new float[dimension];
        // As the ClusterANN data-blind int8 (tiered RaBitQ) backend: Hadamard-rotate every vector, for every metric.
        final HadamardRotation rotation = HadamardRotation.create(dimension, HadamardRotation.DEFAULT_SEED);
        final java.util.function.Function<float[], float[]> prepare = x -> {
            rotation.rotate(cosine ? normalized(x, scratch) : x, rotated);
            return rotated;
        };

        // Pass 1: the segment centroid (in the rotated space), the one "cluster" every residual is taken against.
        final double[] sum = new double[dimension];
        final int[] count = { 0 };
        vectors.forEach(x -> {
            final float[] v = prepare.apply(x);
            for (int d = 0; d < dimension; d++) {
                sum[d] += v[d];
            }
            count[0]++;
        });
        final float[] centroid = new float[dimension];
        if (count[0] > 0) {
            for (int d = 0; d < dimension; d++) {
                centroid[d] = (float) (sum[d] / count[0]);
            }
        }

        // Pass 2: the records.
        final long coarseOffset = coarse.getFilePointer();
        final long refineOffset = refine.getFilePointer();
        final TieredRaBitQRecords.Encoder encoder = new TieredRaBitQRecords.Encoder(centroid);
        vectors.forEach(x -> encoder.write(prepare.apply(x), coarse, refine));

        meta.writeInt(fieldInfo.number);
        meta.writeVInt(dimension);
        meta.writeInt(count[0]);
        meta.writeLong(rotation.seed());
        for (float c : centroid) {
            meta.writeInt(Float.floatToIntBits(c));
        }
        meta.writeVLong(coarseOffset);
        meta.writeVLong(coarse.getFilePointer() - coarseOffset);
        meta.writeVLong(refineOffset);
        meta.writeVLong(refine.getFilePointer() - refineOffset);
    }

    /** {@code x / |x|} into {@code out}, as Lucene's quantized formats store cosine vectors. */
    static float[] normalized(float[] x, float[] out) {
        double n = 0;
        for (float v : x) {
            n += (double) v * v;
        }
        final float inv = n > 0 ? (float) (1.0 / Math.sqrt(n)) : 0f;
        for (int d = 0; d < x.length; d++) {
            out[d] = x[d] * inv;
        }
        return out;
    }

    @Override
    public void finish() throws IOException {
        if (finished) {
            throw new IllegalStateException("already finished");
        }
        finished = true;
        raw.finish();
        meta.writeInt(-1);
        CodecUtil.writeFooter(meta);
        CodecUtil.writeFooter(coarse);
        CodecUtil.writeFooter(refine);
    }

    @Override
    public long ramBytesUsed() {
        return SHALLOW_RAM_BYTES_USED + raw.ramBytesUsed();
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(meta, coarse, refine, raw);
    }
}
