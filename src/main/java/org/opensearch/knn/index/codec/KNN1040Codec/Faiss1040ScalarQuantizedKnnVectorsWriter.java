/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.NonNull;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.util.IORunnable;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.util.IOFunction;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.opensearch.knn.index.codec.nativeindex.AbstractNativeEnginesKnnVectorsWriter;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexWriter;

import java.io.IOException;
import java.util.List;

import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;

/**
 * Writer for Faiss SQ vector fields. Unlike {@link org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsWriter}
 * which handles multiple fields, this writer handles exactly one field per format instance
 * (since each SQ field gets its own dedicated format via per-field routing).
 *
 * <p>Write path:
 * <ol>
 *   <li>Flat vectors are written by Lucene's SQ flat writer (.vec + .veq/.vemq files)</li>
 *   <li>HNSW graph is built by native Faiss via {@link NativeIndexWriter} (.faiss file)</li>
 * </ol>
 *
 * <p>No quantization training is needed — Lucene's flat format handles quantization internally.
 */
@Log4j2
class Faiss1040ScalarQuantizedKnnVectorsWriter extends AbstractNativeEnginesKnnVectorsWriter {
    private static final long SHALLOW_SIZE = RamUsageEstimator.shallowSizeOfInstance(Faiss1040ScalarQuantizedKnnVectorsWriter.class);

    private final SegmentWriteState segmentWriteState;
    private final FlatVectorsWriter flatVectorsWriter;
    // Single field — SQ gets a dedicated format per field via BasePerFieldKnnVectorsFormat
    private FlatFieldVectorsWriter<?> fieldWriter;
    private FieldInfo fieldInfo;
    private boolean finished;
    private final IOFunction<SegmentReadState, FlatVectorsReader> quantizedFlatVectorsReaderSupplier;
    private final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory;

    Faiss1040ScalarQuantizedKnnVectorsWriter(
        @NonNull SegmentWriteState segmentWriteState,
        @NonNull FlatVectorsWriter flatVectorsWriter,
        @NonNull IOFunction<SegmentReadState, FlatVectorsReader> quantizedFlatVectorsReaderSupplier,
        @NonNull NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory
    ) {
        this.segmentWriteState = segmentWriteState;
        this.flatVectorsWriter = flatVectorsWriter;
        this.quantizedFlatVectorsReaderSupplier = quantizedFlatVectorsReaderSupplier;
        this.nativeIndexBuildStrategyFactory = nativeIndexBuildStrategyFactory;
    }

    /**
     * Only one field is expected per format instance. Throws if called more than once.
     */
    @Override
    public KnnFieldVectorsWriter<?> addField(final FieldInfo newFieldInfo) throws IOException {
        if (this.fieldWriter != null) {
            throw new IllegalStateException(
                Faiss1040ScalarQuantizedKnnVectorsWriter.class.getSimpleName()
                    + " supports only a single field, but addField was called for ["
                    + newFieldInfo.name
                    + "] after ["
                    + this.fieldInfo.name
                    + "]"
            );
        }
        this.fieldInfo = newFieldInfo;
        this.fieldWriter = flatVectorsWriter.addField(newFieldInfo);
        return fieldWriter;
    }

    /**
     * Flushes flat vectors first (Lucene SQ format), then builds the native HNSW graph.
     *
     * <p>The flat writer is flushed, finished, and closed before the native build so that
     * the .vec and .veb files are fully written and file handles released. The writer then
     * opens a FlatVectorsReader to extract QuantizedByteVectorValues (via reflection on
     * Lucene internals) and passes it to the build strategy. The reader is closed after
     * the native build completes.
     */
    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
        // Flush, finish, and close the flat vectors writer so that the .vec and .veb files
        // are fully written and file handles are released.
        flatVectorsWriter.flush(maxDoc, sortMap);
        flatVectorsWriter.finish();
        IOUtils.close(flatVectorsWriter);

        if (fieldWriter == null) {
            return;
        }

        // Open a reader on the just-written flat files, extract QuantizedByteVectorValues,
        // and pass it to the build strategy. The writer owns the reader lifecycle.
        final FlatVectorsReader flatVectorsReader = openFlatVectorsReader();
        try {
            final QuantizedByteVectorValues quantizedValues = KNN1040ScalarQuantizedUtils.extractQuantizedByteVectorValues(
                flatVectorsReader.getFloatVectorValues(fieldInfo.getName())
            );
            doFlush(
                fieldInfo,
                fieldWriter,
                fieldWriter.getVectors(),
                null,
                null,
                segmentWriteState,
                nativeIndexBuildStrategyFactory,
                quantizedValues
            );
            try {
                writeQuantizationErrorFile(fieldWriter.getVectors(), quantizedValues);
            } catch (Exception | AssertionError e) {
                log.warn("Failed to write quantization error file for field [{}], partial rescoring will be unavailable", fieldInfo.name, e);
            }
        } finally {
            IOUtils.close(flatVectorsReader);
        }
    }

    /**
     * Merges flat vectors first, then builds the native HNSW graph for the merged segment.
     */
    @Override
    public IORunnable mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        // Setting field info
        this.fieldInfo = fieldInfo;

        // Merge, finish, and close the flat writer so that files are readable.
        IORunnable mergeRunnable = flatVectorsWriter.mergeOneField(fieldInfo, mergeState);

        if (mergeRunnable != null) mergeRunnable.run();
        flatVectorsWriter.finish();
        IOUtils.close(flatVectorsWriter);

        // Open a reader on the merged flat files, extract QuantizedByteVectorValues,
        // and pass it to the build strategy. The writer owns the reader lifecycle.
        final FlatVectorsReader flatVectorsReader = openFlatVectorsReader();
        try {
            final FloatVectorValues floatVectorValues = flatVectorsReader.getFloatVectorValues(fieldInfo.getName());
            if (floatVectorValues == null || floatVectorValues.size() == 0) {
                log.debug("No scalar-quantized vectors found for field [{}], skipping native build", fieldInfo.getName());
                return null;
            }
            final QuantizedByteVectorValues quantizedValues = KNN1040ScalarQuantizedUtils.extractQuantizedByteVectorValues(
                floatVectorValues
            );
            doMergeOneField(fieldInfo, mergeState, null, null, segmentWriteState, nativeIndexBuildStrategyFactory, quantizedValues);
            try {
                writeQuantizationErrorFileFromValues(floatVectorValues, quantizedValues);
            } catch (Exception | AssertionError e) {
                log.warn("Failed to write quantization error file for field [{}] during merge, partial rescoring will be unavailable", fieldInfo.name, e);
            }
        } finally {
            IOUtils.close(flatVectorsReader);
        }
        return null;
    }

    @Override
    public void finish() throws IOException {
        if (finished) {
            throw new IllegalStateException(Faiss1040ScalarQuantizedKnnVectorsWriter.class.getSimpleName() + " is already finished");
        }
        finished = true;
        // flatVectorsWriter.finish() and close() are already called in flush/mergeOneField
        // before the native build. No additional finalization needed here.
    }

    /**
     * Computes and writes per-vector Cauchy-Schwarz correction factors from the in-memory vector list (flush path).
     */
    @SuppressWarnings("unchecked")
    private void writeQuantizationErrorFile(List<?> vectors, QuantizedByteVectorValues quantizedValues) throws IOException {
        if (vectors.isEmpty()) {
            return;
        }
        final float[] centroid = quantizedValues.getCentroid();
        final OptimizedScalarQuantizer quantizer = quantizedValues.getQuantizer();
        final int dim = centroid.length;
        final byte[] quantizedScratch = new byte[dim];

        try (QuantizationErrorFile.Writer writer = new QuantizationErrorFile.Writer(segmentWriteState, fieldInfo.name)) {
            for (Object vecObj : vectors) {
                float[] rawVector = (float[]) vecObj;
                float[] vectorCopy = rawVector.clone();
                OptimizedScalarQuantizer.QuantizationResult result = quantizer.scalarQuantize(
                    vectorCopy, quantizedScratch, (byte) 1, centroid
                );
                // vectorCopy is now the residual (x - c), compute its norm
                float residualNormSq = 0f;
                for (int i = 0; i < dim; i++) {
                    residualNormSq += vectorCopy[i] * vectorCopy[i];
                }
                float residualNorm = (float) Math.sqrt(residualNormSq);

                // Dequantize to get reconstructed residual, then compute error norm
                float[] dequantized = new float[dim];
                OptimizedScalarQuantizer.deQuantize(
                    quantizedScratch, dequantized, (byte) 1, result.lowerInterval(), result.upperInterval(), centroid
                );
                // Error = dequantized - rawVector (equivalently, dequantized_residual - residual)
                float errorNormSq = 0f;
                for (int i = 0; i < dim; i++) {
                    float err = dequantized[i] - rawVector[i];
                    errorNormSq += err * err;
                }
                float errorNorm = (float) Math.sqrt(errorNormSq);

                writer.writeVector(errorNorm, residualNorm);
            }
        }
    }

    /**
     * Computes and writes per-vector correction factors from FloatVectorValues (merge path).
     */
    private void writeQuantizationErrorFileFromValues(FloatVectorValues floatVectorValues, QuantizedByteVectorValues quantizedValues)
        throws IOException {
        final float[] centroid = quantizedValues.getCentroid();
        final OptimizedScalarQuantizer quantizer = quantizedValues.getQuantizer();
        final int dim = centroid.length;
        final byte[] quantizedScratch = new byte[dim];

        try (QuantizationErrorFile.Writer writer = new QuantizationErrorFile.Writer(segmentWriteState, fieldInfo.name)) {
            for (int ord = 0; ord < floatVectorValues.size(); ord++) {
                float[] rawVector = floatVectorValues.vectorValue(ord);
                float[] vectorCopy = rawVector.clone();
                OptimizedScalarQuantizer.QuantizationResult result = quantizer.scalarQuantize(
                    vectorCopy, quantizedScratch, (byte) 1, centroid
                );
                float residualNormSq = 0f;
                for (int i = 0; i < dim; i++) {
                    residualNormSq += vectorCopy[i] * vectorCopy[i];
                }
                float residualNorm = (float) Math.sqrt(residualNormSq);

                float[] dequantized = new float[dim];
                OptimizedScalarQuantizer.deQuantize(
                    quantizedScratch, dequantized, (byte) 1, result.lowerInterval(), result.upperInterval(), centroid
                );
                float errorNormSq = 0f;
                for (int i = 0; i < dim; i++) {
                    float err = dequantized[i] - rawVector[i];
                    errorNormSq += err * err;
                }
                float errorNorm = (float) Math.sqrt(errorNormSq);

                writer.writeVector(errorNorm, residualNorm);
            }
        }
    }

    @Override
    public void close() throws IOException {
        // flatVectorsWriter is already closed in flush/mergeOneField.
        // IOUtils.close is safe to call on an already-closed resource.
        IOUtils.close(flatVectorsWriter);
    }

    @Override
    public long ramBytesUsed() {
        return SHALLOW_SIZE + flatVectorsWriter.ramBytesUsed() + (fieldWriter != null ? fieldWriter.ramBytesUsed() : 0);
    }

    /**
     * Opens a FlatVectorsReader scoped to this single field from the already-written flat files.
     */
    private FlatVectorsReader openFlatVectorsReader() throws IOException {
        final SegmentReadState readState = new SegmentReadState(
            segmentWriteState.directory,
            segmentWriteState.segmentInfo,
            new FieldInfos(new FieldInfo[] { fieldInfo }),
            segmentWriteState.context,
            segmentWriteState.segmentSuffix
        );
        return quantizedFlatVectorsReaderSupplier.apply(readState);
    }
}
