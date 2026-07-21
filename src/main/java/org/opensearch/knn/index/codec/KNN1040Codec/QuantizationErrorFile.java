/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.CompoundDirectory;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;

import java.io.Closeable;
import java.io.IOException;

/**
 * Sidecar file storing per-vector Cauchy-Schwarz correction factors for partial rescoring
 * of quantized radial search.
 *
 * <p>For each vector, two floats are stored:
 * <ul>
 *   <li>{@code ‖E_x‖} — L2 norm of the quantization error: {@code quantized(x - c) - (x - c)}</li>
 *   <li>{@code ‖x - c‖} — L2 norm of the residual from the centroid</li>
 * </ul>
 *
 * <p>These values allow computing a lower bound on the true inner product score without
 * reading the full-precision vector:
 * <pre>
 *   ⟨q, x⟩ ≥ S_approx - (‖q - c‖·‖E_x‖ + ‖e_q + E_q‖·‖x - c‖ + ‖e_q + E_q‖·‖E_x‖)
 * </pre>
 *
 * <p>File format: {@code [codec header][numVectors (int)][errorNorm_0, residualNorm_0, ...][codec footer]}
 * Each norm is stored as a 4-byte float.
 */
public final class QuantizationErrorFile {
    public static final String EXTENSION = "qef";
    static final String CODEC_NAME = "QuantizationErrorFile";
    static final int VERSION_CURRENT = 0;

    private QuantizationErrorFile() {}

    /**
     * Writer for the quantization error file.
     */
    public static class Writer implements Closeable {
        private final IndexOutput output;
        private int count;

        public Writer(SegmentWriteState state, String fieldName) throws IOException {
            String fileName = getFileName(state.segmentInfo.name, state.segmentSuffix, fieldName);
            this.output = state.directory.createOutput(fileName, state.context);
            CodecUtil.writeIndexHeader(output, CODEC_NAME, VERSION_CURRENT, state.segmentInfo.getId(), state.segmentSuffix);
            output.writeInt(0); // placeholder for numVectors (actual count written before footer)
            this.count = 0;
        }

        public void writeVector(float errorNorm, float residualNorm) throws IOException {
            output.writeInt(Float.floatToIntBits(errorNorm));
            output.writeInt(Float.floatToIntBits(residualNorm));
            count++;
        }

        @Override
        public void close() throws IOException {
            // Write the count at the end (before footer) since we can't seek back
            output.writeInt(count);
            CodecUtil.writeFooter(output);
            output.close();
        }
    }

    /**
     * Reader for the quantization error file. Provides random access to per-vector correction factors.
     */
    public static class Reader implements Closeable {
        private final IndexInput input;
        private final int numVectors;
        private final long dataOffset;

        public Reader(SegmentReadState state, String fieldName) throws IOException {
            this(
                state.directory.openInput(getFileName(state.segmentInfo.name, state.segmentSuffix, fieldName), state.context),
                state.segmentInfo.getId(),
                state.segmentSuffix
            );
        }

        Reader(IndexInput input, byte[] segmentId, String segmentSuffix) throws IOException {
            this.input = input;
            CodecUtil.checkIndexHeader(input, CODEC_NAME, VERSION_CURRENT, VERSION_CURRENT, segmentId, segmentSuffix);
            input.readInt(); // skip numVectors placeholder
            this.dataOffset = input.getFilePointer();
            // Read count from the end (just before the 16-byte footer)
            long fileLength = input.length();
            long countOffset = fileLength - CodecUtil.footerLength() - 4;
            if (countOffset < dataOffset) {
                this.numVectors = 0;
                return;
            }
            input.seek(countOffset);
            this.numVectors = input.readInt();
        }

        public int size() {
            return numVectors;
        }

        public float getErrorNorm(int ord) throws IOException {
            if (ord < 0 || ord >= numVectors) return 0f;
            input.seek(dataOffset + (long) ord * 8);
            return Float.intBitsToFloat(input.readInt());
        }

        public float getResidualNorm(int ord) throws IOException {
            if (ord < 0 || ord >= numVectors) return 0f;
            input.seek(dataOffset + (long) ord * 8 + 4);
            return Float.intBitsToFloat(input.readInt());
        }

        @Override
        public void close() throws IOException {
            input.close();
        }
    }

    /**
     * Reader that also owns a CompoundDirectory, closing it when the reader is closed.
     */
    private static class CfsReader extends Reader {
        private final CompoundDirectory cfsDir;

        CfsReader(IndexInput input, byte[] segmentId, String segmentSuffix, CompoundDirectory cfsDir) throws IOException {
            super(input, segmentId, segmentSuffix);
            this.cfsDir = cfsDir;
        }

        @Override
        public void close() throws IOException {
            try {
                super.close();
            } finally {
                cfsDir.close();
            }
        }
    }

    /**
     * Opens a Reader by searching for the .qef file matching the given field.
     * Returns null if no .qef file exists for this segment/field (i.e., not a quantized index).
     */
    public static Reader openIfExists(SegmentReader segmentReader, String fieldName) throws IOException {
        final SegmentInfo segInfo = segmentReader.getSegmentInfo().info;
        final Directory rawDir = segmentReader.directory();
        final String segName = segInfo.name;
        final String[] candidates = {
            segName + "_Faiss1040ScalarQuantizedKnnVectorsFormat_0_" + fieldName + "." + EXTENSION,
            segName + "_" + fieldName + "." + EXTENSION,
        };

        // Try the raw directory first (non-compound segments)
        for (String candidate : candidates) {
            try {
                final IndexInput input = rawDir.openInput(candidate, IOContext.DEFAULT);
                return new Reader(input, segInfo.getId(), extractSegmentSuffix(candidate, segName, fieldName));
            } catch (java.io.FileNotFoundException | java.nio.file.NoSuchFileException e) {
                // Try next candidate
            }
        }

        // If segment uses compound file, open CFS and look inside
        if (segInfo.getUseCompoundFile()) {
            final CompoundDirectory cfsDir = segInfo.getCodec().compoundFormat().getCompoundReader(rawDir, segInfo);
            try {
                for (String candidate : candidates) {
                    try {
                        final IndexInput input = cfsDir.openInput(candidate, IOContext.DEFAULT);
                        return new CfsReader(input, segInfo.getId(), extractSegmentSuffix(candidate, segName, fieldName), cfsDir);
                    } catch (java.io.FileNotFoundException | java.nio.file.NoSuchFileException e) {
                        // Try next candidate
                    }
                }
            } catch (Exception e) {
                cfsDir.close();
                throw e;
            }
            cfsDir.close();
        }
        return null;
    }

    private static String extractSegmentSuffix(String fileName, String segName, String fieldName) {
        // fileName: {segName}_{suffix}_{fieldName}.qef or {segName}_{fieldName}.qef
        String withoutExt = fileName.substring(0, fileName.length() - EXTENSION.length() - 1);
        String afterSegName = withoutExt.substring(segName.length() + 1);
        if (afterSegName.equals(fieldName)) {
            return "";
        }
        // afterSegName = "{suffix}_{fieldName}"
        return afterSegName.substring(0, afterSegName.length() - fieldName.length() - 1);
    }

    private static String getFileName(String segmentName, String segmentSuffix, String fieldName) {
        if (segmentSuffix != null && !segmentSuffix.isEmpty()) {
            return segmentName + "_" + segmentSuffix + "_" + fieldName + "." + EXTENSION;
        }
        return segmentName + "_" + fieldName + "." + EXTENSION;
    }
}
