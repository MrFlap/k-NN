/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.clumping;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Writes a .clump sidecar file that stores marker-to-hidden vector mappings with
 * all vector data inline for sequential reads during expansion.
 * See {@link ClumpFileFormat} for the binary layout.
 * <p>
 * The writer reads marker-to-hidden assignments from a flat temporary file
 * ({@code .clumpassign}) where the int at offset {@code 4 * docId} is the marker
 * doc ID for that vector. Markers point to themselves. Hidden vector data is read
 * from a separate spill file ({@code .clumptmp}) that contains entries in encounter
 * order, each of fixed size ({@code 4 + vectorBytes} bytes — for SQ the vectorBytes
 * includes the packed correction factors).
 * <p>
 * To avoid O(H) heap, the writer uses a two-pass approach with O(M) auxiliary arrays:
 * <ol>
 *   <li>Pass 1: count hidden vectors per marker and compute byte offsets.</li>
 *   <li>Write the clump file header, marker table, and marker vectors sequentially.</li>
 *   <li>Pass 2: scan the spill file once, writing each hidden entry directly to its
 *       correct position in a temp clump-data file via {@link RandomAccessFile}.
 *       Per-marker cursors ({@code int[M]}) track the next write slot.</li>
 *   <li>Append the temp clump-data file contents to the clump {@link IndexOutput}.</li>
 * </ol>
 */
@Log4j2
public final class ClumpFileWriter {

    private ClumpFileWriter() {}

    /**
     * Writes a FLOAT/BYTE/FP16 clump file. For the SQ_1BIT path use
     * {@link #writeClumpFileSq}.
     */
    public static void writeClumpFile(
        SegmentWriteState state,
        String fieldName,
        int dimension,
        byte vectorDataType,
        List<Integer> markerDocIds,
        List<Object> markerVectors,
        IndexInput assignInput,
        IndexInput tempInput,
        int totalHidden
    ) throws IOException {
        if (vectorDataType == ClumpFileFormat.VECTOR_TYPE_SQ_1BIT) {
            throw new IllegalArgumentException("SQ_1BIT clump files must be written via writeClumpFileSq");
        }
        writeClumpFileInternal(
            state, fieldName, dimension, vectorDataType,
            0, null, 0.0f,
            markerDocIds, markerVectors,
            assignInput, tempInput, totalHidden
        );
    }

    /**
     * Writes a SQ_1BIT clump file. Marker vectors must be {@link SqVectorEntry} instances.
     * The quantizer's {@code centroid} and {@code centroidDp} are persisted in the header so the
     * query-time reader can quantize a fresh query vector without reopening the segment's
     * {@code QuantizedByteVectorValues}.
     */
    public static void writeClumpFileSq(
        SegmentWriteState state,
        String fieldName,
        int dimension,
        int quantizedVecBytes,
        float[] centroid,
        float centroidDp,
        List<Integer> markerDocIds,
        List<Object> markerVectors,
        IndexInput assignInput,
        IndexInput tempInput,
        int totalHidden
    ) throws IOException {
        writeClumpFileInternal(
            state, fieldName, dimension, ClumpFileFormat.VECTOR_TYPE_SQ_1BIT,
            quantizedVecBytes, centroid, centroidDp,
            markerDocIds, markerVectors,
            assignInput, tempInput, totalHidden
        );
    }

    private static void writeClumpFileInternal(
        SegmentWriteState state,
        String fieldName,
        int dimension,
        byte vectorDataType,
        int quantizedVecBytes,
        float[] centroid,
        float centroidDp,
        List<Integer> markerDocIds,
        List<Object> markerVectors,
        IndexInput assignInput,
        IndexInput tempInput,
        int totalHidden
    ) throws IOException {
        int numMarkers = markerDocIds.size();
        int vectorSize = ClumpFileFormat.vectorBytes(dimension, vectorDataType, quantizedVecBytes);
        // Spill file uses an interleaved format (docId + vector entry per hidden vector)
        int spillEntrySize = Integer.BYTES + vectorSize;

        // Build marker doc ID → marker index lookup
        Map<Integer, Integer> markerDocIdToIndex = new HashMap<>(numMarkers);
        for (int i = 0; i < numMarkers; i++) {
            markerDocIdToIndex.put(markerDocIds.get(i), i);
        }

        // Pass 1: count numHiddenPerMarker — O(M) heap
        int[] numHiddenPerMarker = new int[numMarkers];
        for (int h = 0; h < totalHidden; h++) {
            long entryOffset = (long) h * spillEntrySize;
            tempInput.seek(entryOffset);
            int hiddenDocId = tempInput.readInt();

            assignInput.seek((long) hiddenDocId * Integer.BYTES);
            int markerDocId = assignInput.readInt();

            Integer markerIndex = markerDocIdToIndex.get(markerDocId);
            if (markerIndex != null) {
                numHiddenPerMarker[markerIndex]++;
            }
        }

        // Per-marker layout in temp clump-data file: [markerVector][docIdBlock][vectorBlock]
        long[] docIdBlockOffset = new long[numMarkers];
        long[] vectorBlockOffset = new long[numMarkers];
        long runningOffset = 0;
        for (int i = 0; i < numMarkers; i++) {
            runningOffset += vectorSize; // marker vector
            docIdBlockOffset[i] = runningOffset;
            runningOffset += (long) numHiddenPerMarker[i] * Integer.BYTES;
            vectorBlockOffset[i] = runningOffset;
            runningOffset += (long) numHiddenPerMarker[i] * vectorSize;
        }
        long totalClumpDataSize = runningOffset;

        String clumpFileName = buildClumpFileName(state.segmentInfo.name, fieldName);
        Path clumpDataTempPath = Files.createTempFile("clumpdata", ".tmp");

        try {
            int[] docIdCursors = new int[numMarkers];
            int[] vectorCursors = new int[numMarkers];

            try (RandomAccessFile raf = new RandomAccessFile(clumpDataTempPath.toFile(), "rw")) {
                raf.setLength(totalClumpDataSize);

                // Marker vectors at the start of each marker's clump data region
                for (int i = 0; i < numMarkers; i++) {
                    long mvOffset = docIdBlockOffset[i] - vectorSize;
                    raf.seek(mvOffset);
                    writeVectorToRaf(raf, markerVectors.get(i), vectorDataType, vectorSize);
                }

                // Scatter hidden entries into separate doc-ID and vector blocks
                byte[] vecBuf = new byte[vectorSize];
                byte[] docIdBuf = new byte[Integer.BYTES];
                for (int h = 0; h < totalHidden; h++) {
                    long spillOffset = (long) h * spillEntrySize;
                    tempInput.seek(spillOffset);
                    int hiddenDocId = tempInput.readInt();

                    assignInput.seek((long) hiddenDocId * Integer.BYTES);
                    int markerDocId = assignInput.readInt();

                    Integer markerIndex = markerDocIdToIndex.get(markerDocId);
                    if (markerIndex == null) {
                        continue;
                    }

                    long docIdDest = docIdBlockOffset[markerIndex]
                        + (long) docIdCursors[markerIndex] * Integer.BYTES;
                    docIdBuf[0] = (byte) hiddenDocId;
                    docIdBuf[1] = (byte) (hiddenDocId >> 8);
                    docIdBuf[2] = (byte) (hiddenDocId >> 16);
                    docIdBuf[3] = (byte) (hiddenDocId >> 24);
                    raf.seek(docIdDest);
                    raf.write(docIdBuf);
                    docIdCursors[markerIndex]++;

                    long vecDest = vectorBlockOffset[markerIndex]
                        + (long) vectorCursors[markerIndex] * vectorSize;
                    tempInput.seek(spillOffset + Integer.BYTES); // skip docId
                    tempInput.readBytes(vecBuf, 0, vectorSize);
                    raf.seek(vecDest);
                    raf.write(vecBuf);
                    vectorCursors[markerIndex]++;
                }
            }

            // Assemble the final clump file
            try (IndexOutput output = state.directory.createOutput(clumpFileName, state.context)) {
                // Common header
                output.writeInt(numMarkers);
                output.writeInt(dimension);
                output.writeByte(vectorDataType);

                // SQ-specific header extension
                if (vectorDataType == ClumpFileFormat.VECTOR_TYPE_SQ_1BIT) {
                    output.writeInt(quantizedVecBytes);
                    output.writeInt(Float.floatToRawIntBits(centroidDp));
                    for (float v : centroid) {
                        output.writeInt(Float.floatToRawIntBits(v));
                    }
                }

                // Marker table — clumpDataOffset points to the start of each marker's data
                long clumpDataBase = ClumpFileFormat.clumpDataStart(numMarkers, vectorDataType, dimension);
                long currentOffset = clumpDataBase;
                for (int i = 0; i < numMarkers; i++) {
                    output.writeInt(markerDocIds.get(i));
                    output.writeInt(numHiddenPerMarker[i]);
                    output.writeLong(currentOffset);
                    currentOffset += vectorSize
                        + (long) numHiddenPerMarker[i] * Integer.BYTES
                        + (long) numHiddenPerMarker[i] * vectorSize;
                }

                appendFileToOutput(clumpDataTempPath, output);
                CodecUtil.writeFooter(output);
            }

            log.debug(
                "Wrote clump file {} with {} markers, {} hidden vectors, dim={}, type={}",
                clumpFileName, numMarkers, totalHidden, dimension,
                typeLabel(vectorDataType)
            );
        } finally {
            Files.deleteIfExists(clumpDataTempPath);
        }
    }

    private static String typeLabel(byte t) {
        switch (t) {
            case ClumpFileFormat.VECTOR_TYPE_FLOAT: return "float";
            case ClumpFileFormat.VECTOR_TYPE_BYTE:  return "byte";
            case ClumpFileFormat.VECTOR_TYPE_FP16:  return "fp16";
            case ClumpFileFormat.VECTOR_TYPE_SQ_1BIT: return "sq1bit";
            default: return "type" + t;
        }
    }

    /**
     * Writes a single hidden entry (docId + vector) to a temp IndexOutput for spilling.
     * {@code vector} must match the {@code vectorDataType}: {@code float[]} for FLOAT/FP16,
     * {@code byte[]} for BYTE, {@link SqVectorEntry} for SQ_1BIT.
     *
     * @return the file offset where this entry was written
     */
    public static long writeHiddenEntryToTemp(IndexOutput tempOutput, int docId, Object vector, byte vectorDataType) throws IOException {
        long offset = tempOutput.getFilePointer();
        tempOutput.writeInt(docId);
        writeVector(tempOutput, vector, vectorDataType);
        return offset;
    }

    private static void writeVector(IndexOutput output, Object vector, byte vectorDataType) throws IOException {
        if (vectorDataType == ClumpFileFormat.VECTOR_TYPE_FLOAT) {
            float[] fv = (float[]) vector;
            for (float v : fv) {
                output.writeInt(Float.floatToIntBits(v));
            }
        } else if (vectorDataType == ClumpFileFormat.VECTOR_TYPE_FP16) {
            float[] fv = (float[]) vector;
            for (float v : fv) {
                output.writeShort(Float.floatToFloat16(v));
            }
        } else if (vectorDataType == ClumpFileFormat.VECTOR_TYPE_SQ_1BIT) {
            SqVectorEntry sq = (SqVectorEntry) vector;
            output.writeBytes(sq.code, sq.code.length);
            output.writeInt(Float.floatToRawIntBits(sq.lowerInterval));
            output.writeInt(Float.floatToRawIntBits(sq.upperInterval));
            output.writeInt(Float.floatToRawIntBits(sq.additionalCorrection));
            output.writeInt(sq.quantizedComponentSum);
        } else {
            byte[] bv = (byte[]) vector;
            output.writeBytes(bv, bv.length);
        }
    }

    /**
     * Writes a vector to a RandomAccessFile in little-endian format (matching Lucene's
     * DataOutput convention).
     *
     * @param vectorSize expected total bytes for the entry; used for safety checks
     */
    private static void writeVectorToRaf(RandomAccessFile raf, Object vector, byte vectorDataType, int vectorSize) throws IOException {
        if (vectorDataType == ClumpFileFormat.VECTOR_TYPE_FLOAT) {
            float[] fv = (float[]) vector;
            byte[] buf = new byte[fv.length * Float.BYTES];
            for (int i = 0; i < fv.length; i++) {
                int bits = Float.floatToIntBits(fv[i]);
                int off = i * Float.BYTES;
                buf[off] = (byte) bits;
                buf[off + 1] = (byte) (bits >> 8);
                buf[off + 2] = (byte) (bits >> 16);
                buf[off + 3] = (byte) (bits >> 24);
            }
            raf.write(buf);
        } else if (vectorDataType == ClumpFileFormat.VECTOR_TYPE_FP16) {
            float[] fv = (float[]) vector;
            byte[] buf = new byte[fv.length * Short.BYTES];
            for (int i = 0; i < fv.length; i++) {
                short fp16 = Float.floatToFloat16(fv[i]);
                int off = i * Short.BYTES;
                buf[off] = (byte) fp16;
                buf[off + 1] = (byte) (fp16 >> 8);
            }
            raf.write(buf);
        } else if (vectorDataType == ClumpFileFormat.VECTOR_TYPE_SQ_1BIT) {
            SqVectorEntry sq = (SqVectorEntry) vector;
            byte[] buf = new byte[vectorSize];
            int off = 0;
            System.arraycopy(sq.code, 0, buf, off, sq.code.length);
            off += sq.code.length;
            off = writeFloatLE(buf, off, sq.lowerInterval);
            off = writeFloatLE(buf, off, sq.upperInterval);
            off = writeFloatLE(buf, off, sq.additionalCorrection);
            writeIntLE(buf, off, sq.quantizedComponentSum);
            raf.write(buf);
        } else {
            byte[] bv = (byte[]) vector;
            raf.write(bv);
        }
    }

    private static int writeFloatLE(byte[] buf, int off, float v) {
        return writeIntLE(buf, off, Float.floatToRawIntBits(v));
    }

    private static int writeIntLE(byte[] buf, int off, int bits) {
        buf[off]     = (byte) bits;
        buf[off + 1] = (byte) (bits >>> 8);
        buf[off + 2] = (byte) (bits >>> 16);
        buf[off + 3] = (byte) (bits >>> 24);
        return off + 4;
    }

    private static void appendFileToOutput(Path filePath, IndexOutput output) throws IOException {
        byte[] buffer = new byte[64 * 1024];
        try (java.io.FileInputStream fis = new java.io.FileInputStream(filePath.toFile())) {
            int bytesRead;
            while ((bytesRead = fis.read(buffer)) != -1) {
                output.writeBytes(buffer, bytesRead);
            }
        }
    }

    public static String buildClumpFileName(String segmentName, String fieldName) {
        return segmentName + "_" + fieldName + ClumpFileFormat.CLUMP_FILE_EXTENSION;
    }

    public static String buildTempFileName(String segmentName, String fieldName) {
        return segmentName + "_" + fieldName + ".clumptmp";
    }
}
