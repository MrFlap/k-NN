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
 * order, each of fixed size ({@code 4 + dimension * elementSize} bytes).
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
     * Writes the clump file using O(M) heap. Hidden entries are placed into their
     * correct positions via a seekable temp file, avoiding any O(H) in-memory structures.
     *
     * @param state          Segment write state for creating output files
     * @param fieldName      The vector field name
     * @param dimension      The vector dimension
     * @param vectorDataType The vector data type code (see {@link ClumpFileFormat})
     * @param markerDocIds   Ordered list of marker doc IDs
     * @param markerVectors  Marker vectors in same order as markerDocIds (float[] or byte[])
     * @param assignInput    IndexInput for the assign file (flat int array, 4 bytes per doc ID)
     * @param tempInput      IndexInput for the hidden vector spill file
     * @param totalHidden    Total number of hidden vectors in the spill file
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
        int numMarkers = markerDocIds.size();
        int vectorSize = ClumpFileFormat.vectorBytes(dimension, vectorDataType);
        int hiddenEntrySize = ClumpFileFormat.hiddenEntryBytes(dimension, vectorDataType);

        // Build marker doc ID → marker index lookup
        Map<Integer, Integer> markerDocIdToIndex = new HashMap<>(numMarkers);
        for (int i = 0; i < numMarkers; i++) {
            markerDocIdToIndex.put(markerDocIds.get(i), i);
        }

        // Pass 1: count numHiddenPerMarker — O(M) heap
        int[] numHiddenPerMarker = new int[numMarkers];
        for (int h = 0; h < totalHidden; h++) {
            long entryOffset = (long) h * hiddenEntrySize;
            tempInput.seek(entryOffset);
            int hiddenDocId = tempInput.readInt();

            assignInput.seek((long) hiddenDocId * Integer.BYTES);
            int markerDocId = assignInput.readInt();

            Integer markerIndex = markerDocIdToIndex.get(markerDocId);
            if (markerIndex != null) {
                numHiddenPerMarker[markerIndex]++;
            }
        }

        // Compute each marker's byte offset within the clump data section.
        // The clump data for marker i is: markerVector + numHidden[i] * hiddenEntry.
        // The hidden entries for marker i start at clumpDataOffset[i] + vectorSize.
        long[] hiddenSectionOffset = new long[numMarkers];
        long runningOffset = 0;
        for (int i = 0; i < numMarkers; i++) {
            hiddenSectionOffset[i] = runningOffset + vectorSize;
            runningOffset += vectorSize + (long) numHiddenPerMarker[i] * hiddenEntrySize;
        }
        long totalClumpDataSize = runningOffset;

        // Write clump file: header + marker table + marker vectors, then hidden entries
        // via a seekable temp file.
        String clumpFileName = buildClumpFileName(state.segmentInfo.name, fieldName);
        Path clumpDataTempPath = Files.createTempFile("clumpdata", ".tmp");

        try {
            // Pass 2: write hidden entries to the seekable temp file at computed offsets.
            // Per-marker cursors track how many entries have been placed for each marker.
            int[] cursors = new int[numMarkers];
            byte[] entryBuf = new byte[hiddenEntrySize];

            try (RandomAccessFile raf = new RandomAccessFile(clumpDataTempPath.toFile(), "rw")) {
                // Pre-allocate the file to its final size
                raf.setLength(totalClumpDataSize);

                // Write marker vectors at their known offsets
                long markerOffset = 0;
                for (int i = 0; i < numMarkers; i++) {
                    raf.seek(markerOffset);
                    writeVectorToRaf(raf, markerVectors.get(i), vectorDataType);
                    markerOffset += vectorSize + (long) numHiddenPerMarker[i] * hiddenEntrySize;
                }

                // Scan spill file once, placing each hidden entry at its correct position
                for (int h = 0; h < totalHidden; h++) {
                    long spillOffset = (long) h * hiddenEntrySize;
                    tempInput.seek(spillOffset);
                    int hiddenDocId = tempInput.readInt();

                    assignInput.seek((long) hiddenDocId * Integer.BYTES);
                    int markerDocId = assignInput.readInt();

                    Integer markerIndex = markerDocIdToIndex.get(markerDocId);
                    if (markerIndex == null) {
                        continue;
                    }

                    // Compute destination: marker's hidden section start + cursor * entrySize
                    long destOffset = hiddenSectionOffset[markerIndex]
                        + (long) cursors[markerIndex] * hiddenEntrySize;
                    cursors[markerIndex]++;

                    // Read the full entry from the spill file and write to the temp file
                    tempInput.seek(spillOffset);
                    tempInput.readBytes(entryBuf, 0, hiddenEntrySize);
                    raf.seek(destOffset);
                    raf.write(entryBuf);
                }
            }

            // Assemble the final clump file: header + marker table + clump data from temp
            try (IndexOutput output = state.directory.createOutput(clumpFileName, state.context)) {
                // Header
                output.writeInt(numMarkers);
                output.writeInt(dimension);
                output.writeByte(vectorDataType);

                // Marker table
                long clumpDataBase = ClumpFileFormat.clumpDataStart(numMarkers);
                long currentOffset = clumpDataBase;
                for (int i = 0; i < numMarkers; i++) {
                    output.writeInt(markerDocIds.get(i));
                    output.writeInt(numHiddenPerMarker[i]);
                    output.writeLong(currentOffset);
                    currentOffset += vectorSize + (long) numHiddenPerMarker[i] * hiddenEntrySize;
                }

                // Append clump data from the temp file
                appendFileToOutput(clumpDataTempPath, output);

                CodecUtil.writeFooter(output);
            }

            log.debug(
                "Wrote clump file {} with {} markers, {} hidden vectors, dim={}, type={}",
                clumpFileName, numMarkers, totalHidden, dimension,
                vectorDataType == ClumpFileFormat.VECTOR_TYPE_FLOAT ? "float" : "byte"
            );
        } finally {
            Files.deleteIfExists(clumpDataTempPath);
        }
    }

    /**
     * Writes a single hidden entry (docId + vector) to a temp IndexOutput for spilling.
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
        } else {
            byte[] bv = (byte[]) vector;
            output.writeBytes(bv, bv.length);
        }
    }

    /**
     * Writes a vector to a RandomAccessFile in little-endian format (matching Lucene's
     * DataOutput convention). Float vectors are converted to their IEEE 754 byte
     * representation in little-endian order.
     */
    private static void writeVectorToRaf(RandomAccessFile raf, Object vector, byte vectorDataType) throws IOException {
        if (vectorDataType == ClumpFileFormat.VECTOR_TYPE_FLOAT) {
            float[] fv = (float[]) vector;
            byte[] buf = new byte[fv.length * Float.BYTES];
            for (int i = 0; i < fv.length; i++) {
                int bits = Float.floatToIntBits(fv[i]);
                int off = i * Float.BYTES;
                // Little-endian: least significant byte first
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
                // Little-endian
                buf[off] = (byte) fp16;
                buf[off + 1] = (byte) (fp16 >> 8);
            }
            raf.write(buf);
        } else {
            byte[] bv = (byte[]) vector;
            raf.write(bv);
        }
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
