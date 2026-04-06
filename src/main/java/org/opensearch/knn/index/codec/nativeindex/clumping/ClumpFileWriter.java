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
        // Spill file still uses the old interleaved format (docId + vector per entry)
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

        // v3 layout per marker: [markerVector][docId0..docIdN][vec0..vecN]
        // Compute each marker's offsets within the clump data section.
        long[] docIdBlockOffset = new long[numMarkers];  // offset of doc ID block within temp file
        long[] vectorBlockOffset = new long[numMarkers];  // offset of vector block within temp file
        long runningOffset = 0;
        for (int i = 0; i < numMarkers; i++) {
            // marker vector
            runningOffset += vectorSize;
            // doc ID block
            docIdBlockOffset[i] = runningOffset;
            runningOffset += (long) numHiddenPerMarker[i] * Integer.BYTES;
            // vector block
            vectorBlockOffset[i] = runningOffset;
            runningOffset += (long) numHiddenPerMarker[i] * vectorSize;
        }
        long totalClumpDataSize = runningOffset;

        // Write clump file: header + marker table + clump data via seekable temp file.
        String clumpFileName = buildClumpFileName(state.segmentInfo.name, fieldName);
        Path clumpDataTempPath = Files.createTempFile("clumpdata", ".tmp");

        try {
            // Pass 2: write marker vectors, then scatter hidden doc IDs and vectors
            // into their separate blocks within the temp file.
            int[] docIdCursors = new int[numMarkers];
            int[] vectorCursors = new int[numMarkers];

            try (RandomAccessFile raf = new RandomAccessFile(clumpDataTempPath.toFile(), "rw")) {
                // Pre-allocate the file to its final size
                raf.setLength(totalClumpDataSize);

                // Write marker vectors at their known offsets (start of each marker's clump data)
                for (int i = 0; i < numMarkers; i++) {
                    long mvOffset = docIdBlockOffset[i] - vectorSize; // marker vec is right before doc ID block
                    raf.seek(mvOffset);
                    writeVectorToRaf(raf, markerVectors.get(i), vectorDataType);
                }

                // Scan spill file once, placing doc IDs and vectors into separate blocks
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

                    // Write doc ID to the doc ID block
                    long docIdDest = docIdBlockOffset[markerIndex]
                        + (long) docIdCursors[markerIndex] * Integer.BYTES;
                    docIdBuf[0] = (byte) hiddenDocId;
                    docIdBuf[1] = (byte) (hiddenDocId >> 8);
                    docIdBuf[2] = (byte) (hiddenDocId >> 16);
                    docIdBuf[3] = (byte) (hiddenDocId >> 24);
                    raf.seek(docIdDest);
                    raf.write(docIdBuf);
                    docIdCursors[markerIndex]++;

                    // Write vector to the vector block
                    long vecDest = vectorBlockOffset[markerIndex]
                        + (long) vectorCursors[markerIndex] * vectorSize;
                    tempInput.seek(spillOffset + Integer.BYTES); // skip docId in spill entry
                    tempInput.readBytes(vecBuf, 0, vectorSize);
                    raf.seek(vecDest);
                    raf.write(vecBuf);
                    vectorCursors[markerIndex]++;
                }
            }

            // Assemble the final clump file: header + marker table + clump data from temp
            try (IndexOutput output = state.directory.createOutput(clumpFileName, state.context)) {
                // Header
                output.writeInt(numMarkers);
                output.writeInt(dimension);
                output.writeByte(vectorDataType);

                // Marker table — clumpDataOffset points to the start of each marker's data
                long clumpDataBase = ClumpFileFormat.clumpDataStart(numMarkers);
                long currentOffset = clumpDataBase;
                for (int i = 0; i < numMarkers; i++) {
                    output.writeInt(markerDocIds.get(i));
                    output.writeInt(numHiddenPerMarker[i]);
                    output.writeLong(currentOffset);
                    currentOffset += vectorSize
                        + (long) numHiddenPerMarker[i] * Integer.BYTES
                        + (long) numHiddenPerMarker[i] * vectorSize;
                }

                // Append clump data from the temp file
                appendFileToOutput(clumpDataTempPath, output);

                CodecUtil.writeFooter(output);
            }

            log.debug(
                "Wrote v3 clump file {} with {} markers, {} hidden vectors, dim={}, type={}",
                clumpFileName, numMarkers, totalHidden, dimension,
                vectorDataType == ClumpFileFormat.VECTOR_TYPE_FP16 ? "fp16"
                    : vectorDataType == ClumpFileFormat.VECTOR_TYPE_FLOAT ? "float" : "byte"
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

    /**
     * Writes a v4 Huffman-compressed clump file for FP16 vectors. The Huffman tree is
     * built from all hidden + marker FP16 symbol frequencies, serialized into the header,
     * and vector blocks are stored compressed. Doc IDs remain uncompressed.
     * <p>
     * The clump data layout per marker in v4:
     * <pre>
     *   [marker vector — raw FP16, D * 2 bytes]
     *   [doc ID block — numHidden * 4 bytes, uncompressed]
     *   [compressed vector block — variable length]
     * </pre>
     * The reader determines compressed block boundaries via adjacent marker offsets:
     * start = offset[i] + vectorSize + docIdBlockSize, end = offset[i+1] (or clump data end).
     */
    public static void writeClumpFileHuffman(
        SegmentWriteState state,
        String fieldName,
        int dimension,
        List<Integer> markerDocIds,
        List<Object> markerVectors,
        IndexInput assignInput,
        IndexInput tempInput,
        int totalHidden
    ) throws IOException {
        byte vectorDataType = ClumpFileFormat.VECTOR_TYPE_FP16;
        int numMarkers = markerDocIds.size();
        int vectorSize = ClumpFileFormat.vectorBytes(dimension, vectorDataType);
        int spillEntrySize = Integer.BYTES + vectorSize;

        // Build marker doc ID → marker index lookup
        Map<Integer, Integer> markerDocIdToIndex = new HashMap<>(numMarkers);
        for (int i = 0; i < numMarkers; i++) {
            markerDocIdToIndex.put(markerDocIds.get(i), i);
        }

        // Pass 1: count numHiddenPerMarker and collect FP16 symbol frequencies
        int[] numHiddenPerMarker = new int[numMarkers];
        long[] symbolFrequencies = new long[HuffmanCodec.NUM_SYMBOLS];
        byte[] vecBuf = new byte[vectorSize];

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

            // Read vector bytes and tally FP16 symbol frequencies
            tempInput.seek(entryOffset + Integer.BYTES);
            tempInput.readBytes(vecBuf, 0, vectorSize);
            java.nio.ByteBuffer vbb = java.nio.ByteBuffer.wrap(vecBuf).order(java.nio.ByteOrder.BIG_ENDIAN);
            for (int d = 0; d < dimension; d++) {
                int sym = Short.toUnsignedInt(vbb.getShort());
                symbolFrequencies[sym]++;
            }
        }

        // Also tally marker vector frequencies
        for (Object mv : markerVectors) {
            float[] fv = (float[]) mv;
            for (float v : fv) {
                int sym = Short.toUnsignedInt(Float.floatToFloat16(v));
                symbolFrequencies[sym]++;
            }
        }

        // Build Huffman codec
        HuffmanCodec huffman = HuffmanCodec.buildFromFrequencies(symbolFrequencies);

        // Pass 2: build the clump data with compressed vector blocks.
        // We need to write marker vectors (raw), doc IDs (raw), and compressed vectors.
        // Since compressed sizes are variable, we must compute them before writing the marker table.

        // First, compress all marker vectors and hidden vector blocks per marker.
        byte[][] compressedMarkerVecs = new byte[numMarkers][];
        for (int i = 0; i < numMarkers; i++) {
            float[] fv = (float[]) markerVectors.get(i);
            short[] fp16 = new short[dimension];
            for (int d = 0; d < dimension; d++) {
                fp16[d] = Float.floatToFloat16(fv[d]);
            }
            compressedMarkerVecs[i] = huffman.encode(fp16);
        }

        // Collect hidden vectors per marker, compress each marker's vector block
        // We'll re-scan the spill file, grouping vectors by marker.
        // To keep heap bounded, we process one marker at a time via sorted spill entries.

        // Build per-marker hidden doc IDs and compressed vector blocks via temp file
        int[][] hiddenDocIdsPerMarker = new int[numMarkers][];
        byte[][] compressedVecBlocks = new byte[numMarkers][];
        int[] docIdCursors = new int[numMarkers];

        // Allocate doc ID arrays
        for (int i = 0; i < numMarkers; i++) {
            hiddenDocIdsPerMarker[i] = new int[numHiddenPerMarker[i]];
        }

        // Collect per-marker FP16 bytes for compression
        byte[][] rawVecBlocksPerMarker = new byte[numMarkers][];
        int[] vecCursors = new int[numMarkers];
        for (int i = 0; i < numMarkers; i++) {
            rawVecBlocksPerMarker[i] = new byte[numHiddenPerMarker[i] * vectorSize];
        }

        for (int h = 0; h < totalHidden; h++) {
            long entryOffset = (long) h * spillEntrySize;
            tempInput.seek(entryOffset);
            int hiddenDocId = tempInput.readInt();

            assignInput.seek((long) hiddenDocId * Integer.BYTES);
            int markerDocId = assignInput.readInt();

            Integer markerIndex = markerDocIdToIndex.get(markerDocId);
            if (markerIndex == null) {
                continue;
            }

            hiddenDocIdsPerMarker[markerIndex][docIdCursors[markerIndex]++] = hiddenDocId;

            tempInput.seek(entryOffset + Integer.BYTES);
            int destOffset = vecCursors[markerIndex] * vectorSize;
            tempInput.readBytes(rawVecBlocksPerMarker[markerIndex], destOffset, vectorSize);
            vecCursors[markerIndex]++;
        }

        // Compress each marker's vector block
        for (int i = 0; i < numMarkers; i++) {
            if (numHiddenPerMarker[i] > 0) {
                compressedVecBlocks[i] = huffman.encodeFromBytes(rawVecBlocksPerMarker[i]);
            } else {
                compressedVecBlocks[i] = new byte[0];
            }
            rawVecBlocksPerMarker[i] = null; // free raw data
        }

        // Compute clump data offsets. v4 layout per marker:
        //   [raw marker vector: vectorSize bytes]
        //   [doc ID block: numHidden * 4 bytes]
        //   [compressed vector block: variable bytes]
        int huffmanTreeSize = huffman.serializedSizeBytes();
        long headerSize = ClumpFileFormat.headerBytesV4(huffmanTreeSize);
        long markerTableSize = (long) numMarkers * ClumpFileFormat.MARKER_TABLE_ENTRY_BYTES;
        long clumpDataBase = headerSize + markerTableSize;

        long[] clumpDataOffsets = new long[numMarkers];
        long currentOffset = clumpDataBase;
        for (int i = 0; i < numMarkers; i++) {
            clumpDataOffsets[i] = currentOffset;
            currentOffset += vectorSize  // raw marker vector
                + (long) numHiddenPerMarker[i] * Integer.BYTES  // doc ID block
                + compressedVecBlocks[i].length;  // compressed vector block
        }

        // Write the final clump file
        String clumpFileName = buildClumpFileName(state.segmentInfo.name, fieldName);
        try (IndexOutput output = state.directory.createOutput(clumpFileName, state.context)) {
            // v4 Header
            output.writeByte(ClumpFileFormat.FORMAT_VERSION_V4_HUFFMAN);
            output.writeInt(numMarkers);
            output.writeInt(dimension);
            output.writeByte(vectorDataType);
            output.writeInt(huffmanTreeSize);
            huffman.serialize(output);

            // Marker table
            for (int i = 0; i < numMarkers; i++) {
                output.writeInt(markerDocIds.get(i));
                output.writeInt(numHiddenPerMarker[i]);
                output.writeLong(clumpDataOffsets[i]);
            }

            // Clump data
            for (int i = 0; i < numMarkers; i++) {
                // Raw marker vector (FP16)
                writeVector(output, markerVectors.get(i), vectorDataType);

                // Doc ID block (little-endian int32s)
                for (int j = 0; j < numHiddenPerMarker[i]; j++) {
                    output.writeInt(hiddenDocIdsPerMarker[i][j]);
                }

                // Compressed vector block
                output.writeBytes(compressedVecBlocks[i], compressedVecBlocks[i].length);
            }

            CodecUtil.writeFooter(output);
        }

        log.debug(
            "Wrote v4 Huffman clump file {} with {} markers, {} hidden vectors, dim={}, huffmanTreeSize={}",
            clumpFileName, numMarkers, totalHidden, dimension, huffmanTreeSize
        );
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
