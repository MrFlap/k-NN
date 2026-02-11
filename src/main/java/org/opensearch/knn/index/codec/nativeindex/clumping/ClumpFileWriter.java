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
import java.util.List;

/**
 * Writes a .clump sidecar file that stores marker-to-hidden vector mappings with
 * all vector data inline for sequential reads during expansion.
 * See {@link ClumpFileFormat} for the binary layout.
 * <p>
 * The writer supports a streaming mode that avoids holding all hidden vectors in
 * memory simultaneously. Hidden vector data is first spilled to a temporary file
 * during assignment, then assembled into the final clump file in marker order.
 */
@Log4j2
public final class ClumpFileWriter {

    private ClumpFileWriter() {}

    /**
     * Writes the clump file using a temp file that contains spilled hidden vector data.
     * This avoids holding all hidden vectors in heap simultaneously.
     * <p>
     * The temp file contains hidden entries written in doc-ID encounter order. Each entry
     * is (docId int + vector bytes). The {@code hiddenEntryLocations} list provides
     * (markerIndex, tempFileOffset) pairs that allow reading entries back in marker order.
     *
     * @param state                Segment write state for creating output files
     * @param fieldName            The vector field name
     * @param dimension            The vector dimension
     * @param vectorDataType       The vector data type code (see {@link ClumpFileFormat})
     * @param markerDocIds         Ordered list of marker doc IDs
     * @param markerVectors        Marker vectors in same order as markerDocIds (float[] or byte[])
     * @param numHiddenPerMarker   Number of hidden vectors assigned to each marker (parallel to markerDocIds)
     * @param hiddenEntryLocations List of (markerIndex, tempFileOffset) for each hidden entry in the temp file
     * @param tempInput            IndexInput for reading spilled hidden vector data
     */
    public static void writeClumpFile(
        SegmentWriteState state,
        String fieldName,
        int dimension,
        byte vectorDataType,
        List<Integer> markerDocIds,
        List<Object> markerVectors,
        int[] numHiddenPerMarker,
        List<HiddenEntryLocation> hiddenEntryLocations,
        IndexInput tempInput
    ) throws IOException {
        String clumpFileName = buildClumpFileName(state.segmentInfo.name, fieldName);
        int numMarkers = markerDocIds.size();
        int vectorSize = ClumpFileFormat.vectorBytes(dimension, vectorDataType);
        int hiddenEntrySize = ClumpFileFormat.hiddenEntryBytes(dimension, vectorDataType);

        // Sort hidden entry locations by markerIndex so we can write clump data in marker order.
        // Within each marker group, entries retain their encounter order.
        hiddenEntryLocations.sort((a, b) -> Integer.compare(a.getMarkerIndex(), b.getMarkerIndex()));

        try (IndexOutput output = state.directory.createOutput(clumpFileName, state.context)) {
            // Header
            output.writeInt(numMarkers);
            output.writeInt(dimension);
            output.writeByte(vectorDataType);

            // Compute clump data offsets for each marker
            long clumpDataBase = ClumpFileFormat.clumpDataStart(numMarkers);
            long currentOffset = clumpDataBase;

            // Write marker table: (markerDocId, numHidden, clumpDataOffset) per marker
            for (int i = 0; i < numMarkers; i++) {
                output.writeInt(markerDocIds.get(i));
                output.writeInt(numHiddenPerMarker[i]);
                output.writeLong(currentOffset);

                // Each clump: marker vector + numHidden * (docId + vector)
                currentOffset += vectorSize + (long) numHiddenPerMarker[i] * hiddenEntrySize;
            }

            // Write clump data for each marker
            int totalHidden = 0;
            int locationIdx = 0;

            for (int i = 0; i < numMarkers; i++) {
                // Write marker vector from memory
                writeVector(output, markerVectors.get(i), vectorDataType);

                // Write hidden entries by reading from temp file at recorded offsets
                int numHidden = numHiddenPerMarker[i];
                for (int j = 0; j < numHidden; j++) {
                    HiddenEntryLocation loc = hiddenEntryLocations.get(locationIdx++);
                    // Seek to the hidden entry in the temp file and copy it
                    tempInput.seek(loc.getTempFileOffset());
                    int hiddenDocId = tempInput.readInt();
                    output.writeInt(hiddenDocId);
                    // Copy vector bytes directly
                    copyBytes(tempInput, output, vectorSize);
                }
                totalHidden += numHidden;
            }

            CodecUtil.writeFooter(output);
            log.debug(
                "Wrote clump file {} with {} markers, {} hidden vectors, dim={}, type={}",
                clumpFileName, numMarkers, totalHidden, dimension,
                vectorDataType == ClumpFileFormat.VECTOR_TYPE_FLOAT ? "float" : "byte"
            );
        }
    }

    /**
     * Writes a single hidden entry (docId + vector) to a temp IndexOutput for spilling.
     *
     * @return the file offset where this entry was written (for later retrieval)
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
        } else {
            byte[] bv = (byte[]) vector;
            output.writeBytes(bv, bv.length);
        }
    }

    private static void copyBytes(IndexInput input, IndexOutput output, int numBytes) throws IOException {
        // Copy in chunks to avoid allocating a huge buffer
        byte[] buffer = new byte[Math.min(numBytes, 8192)];
        int remaining = numBytes;
        while (remaining > 0) {
            int toRead = Math.min(remaining, buffer.length);
            input.readBytes(buffer, 0, toRead);
            output.writeBytes(buffer, toRead);
            remaining -= toRead;
        }
    }

    public static String buildClumpFileName(String segmentName, String fieldName) {
        return segmentName + "_" + fieldName + ClumpFileFormat.CLUMP_FILE_EXTENSION;
    }

    public static String buildTempFileName(String segmentName, String fieldName) {
        return segmentName + "_" + fieldName + ".clumptmp";
    }
}
