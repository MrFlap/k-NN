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
import java.util.ArrayList;
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
 */
@Log4j2
public final class ClumpFileWriter {

    private ClumpFileWriter() {}

    /**
     * Writes the clump file by reading assignments from the assign file and hidden
     * vector data from the spill file.
     * <p>
     * The assign file is a flat array of int32 values where the int at offset
     * {@code 4 * docId} is the marker doc ID for that vector. Markers point to
     * themselves; unoccupied slots contain {@link ClumpingIndexBuildStrategy#UNASSIGNED}.
     * <p>
     * The spill file contains hidden entries in encounter order, each of fixed size
     * ({@code 4 + vectorSize} bytes: docId int + vector bytes).
     * <p>
     * The method performs two passes over the spill file:
     * <ol>
     *   <li>First pass: scan the spill file to build a map from marker doc ID to the
     *       list of spill-file byte offsets for its hidden entries. This uses the assign
     *       file to look up each hidden entry's marker.</li>
     *   <li>Second pass: for each marker, seek to each of its hidden entries in the spill
     *       file and copy them into the clump file.</li>
     * </ol>
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
        String clumpFileName = buildClumpFileName(state.segmentInfo.name, fieldName);
        int numMarkers = markerDocIds.size();
        int vectorSize = ClumpFileFormat.vectorBytes(dimension, vectorDataType);
        int hiddenEntrySize = ClumpFileFormat.hiddenEntryBytes(dimension, vectorDataType);

        // Build a map from marker doc ID to its index in markerDocIds for fast lookup
        Map<Integer, Integer> markerDocIdToIndex = new HashMap<>(numMarkers);
        for (int i = 0; i < numMarkers; i++) {
            markerDocIdToIndex.put(markerDocIds.get(i), i);
        }

        // First pass over the spill file: read each hidden entry's doc ID, look up its
        // marker from the assign file, and record the spill-file offset grouped by marker.
        // This builds per-marker lists of spill offsets — much smaller than the previous
        // approach since we only store longs (8 bytes each) rather than full HiddenEntryLocation
        // objects, and the grouping happens naturally.
        @SuppressWarnings("unchecked")
        List<Long>[] spillOffsetsByMarker = new List[numMarkers];
        for (int i = 0; i < numMarkers; i++) {
            spillOffsetsByMarker[i] = new ArrayList<>();
        }

        for (int h = 0; h < totalHidden; h++) {
            long entryOffset = (long) h * hiddenEntrySize;
            tempInput.seek(entryOffset);
            int hiddenDocId = tempInput.readInt();

            // Look up the marker for this hidden doc ID from the assign file
            assignInput.seek((long) hiddenDocId * Integer.BYTES);
            int markerDocId = assignInput.readInt();

            Integer markerIndex = markerDocIdToIndex.get(markerDocId);
            if (markerIndex != null) {
                spillOffsetsByMarker[markerIndex].add(entryOffset);
            } else {
                log.warn("Hidden doc {} assigned to unknown marker {}, skipping", hiddenDocId, markerDocId);
            }
        }

        // Compute numHiddenPerMarker from the grouped offsets
        int[] numHiddenPerMarker = new int[numMarkers];
        for (int i = 0; i < numMarkers; i++) {
            numHiddenPerMarker[i] = spillOffsetsByMarker[i].size();
        }

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
            int writtenHidden = 0;

            for (int i = 0; i < numMarkers; i++) {
                // Write marker vector from memory
                writeVector(output, markerVectors.get(i), vectorDataType);

                // Write hidden entries by seeking to each spill offset
                for (long spillOffset : spillOffsetsByMarker[i]) {
                    tempInput.seek(spillOffset);
                    int hiddenDocId = tempInput.readInt();
                    output.writeInt(hiddenDocId);
                    // Copy vector bytes directly
                    copyBytes(tempInput, output, vectorSize);
                    writtenHidden++;
                }
            }

            CodecUtil.writeFooter(output);
            log.debug(
                "Wrote clump file {} with {} markers, {} hidden vectors, dim={}, type={}",
                clumpFileName, numMarkers, writtenHidden, dimension,
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
