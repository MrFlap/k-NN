/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.padrotate;

import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;

import java.io.IOException;
import java.util.Collections;
import java.util.List;

/**
 * Per-field float-vector writer that streams vectors to a temp file during indexing instead of
 * accumulating them in RAM. Accumulates a running D-dim centroid sum on the fly, so at flush
 * time only the temp file needs to be re-read once to quantize each vector.
 */
final class StreamingFieldVectorsWriter extends FlatFieldVectorsWriter<float[]> {

    private static final long SHALLOW_RAM_BYTES = RamUsageEstimator.shallowSizeOfInstance(StreamingFieldVectorsWriter.class);

    private final FieldInfo fieldInfo;
    private final int dim;
    private final Directory tempDirectory;
    private final IOContext ioContext;
    private final DocsWithFieldSet docsWithField;
    private final double[] dimensionSums;
    private int lastDocID = -1;
    private int vectorCount = 0;
    private IndexOutput tempOutput;
    private String tempFileName;
    private boolean finished;

    StreamingFieldVectorsWriter(FieldInfo fieldInfo, Directory tempDirectory, IOContext ioContext) throws IOException {
        this.fieldInfo = fieldInfo;
        this.dim = fieldInfo.getVectorDimension();
        this.tempDirectory = tempDirectory;
        this.ioContext = ioContext;
        this.docsWithField = new DocsWithFieldSet();
        this.dimensionSums = new double[dim];
        this.tempOutput = tempDirectory.createTempOutput("padrotate_vectors_" + fieldInfo.name, "tmp", ioContext);
        this.tempFileName = tempOutput.getName();
    }

    @Override
    public void addValue(int docID, float[] vectorValue) throws IOException {
        if (finished) {
            throw new IllegalStateException("already finished, cannot add more values");
        }
        if (docID == lastDocID) {
            throw new IllegalArgumentException(
                "vector field \"" + fieldInfo.name + "\" appears more than once in a document (only one value allowed per field)"
            );
        }
        assert docID > lastDocID;
        if (vectorValue.length != dim) {
            throw new IllegalArgumentException("expected dim=" + dim + ", got vector.length=" + vectorValue.length);
        }
        docsWithField.add(docID);
        lastDocID = docID;
        vectorCount++;
        for (int i = 0; i < dim; i++) {
            dimensionSums[i] += vectorValue[i];
            tempOutput.writeInt(Float.floatToIntBits(vectorValue[i]));
        }
    }

    @Override
    public float[] copyValue(float[] vectorValue) {
        // Values are streamed to disk; no in-memory copy is needed. This method is present to
        // satisfy the KnnFieldVectorsWriter contract and isn't used by our writer.
        return vectorValue.clone();
    }

    /**
     * Finalize the temp file so it can be opened for reading. Called during flush, before
     * quantization begins.
     */
    void closeTempOutputForReading() throws IOException {
        if (tempOutput != null) {
            try {
                tempOutput.close();
            } finally {
                tempOutput = null;
            }
        }
    }

    IndexInput openTempInput() throws IOException {
        return tempDirectory.openInput(tempFileName, ioContext);
    }

    void deleteTempFile() {
        if (tempFileName != null) {
            try {
                tempDirectory.deleteFile(tempFileName);
            } catch (IOException ignored) {
                // Temp file delete failure is non-fatal.
            } finally {
                tempFileName = null;
            }
        }
    }

    int dim() {
        return dim;
    }

    int vectorCount() {
        return vectorCount;
    }

    /**
     * Returns the D-dim centroid of all indexed vectors, or a zero vector if no vectors were
     * added. Lazily computed from the running dimension sums.
     */
    float[] computeCentroid() {
        float[] centroid = new float[dim];
        if (vectorCount == 0) {
            return centroid;
        }
        for (int i = 0; i < dim; i++) {
            centroid[i] = (float) (dimensionSums[i] / vectorCount);
        }
        return centroid;
    }

    FieldInfo fieldInfo() {
        return fieldInfo;
    }

    @Override
    public List<float[]> getVectors() {
        // We stream vectors to disk. Callers that need random access should go through the
        // temp file via openTempInput(). Returning an empty list here makes
        // ramBytesUsed() in parent callers correct without allocating.
        return Collections.emptyList();
    }

    @Override
    public DocsWithFieldSet getDocsWithFieldSet() {
        return docsWithField;
    }

    @Override
    public void finish() throws IOException {
        if (finished) {
            return;
        }
        finished = true;
        // The temp output is closed separately via closeTempOutputForReading(); we leave cleanup
        // of the file itself to the writer's flush path.
    }

    @Override
    public boolean isFinished() {
        return finished;
    }

    @Override
    public long ramBytesUsed() {
        return SHALLOW_RAM_BYTES + docsWithField.ramBytesUsed() + (long) dimensionSums.length * Double.BYTES;
    }

    /**
     * Ensure temp-file resources are cleaned up on abort paths.
     */
    void closeOnAbort() {
        IOUtils.closeWhileHandlingException(tempOutput);
        tempOutput = null;
        deleteTempFile();
    }
}
