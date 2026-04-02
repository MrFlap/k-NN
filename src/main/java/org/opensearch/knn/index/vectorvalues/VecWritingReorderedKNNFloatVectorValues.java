/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.vectorvalues;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.store.IndexOutput;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * A {@link ReorderedKNNFloatVectorValues} that also writes each vector to a .vec {@link IndexOutput}
 * as it yields vectors to the FAISS graph builder. This fuses the .vec write with the FAISS build
 * into a single pass over the permuted vectors, eliminating the separate .vec write loop and
 * halving the random reads from source segments.
 * <p>
 * After the FAISS build completes (which exhausts this iterator), call {@link #getVecBytesWritten()}
 * to get the total bytes written to the .vec output.
 */
public class VecWritingReorderedKNNFloatVectorValues extends ReorderedKNNFloatVectorValues {

    private final IndexOutput vecOutput;
    private ByteBuffer cachedBuffer;
    private long vecBytesWritten = 0;

    public VecWritingReorderedKNNFloatVectorValues(
        FloatVectorValues mergedRandomAccess,
        int[] permutation,
        int[] mergedOrdToDocId,
        IndexOutput vecOutput
    ) {
        super(mergedRandomAccess, permutation, mergedOrdToDocId);
        this.vecOutput = vecOutput;
    }

    @Override
    public float[] getVector() throws IOException {
        float[] vector = super.getVector();
        writeVectorToOutput(vector);
        return vector;
    }

    private void writeVectorToOutput(float[] vector) throws IOException {
        int numBytes = vector.length * Float.BYTES;
        if (cachedBuffer == null || cachedBuffer.capacity() != numBytes) {
            cachedBuffer = ByteBuffer.allocate(numBytes).order(ByteOrder.LITTLE_ENDIAN);
        }
        cachedBuffer.clear();
        cachedBuffer.asFloatBuffer().put(vector);
        vecOutput.writeBytes(cachedBuffer.array(), numBytes);
        vecBytesWritten += numBytes;
    }

    public long getVecBytesWritten() {
        return vecBytesWritten;
    }
}
