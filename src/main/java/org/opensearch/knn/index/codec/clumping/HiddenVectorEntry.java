/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.clumping;

import lombok.Builder;
import lombok.Value;

/**
 * Represents a hidden vector with its metadata.
 * 
 * A hidden vector is a vector that is NOT indexed in the main k-NN index (FAISS/HNSW)
 * but is stored in a separate clumping file on disk. Each hidden vector is associated
 * with its closest marker vector, which serves as its representative in the main index.
 * 
 * During search, hidden vectors are retrieved from the clumping file based on their
 * associated marker vectors and included in the candidate set for rescoring.
 * 
 * @see ClumpingFileWriter for writing hidden vectors to disk
 * @see ClumpingFileReader for reading hidden vectors from disk
 */
@Value
@Builder
public class HiddenVectorEntry {

    /**
     * The document ID of this hidden vector.
     * This is the Lucene document ID that uniquely identifies the document
     * containing this vector within a segment.
     */
    int docId;

    /**
     * The vector data for this hidden vector.
     * This is the actual float array representing the vector that was not
     * indexed in the main k-NN index.
     */
    float[] vector;

    /**
     * The document ID of the marker vector associated with this hidden vector.
     * The marker is the closest indexed vector to this hidden vector, determined
     * during indexing using exact distance calculation.
     */
    int markerDocId;
}
