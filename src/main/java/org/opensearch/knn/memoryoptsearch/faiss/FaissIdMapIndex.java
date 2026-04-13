/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.packed.DirectMonotonicReader;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissBinaryHnswIndex;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissBinaryIndex;
import org.opensearch.knn.memoryoptsearch.faiss.vectorvalues.FaissFloatVectorValues;

import java.io.IOException;

/**
 * A FAISS index with an ID mapping that maps the internal vector ID to a logical ID, along with the actual vector index.
 * It first delegates the vector search to its nested vector index, then transforms the vector ID into a logical index that is
 * understandable by upstream components. This is particularly useful when not all Lucene documents are indexed with a vector field.
 * For example, if 70% of the documents have a vector field and the remaining 30% do not, the FAISS vector index will still assign
 * increasing and continuous vector IDs starting from 0.
 * However, these IDs only cover the sparse 30% of Lucene documents, so an ID mapping is needed to convert the internal physical vector ID
 * into the corresponding Lucene document ID.
 * If the mapping is an identity mapping, where each `i` is mapped to itself, we omit storing it to save memory.
 */
public class FaissIdMapIndex extends FaissBinaryIndex implements FaissHNSWProvider {
    public static final String IXMP = "IxMp";
    public static final String IBMP = "IBMp";

    @Getter
    private FaissIndex nestedIndex;
    private FaissHNSWProvider hnswGetter;
    private DirectMonotonicReader idMappingReader;
    // Raw ord-to-doc mapping array for reorder code that needs direct array access.
    // Null when the mapping is identity (dense case).
    @Getter
    private int[] ordToDocs;

    public FaissIdMapIndex(final String indexType) {
        super(indexType);
    }

    /**
     * Partially load id mapping and its nested index to which vector searching will be delegated.
     * Faiss deserialization code :
     * <a href="https://github.com/facebookresearch/faiss/blob/main/faiss/impl/index_read.cpp#L1088">IndexIDMap.h</a>
     * @param input An input stream for a FAISS HNSW graph file, allowing access to the neighbor list and vector locations.
     * @throws IOException
     */
    @Override
    protected void doLoad(IndexInput input) throws IOException {
        if (indexType.equals(IXMP)) {
            readCommonHeader(input);
        } else {
            readBinaryCommonHeader(input);
        }
        final FaissIndex nestedIndex = FaissIndex.load(input);

        if (nestedIndex instanceof AbstractFaissHNSWIndex || nestedIndex instanceof FaissBinaryHnswIndex) {
            this.nestedIndex = nestedIndex;
            this.hnswGetter = (FaissHNSWProvider) nestedIndex;
        } else {
            throw new IllegalStateException("Invalid nested HNSW index type, got index type=" + nestedIndex.getIndexType());
        }

        final int numElements = Math.toIntExact(input.readLong());

        // This is a mapping table that maps internal vector id to Lucene document id.
        // In dense case where all documents having at least one KNN field which also has exactly one vector, we don't need this.
        // However, in a sparse case where not all documents having KNN field, we need this mapping table to convert inner vector id to
        // Lucene document id.
        // Another case is parent-child nested case. In which, this mapping table will map internal vector id to parent document id.
        // NOTE : If the mapping is an identity function that maps `i` to `i`, then the reader will be null.
        idMappingReader = MonotonicIntegerSequenceEncoder.encode(numElements, input);

        // Populate ordToDocs array for reorder code that needs direct array access.
        // If idMappingReader is null (identity mapping), ordToDocs stays null.
        if (idMappingReader != null) {
            ordToDocs = new int[numElements];
            boolean isIdentical = true;
            for (int i = 0; i < numElements; i++) {
                ordToDocs[i] = (int) idMappingReader.get(i);
                if (ordToDocs[i] != i) {
                    isIdentical = false;
                }
            }
            if (isIdentical) {
                ordToDocs = null;
            }
        }
    }

    @Override
    public VectorEncoding getVectorEncoding() {
        return nestedIndex.getVectorEncoding();
    }

    @Override
    public FloatVectorValues getFloatValues(IndexInput indexInput) throws IOException {
        if (idMappingReader == null) {
            return nestedIndex.getFloatValues(indexInput);
        }
        return sparseFloatValues(indexInput);
    }

    @Override
    public ByteVectorValues getByteValues(IndexInput indexInput) throws IOException {
        if (idMappingReader == null) {
            return nestedIndex.getByteValues(indexInput);
        }
        return sparseByteValues(indexInput);
    }

    /**
     * For sparse or nested cases, {@link ByteVectorValues} needs to be wrapped to correctly map an internal vector ID to a
     * Lucene document ID.
     */
    private ByteVectorValues sparseByteValues(IndexInput indexInput) throws IOException {
        final ByteVectorValues vectorValues = nestedIndex.getByteValues(indexInput);

        @RequiredArgsConstructor
        class SparseByteVectorValuesImpl extends ByteVectorValues {
            private final ByteVectorValues vectorValues;

            @Override
            public byte[] vectorValue(int internalVectorId) throws IOException {
                return vectorValues.vectorValue(internalVectorId);
            }

            @Override
            public int dimension() {
                return vectorValues.dimension();
            }

            @Override
            public int ordToDoc(int internalVectorId) {
                return (int) idMappingReader.get(internalVectorId);
            }

            @Override
            public Bits getAcceptOrds(final Bits acceptDocs) {
                if (acceptDocs != null) {
                    return new Bits() {
                        @Override
                        public boolean get(int internalVectorId) {
                            return acceptDocs.get((int) idMappingReader.get(internalVectorId));
                        }

                        @Override
                        public int length() {
                            return vectorValues.size();
                        }
                    };
                }
                return null;
            }

            @Override
            public int size() {
                return vectorValues.size();
            }

            @Override
            public ByteVectorValues copy() throws IOException {
                return new SparseByteVectorValuesImpl(vectorValues.copy());
            }
        }

        return new SparseByteVectorValuesImpl(vectorValues);
    }

    private FloatVectorValues sparseFloatValues(IndexInput indexInput) throws IOException {
        final FloatVectorValues vectorValues = nestedIndex.getFloatValues(indexInput);
        return new FaissFloatVectorValues.SparseFloatVectorValuesImpl(vectorValues, idMappingReader);
    }

    @Override
    public FaissHNSW getFaissHnsw() {
        return hnswGetter.getFaissHnsw();
    }
}
