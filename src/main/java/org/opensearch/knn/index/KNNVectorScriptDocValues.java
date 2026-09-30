/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import java.util.Locale;

import java.io.IOException;
import java.util.Objects;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.knn.index.mapper.RandomOrthogonalVectorTransformer;
import org.opensearch.ExceptionsHelper;
import org.opensearch.index.fielddata.ScriptDocValues;

@RequiredArgsConstructor(access = AccessLevel.PRIVATE)
public abstract class KNNVectorScriptDocValues<T> extends ScriptDocValues<T> {

    private final DocIdSetIterator vectorValues;
    private final String fieldName;
    @Getter
    private final VectorDataType vectorDataType;
    private boolean docExists = false;
    private int lastDocID = -1;

    @Override
    public void setNextDocId(int docId) throws IOException {
        if (docId < lastDocID) {
            throw new IllegalArgumentException("docs were sent out-of-order: lastDocID=" + lastDocID + " vs docID=" + docId);
        }
        lastDocID = docId;
        int curDocID = vectorValues.docID();
        if (lastDocID > curDocID) {
            curDocID = vectorValues.advance(docId);
        }
        docExists = lastDocID == curDocID;
    }

    public T getValue() {
        if (!docExists) {
            String errorMessage = String.format(
                Locale.ROOT,
                "One of the document doesn't have a value for field '%s'. "
                    + "This can be avoided by checking if a document has a value for the field or not "
                    + "by doc['%s'].size() == 0 ? 0 : {your script}",
                fieldName,
                fieldName
            );
            throw new IllegalStateException(errorMessage);
        }
        try {
            return doGetValue();
        } catch (IOException e) {
            throw ExceptionsHelper.convertToOpenSearchException(e);
        }
    }

    protected abstract T doGetValue() throws IOException;

    @Override
    public int size() {
        return docExists ? 1 : 0;
    }

    @Override
    public T get(int i) {
        throw new UnsupportedOperationException("knn vector does not support this operation");
    }

    /**
     * Creates a KNNVectorScriptDocValues object based on the provided parameters.
     *
     * @param knnVectorValues          The DocIdSetIterator representing the vector values.
     * @param fieldName       The name of the field.
     * @param vectorDataType  The data type of the vector.
     * @return A KNNVectorScriptDocValues object based on the type of the values.
     * @throws IllegalArgumentException If the type of values is unsupported.
     */
    public static KNNVectorScriptDocValues<?> create(KnnVectorValues knnVectorValues, String fieldName, VectorDataType vectorDataType) {
        return create(knnVectorValues, fieldName, vectorDataType, false);
    }

    /**
     * Creates a KNNVectorScriptDocValues object based on the provided parameters.
     *
     * @param knnVectorValues          The DocIdSetIterator representing the vector values.
     * @param fieldName       The name of the field.
     * @param vectorDataType  The data type of the vector.
     * @param undoRandomOrthogonalTransform Whether float vectors are stored transformed and must be returned with the random
     *                                      orthogonal transform undone.
     * @return A KNNVectorScriptDocValues object based on the type of the values.
     * @throws IllegalArgumentException If the type of values is unsupported.
     */
    public static KNNVectorScriptDocValues<?> create(
        KnnVectorValues knnVectorValues,
        String fieldName,
        VectorDataType vectorDataType,
        boolean undoRandomOrthogonalTransform
    ) {
        Objects.requireNonNull(knnVectorValues, "values must not be null");
        if (knnVectorValues instanceof FloatVectorValues) {
            return new KNNFloatVectorScriptDocValues(
                (FloatVectorValues) knnVectorValues,
                fieldName,
                vectorDataType,
                undoRandomOrthogonalTransform
            );
        } else if (knnVectorValues instanceof ByteVectorValues) {
            return new KNNByteVectorScriptDocValues((ByteVectorValues) knnVectorValues, fieldName, vectorDataType);
        } else {
            throw new IllegalArgumentException("Unsupported values type: " + knnVectorValues.getClass());
        }
    }

    public static KNNVectorScriptDocValues<?> create(DocIdSetIterator docIdSetIterator, String fieldName, VectorDataType vectorDataType) {
        return create(docIdSetIterator, fieldName, vectorDataType, false);
    }

    public static KNNVectorScriptDocValues<?> create(
        DocIdSetIterator docIdSetIterator,
        String fieldName,
        VectorDataType vectorDataType,
        boolean undoRandomOrthogonalTransform
    ) {
        Objects.requireNonNull(docIdSetIterator, "values must not be null");
        if (docIdSetIterator instanceof BinaryDocValues) {
            return new KNNNativeVectorScriptDocValues<>(
                (BinaryDocValues) docIdSetIterator,
                fieldName,
                vectorDataType,
                undoRandomOrthogonalTransform
            );
        } else {
            throw new IllegalArgumentException("Unsupported values type: " + docIdSetIterator.getClass());
        }

    }

    private static final class KNNByteVectorScriptDocValues extends KNNVectorScriptDocValues<byte[]> {
        private final ByteVectorValues values;
        private final KnnVectorValues.DocIndexIterator iterator;

        KNNByteVectorScriptDocValues(ByteVectorValues values, String field, VectorDataType type) {
            super(values.iterator(), field, type);
            this.values = values;
            this.iterator = super.vectorValues instanceof KnnVectorValues.DocIndexIterator
                ? (KnnVectorValues.DocIndexIterator) super.vectorValues
                : values.iterator();
        }

        @Override
        protected byte[] doGetValue() throws IOException {
            int docId = this.iterator.index();
            if (docId == KnnVectorValues.DocIndexIterator.NO_MORE_DOCS) {
                throw new IllegalStateException("No more ordinals to retrieve vector values.");
            }

            try {
                return values.vectorValue(docId);
            } catch (IOException e) {
                throw ExceptionsHelper.convertToOpenSearchException(e);
            }
        }

    }

    private static final class KNNFloatVectorScriptDocValues extends KNNVectorScriptDocValues<float[]> {
        private final FloatVectorValues values;
        private final KnnVectorValues.DocIndexIterator iterator;
        private final boolean undoRandomOrthogonalTransform;

        KNNFloatVectorScriptDocValues(FloatVectorValues values, String field, VectorDataType type, boolean undoRandomOrthogonalTransform) {
            super(values.iterator(), field, type);
            this.values = values;
            this.undoRandomOrthogonalTransform = undoRandomOrthogonalTransform;
            this.iterator = super.vectorValues instanceof KnnVectorValues.DocIndexIterator
                ? (KnnVectorValues.DocIndexIterator) super.vectorValues
                : values.iterator();
        }

        @Override
        protected float[] doGetValue() throws IOException {
            int ord = iterator.index();    // Fetch ordinal (index of vector)
            if (ord == KnnVectorValues.DocIndexIterator.NO_MORE_DOCS) {
                throw new IllegalStateException("No more ordinals to retrieve vector values.");
            }
            final float[] vector = values.vectorValue(ord);
            // vectorValue may return a buffer owned by the reader, so undo the transform on a copy.
            return undoRandomOrthogonalTransform
                ? RandomOrthogonalVectorTransformer.forDimension(vector.length).inverseTransform(vector, false)
                : vector;
        }
    }

    private static final class KNNNativeVectorScriptDocValues<T> extends KNNVectorScriptDocValues<T> {
        private final BinaryDocValues values;
        private final boolean undoRandomOrthogonalTransform;

        KNNNativeVectorScriptDocValues(BinaryDocValues values, String field, VectorDataType type, boolean undoRandomOrthogonalTransform) {
            super(values, field, type);
            this.values = values;
            this.undoRandomOrthogonalTransform = undoRandomOrthogonalTransform;
        }

        @Override
        protected T doGetValue() throws IOException {
            final T vector = getVectorDataType().getVectorFromBytesRef(values.binaryValue());
            // Deserialization returns a new array, so the transform can be undone in place.
            if (undoRandomOrthogonalTransform && vector instanceof float[] floats) {
                RandomOrthogonalVectorTransformer.forDimension(floats.length).inverseTransform(floats, true);
            }
            return vector;
        }
    }

    /**
     * Creates an empty KNNVectorScriptDocValues object based on the provided field name and vector data type.
     *
     * @param fieldName The name of the field.
     * @param type      The data type of the vector.
     * @return An empty KNNVectorScriptDocValues object.
     */
    public static KNNVectorScriptDocValues<?> emptyValues(String fieldName, VectorDataType type) {
        if (type == VectorDataType.FLOAT || type == VectorDataType.HALF_FLOAT) {
            return new KNNVectorScriptDocValues<float[]>(DocIdSetIterator.empty(), fieldName, type) {
                @Override
                protected float[] doGetValue() throws IOException {
                    throw new UnsupportedOperationException("empty values");
                }
            };
        }
        return new KNNVectorScriptDocValues<byte[]>(DocIdSetIterator.empty(), fieldName, type) {
            @Override
            protected byte[] doGetValue() throws IOException {
                throw new UnsupportedOperationException("empty values");
            }
        };
    }
}
