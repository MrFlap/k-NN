/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.analysis.MockAnalyzer;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;

import java.io.IOException;
import java.util.List;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class KnnVectorValuesFetcherTests extends KNNTestCase {
    private static final String FIELD_NAME = "test-vector";

    public void testFetch_whenFieldIsNotTransformed_thenReturnsStoredVector() throws IOException {
        final float[] vector = { 1.0f, 2.0f, 3.0f };
        assertArrayEquals(vector, fetch(vector, false), 0f);
    }

    public void testFetch_whenFieldIsRandomOrthogonallyTransformed_thenReturnsOriginalVector() throws IOException {
        final float[] original = { 1.0f, -2.0f, 3.0f, 0.25f, 9.0f, -4.0f, 6.5f };
        final float[] stored = RandomOrthogonalVectorTransformer.forDimension(original.length).transform(original, false);
        assertArrayEquals(original, fetch(stored, true), 1e-5f);
    }

    private float[] fetch(final float[] storedVector, final boolean transformed) throws IOException {
        final FieldType fieldType = new FieldType(
            KnnFloatVectorField.createFieldType(storedVector.length, VectorSimilarityFunction.EUCLIDEAN)
        );
        if (transformed) {
            fieldType.putAttribute(KNNConstants.RANDOM_ORTHOGONAL_TRANSFORM, KNNConstants.RANDOM_ORTHOGONAL_TRANSFORM_FWHH_V1);
        }
        fieldType.freeze();

        try (Directory directory = newDirectory()) {
            try (IndexWriter writer = new IndexWriter(directory, newIndexWriterConfig(new MockAnalyzer(random())))) {
                final Document doc = new Document();
                doc.add(new KnnFloatVectorField(FIELD_NAME, storedVector, fieldType));
                writer.addDocument(doc);
                writer.commit();
            }
            try (DirectoryReader reader = DirectoryReader.open(directory)) {
                final LeafReader leafReader = reader.leaves().get(0).reader();
                final KNNVectorFieldType mappedFieldType = mock(KNNVectorFieldType.class);
                when(mappedFieldType.name()).thenReturn(FIELD_NAME);

                final List<Object> values = new KnnVectorValuesFetcher(mappedFieldType, FIELD_NAME).fetch(leafReader, 0);
                assertEquals(1, values.size());
                return (float[]) values.get(0);
            }
        }
    }
}
