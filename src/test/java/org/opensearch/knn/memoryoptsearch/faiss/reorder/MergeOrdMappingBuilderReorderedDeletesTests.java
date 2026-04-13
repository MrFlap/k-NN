/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.util.Bits;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Test that MergeOrdMappingBuilder correctly handles a reordered source segment with deletions.
 * In a reordered segment, ordToDoc(ord) != ord — it returns the doc ID from the ordToDocMap.
 * When docs are deleted, docMap.get(docId) returns -1 for those docs.
 */
public class MergeOrdMappingBuilderReorderedDeletesTests extends OpenSearchTestCase {

    private static final String FIELD = "vec";

    /**
     * Simulates: reordered segment with 100 vectors, 10 deleted docs.
     * ordToDoc returns a shuffled mapping (not identity).
     * docMap returns -1 for the 10 deleted doc IDs.
     * Expected: totalLiveDocs = 90, liveLocalOrds has 90 entries.
     */
    public void testReorderedSegmentWithDeletions() throws Exception {
        int numVectors = 100;
        int numDeleted = 10;

        // Simulate a reordered ordToDoc: ord -> docId is a permutation
        // e.g., ord 0 -> doc 50, ord 1 -> doc 23, etc.
        int[] ordToDocMap = new int[numVectors];
        for (int i = 0; i < numVectors; i++) {
            ordToDocMap[i] = (i * 7 + 13) % numVectors;  // simple permutation
        }

        // Delete docs with IDs 0-9
        Set<Integer> deletedDocIds = new HashSet<>();
        for (int i = 0; i < numDeleted; i++) {
            deletedDocIds.add(i);
        }

        // Mock FloatVectorValues with non-identity ordToDoc
        FloatVectorValues fvv = mock(FloatVectorValues.class);
        when(fvv.size()).thenReturn(numVectors);
        for (int ord = 0; ord < numVectors; ord++) {
            when(fvv.ordToDoc(ord)).thenReturn(ordToDocMap[ord]);
        }

        KnnVectorsReader reader = mock(KnnVectorsReader.class);
        when(reader.getFloatVectorValues(FIELD)).thenReturn(fvv);

        // docMap: returns -1 for deleted docs, otherwise sequential merged IDs
        int[] docIdToMergedId = new int[numVectors];
        int nextMergedId = 0;
        for (int docId = 0; docId < numVectors; docId++) {
            if (deletedDocIds.contains(docId)) {
                docIdToMergedId[docId] = -1;
            } else {
                docIdToMergedId[docId] = nextMergedId++;
            }
        }
        MergeState.DocMap docMap = sourceDocId -> docIdToMergedId[sourceDocId];

        // liveDocs marks deleted docs
        Bits liveDocs = mock(Bits.class);

        MergeState mergeState = new MergeState(
            new MergeState.DocMap[] { docMap },
            mock(org.apache.lucene.index.SegmentInfo.class),
            null,
            new org.apache.lucene.codecs.StoredFieldsReader[1],
            new org.apache.lucene.codecs.TermVectorsReader[1],
            new org.apache.lucene.codecs.NormsProducer[1],
            new org.apache.lucene.codecs.DocValuesProducer[1],
            new org.apache.lucene.index.FieldInfos[1],
            new Bits[] { liveDocs },
            new org.apache.lucene.codecs.FieldsProducer[1],
            new org.apache.lucene.codecs.PointsReader[1],
            new KnnVectorsReader[] { reader },
            new int[] { numVectors },
            org.apache.lucene.util.InfoStream.NO_OUTPUT,
            null,
            false
        );

        FieldInfo fieldInfo = new FieldInfo(
            FIELD, 0, false, false, false,
            org.apache.lucene.index.IndexOptions.NONE,
            org.apache.lucene.index.DocValuesType.NONE,
            org.apache.lucene.index.DocValuesSkipIndexType.NONE,
            -1, java.util.Collections.emptyMap(), 0, 0, 0, 0,
            org.apache.lucene.index.VectorEncoding.FLOAT32,
            org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN,
            false, false
        );

        MergeOrdMappingBuilder.MergeOrdMapping result = MergeOrdMappingBuilder.build(mergeState, fieldInfo);

        // Should have 90 live docs
        assertEquals(numVectors - numDeleted, result.totalLiveDocs());
        assertEquals(numVectors - numDeleted, result.mergedOrdToDocId().length);

        // No merged doc ID should be -1
        for (int docId : result.mergedOrdToDocId()) {
            assertTrue("Merged doc ID should not be -1, got " + docId, docId >= 0);
        }

        // liveLocalOrds should exist (hasDeletions=true) and have 90 entries
        assertNotNull(result.liveLocalOrds()[0]);
        assertEquals(numVectors - numDeleted, result.liveLocalOrds()[0].length);

        // Each liveLocalOrd should map to a non-deleted doc via ordToDoc
        for (int liveOrd : result.liveLocalOrds()[0]) {
            int docId = ordToDocMap[liveOrd];
            assertFalse("Live ord " + liveOrd + " maps to deleted doc " + docId, deletedDocIds.contains(docId));
        }
    }
}
