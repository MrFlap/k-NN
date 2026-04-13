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
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class MergeOrdMappingBuilderTests extends OpenSearchTestCase {

    private static final String FIELD_NAME = "test_field";

    /**
     * 3 segments with sizes (5, 7, 3), no deletions.
     * DocMap assigns merged doc IDs sequentially: seg0 gets [0..4], seg1 gets [5..11], seg2 gets [12..14].
     */
    public void testThreeSegmentsNoDeletions() throws Exception {
        int[] segSizes = { 5, 7, 3 };
        int totalDocs = 15;

        MergeState mergeState = buildMergeState(segSizes, null, false);
        FieldInfo fieldInfo = createFieldInfo(FIELD_NAME);

        MergeOrdMappingBuilder.MergeOrdMapping result = MergeOrdMappingBuilder.build(mergeState, fieldInfo);

        assertEquals(totalDocs, result.totalLiveDocs());
        assertEquals(totalDocs, result.mergedOrdToDocId().length);

        // All merged doc IDs should be present exactly once
        Set<Integer> docIds = new HashSet<>();
        for (int docId : result.mergedOrdToDocId()) {
            assertTrue("Duplicate merged doc ID: " + docId, docIds.add(docId));
        }
        assertEquals(totalDocs, docIds.size());

        // segmentStarts should be [0, 5, 12]
        assertArrayEquals(new int[] { 0, 5, 12, 15 }, result.segmentStarts());

        // No liveDocs set, so liveLocalOrds should all be null
        for (int[] ords : result.liveLocalOrds()) {
            assertNull(ords);
        }
    }

    /**
     * Segment with deletions: 3 segments, middle segment has some docs deleted (mapped to -1).
     */
    public void testSegmentWithDeletions() throws Exception {
        int[] segSizes = { 3, 5, 2 };
        // In seg1, localOrds 1 and 3 are deleted (docMap returns -1)
        int[][] deletedLocalOrds = { null, new int[] { 1, 3 }, null };

        MergeState mergeState = buildMergeStateWithDeletions(segSizes, deletedLocalOrds, false);
        FieldInfo fieldInfo = createFieldInfo(FIELD_NAME);

        MergeOrdMappingBuilder.MergeOrdMapping result = MergeOrdMappingBuilder.build(mergeState, fieldInfo);

        // seg0: 3 live, seg1: 3 live (5-2), seg2: 2 live => total 8
        assertEquals(8, result.totalLiveDocs());
        assertEquals(8, result.mergedOrdToDocId().length);

        // No doc ID should be -1
        for (int docId : result.mergedOrdToDocId()) {
            assertTrue("Merged doc ID should not be -1", docId >= 0);
        }

        // seg1 should have liveLocalOrds tracking the 3 surviving ords: {0, 2, 4}
        assertNotNull(result.liveLocalOrds()[1]);
        assertArrayEquals(new int[] { 0, 2, 4 }, result.liveLocalOrds()[1]);

        // seg0 and seg2 have no liveDocs set, so liveLocalOrds should be null
        assertNull(result.liveLocalOrds()[0]);
        assertNull(result.liveLocalOrds()[2]);
    }

    /**
     * needsIndexSort: merged doc IDs should be sorted ascending.
     */
    public void testNeedsIndexSort() throws Exception {
        int[] segSizes = { 3, 4 };

        MergeState mergeState = buildMergeStateReversed(segSizes);
        FieldInfo fieldInfo = createFieldInfo(FIELD_NAME);

        MergeOrdMappingBuilder.MergeOrdMapping result = MergeOrdMappingBuilder.build(mergeState, fieldInfo);

        assertEquals(7, result.totalLiveDocs());

        // Verify sorted ascending
        int[] docIds = result.mergedOrdToDocId();
        for (int i = 1; i < docIds.length; i++) {
            assertTrue("mergedOrdToDocId not sorted at index " + i, docIds[i - 1] <= docIds[i]);
        }
    }

    // --- Helper methods ---

    private FieldInfo createFieldInfo(String name) {
        return new FieldInfo(
            name, 0, false, false, false,
            org.apache.lucene.index.IndexOptions.NONE,
            org.apache.lucene.index.DocValuesType.NONE,
            org.apache.lucene.index.DocValuesSkipIndexType.NONE,
            -1, java.util.Collections.emptyMap(), 0, 0, 0, 0,
            org.apache.lucene.index.VectorEncoding.FLOAT32,
            org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN,
            false, false
        );
    }

    private FloatVectorValues mockFloatVectorValues(int size) throws IOException {
        FloatVectorValues fvv = mock(FloatVectorValues.class);
        when(fvv.size()).thenReturn(size);
        for (int i = 0; i < size; i++) {
            when(fvv.ordToDoc(i)).thenReturn(i);
        }
        return fvv;
    }

    private MergeState buildMergeState(int[] segSizes, int[][] deletedLocalOrds, boolean needsIndexSort) throws Exception {
        int numSegs = segSizes.length;

        KnnVectorsReader[] readers = new KnnVectorsReader[numSegs];
        MergeState.DocMap[] docMaps = new MergeState.DocMap[numSegs];
        Bits[] liveDocs = new Bits[numSegs];
        int[] maxDocs = new int[numSegs];

        int offset = 0;
        for (int seg = 0; seg < numSegs; seg++) {
            FloatVectorValues fvv = mockFloatVectorValues(segSizes[seg]);
            KnnVectorsReader reader = mock(KnnVectorsReader.class);
            when(reader.getFloatVectorValues(FIELD_NAME)).thenReturn(fvv);
            readers[seg] = reader;

            final int segOffset = offset;
            docMaps[seg] = sourceDocId -> segOffset + sourceDocId;
            liveDocs[seg] = null;
            maxDocs[seg] = segSizes[seg];
            offset += segSizes[seg];
        }

        return createMergeState(docMaps, readers, liveDocs, maxDocs, needsIndexSort);
    }

    private MergeState buildMergeStateWithDeletions(int[] segSizes, int[][] deletedLocalOrds, boolean needsIndexSort) throws Exception {
        int numSegs = segSizes.length;

        KnnVectorsReader[] readers = new KnnVectorsReader[numSegs];
        MergeState.DocMap[] docMaps = new MergeState.DocMap[numSegs];
        Bits[] liveDocs = new Bits[numSegs];
        int[] maxDocs = new int[numSegs];

        int offset = 0;
        for (int seg = 0; seg < numSegs; seg++) {
            FloatVectorValues fvv = mockFloatVectorValues(segSizes[seg]);
            KnnVectorsReader reader = mock(KnnVectorsReader.class);
            when(reader.getFloatVectorValues(FIELD_NAME)).thenReturn(fvv);
            readers[seg] = reader;

            final Set<Integer> deleted;
            if (deletedLocalOrds != null && deletedLocalOrds[seg] != null) {
                deleted = new HashSet<>();
                for (int d : deletedLocalOrds[seg]) {
                    deleted.add(d);
                }
                liveDocs[seg] = mock(Bits.class);
            } else {
                deleted = Set.of();
                liveDocs[seg] = null;
            }

            // Precompute stable mapping
            int[] mapping = new int[segSizes[seg]];
            int mergedId = offset;
            for (int i = 0; i < segSizes[seg]; i++) {
                mapping[i] = deleted.contains(i) ? -1 : mergedId++;
            }
            docMaps[seg] = sourceDocId -> mapping[sourceDocId];
            maxDocs[seg] = segSizes[seg];
            offset = mergedId;
        }

        return createMergeState(docMaps, readers, liveDocs, maxDocs, needsIndexSort);
    }

    private MergeState buildMergeStateReversed(int[] segSizes) throws Exception {
        int numSegs = segSizes.length;
        int totalDocs = Arrays.stream(segSizes).sum();

        KnnVectorsReader[] readers = new KnnVectorsReader[numSegs];
        MergeState.DocMap[] docMaps = new MergeState.DocMap[numSegs];
        Bits[] liveDocs = new Bits[numSegs];
        int[] maxDocs = new int[numSegs];

        int reverseOffset = totalDocs - 1;
        for (int seg = 0; seg < numSegs; seg++) {
            FloatVectorValues fvv = mockFloatVectorValues(segSizes[seg]);
            KnnVectorsReader reader = mock(KnnVectorsReader.class);
            when(reader.getFloatVectorValues(FIELD_NAME)).thenReturn(fvv);
            readers[seg] = reader;

            final int segReverseStart = reverseOffset;
            docMaps[seg] = sourceDocId -> segReverseStart - sourceDocId;
            liveDocs[seg] = null;
            maxDocs[seg] = segSizes[seg];
            reverseOffset -= segSizes[seg];
        }

        return createMergeState(docMaps, readers, liveDocs, maxDocs, true);
    }

    private MergeState createMergeState(
        MergeState.DocMap[] docMaps,
        KnnVectorsReader[] readers,
        Bits[] liveDocs,
        int[] maxDocs,
        boolean needsIndexSort
    ) {
        int numSegs = readers.length;
        return new MergeState(
            docMaps,
            mock(org.apache.lucene.index.SegmentInfo.class),
            null,  // mergeFieldInfos
            new org.apache.lucene.codecs.StoredFieldsReader[numSegs],
            new org.apache.lucene.codecs.TermVectorsReader[numSegs],
            new org.apache.lucene.codecs.NormsProducer[numSegs],
            new org.apache.lucene.codecs.DocValuesProducer[numSegs],
            new org.apache.lucene.index.FieldInfos[numSegs],
            liveDocs,
            new org.apache.lucene.codecs.FieldsProducer[numSegs],
            new org.apache.lucene.codecs.PointsReader[numSegs],
            readers,
            maxDocs,
            org.apache.lucene.util.InfoStream.NO_OUTPUT,
            null,  // intraMergeTaskExecutor
            needsIndexSort
        );
    }
}
