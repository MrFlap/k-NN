/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.MergeState;

import java.io.IOException;
import java.util.Arrays;

/**
 * Builds a mapping from merged ordinals to merged doc IDs across all source segments,
 * accounting for deleted documents and optional index sorting.
 */
public class MergeOrdMappingBuilder {

    public record MergeOrdMapping(int[] mergedOrdToDocId, int[] segmentStarts, int[][] liveLocalOrds, int totalLiveDocs) {}

    public static MergeOrdMapping build(MergeState mergeState, FieldInfo fieldInfo) throws IOException {
        final int numSegments = mergeState.knnVectorsReaders.length;
        final int[] segmentStarts = new int[numSegments + 1];
        final int[][] liveLocalOrds = new int[numSegments][];

        // Compute max possible size for mergedOrdToDocId
        int maxDocs = 0;
        for (int seg = 0; seg < numSegments; seg++) {
            maxDocs += mergeState.maxDocs[seg];
        }
        int[] mergedOrdToDocId = new int[maxDocs];

        // Single pass: build all mappings at once
        int totalLiveDocs = 0;
        for (int seg = 0; seg < numSegments; seg++) {
            segmentStarts[seg] = totalLiveDocs;

            if (mergeState.knnVectorsReaders[seg] == null) {
                continue;
            }

            FloatVectorValues segValues = mergeState.knnVectorsReaders[seg].getFloatVectorValues(fieldInfo.name);
            if (segValues == null) {
                continue;
            }

            int segSize = segValues.size();
            boolean hasDeletions = mergeState.liveDocs[seg] != null;
            int[] tempLiveOrds = hasDeletions ? new int[segSize] : null;
            int liveCount = 0;
            MergeState.DocMap docMap = mergeState.docMaps[seg];

            for (int localOrd = 0; localOrd < segSize; localOrd++) {
                int sourceDocId = segValues.ordToDoc(localOrd);
                int mergedDocId = docMap.get(sourceDocId);
                if (mergedDocId == -1) {
                    continue;
                }
                mergedOrdToDocId[totalLiveDocs + liveCount] = mergedDocId;
                if (hasDeletions) {
                    tempLiveOrds[liveCount] = localOrd;
                }
                liveCount++;
            }

            if (hasDeletions) {
                liveLocalOrds[seg] = Arrays.copyOf(tempLiveOrds, liveCount);
            }
            totalLiveDocs += liveCount;
        }
        segmentStarts[numSegments] = totalLiveDocs;

        // Trim to actual size
        if (mergedOrdToDocId.length != totalLiveDocs) {
            mergedOrdToDocId = Arrays.copyOf(mergedOrdToDocId, totalLiveDocs);
        }

        if (mergeState.needsIndexSort) {
            Arrays.sort(mergedOrdToDocId);
        }

        return new MergeOrdMapping(mergedOrdToDocId, segmentStarts, liveLocalOrds, totalLiveDocs);
    }
}
