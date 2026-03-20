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

        // First pass: count total live docs and collect live local ords
        int totalLiveDocs = 0;
        for (int seg = 0; seg < numSegments; seg++) {
            segmentStarts[seg] = totalLiveDocs;

            if (mergeState.knnVectorsReaders[seg] == null) {
                liveLocalOrds[seg] = null;
                continue;
            }

            FloatVectorValues segValues = mergeState.knnVectorsReaders[seg].getFloatVectorValues(fieldInfo.name);
            if (segValues == null) {
                liveLocalOrds[seg] = null;
                continue;
            }

            int segSize = segValues.size();
            int liveCount = 0;
            int[] tempLiveOrds = new int[segSize];

            for (int localOrd = 0; localOrd < segSize; localOrd++) {
                int sourceDocId = segValues.ordToDoc(localOrd);
                int mergedDocId = mergeState.docMaps[seg].get(sourceDocId);
                if (mergedDocId == -1) {
                    continue;
                }
                tempLiveOrds[liveCount++] = localOrd;
            }

            if (mergeState.liveDocs[seg] != null) {
                liveLocalOrds[seg] = Arrays.copyOf(tempLiveOrds, liveCount);
            } else {
                liveLocalOrds[seg] = null;
            }
            totalLiveDocs += liveCount;
        }
        segmentStarts[numSegments] = totalLiveDocs;

        // Second pass: build mergedOrdToDocId
        int[] mergedOrdToDocId = new int[totalLiveDocs];
        int idx = 0;
        for (int seg = 0; seg < numSegments; seg++) {
            if (mergeState.knnVectorsReaders[seg] == null) {
                continue;
            }

            FloatVectorValues segValues = mergeState.knnVectorsReaders[seg].getFloatVectorValues(fieldInfo.name);
            if (segValues == null) {
                continue;
            }

            int segSize = segValues.size();
            for (int localOrd = 0; localOrd < segSize; localOrd++) {
                int sourceDocId = segValues.ordToDoc(localOrd);
                int mergedDocId = mergeState.docMaps[seg].get(sourceDocId);
                if (mergedDocId == -1) {
                    continue;
                }
                mergedOrdToDocId[idx++] = mergedDocId;
            }
        }

        if (mergeState.needsIndexSort) {
            Arrays.sort(mergedOrdToDocId);
        }

        return new MergeOrdMapping(mergedOrdToDocId, segmentStarts, liveLocalOrds, totalLiveDocs);
    }
}
