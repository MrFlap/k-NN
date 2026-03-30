/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.util.Bits;

import java.io.IOException;
import java.util.Arrays;

/**
 * Builds a mapping from merged ordinals to merged doc IDs across all source segments,
 * accounting for deleted documents and optional index sorting.
 */
public class MergeOrdMappingBuilder {
    private static final Logger log = LogManager.getLogger(MergeOrdMappingBuilder.class);

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
            if (hasDeletions) {
                Bits ld = mergeState.liveDocs[seg];
                StringBuilder sb = new StringBuilder();
                int deadCount = 0;
                int ldLen = ld.length();
                for (int d = 0; d < ldLen; d++) {
                    if (!ld.get(d)) {
                        if (deadCount < 20) sb.append(d).append(",");
                        deadCount++;
                    }
                }
                log.info("[MergeOrdMapping] seg={} liveDocs class={} len={} segSize={} deadDocs(first20)=[{}] totalDead={}",
                    seg, ld.getClass().getSimpleName(), ldLen, segSize, sb, deadCount);
            }
            int[] tempLiveOrds = hasDeletions ? new int[segSize] : null;
            int liveCount = 0;
            int skippedCount = 0;
            MergeState.DocMap docMap = mergeState.docMaps[seg];

            for (int localOrd = 0; localOrd < segSize; localOrd++) {
                int sourceDocId = segValues.ordToDoc(localOrd);
                int mergedDocId = docMap.get(sourceDocId);
                if (mergedDocId == -1) {
                    if (skippedCount < 5) {
                        log.info("[MergeOrdMapping] seg={} SKIPPED localOrd={} sourceDocId={}", seg, localOrd, sourceDocId);
                    }
                    skippedCount++;
                    continue;
                }
                if (hasDeletions && liveCount < 3) {
                    log.info("[MergeOrdMapping] seg={} KEPT localOrd={} sourceDocId={} mergedDocId={}", seg, localOrd, sourceDocId, mergedDocId);
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

            log.info("[MergeOrdMapping] seg={} segSize={} hasDeletions={} liveCount={} skipped={} totalLiveDocs={} valuesClass={} ordToDoc0={}",
                seg, segSize, hasDeletions, liveCount, skippedCount, totalLiveDocs,
                segValues.getClass().getSimpleName(), segSize > 0 ? segValues.ordToDoc(0) : -1);
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
