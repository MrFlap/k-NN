/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from org.apache.lucene.misc.index.BPReorderingMergePolicy
 * to avoid requiring the lucene-misc dependency.
 */

package org.opensearch.knn.index.engine.merge;

import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.FilterMergePolicy;
import org.apache.lucene.index.MergePolicy;
import org.apache.lucene.index.MergeTrigger;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.misc.index.AbstractBPReorderer;
import org.apache.lucene.misc.index.IndexReorderer;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.SetOnce;

import java.io.IOException;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.Executor;

/**
 * A merge policy that reorders merged segments according to an {@link IndexReorderer}.
 * When reordering doesn't have enough RAM, it simply skips reordering in order not to fail the merge.
 */
public final class BPReorderingMergePolicy extends FilterMergePolicy {

    static final String REORDERED = "bp.reordered";

    private final IndexReorderer reorderer;
    private int minNaturalMergeNumDocs = 1;

    /**
     * @param in the merge policy to use to compute merges
     * @param reorderer the {@link IndexReorderer} to use to renumber doc IDs
     */
    public BPReorderingMergePolicy(MergePolicy in, IndexReorderer reorderer) {
        super(in);
        this.reorderer = reorderer;
    }

    /**
     * Set the minimum number of docs that a merge must produce for the resulting segment to be reordered.
     */
    public void setMinNaturalMergeNumDocs(int minNaturalMergeNumDocs) {
        if (minNaturalMergeNumDocs < 1) {
            throw new IllegalArgumentException("minNaturalMergeNumDocs must be at least 1, got " + minNaturalMergeNumDocs);
        }
        this.minNaturalMergeNumDocs = minNaturalMergeNumDocs;
    }

    private MergeSpecification maybeReorder(MergeSpecification spec, boolean forced, SegmentInfos infos) {
        if (spec == null) {
            return null;
        }

        final int minNumDocs;
        if (forced) {
            minNumDocs = 1;
        } else {
            minNumDocs = this.minNaturalMergeNumDocs;
        }

        MergeSpecification newSpec = new MergeSpecification();
        for (OneMerge oneMerge : spec.merges) {
            newSpec.add(new OneMerge(oneMerge) {
                private final SetOnce<Boolean> reordered = new SetOnce<>();

                @Override
                public CodecReader wrapForMerge(CodecReader reader) throws IOException {
                    return oneMerge.wrapForMerge(reader);
                }

                @Override
                public Sorter.DocMap reorder(CodecReader reader, Directory dir, Executor executor) throws IOException {
                    Sorter.DocMap docMap = null;
                    if (reader.numDocs() >= minNumDocs) {
                        try {
                            docMap = reorderer.computeDocMap(reader, dir, executor);
                        } catch (AbstractBPReorderer.NotEnoughRAMException e) {
                            // skip reordering, not enough RAM
                        }
                    }
                    reordered.set(docMap != null);
                    return docMap;
                }

                @Override
                public void setMergeInfo(SegmentCommitInfo info) {
                    Boolean wasReordered = this.reordered.get();
                    if (wasReordered == null) {
                        wasReordered = false;
                    }
                    info.info.addDiagnostics(Collections.singletonMap(REORDERED, Boolean.toString(wasReordered)));
                    super.setMergeInfo(info);
                }
            });
        }
        return newSpec;
    }

    @Override
    public MergeSpecification findMerges(MergeTrigger mergeTrigger, SegmentInfos segmentInfos, MergeContext mergeContext)
        throws IOException {
        return maybeReorder(super.findMerges(mergeTrigger, segmentInfos, mergeContext), false, segmentInfos);
    }

    @Override
    public MergeSpecification findForcedMerges(
        SegmentInfos segmentInfos,
        int maxSegmentCount,
        Map<SegmentCommitInfo, Boolean> segmentsToMerge,
        MergeContext mergeContext
    ) throws IOException {
        return maybeReorder(super.findForcedMerges(segmentInfos, maxSegmentCount, segmentsToMerge, mergeContext), true, segmentInfos);
    }

    @Override
    public MergeSpecification findForcedDeletesMerges(SegmentInfos segmentInfos, MergeContext mergeContext) throws IOException {
        return maybeReorder(super.findForcedDeletesMerges(segmentInfos, mergeContext), true, segmentInfos);
    }

    @Override
    public MergeSpecification findFullFlushMerges(MergeTrigger mergeTrigger, SegmentInfos segmentInfos, MergeContext mergeContext)
        throws IOException {
        return maybeReorder(super.findFullFlushMerges(mergeTrigger, segmentInfos, mergeContext), false, segmentInfos);
    }

    @Override
    public MergeSpecification findMerges(CodecReader... readers) throws IOException {
        return maybeReorder(super.findMerges(readers), true, null);
    }
}
