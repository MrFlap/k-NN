/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.merge;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.MergePolicy;
import org.opensearch.index.engine.EngineConfig;
import org.opensearch.index.engine.EngineFactory;
import org.opensearch.index.engine.Engine;
import org.opensearch.index.engine.InternalEngine;

/**
 * Engine factory that wraps the default merge policy with {@link BPReorderingMergePolicy}
 * to reorder documents by vector similarity during merges, improving disk cache locality
 * for vector search workloads.
 */
@Log4j2
public class KNNEngineFactory implements EngineFactory {

    /** Minimum number of docs in a merged segment before reordering kicks in. */
    private static final int DEFAULT_MIN_MERGE_DOCS_FOR_REORDER = 10_000;

    @Override
    public Engine newReadWriteEngine(EngineConfig config) {
        MergePolicy originalPolicy = config.getMergePolicy();
        BPReorderingMergePolicy reorderingPolicy = new BPReorderingMergePolicy(originalPolicy, new KNNIndexReorderer());
        reorderingPolicy.setMinNaturalMergeNumDocs(DEFAULT_MIN_MERGE_DOCS_FOR_REORDER);

        EngineConfig wrappedConfig = config.toBuilder().mergePolicy(reorderingPolicy).build();
        log.info("KNN BP vector reorder merge policy enabled for shard [{}]", config.getShardId());
        return new InternalEngine(wrappedConfig);
    }
}
