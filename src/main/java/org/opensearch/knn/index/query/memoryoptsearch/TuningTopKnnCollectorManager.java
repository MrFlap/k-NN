/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.memoryoptsearch;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.search.knn.MultiLeafKnnCollector;
import org.apache.lucene.util.hnsw.BlockingFloatHeap;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentParser;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Map;

//
// TuningTopKnnCollectorManager
//

public class TuningTopKnnCollectorManager implements KnnCollectorManager {
    private final int originalK;
    private final int newK;
    private final float kExpansion;
    private final float greediness;
    private final BlockingFloatHeap globalScoreQueue;

    public TuningTopKnnCollectorManager(int originalK, IndexSearcher indexSearcher) {
        // Save orignal K
        this.originalK = originalK;

        // Set new parameters from a file
        // the file is located at NFS mount point.
        Map<String, Object> parameters = loadParametersFromJson("/efs/hptet/parameters.json");
        final int faissDefaultEfSearch = 100;
        this.kExpansion = (float) parameters.get("kExpansion");
        this.newK = (int) (faissDefaultEfSearch * kExpansion);
        this.greediness = (float) parameters.get("greediness");

        // Create global queue
        final boolean isMultiSegments = indexSearcher.getIndexReader().leaves().size() > 1;
        this.globalScoreQueue = isMultiSegments ? new BlockingFloatHeap(originalK) : null;
    }

    private Map<String, Object> loadParametersFromJson(String filePath) {
        try {
            String content = new String(Files.readAllBytes(Paths.get(filePath)));
            try (XContentParser parser = JsonXContent.jsonXContent.createParser(
                    NamedXContentRegistry.EMPTY,
                    DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                    content)) {
                return parser.map();
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to load parameters from " + filePath, e);
        }
    }

    public KnnCollector newCollector(int visitedLimit, KnnSearchStrategy searchStrategy, LeafReaderContext context) throws IOException {
        if (globalScoreQueue == null) {
            return new TopKnnCollector(originalK, visitedLimit, searchStrategy);
        } else {
            return new MultiLeafKnnCollector(
                    newK,         // Assign new k
                    greediness,   // Assign new greediness
                    256,          // Default value
                    globalScoreQueue,
                    new TakeTopKnnCollector(originalK, newK, visitedLimit, searchStrategy));
        }
    }
}