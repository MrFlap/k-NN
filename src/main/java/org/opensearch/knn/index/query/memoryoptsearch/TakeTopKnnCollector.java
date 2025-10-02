/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.memoryoptsearch;


import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.knn.KnnSearchStrategy;

//
// TakeTopKnnCollector :
//   Since `k` is expanded, we need to make sure return exactly `k` results
//   that was requested at the beginning.
//
public class TakeTopKnnCollector extends TopKnnCollector {
    private int originalK;

    public TakeTopKnnCollector(int originalK, int k, int visitLimit, KnnSearchStrategy searchStrategy) {
        super(k, visitLimit, searchStrategy);
    }

    public TopDocs topDocs() {
        assert this.queue.size() <= this.k() : "Tried to collect more results than the maximum number allowed";

        // Make sure only returning `originalK` results.
        final ScoreDoc[] scoreDocs = new ScoreDoc[originalK];

        for(int i = 1; i <= scoreDocs.length; ++i) {
            scoreDocs[scoreDocs.length - i] = new ScoreDoc(this.queue.topNode(), this.queue.topScore());
            this.queue.pop();
        }

        TotalHits.Relation relation = this.earlyTerminated() ? TotalHits.Relation.GREATER_THAN_OR_EQUAL_TO : TotalHits.Relation.EQUAL_TO;
        return new TopDocs(new TotalHits(this.visitedCount(), relation), scoreDocs);
    }
}