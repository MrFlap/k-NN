/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.clumping;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOSupplier;
import org.opensearch.common.StopWatch;
import org.opensearch.knn.index.query.KNNQuery;
import org.opensearch.knn.index.query.KNNWeight;
import org.opensearch.knn.index.query.PerLeafResult;
import org.opensearch.knn.index.query.ResultUtil;
import org.opensearch.knn.index.query.common.QueryUtils;
import org.opensearch.knn.indices.ModelDao;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Callable;

/**
 * A k-NN query that supports clumping-based search optimization.
 * 
 * This query wraps a standard KNNQuery and adds clumping expansion logic:
 * 1. Execute the underlying k-NN search on marker vectors
 * 2. Expand results to include hidden vectors associated with found markers
 * 3. Re-score and return top-k from combined results
 * 
 * This is similar to how NativeEngineKnnVectorQuery handles rescoring,
 * but instead of oversampling, it expands marker results to hidden vectors.
 */
@Log4j2
@Getter
@RequiredArgsConstructor
public class ClumpingKnnVectorQuery extends Query {

    private final KNNQuery knnQuery;
    private final QueryUtils queryUtils;
    private final ClumpingContext clumpingContext;
    private final ModelDao modelDao;

    @Override
    public Weight createWeight(IndexSearcher indexSearcher, ScoreMode scoreMode, float boost) throws IOException {
        // Create the underlying weight
        final KNNWeight knnWeight = (KNNWeight) knnQuery.createWeight(indexSearcher, scoreMode, boost);

        // Run search on marker vectors
        final IndexReader reader = indexSearcher.getIndexReader();
        List<LeafReaderContext> leafReaderContexts = reader.leaves();
        final int k = knnQuery.getK();

        // Execute search on all leaves
        List<PerLeafResult> perLeafResults = doSearch(indexSearcher, leafReaderContexts, knnWeight, k);

        // If clumping is enabled, expand marker results to include hidden vectors
        if (clumpingContext != null && clumpingContext.isEffectivelyEnabled()) {
            StopWatch stopWatch = new StopWatch().start();
            perLeafResults = expandClumpedResults(indexSearcher, leafReaderContexts, perLeafResults, k);
            long expansionTime = stopWatch.stop().totalTime().millis();
            log.debug(
                "Clumping expansion took {} ms for {} segments",
                expansionTime,
                leafReaderContexts.size()
            );
        }

        // Reduce to top-k across all segments
        ResultUtil.reduceToTopK(perLeafResults, k);

        // Build final TopDocs
        TopDocs[] topDocs = new TopDocs[perLeafResults.size()];
        for (int i = 0; i < perLeafResults.size(); i++) {
            TopDocs leafTopDocs = perLeafResults.get(i).getResult();
            for (ScoreDoc scoreDoc : leafTopDocs.scoreDocs) {
                scoreDoc.doc += leafReaderContexts.get(i).docBase;
            }
            topDocs[i] = leafTopDocs;
        }

        TopDocs topK = TopDocs.merge(k, topDocs);

        if (topK.scoreDocs.length == 0) {
            return new MatchNoDocsQuery().createWeight(indexSearcher, scoreMode, boost);
        }

        return queryUtils.createDocAndScoreQuery(reader, topK, knnWeight).createWeight(indexSearcher, scoreMode, boost);
    }

    private List<PerLeafResult> doSearch(
        IndexSearcher indexSearcher,
        List<LeafReaderContext> leafReaderContexts,
        KNNWeight knnWeight,
        int k
    ) throws IOException {
        List<Callable<PerLeafResult>> tasks = new ArrayList<>(leafReaderContexts.size());
        for (LeafReaderContext leafReaderContext : leafReaderContexts) {
            tasks.add(() -> searchLeaf(leafReaderContext, knnWeight, k));
        }
        return indexSearcher.getTaskExecutor().invokeAll(tasks);
    }

    private PerLeafResult searchLeaf(LeafReaderContext ctx, KNNWeight queryWeight, int k) throws IOException {
        final PerLeafResult perLeafResult = queryWeight.searchLeaf(ctx, k);
        final Bits liveDocs = ctx.reader().getLiveDocs();
        if (liveDocs != null) {
            List<ScoreDoc> list = new ArrayList<>();
            for (ScoreDoc scoreDoc : perLeafResult.getResult().scoreDocs) {
                if (liveDocs.get(scoreDoc.doc)) {
                    list.add(scoreDoc);
                }
            }
            ScoreDoc[] filteredScoreDoc = list.toArray(new ScoreDoc[0]);
            org.apache.lucene.search.TotalHits totalHits = new org.apache.lucene.search.TotalHits(
                filteredScoreDoc.length,
                org.apache.lucene.search.TotalHits.Relation.EQUAL_TO
            );
            perLeafResult.setResult(new TopDocs(totalHits, filteredScoreDoc));
        }
        return perLeafResult;
    }

    private List<PerLeafResult> expandClumpedResults(
        IndexSearcher indexSearcher,
        List<LeafReaderContext> leafReaderContexts,
        List<PerLeafResult> perLeafResults,
        int k
    ) throws IOException {
        ClumpingSearchHandler handler = new ClumpingSearchHandler(clumpingContext, modelDao);
        
        if (!handler.isClumpingEnabled()) {
            return perLeafResults;
        }

        List<Callable<PerLeafResult>> expansionTasks = new ArrayList<>(leafReaderContexts.size());
        
        for (int i = 0; i < perLeafResults.size(); i++) {
            final int index = i;
            final LeafReaderContext leafReaderContext = leafReaderContexts.get(i);
            final PerLeafResult perLeafResult = perLeafResults.get(i);
            
            expansionTasks.add(() -> handler.expandClumpedResults(
                leafReaderContext,
                perLeafResult,
                knnQuery.getQueryVector(),
                knnQuery.getField(),
                k
            ));
        }

        return indexSearcher.getTaskExecutor().invokeAll(expansionTasks);
    }

    @Override
    public String toString(String field) {
        return this.getClass().getSimpleName() + "[" + field + "]..." 
            + KNNQuery.class.getSimpleName() + "[" + knnQuery.toString() + "]"
            + " clumping=" + (clumpingContext != null ? clumpingContext.getClumpingFactor() : "disabled");
    }

    @Override
    public void visit(QueryVisitor visitor) {
        visitor.visitLeaf(this);
    }

    @Override
    public boolean equals(Object obj) {
        if (!sameClassAs(obj)) {
            return false;
        }
        ClumpingKnnVectorQuery other = (ClumpingKnnVectorQuery) obj;
        return knnQuery.equals(other.knnQuery) 
            && Objects.equals(clumpingContext, other.clumpingContext);
    }

    @Override
    public int hashCode() {
        return Objects.hash(classHash(), knnQuery.hashCode(), clumpingContext);
    }
}
