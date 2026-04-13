/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.util.hnsw.HnswGraph;

import java.io.IOException;
import java.util.function.Consumer;
import java.util.function.Function;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

public final class HnswGraphHelper {
    public static int getOutDegree(final HnswGraph graph, final int index) throws IOException {
        graph.seek(0, index);
        return graph.neighborCount();
    }

    public static void forAllOutgoingNodes(
        final HnswGraph graph,
        final int index,
        final int minNeighbors,
        final Consumer<Integer> vertexIdConsumer
    ) throws IOException {
        int w;
        graph.seek(0, index);
        if (graph.neighborCount() > minNeighbors) {
            while ((w = graph.nextNeighbor()) != NO_MORE_DOCS) {
                vertexIdConsumer.accept(w);
            }
        }
    }

    public static void forAllOutgoingNodes(
        final HnswGraph graph,
        final int index,
        final int minNeighbors,
        final Function<Integer, Boolean> vertexIdConsumer
    ) throws IOException {
        int w;
        graph.seek(0, index);
        if (graph.neighborCount() > minNeighbors) {
            while ((w = graph.nextNeighbor()) != NO_MORE_DOCS) {
                final boolean continu = vertexIdConsumer.apply(w);
                if (continu == false) {
                    break;
                }
            }
        }
    }
}
