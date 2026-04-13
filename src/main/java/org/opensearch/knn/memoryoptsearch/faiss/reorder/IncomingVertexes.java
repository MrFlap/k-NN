/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.util.hnsw.HnswGraph;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

public class IncomingVertexes {
    public final int[] incomingVertices;
    public final int[] offsets;

    public IncomingVertexes(int[] incomingVertices, int[] offsets) {
        this.incomingVertices = incomingVertices;
        this.offsets = offsets;
    }

    public int getDegree(int index) {
        return offsets[index * 2 + 1] - offsets[index * 2];
    }

    public int startOffset(int index) {
        return offsets[index * 2];
    }

    public int endOffset(int index) {
        // Exclusive
        return offsets[index * 2 + 1];
    }

    public static IncomingVertexes collectIncomingVertices(HnswGraph graph) throws IOException {
        final int numNodes = graph.size();

        // Build incoming adjacency lists
        final List<List<Integer>> incomingLists = new ArrayList<>(numNodes);
        for (int i = 0; i < numNodes; i++) {
            incomingLists.add(new ArrayList<>());
        }

        // Collect all outgoing edges to build incoming edges
        for (int node = 0; node < numNodes; node++) {
            graph.seek(0, node);
            int neighbor;
            while ((neighbor = graph.nextNeighbor()) != NO_MORE_DOCS) {
                incomingLists.get(neighbor).add(node);
            }
        }

        // Count total incoming vertices
        final int totalNumEdges = incomingLists.stream().mapToInt(List::size).sum();

        final int[] vertices = new int[totalNumEdges];
        final int[] offsets = new int[numNodes * 2];
        int vertexIndex = 0;

        for (int node = 0; node < numNodes; node++) {
            // Start
            offsets[node * 2] = vertexIndex;

            for (int incoming : incomingLists.get(node)) {
                vertices[vertexIndex++] = incoming;
            }

            // Exclusive end
            offsets[node * 2 + 1] = vertexIndex;
        }

        return new IncomingVertexes(vertices, offsets);
    }
}
