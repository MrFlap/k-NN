/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.opensearch.knn.memoryoptsearch.faiss.FaissHNSW;
import org.opensearch.knn.memoryoptsearch.faiss.FaissHnswGraph;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIdMapIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndex;
import org.opensearch.knn.memoryoptsearch.faiss.FaissMemoryOptimizedSearcher;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissBinaryHnswIndex;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissBinaryIndex;
import org.opensearch.knn.memoryoptsearch.faiss.binary.FaissIndexBinaryFlat;

import java.io.IOException;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicReference;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

public class GOrderVectorReorder {
    public static void main(String... args) throws IOException {
        final String faissGraphDir = "/Users/kdooyong/workspace/Gorder/kdy";
        final String faissGraphFileName = "_11_165_target_field.faiss";
        final int window = 16;

        try (final Directory directory = new MMapDirectory(Path.of(faissGraphDir));) {
            try (final IndexInput indexInput = directory.openInput(faissGraphFileName, IOContext.DEFAULT)) {
                final FaissIndex faissIndex = FaissIndex.load(indexInput);
                final int numVectors = faissIndex.getTotalNumberOfVectors();
                System.out.println("Total #vectors: " + numVectors);
                if (faissIndex instanceof FaissIdMapIndex idMapIndex) {
                    // Get the permutation
                    final FaissHNSW faissHNSW = idMapIndex.getFaissHnsw();
                    // Permutation -> newOrd2Old
                    final long s = System.nanoTime();
                    final int[] permutation = getPermutation(faissHNSW, indexInput, window);
                    final long e = System.nanoTime();
                    System.out.println("Reordering took : " + (e - s) / 1e6 + "ms");
                    final ReorderOrdMap reorderOrdMap = new ReorderOrdMap(permutation);
                    System.out.println(Arrays.toString(permutation));

                    // Transform the index
                    if (true) {
                        try {
                            directory.deleteFile(faissGraphFileName + ".reorder");
                        } catch (NoSuchFileException x) {}
                        try (final IndexOutput indexOutput = directory.createOutput(faissGraphFileName + ".reorder", IOContext.DEFAULT)) {
                            FaissIndexReorderTransformer.transform(faissIndex, indexInput, indexOutput, reorderOrdMap);
                        }
                        try (
                            final IndexInput indexInputForNewGraph = directory.openInput(faissGraphFileName + ".reorder", IOContext.DEFAULT)
                        ) {
                            final FaissIndex newFaissIndex = FaissIndex.load(indexInputForNewGraph);
                            validate(faissIndex, indexInput, newFaissIndex, indexInputForNewGraph, reorderOrdMap);

                            searchTest(
                                ((FaissBinaryHnswIndex) idMapIndex.getNestedIndex()).getStorage(),
                                indexInput,
                                indexInputForNewGraph
                            );
                        }
                    }
                } else {
                    throw new IllegalStateException("faissIndex is not FaissIdMapIndex! Actual type: " + faissIndex.getClass());
                }
            }
        }
    }

    private static void searchTest(final FaissBinaryIndex storage, final IndexInput indexInput, final IndexInput indexInputNew)
        throws IOException {
        // Prepare
        int N = 10000;
        int topK = 30;
        for (int k = 0; k < N; ++k) {
            byte[] query = new byte[storage.getCodeSize()];
            for (int i = 0; i < query.length; ++i) {
                query[i] = (byte) ThreadLocalRandom.current().nextInt();
            }
            final AcceptDocs acceptDocs = AcceptDocs.fromLiveDocs(null, storage.getTotalNumberOfVectors());

            // Search on the original index
            indexInput.seek(0);
            final FaissMemoryOptimizedSearcher searcher = new FaissMemoryOptimizedSearcher(indexInput, null, null, null);
            final TopKnnCollector topKnnCollector = new TopKnnCollector(topK, Integer.MAX_VALUE);
            final long s = System.nanoTime();
            searcher.search(query, topKnnCollector, acceptDocs);
            final long e = System.nanoTime();
            final TopDocs topDocs = topKnnCollector.topDocs();

            // Search on the new index
            indexInputNew.seek(0);
            final FaissMemoryOptimizedSearcher searcherNew = new FaissMemoryOptimizedSearcher(indexInputNew, null, null, null);
            final TopKnnCollector topKnnCollectorNew = new TopKnnCollector(topK, Integer.MAX_VALUE);
            final long sNew = System.nanoTime();
            searcherNew.search(query, topKnnCollectorNew, acceptDocs);
            final long eNew = System.nanoTime();
            final TopDocs topDocsNew = topKnnCollectorNew.topDocs();
            final float lastScore = topDocs.scoreDocs[topDocs.scoreDocs.length - 1].score;
            final float lastScoreRescore = topDocsNew.scoreDocs[topDocsNew.scoreDocs.length - 1].score;
            System.out.println(
                "Original vs Reordered took: "
                    + ((e - s) / 1000)
                    + " vs "
                    + ((eNew - sNew) / 1000)
                    + ", Last recall vs Reordered last recall: "
                    + lastScore
                    + " vs "
                    + lastScoreRescore
                    + "[GE?"
                    + (lastScoreRescore >= lastScore)
                    + "]"
            );

            assert topDocs.scoreDocs.length == topDocsNew.scoreDocs.length;
            assert Math.abs(lastScoreRescore - lastScore) < 1e-3;
        }
    }

    private static void validate(
        FaissIndex faissIndex,
        IndexInput indexInput,
        FaissIndex faissIndexNew,
        IndexInput indexInputNew,
        ReorderOrdMap reorderOrdMap
    ) throws IOException {
        final FaissIdMapIndex idMapIndex = (FaissIdMapIndex) faissIndex;
        final FaissBinaryHnswIndex hnswIndex = (FaissBinaryHnswIndex) idMapIndex.getNestedIndex();
        final FaissHNSW faissHNSW = hnswIndex.getFaissHnsw();
        final FaissIndexBinaryFlat binaryFlat = (FaissIndexBinaryFlat) hnswIndex.getStorage();

        final FaissIdMapIndex idMapIndexNew = (FaissIdMapIndex) faissIndexNew;
        final FaissBinaryHnswIndex hnswIndexNew = (FaissBinaryHnswIndex) idMapIndexNew.getNestedIndex();
        final FaissHNSW faissHNSWNew = hnswIndexNew.getFaissHnsw();
        final FaissIndexBinaryFlat binaryFlatNew = (FaissIndexBinaryFlat) hnswIndexNew.getStorage();

        // Storage level check
        assert binaryFlat.getCodeSize() == binaryFlatNew.getCodeSize();
        assert binaryFlat.getTotalNumberOfVectors() == binaryFlatNew.getTotalNumberOfVectors();
        final ByteVectorValues byteVectorValues = binaryFlat.getByteValues(indexInput);
        final ByteVectorValues byteVectorValuesNew = binaryFlatNew.getByteValues(indexInputNew);
        for (int i = 0; i < binaryFlat.getTotalNumberOfVectors(); ++i) {
            byte[] vec1 = byteVectorValuesNew.vectorValue(i);
            final int oldOrd = reorderOrdMap.newOrd2Old[i];
            byte[] vec = byteVectorValues.vectorValue(oldOrd);
            assert Arrays.equals(vec, vec1);
        }

        // HNSW graph check
        assert faissHNSW.getEntryPoint() == reorderOrdMap.newOrd2Old[faissHNSWNew.getEntryPoint()];
        assert faissHNSW.getEfConstruct() == faissHNSWNew.getEfConstruct();
        assert faissHNSW.getEfSearch() == faissHNSWNew.getEfSearch();
        assert faissHNSW.getMaxLevel() == faissHNSWNew.getMaxLevel();
        assert faissHNSW.getTotalNumberOfVectors() == faissHNSWNew.getTotalNumberOfVectors();

        // assignProbas check
        {
            indexInput.seek(faissHNSW.getAssignProbas().getBaseOffset());
            indexInputNew.seek(faissHNSWNew.getAssignProbas().getBaseOffset());
            assert faissHNSW.getAssignProbas().getSectionSize() == faissHNSWNew.getAssignProbas().getSectionSize();
            long size = faissHNSW.getAssignProbas().getSectionSize();
            while (size > 0) {
                assert indexInput.readByte() == indexInputNew.readByte();
                --size;
            }
        }

        // cumNumberNeighborPerLevel check
        assert Arrays.equals(faissHNSW.getCumNumberNeighborPerLevel(), faissHNSWNew.getCumNumberNeighborPerLevel());

        // levels checks
        {
            for (int oldOrd = 0; oldOrd < faissHNSW.getTotalNumberOfVectors(); ++oldOrd) {
                indexInput.seek(faissHNSW.getLevels().getBaseOffset() + Integer.BYTES * oldOrd);
                final int level = indexInput.readInt();

                final int newOrd = reorderOrdMap.oldOrd2New[oldOrd];
                indexInputNew.seek(faissHNSWNew.getLevels().getBaseOffset() + Integer.BYTES * newOrd);
                final int newLevel = indexInputNew.readInt();
                assert level == newLevel;
            }
        }

        // Neighbor list check
        {
            FaissHnswGraph graph = new FaissHnswGraph(faissHNSW, indexInput);
            FaissHnswGraph graphNew = new FaissHnswGraph(faissHNSWNew, indexInputNew);

            for (int level = 0; level < faissHNSW.getMaxLevel(); ++level) {
                // Num vector check
                HnswGraph.NodesIterator nodesIterator = graph.getNodesOnLevel(level);
                int numVectors = 0;
                while (nodesIterator.hasNext()) {
                    ++numVectors;
                    nodesIterator.next();
                }

                HnswGraph.NodesIterator nodesIteratorNew = graphNew.getNodesOnLevel(level);
                int numVectorsNew = 0;
                while (nodesIteratorNew.hasNext()) {
                    ++numVectorsNew;
                    nodesIteratorNew.next();
                }

                assert numVectors == numVectorsNew;

                // Neighbor list check
                nodesIterator = graph.getNodesOnLevel(level);

                List<Integer> neighbors = new ArrayList<>();
                List<Integer> neighborsNew = new ArrayList<>();
                while (nodesIterator.hasNext()) {
                    final int oldOrd = nodesIterator.nextInt();
                    graph.seek(level, oldOrd);
                    neighbors.clear();
                    while (true) {
                        int neighbor = graph.nextNeighbor();
                        if (neighbor != NO_MORE_DOCS) {
                            neighbors.add(neighbor);
                        } else {
                            break;
                        }
                    }

                    final int newOrd = reorderOrdMap.oldOrd2New[oldOrd];
                    graphNew.seek(level, newOrd);
                    neighborsNew.clear();
                    while (true) {
                        int neighbor = graphNew.nextNeighbor();
                        if (neighbor != NO_MORE_DOCS) {
                            neighborsNew.add(neighbor);
                        } else {
                            break;
                        }
                    }

                    assert neighbors.size() == neighborsNew.size();

                    final Set<Integer> neighborsNewSet = new HashSet<>(neighborsNew);
                    for (int oldNeighbor : neighbors) {
                        final int newNeighbor = reorderOrdMap.oldOrd2New[oldNeighbor];
                        assert neighborsNewSet.contains(newNeighbor);
                    }
                }
            }
        }

        // Id map check
        assert idMapIndex.getTotalNumberOfVectors() == idMapIndexNew.getTotalNumberOfVectors();
        int[] ordToDoc = idMapIndex.getOrdToDocs();
        int[] ordToDocNew = idMapIndexNew.getOrdToDocs();
        for (int oldOrd = 0; oldOrd < idMapIndex.getTotalNumberOfVectors(); ++oldOrd) {
            final int newOrd = reorderOrdMap.oldOrd2New[oldOrd];
            final int doc;
            if (ordToDoc == null) {
                doc = oldOrd;
            } else {
                doc = ordToDoc[oldOrd];
            }
            final int newDoc = ordToDocNew[newOrd];
            assert doc == newDoc;
        }
    }

    private static int[] getPermutation(FaissHNSW faissHNSW, IndexInput indexInput, int window) throws IOException {
        // Prepare
        final FaissHnswGraph faissHnswGraph = new FaissHnswGraph(faissHNSW, indexInput);
        final int numVectors = Math.toIntExact(faissHNSW.getTotalNumberOfVectors());

        // Construct incoming vertexes
        final IncomingVertexes incomingVertexes = IncomingVertexes.collectIncomingVertices(faissHnswGraph);

        // Isolated vectors
        List<Integer> isolatedVertexes = new ArrayList<>();

        // Heap
        final UnitHeap unitHeap = new UnitHeap(numVectors);

        // Bit set
        final BitSet popvExist = new BitSet(numVectors);

        // New permutation
        int orderIndex = 0;
        int[] newPermutation = new int[numVectors];

        // Find the first vertex with the largest indegree
        int veryFirstVertex = -1;
        int maxInDegreeFound = -1;
        for (int i = 0; i < numVectors; i++) {
            final int inDegree = incomingVertexes.getDegree(i);
            if (inDegree > maxInDegreeFound) {
                maxInDegreeFound = inDegree;
                veryFirstVertex = i;
            } else if (inDegree + HnswGraphHelper.getOutDegree(faissHnswGraph, i) == 0) {
                unitHeap.update[i] = Integer.MAX_VALUE / 2;
                isolatedVertexes.add(i);
                unitHeap.deleteElement(i);
            }
        }

        // Push the first vertex and remove it from the queue
        newPermutation[orderIndex++] = veryFirstVertex;
        unitHeap.deleteElement(veryFirstVertex);

        final AtomicReference<Boolean> skipIterationForCommonNeighbor = new AtomicReference<>(false);
        final int untilOrderIndex = (numVectors - isolatedVertexes.size());
        while (true) {
            final int ve = newPermutation[orderIndex - 1];
            final int vb = (orderIndex > (window + 1)) ? newPermutation[orderIndex - window - 1] : -1;

            // For incoming vertexes to `ve`
            // e.g.
            // A -> V_e -> P
            // -> X
            // Then increase counter for A, X and P.
            for (int i = incomingVertexes.startOffset(ve), limit1 = incomingVertexes.endOffset(ve); i < limit1; ++i) {
                // Increase incoming vertex count
                final int u = incomingVertexes.incomingVertices[i];
                if (unitHeap.update[u] == 0) {
                    unitHeap.increaseKey(u);
                } else {
                    unitHeap.update[u] += 1;
                }

                // Increase sibling count
                if (HnswGraphHelper.getOutDegree(faissHnswGraph, u) > 1) {
                    skipIterationForCommonNeighbor.set(false);
                    if (vb != -1) {
                        // We have a leaving node from the window
                        HnswGraphHelper.forAllOutgoingNodes(faissHnswGraph, u, 0, (w) -> {
                            if (w == vb) {
                                skipIterationForCommonNeighbor.set(true);
                                // Stop the loop
                                return false;
                            }
                            return true;
                        });
                    }

                    if (skipIterationForCommonNeighbor.get() == false) {
                        // If `popv` (e.g. v_b) and v (e.g. v_max) are NOT sibling, then do below:
                        HnswGraphHelper.forAllOutgoingNodes(faissHnswGraph, u, 0, (w) -> { unitHeap.update[w] += 1; });
                    } else {
                        popvExist.set(u);
                    }  // End if
                }
            }  // End for

            // For the vertexes pointed by vertex of `veryFirstVertex`
            HnswGraphHelper.forAllOutgoingNodes(faissHnswGraph, ve, 0, (w) -> {
                if (unitHeap.update[w] == 0) {
                    unitHeap.increaseKey(w);
                } else {
                    unitHeap.update[w] += 1;
                }
            });

            if (vb != -1) {
                // For incoming vertexes to `vb`
                // e.g.
                // A -> V_b
                // -> X
                // Then increase counter for A and X
                for (int i = incomingVertexes.startOffset(vb), limit1 = incomingVertexes.endOffset(vb); i < limit1; ++i) {
                    final int u = incomingVertexes.incomingVertices[i];
                    unitHeap.update[u] -= 1;

                    // Increase sibling count
                    if (HnswGraphHelper.getOutDegree(faissHnswGraph, u) > 1) {
                        if (popvExist.get(u) == false) {
                            HnswGraphHelper.forAllOutgoingNodes(faissHnswGraph, u, 1, (w) -> { unitHeap.update[w] -= 1; });
                        } else {
                            popvExist.set(u, false);
                        }
                    }
                }  // End for

                // For the vertexes pointed by vertex of `veryFirstVertex`
                HnswGraphHelper.forAllOutgoingNodes(faissHnswGraph, vb, 0, (w) -> { unitHeap.update[w] -= 1; });
            }  // End if

            if (orderIndex < untilOrderIndex) {
                final int vmax = unitHeap.extractMax();
                newPermutation[orderIndex++] = vmax;
            } else {
                break;
            }
        }  // End while

        // Insert isolated vertexes
        for (final int isolatedIndex : isolatedVertexes) {
            newPermutation[orderIndex++] = isolatedIndex;
        }

        return newPermutation;
    }
}
