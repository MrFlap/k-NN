/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.Bits;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

public class SparseReorderedOffHeapFloatVectorValuesTests extends OpenSearchTestCase {

    private static final int DIM = 4;
    private static final VectorSimilarityFunction SIM = VectorSimilarityFunction.EUCLIDEAN;

    static class TestFixture implements AutoCloseable {
        final ByteBuffersDirectory dir;
        final int numVectors;
        final int maxDoc;
        final int[] ordToDocMap;
        final float[][] vectors;
        final Set<Integer> docsWithVectors;
        final ReorderedOffHeapFloatVectorValues111 vectorValues;
        private final IndexInput vecInput;

        TestFixture(int numVectors, int maxDoc, Set<Integer> docsWithVectors, int[] permutation) throws IOException {
            this.dir = new ByteBuffersDirectory();
            this.numVectors = numVectors;
            this.maxDoc = maxDoc;
            this.docsWithVectors = docsWithVectors;

            int[] sortedDocs = docsWithVectors.stream().mapToInt(Integer::intValue).sorted().toArray();
            assertEquals(numVectors, sortedDocs.length);

            this.vectors = new float[numVectors][DIM];
            for (int i = 0; i < numVectors; i++) {
                for (int d = 0; d < DIM; d++) vectors[i][d] = i * 10f + d;
            }

            int[] oldOrd2New = new int[numVectors];
            for (int newOrd = 0; newOrd < numVectors; newOrd++) {
                oldOrd2New[permutation[newOrd]] = newOrd;
            }

            this.ordToDocMap = new int[numVectors];
            for (int newOrd = 0; newOrd < numVectors; newOrd++) {
                ordToDocMap[newOrd] = sortedDocs[permutation[newOrd]];
            }

            ByteBuffer buf = ByteBuffer.allocate(DIM * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
            try (IndexOutput vecOut = dir.createOutput("test.vec", IOContext.DEFAULT)) {
                for (int newOrd = 0; newOrd < numVectors; newOrd++) {
                    buf.asFloatBuffer().put(vectors[permutation[newOrd]]);
                    vecOut.writeBytes(buf.array(), buf.array().length);
                }
            }

            int numBytesPerValue = Integer.BYTES - (Integer.numberOfLeadingZeros(maxDoc) / Byte.SIZE);
            int sentinel = (1 << (8 * numBytesPerValue)) - 1;

            try (IndexOutput metaOut = dir.createOutput("test.vemf", IOContext.DEFAULT)) {
                FixedBlockSkipListIndexBuilder builder = new FixedBlockSkipListIndexBuilder(metaOut, maxDoc);
                int vecIdx = 0;
                for (int doc = 0; doc <= maxDoc; doc++) {
                    if (vecIdx < numVectors && sortedDocs[vecIdx] == doc) {
                        builder.add(doc, oldOrd2New[vecIdx]);
                        vecIdx++;
                    } else {
                        builder.add(doc, sentinel);
                    }
                }
                builder.finish();
            }

            FixedBlockSkipListIndexReader skipListReader;
            try (IndexInput metaIn = dir.openInput("test.vemf", IOContext.DEFAULT)) {
                skipListReader = new FixedBlockSkipListIndexReader(metaIn, maxDoc);
            }

            this.vecInput = dir.openInput("test.vec", IOContext.DEFAULT);

            boolean isDense = (numVectors == maxDoc + 1);
            if (isDense) {
                this.vectorValues = ReorderedOffHeapFloatVectorValues111.load(
                    SIM, DefaultFlatVectorScorer.INSTANCE, skipListReader,
                    DIM, 0L, vecInput.length(), vecInput, numVectors, ordToDocMap
                );
            } else {
                this.vectorValues = ReorderedOffHeapFloatVectorValues111.loadSparse(
                    SIM, DefaultFlatVectorScorer.INSTANCE, skipListReader,
                    DIM, 0L, vecInput.length(), vecInput, numVectors, maxDoc, ordToDocMap
                );
            }
        }

        @Override
        public void close() throws IOException {
            vecInput.close();
            dir.close();
        }
    }

    private TestFixture buildDense(int n) throws IOException {
        Set<Integer> docs = new HashSet<>();
        for (int i = 0; i < n; i++) docs.add(i);
        return new TestFixture(n, n - 1, docs, shuffled(n, 42));
    }

    private TestFixture buildSparse(int n, long seed) throws IOException {
        int totalDocs = n * 10;
        Random rng = new Random(seed);
        Set<Integer> docs = new HashSet<>();
        while (docs.size() < n) docs.add(rng.nextInt(totalDocs));
        int maxDoc = docs.stream().mapToInt(Integer::intValue).max().orElse(0);
        return new TestFixture(n, maxDoc, docs, shuffled(n, seed));
    }

    private static int[] shuffled(int n, long seed) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) p[i] = i;
        Random rng = new Random(seed);
        for (int i = n - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int t = p[i]; p[i] = p[j]; p[j] = t;
        }
        return p;
    }

    // ===== Test 2: iterator nextDoc =====

    public void testDenseIteratorNextDoc() throws IOException {
        try (TestFixture f = buildDense(50)) {
            FloatVectorValues.DocIndexIterator it = f.vectorValues.iterator();
            int count = 0;
            int prev = -1;
            for (int d = it.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = it.nextDoc()) {
                assertTrue(d > prev);
                prev = d;
                count++;
            }
            assertEquals(f.numVectors, count);
            assertEquals(f.numVectors, f.vectorValues.size());
        }
    }

    public void testSparseIteratorNextDoc() throws IOException {
        try (TestFixture f = buildSparse(50, 42)) {
            FloatVectorValues.DocIndexIterator it = f.vectorValues.iterator();
            List<Integer> yielded = new ArrayList<>();
            for (int d = it.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = it.nextDoc()) {
                assertTrue(f.docsWithVectors.contains(d));
                yielded.add(d);
            }
            assertEquals(f.numVectors, yielded.size());
            for (int i = 1; i < yielded.size(); i++) assertTrue(yielded.get(i) > yielded.get(i - 1));
            assertEquals(f.numVectors, f.vectorValues.size());
        }
    }

    // ===== Test 3: iterator advance =====

    public void testDenseIteratorAdvance() throws IOException {
        try (TestFixture f = buildDense(50)) {
            FloatVectorValues.DocIndexIterator it = f.vectorValues.iterator();
            assertEquals(10, it.advance(10));
            assertEquals(25, it.advance(25));
            assertEquals(49, it.advance(49));
            assertEquals(DocIdSetIterator.NO_MORE_DOCS, it.advance(50));
        }
    }

    public void testSparseIteratorAdvance() throws IOException {
        try (TestFixture f = buildSparse(50, 42)) {
            int[] sorted = f.docsWithVectors.stream().mapToInt(Integer::intValue).sorted().toArray();

            // Advance to doc with vector
            FloatVectorValues.DocIndexIterator it = f.vectorValues.iterator();
            assertEquals(sorted[10], it.advance(sorted[10]));

            // Advance to doc without vector — find a gap
            for (int i = 0; i < sorted.length - 1; i++) {
                if (sorted[i + 1] - sorted[i] > 1) {
                    it = f.vectorValues.iterator();
                    assertEquals(sorted[i + 1], it.advance(sorted[i] + 1));
                    break;
                }
            }

            // Advance past end
            it = f.vectorValues.iterator();
            assertEquals(DocIdSetIterator.NO_MORE_DOCS, it.advance(f.maxDoc + 1));
        }
    }

    // ===== Test 4: getAcceptOrds =====

    private static Bits evenDocBits(int maxDoc) {
        return new Bits() {
            @Override public boolean get(int index) { return index % 2 == 0; }
            @Override public int length() { return maxDoc + 1; }
        };
    }

    public void testDenseGetAcceptOrds() throws IOException {
        try (TestFixture f = buildDense(50)) {
            Bits accept = evenDocBits(f.maxDoc);
            Bits result = f.vectorValues.getAcceptOrds(accept);
            assertSame(accept, result);
        }
    }

    public void testSparseGetAcceptOrds() throws IOException {
        try (TestFixture f = buildSparse(50, 42)) {
            Bits accept = evenDocBits(f.maxDoc);
            Bits result = f.vectorValues.getAcceptOrds(accept);
            assertNotNull(result);
            for (int ord = 0; ord < f.numVectors; ord++) {
                assertEquals(accept.get(f.ordToDocMap[ord]), result.get(ord));
            }
        }
    }

    // ===== Test 5: getAcceptOrds(null) =====

    public void testDenseGetAcceptOrdsNull() throws IOException {
        try (TestFixture f = buildDense(50)) { assertNull(f.vectorValues.getAcceptOrds(null)); }
    }

    public void testSparseGetAcceptOrdsNull() throws IOException {
        try (TestFixture f = buildSparse(50, 42)) { assertNull(f.vectorValues.getAcceptOrds(null)); }
    }

    // ===== Test 6: ordToDoc =====

    public void testDenseOrdToDoc() throws IOException {
        try (TestFixture f = buildDense(50)) {
            for (int o = 0; o < f.numVectors; o++) assertEquals(f.ordToDocMap[o], f.vectorValues.ordToDoc(o));
        }
    }

    public void testSparseOrdToDoc() throws IOException {
        try (TestFixture f = buildSparse(50, 42)) {
            for (int o = 0; o < f.numVectors; o++) {
                int doc = f.vectorValues.ordToDoc(o);
                assertEquals(f.ordToDocMap[o], doc);
                assertTrue(f.docsWithVectors.contains(doc));
            }
        }
    }

    // ===== Test 7: vectorValue =====

    public void testDenseVectorValue() throws IOException {
        try (TestFixture f = buildDense(20)) {
            for (int o = 0; o < f.numVectors; o++) {
                float[] v = f.vectorValues.vectorValue(o);
                assertNotNull(v);
                assertEquals(DIM, v.length);
            }
        }
    }

    public void testSparseVectorValue() throws IOException {
        try (TestFixture f = buildSparse(20, 42)) {
            for (int o = 0; o < f.numVectors; o++) {
                float[] v = f.vectorValues.vectorValue(o);
                assertNotNull(v);
                assertEquals(DIM, v.length);
            }
            assertEquals(f.numVectors, f.vectorValues.size());
        }
    }

    // ===== Test 10: vectors at high doc IDs =====

    public void testSparseHighDocIds() throws IOException {
        int n = 1000;
        Set<Integer> docs = new HashSet<>();
        for (int i = 9000; i < 9000 + n; i++) docs.add(i);
        try (TestFixture f = new TestFixture(n, 9999, docs, shuffled(n, 77))) {
            FloatVectorValues.DocIndexIterator it = f.vectorValues.iterator();
            assertEquals(9000, it.nextDoc());

            it = f.vectorValues.iterator();
            assertEquals(9000, it.advance(0));

            it = f.vectorValues.iterator();
            assertEquals(9500, it.advance(9500));
        }
    }

    // ===== Test 11: single vector =====

    public void testSingleDense() throws IOException {
        Set<Integer> docs = Set.of(0);
        try (TestFixture f = new TestFixture(1, 0, docs, new int[]{0})) {
            assertEquals(1, f.vectorValues.size());
            assertEquals(0, f.vectorValues.ordToDoc(0));
            FloatVectorValues.DocIndexIterator it = f.vectorValues.iterator();
            assertEquals(0, it.nextDoc());
            assertEquals(DocIdSetIterator.NO_MORE_DOCS, it.nextDoc());
        }
    }

    public void testSingleSparse() throws IOException {
        Set<Integer> docs = Set.of(999);
        try (TestFixture f = new TestFixture(1, 999, docs, new int[]{0})) {
            assertEquals(1, f.vectorValues.size());
            assertEquals(999, f.vectorValues.ordToDoc(0));
            FloatVectorValues.DocIndexIterator it = f.vectorValues.iterator();
            assertEquals(999, it.nextDoc());
            assertEquals(DocIdSetIterator.NO_MORE_DOCS, it.nextDoc());
            it = f.vectorValues.iterator();
            assertEquals(999, it.advance(0));
        }
    }

    // ===== Test 12: consecutive at end =====

    public void testSparseConsecutiveAtEnd() throws IOException {
        int n = 100;
        Set<Integer> docs = new HashSet<>();
        for (int i = 900; i < 1000; i++) docs.add(i);
        try (TestFixture f = new TestFixture(n, 999, docs, shuffled(n, 55))) {
            FloatVectorValues.DocIndexIterator it = f.vectorValues.iterator();
            assertEquals(900, it.nextDoc());
            for (int e = 901; e < 1000; e++) assertEquals(e, it.nextDoc());
            assertEquals(DocIdSetIterator.NO_MORE_DOCS, it.nextDoc());

            it = f.vectorValues.iterator();
            assertEquals(900, it.advance(500));
        }
    }

    // ===== Test 13: scorer =====

    public void testDenseScorer() throws IOException {
        try (TestFixture f = buildDense(10)) {
            VectorScorer s = f.vectorValues.scorer(new float[DIM]);
            int count = 0;
            for (int d = s.iterator().nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = s.iterator().nextDoc()) {
                assertTrue(s.score() >= 0);
                count++;
            }
            assertEquals(f.numVectors, count);
        }
    }

    public void testSparseScorer() throws IOException {
        try (TestFixture f = buildSparse(10, 42)) {
            VectorScorer s = f.vectorValues.scorer(new float[DIM]);
            int count = 0;
            DocIdSetIterator it = s.iterator();
            for (int d = it.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = it.nextDoc()) {
                assertTrue(f.docsWithVectors.contains(d));
                assertTrue(s.score() >= 0);
                count++;
            }
            assertEquals(f.numVectors, count);
        }
    }

    // ===== Test 14: copy =====

    public void testDenseCopy() throws IOException {
        try (TestFixture f = buildDense(10)) {
            FloatVectorValues copy = f.vectorValues.copy();
            assertNotSame(f.vectorValues, copy);
            assertEquals(f.vectorValues.size(), copy.size());
            for (int o = 0; o < f.numVectors; o++) {
                assertArrayEquals(f.vectorValues.vectorValue(o), copy.vectorValue(o), 0f);
            }
        }
    }

    public void testSparseCopy() throws IOException {
        try (TestFixture f = buildSparse(10, 42)) {
            FloatVectorValues copy = f.vectorValues.copy();
            assertNotSame(f.vectorValues, copy);
            assertEquals(f.vectorValues.size(), copy.size());
            for (int o = 0; o < f.numVectors; o++) {
                assertArrayEquals(f.vectorValues.vectorValue(o), copy.vectorValue(o), 0f);
                assertEquals(f.vectorValues.ordToDoc(o), copy.ordToDoc(o));
            }
        }
    }
}
