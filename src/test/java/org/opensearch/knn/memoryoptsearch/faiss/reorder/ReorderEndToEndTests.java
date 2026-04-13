/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsReader;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * End-to-end test verifying that after reordering:
 * 1. The .vec file bytes are different (vectors are physically rearranged)
 * 2. The same set of vectors is still readable (search correctness preserved)
 *
 * Uses a deterministic reverse-permutation strategy so the test is reproducible
 * and requires no JNI.
 */
public class ReorderEndToEndTests extends OpenSearchTestCase {

    private static final String FIELD_NAME = "test_vector";
    private static final int DIMENSION = 4;
    private static final int NUM_VECTORS = 100;

    /**
     * Writes vectors into a Lucene segment, snapshots the .vec bytes, applies a reverse
     * reorder to the .vec/.vemf files, then reads back via the unified reader and verifies:
     * - .vec bytes differ after reorder
     * - The same set of vectors is present (just in reversed order)
     */
    /**
     * Verify temp filenames match Lucene's codec filename pattern.
     * This prevents the "invalid codec filename" error that only surfaces
     * on real clusters with MMapDirectory (not caught by gradle run).
     */
    public void testTempFileNamesMatchLucenePattern() {
        String[] inputs = {
            "_0_NativeEngines990KnnVectorsFormat_0.vec",
            "_1k_165_target_field.faiss",
            "_3q_Faiss1040ScalarQuantizedKnnVectorsFormat_0.vemf",
        };
        java.util.regex.Pattern lucenePattern = java.util.regex.Pattern.compile("_[a-z0-9]+(_.*)?\\..+");
        for (String input : inputs) {
            String temp = SegmentReorderService.tempName(input);
            assertTrue(
                "Temp filename '" + temp + "' (from '" + input + "') must match Lucene codec pattern",
                lucenePattern.matcher(temp).matches()
            );
            // Must not start with the original segment prefix
            assertFalse(
                "Temp filename must not start with original segment prefix",
                temp.startsWith(input.substring(0, input.indexOf('_', 1) + 1))
            );
        }
    }

    public void testReorderProducesDifferentVecFileButSameVectors() throws Exception {
        Path tempDir = createTempDir("reorder-e2e");
        try (MMapDirectory dir = new MMapDirectory(tempDir)) {
            // Step 1: Write vectors into a single segment
            float[][] originalVectors = writeVectors(dir);

            // Step 2: Find the segment and .vec file
            SegmentInfos segmentInfos = SegmentInfos.readLatestCommit(dir);
            assertEquals("Expected exactly one segment", 1, segmentInfos.size());
            SegmentCommitInfo segCommitInfo = segmentInfos.info(0);
            String segmentName = segCommitInfo.info.name;

            String vecFileName = findFile(dir, segmentName, ".vec");
            assertNotNull("Should find .vec file", vecFileName);

            // Step 3: Snapshot .vec bytes before reorder
            byte[] vecBytesBefore = readFileBytes(dir, vecFileName);

            // Step 4: Read vectors before reorder using standard Lucene reader
            FieldInfo fieldInfo = findVectorFieldInfo(dir);
            assertNotNull("Should find vector field info", fieldInfo);
            float[][] vectorsBefore = readVectorsViaStandardReader(dir, segCommitInfo, segmentName, fieldInfo);
            assertEquals(NUM_VECTORS, vectorsBefore.length);
            assertVectorSetsEqual(originalVectors, vectorsBefore);

            // Step 5: Apply reverse reorder to .vec/.vemf
            applyVecReorder(dir, segCommitInfo, segmentName, vectorsBefore, fieldInfo);

            // Step 6: Snapshot .vec bytes after reorder
            byte[] vecBytesAfter = readFileBytes(dir, vecFileName);

            // Step 7: Verify .vec bytes differ
            assertFalse(
                "The .vec file bytes should differ after reordering",
                Arrays.equals(vecBytesBefore, vecBytesAfter)
            );

            // Step 8: Read vectors after reorder using the reordered reader
            float[][] vectorsAfter = readVectorsViaReorderedReader(dir, segCommitInfo, segmentName, fieldInfo);
            assertEquals(NUM_VECTORS, vectorsAfter.length);
            assertVectorSetsEqual(originalVectors, vectorsAfter);

            // Step 9: Verify the order actually changed (reversed)
            assertArrayEquals(
                "First vector after reorder should be last vector before reorder",
                vectorsBefore[NUM_VECTORS - 1],
                vectorsAfter[0],
                0.0f
            );
        }
    }

    private float[][] writeVectors(MMapDirectory dir) throws IOException {
        IndexWriterConfig iwc = new IndexWriterConfig();
        iwc.setUseCompoundFile(false);
        iwc.setCodec(new Lucene104Codec());
        float[][] vectors = new float[NUM_VECTORS][DIMENSION];

        try (IndexWriter writer = new IndexWriter(dir, iwc)) {
            for (int i = 0; i < NUM_VECTORS; i++) {
                for (int d = 0; d < DIMENSION; d++) {
                    vectors[i][d] = i * DIMENSION + d;
                }
                Document doc = new Document();
                doc.add(new KnnFloatVectorField(FIELD_NAME, vectors[i], VectorSimilarityFunction.EUCLIDEAN));
                writer.addDocument(doc);
            }
            writer.commit();
            writer.forceMerge(1);
            writer.commit();
        }
        return vectors;
    }

    /**
     * Read vectors using the standard Lucene99FlatVectorsReader (before reorder).
     */
    private float[][] readVectorsViaStandardReader(MMapDirectory dir, SegmentCommitInfo segCommitInfo,
                                                   String segmentName, FieldInfo fieldInfo) throws IOException {
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });
        String suffix = extractSuffix(dir, segmentName, ".vec");

        SegmentReadState readState = new SegmentReadState(
            dir, segCommitInfo.info, fieldInfos, IOContext.DEFAULT, suffix
        );

        try (Lucene99FlatVectorsReader reader = new Lucene99FlatVectorsReader(readState, DefaultFlatVectorScorer.INSTANCE)) {
            return extractAllVectors(reader.getFloatVectorValues(FIELD_NAME));
        }
    }

    /**
     * Read vectors using the ReorderedLucene99FlatVectorsReader111 (handles reordered codec headers).
     */
    private float[][] readVectorsViaReorderedReader(MMapDirectory dir, SegmentCommitInfo segCommitInfo,
                                                  String segmentName, FieldInfo fieldInfo) throws IOException {
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });
        String suffix = extractSuffix(dir, segmentName, ".vec");

        SegmentReadState readState = new SegmentReadState(
            dir, segCommitInfo.info, fieldInfos, IOContext.DEFAULT, suffix
        );

        try (ReorderedLucene99FlatVectorsReader111 reader = new ReorderedLucene99FlatVectorsReader111(readState,
                FlatVectorScorerUtil.getLucene99FlatVectorsScorer())) {
            return extractAllVectors(reader.getFloatVectorValues(FIELD_NAME));
        }
    }

    private float[][] extractAllVectors(FloatVectorValues fvv) throws IOException {
        assertNotNull("Should have float vector values", fvv);
        int n = fvv.size();
        float[][] result = new float[n][];
        for (int i = 0; i < n; i++) {
            float[] vec = fvv.vectorValue(i);
            result[i] = Arrays.copyOf(vec, vec.length);
        }
        return result;
    }

    /**
     * Apply a reverse-permutation reorder to the .vec and .vemf files.
     * Reads vectors from the original files, writes them in reversed order
     * using ReorderedFlatVectorsWriter, then atomically replaces the originals.
     */
    private void applyVecReorder(MMapDirectory dir, SegmentCommitInfo segCommitInfo,
                                 String segmentName, float[][] vectorsBefore, FieldInfo fieldInfo) throws IOException {
        String suffix = extractSuffix(dir, segmentName, ".vec");
        FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });

        // Compute reverse permutation
        int n = vectorsBefore.length;
        int[] permutation = new int[n];
        for (int i = 0; i < n; i++) {
            permutation[i] = n - 1 - i;
        }
        ReorderOrdMap reorderOrdMap = new ReorderOrdMap(permutation);

        String vecDataFileName = findFile(dir, segmentName, ".vec");
        String vecMetaFileName = findFile(dir, segmentName, ".vemf");
        assertNotNull(vecDataFileName);
        assertNotNull(vecMetaFileName);

        String reorderedVecData = "reorder_tmp_vec_" + vecDataFileName;
        String reorderedVecMeta = "reorder_tmp_meta_" + vecMetaFileName;

        // Read original vectors and write in reordered order
        SegmentReadState readState = new SegmentReadState(
            dir, segCommitInfo.info, fieldInfos, IOContext.DEFAULT, suffix
        );

        try (IndexInput vecMetaInput = dir.openInput(vecMetaFileName, IOContext.DEFAULT)) {
            ReorderedFlatVectorsWriter writer = new ReorderedFlatVectorsWriter(
                dir, reorderedVecMeta, reorderedVecData, n, vecMetaInput
            );

            try (
                writer;
                Lucene99FlatVectorsReader reader = new Lucene99FlatVectorsReader(readState, DefaultFlatVectorScorer.INSTANCE)
            ) {
                FloatVectorValues vectorValues = reader.getFloatVectorValues(FIELD_NAME);
                @SuppressWarnings("rawtypes")
                ReorderedFlatVectorsWriter.ReorderedFlatFieldVectorsWriter fieldWriter = writer.addField(fieldInfo);

                for (int newOrd = 0; newOrd < n; newOrd++) {
                    int oldOrd = reorderOrdMap.newOrd2Old[newOrd];
                    float[] vector = vectorValues.vectorValue(oldOrd);
                    //noinspection unchecked
                    fieldWriter.addValue(oldOrd, vector);
                }
                fieldWriter.finish();
            }
        }

        // Atomic replace
        dir.deleteFile(vecDataFileName);
        dir.deleteFile(vecMetaFileName);
        dir.rename(reorderedVecData, vecDataFileName);
        dir.rename(reorderedVecMeta, vecMetaFileName);
    }

    // --- Helper methods ---

    private FieldInfo findVectorFieldInfo(MMapDirectory dir) throws IOException {
        try (DirectoryReader reader = DirectoryReader.open(dir)) {
            for (LeafReaderContext ctx : reader.leaves()) {
                FieldInfo fi = ctx.reader().getFieldInfos().fieldInfo(FIELD_NAME);
                if (fi != null) return fi;
            }
        }
        return null;
    }

    private String findFile(MMapDirectory dir, String segmentName, String extension) throws IOException {
        for (String file : dir.listAll()) {
            if (file.startsWith(segmentName) && file.endsWith(extension)) {
                return file;
            }
        }
        return null;
    }

    private String extractSuffix(MMapDirectory dir, String segmentName, String extension) throws IOException {
        String file = findFile(dir, segmentName, extension);
        if (file == null) return "";
        // e.g. _0_Lucene99HnswVectorsFormat_0.vec → Lucene99HnswVectorsFormat_0
        String withoutExt = file.substring(0, file.lastIndexOf('.'));
        return withoutExt.substring(segmentName.length() + 1);
    }

    private byte[] readFileBytes(MMapDirectory dir, String fileName) throws IOException {
        try (IndexInput input = dir.openInput(fileName, IOContext.DEFAULT)) {
            byte[] bytes = new byte[(int) input.length()];
            input.readBytes(bytes, 0, bytes.length);
            return bytes;
        }
    }

    private void assertVectorSetsEqual(float[][] expected, float[][] actual) {
        assertEquals("Vector count mismatch", expected.length, actual.length);
        Set<String> expectedSet = new HashSet<>();
        for (float[] v : expected) expectedSet.add(Arrays.toString(v));
        Set<String> actualSet = new HashSet<>();
        for (float[] v : actual) actualSet.add(Arrays.toString(v));
        assertEquals("Vector sets should be identical", expectedSet, actualSet);
    }
}
