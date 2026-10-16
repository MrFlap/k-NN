/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.codec.KNN1040Codec.Faiss1040DataBlindInt8KnnVectorsFormat;
import org.opensearch.knn.index.warmup.WarmableReader;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;

import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * End to end through {@link Faiss1040DataBlindInt8KnnVectorsFormat}: writes float vectors of a {@code bits=8} SQ field,
 * which stores data-blind int8 codes and builds the native Faiss HNSW graph from them, then reads the segment back.
 */
public class Faiss1040DataBlindInt8FormatTests extends KNNTestCase {

    static final int DIMENSIONS = 24;
    static final int NUM_DOCS = 300;
    static final String FIELD_NAME = "target_field";

    @SneakyThrows
    public void testWriteThenRead_whenL2_thenInt8CodesAndGraphAreRead() {
        assertWriteThenRead("l2", VectorSimilarityFunction.EUCLIDEAN);
    }

    @SneakyThrows
    public void testWriteThenRead_whenInnerProduct_thenInt8CodesAndGraphAreRead() {
        assertWriteThenRead("innerproduct", VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT);
    }

    private void assertWriteThenRead(final String spaceType, final VectorSimilarityFunction similarity) throws Exception {
        try (Directory rawDirectory = new MMapDirectory(createTempDir("faiss1040_int8"))) {
            final float[][] vectors = randomVectors(new Random(42), NUM_DOCS, DIMENSIONS);
            final SegmentReadState readState = write(rawDirectory, spaceType, similarity, vectors);
            final FaissMemoryOptimizedSearcherTests.ReadTrackingDirectory spyDirectory =
                new FaissMemoryOptimizedSearcherTests.ReadTrackingDirectory(rawDirectory);
            final SegmentReadState spyReadState = new SegmentReadState(
                spyDirectory,
                readState.segmentInfo,
                readState.fieldInfos,
                IOContext.DEFAULT
            );

            try (KnnVectorsReader reader = new Faiss1040DataBlindInt8KnnVectorsFormat().fieldsReader(spyReadState)) {
                assertEquals(NUM_DOCS, reader.getFloatVectorValues(FIELD_NAME).size());
                spyDirectory.resetReadFlags();
                ((WarmableReader) reader).warmUp(FIELD_NAME);
                assertTrue("Warmup should read the .faiss graph", spyDirectory.wasExtensionRead(".faiss"));
                assertTrue("Warmup should read the int8 codes", spyDirectory.wasExtensionRead(".vdbi"));
                assertTrue("Warmup should read the float vectors", spyDirectory.wasExtensionRead(".vec"));
            }
        }
    }

    /** Flushes {@code vectors} as one dense segment and returns a read state for it. */
    static SegmentReadState write(
        final Directory directory,
        final String spaceType,
        final VectorSimilarityFunction similarity,
        final float[][] vectors
    ) throws Exception {
        final FieldInfo fieldInfo = fieldInfo(spaceType, similarity);
        final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });
        final SegmentInfo segInfo = new SegmentInfo(
            directory,
            Version.LATEST,
            Version.LATEST,
            "_0",
            vectors.length,
            false,
            false,
            null,
            Collections.emptyMap(),
            StringHelper.randomId(),
            Collections.emptyMap(),
            null
        );
        final SegmentWriteState writeState = new SegmentWriteState(
            InfoStream.NO_OUTPUT,
            directory,
            segInfo,
            fieldInfos,
            null,
            IOContext.DEFAULT
        );
        try (KnnVectorsWriter writer = new Faiss1040DataBlindInt8KnnVectorsFormat().fieldsWriter(writeState)) {
            @SuppressWarnings("unchecked")
            final KnnFieldVectorsWriter<float[]> fw = (KnnFieldVectorsWriter<float[]>) writer.addField(fieldInfo);
            for (int doc = 0; doc < vectors.length; doc++) {
                fw.addValue(doc, vectors[doc]);
            }
            writer.flush(vectors.length, null);
            writer.finish();
        }
        final Set<String> files = new HashSet<>();
        for (String file : directory.listAll()) {
            if (file.startsWith("_")) {
                files.add(file);
            }
        }
        segInfo.setFiles(files);
        return new SegmentReadState(directory, segInfo, fieldInfos, IOContext.DEFAULT);
    }

    private static FieldInfo fieldInfo(final String spaceType, final VectorSimilarityFunction similarity) {
        final String parameters = "{\"index_description\":\"BHNSW16,Flat\",\"spaceType\":\""
            + spaceType
            + "\",\"name\":\"hnsw\",\"data_type\":\"float\",\"parameters\":{\"ef_search\":256,\"ef_construction\":256,\"m\":16,"
            + "\"encoder\":{\"name\":\"sq\",\"bits\":8}}}";
        return new FieldInfo(
            FIELD_NAME,
            0,
            false,
            false,
            false,
            IndexOptions.NONE,
            DocValuesType.NONE,
            DocValuesSkipIndexType.NONE,
            -1,
            Map.of(
                KNNConstants.PARAMETERS,
                parameters,
                KNNConstants.VECTOR_DATA_TYPE_FIELD,
                "float",
                KNNConstants.KNN_ENGINE,
                "faiss",
                KNNConstants.SQ_CONFIG,
                "bits=8",
                KNNVectorFieldMapper.KNN_FIELD,
                "true",
                KNNConstants.SPACE_TYPE,
                spaceType
            ),
            0,
            0,
            0,
            DIMENSIONS,
            VectorEncoding.FLOAT32,
            similarity,
            false,
            false
        );
    }

    static float[][] randomVectors(final Random random, final int n, final int dim) {
        final float[][] vectors = new float[n][dim];
        for (float[] v : vectors) {
            for (int d = 0; d < dim; d++) {
                v[d] = (float) random.nextGaussian();
            }
        }
        return vectors;
    }
}
