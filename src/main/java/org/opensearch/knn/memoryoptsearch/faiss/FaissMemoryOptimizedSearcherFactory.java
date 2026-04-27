/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss;

import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.VectorEncoding;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.IOUtils;
import org.opensearch.knn.memoryoptsearch.VectorSearcher;
import org.opensearch.knn.memoryoptsearch.VectorSearcherFactory;

import java.io.IOException;

/**
 * This factory returns {@link VectorSearcher} that performs vector search directly on FAISS index.
 * Note that we pass `RANDOM` as advice to prevent the underlying storage from performing read-ahead. Since vector search naturally accesses
 * random vector locations, read-ahead does not improve performance. By passing the `RANDOM` context, we explicitly indicate that
 * this searcher will access vectors randomly.
 */
@Log4j2
public class FaissMemoryOptimizedSearcherFactory implements VectorSearcherFactory {

    @Override
    public VectorSearcher createVectorSearcher(
        final Directory directory,
        final String fileName,
        final FieldInfo fieldInfo,
        final IOContext ioContext,
        final FlatVectorsReader flatVectorsReader
    ) throws IOException {
        final IndexInput indexInput = directory.openInput(fileName, ioContext);
        try {
            // Try load it. Not all FAISS index types are currently supported at the moment.
            final FaissIndex faissIndex = FaissIndex.load(indexInput);
            FaissFlatIndexFactory.maybeSetFlatBinaryIndex(faissIndex, fieldInfo, flatVectorsReader);

            // When clumping is enabled for an SQ field, the marker index on disk is a plain fp32
            // HNSW (not a binary HNSW with null storage), so the SQ scorer — which extracts
            // QuantizedByteVectorValues from a ScalarQuantizedFloatVectorValues wrapper — doesn't
            // apply. Detect this by checking the loaded index's vector encoding and use the
            // Lucene99 fp32 scorer instead.
            final boolean indexOnDiskIsFp32 = faissIndex.getVectorEncoding() == VectorEncoding.FLOAT32;
            final FlatVectorsScorer vectorScorer;
            if (indexOnDiskIsFp32) {
                vectorScorer = FlatVectorsScorerProvider.getLucene99FlatVectorsScorer();
            } else {
                vectorScorer = FlatVectorsScorerProvider.getFlatVectorsScorer(
                    fieldInfo,
                    faissIndex.getVectorSimilarityFunction(),
                    flatVectorsReader.getFlatVectorScorer()
                );
            }
            return new FaissMemoryOptimizedSearcher(indexInput, faissIndex, fieldInfo, vectorScorer);
        } catch (UnsupportedFaissIndexException e) {
            // Clean up input stream.
            try {
                IOUtils.close(indexInput);
            } catch (IOException ioException) {}

            throw e;
        }
    }

}
