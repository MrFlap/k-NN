/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.opensearch.common.StopWatch;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexWriter;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.VecWritingReorderedKNNFloatVectorValues;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.MergeOrdMappingBuilder;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.MergedRandomAccessFloatVectorValues;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.ReorderedFieldMetaWriter;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.SegmentReorderService;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.VectorReorderStrategy;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.MergeAwareReorderStrategy;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.kmeansreorder.ClusterResult;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.kmeansreorder.ClusterSummary;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.kmeansreorder.ClusterSummaryReader;
import org.opensearch.knn.memoryoptsearch.faiss.reorder.kmeansreorder.ClusterSummaryWriter;
import org.opensearch.knn.plugin.stats.KNNGraphValue;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import static org.opensearch.knn.common.FieldInfoExtractor.extractVectorDataType;
import static org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory.getKNNVectorValuesSupplierForMerge;
import static org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory.getVectorValuesSupplier;

/**
 * A KNNVectorsWriter class for writing the vector data structures and flat vectors for Native Engines.
 */
@Log4j2
public class NativeEngines990KnnVectorsWriter extends KnnVectorsWriter {
    private static final long SHALLOW_SIZE = RamUsageEstimator.shallowSizeOfInstance(NativeEngines990KnnVectorsWriter.class);

    private final SegmentWriteState segmentWriteState;
    private final FlatVectorsWriter flatVectorsWriter;
    private KNN990QuantizationStateWriter quantizationStateWriter;
    private final List<NativeEngineFieldVectorsWriter<?>> fields = new ArrayList<>();
    private boolean finished;
    private final Integer approximateThreshold;
    private final NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory;
    private final VectorReorderStrategy reorderStrategy;
    private final boolean replacementFree;

    // Fields that need reordering after finish() writes footers.
    private final List<FieldInfo> fieldsToReorder = new ArrayList<>();

    // Stored during mergeOneField() so finish() can read source .kcs files.
    private MergeState mergeState;

    // Source cluster summaries loaded during mergeOneField(), keyed by field name.
    // Used in finish() for merge-aware reorder.
    private final java.util.Map<String, List<ClusterSummary>> sourceClusterSummaries = new java.util.HashMap<>();

    public NativeEngines990KnnVectorsWriter(
        SegmentWriteState segmentWriteState,
        FlatVectorsWriter flatVectorsWriter,
        Integer approximateThreshold,
        NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory
    ) {
        this(segmentWriteState, flatVectorsWriter, approximateThreshold, nativeIndexBuildStrategyFactory, null, false);
    }

    public NativeEngines990KnnVectorsWriter(
        SegmentWriteState segmentWriteState,
        FlatVectorsWriter flatVectorsWriter,
        Integer approximateThreshold,
        NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory,
        VectorReorderStrategy reorderStrategy
    ) {
        this(segmentWriteState, flatVectorsWriter, approximateThreshold, nativeIndexBuildStrategyFactory, reorderStrategy, false);
    }

    public NativeEngines990KnnVectorsWriter(
        SegmentWriteState segmentWriteState,
        FlatVectorsWriter flatVectorsWriter,
        Integer approximateThreshold,
        NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory,
        VectorReorderStrategy reorderStrategy,
        boolean replacementFree
    ) {
        this.segmentWriteState = segmentWriteState;
        this.flatVectorsWriter = flatVectorsWriter;
        this.approximateThreshold = approximateThreshold;
        this.nativeIndexBuildStrategyFactory = nativeIndexBuildStrategyFactory;
        this.reorderStrategy = reorderStrategy;
        this.replacementFree = replacementFree;
    }

    @Override
    public KnnFieldVectorsWriter<?> addField(final FieldInfo fieldInfo) throws IOException {
        final NativeEngineFieldVectorsWriter<?> newField = NativeEngineFieldVectorsWriter.create(
            fieldInfo,
            flatVectorsWriter.addField(fieldInfo),
            segmentWriteState.infoStream
        );
        fields.add(newField);
        return newField;
    }

    @Override
    public void flush(int maxDoc, final Sorter.DocMap sortMap) throws IOException {
        flatVectorsWriter.flush(maxDoc, sortMap);

        for (final NativeEngineFieldVectorsWriter<?> field : fields) {
            final FieldInfo fieldInfo = field.getFieldInfo();
            final VectorDataType vectorDataType = extractVectorDataType(fieldInfo);
            int totalLiveDocs = field.getVectors().size();
            if (totalLiveDocs == 0) {
                log.debug("[Flush] No live docs for field {}", fieldInfo.getName());
                continue;
            }
            final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier = getVectorValuesSupplier(
                vectorDataType,
                field.getFlatFieldVectorsWriter().getDocsWithFieldSet(),
                field.getVectors()
            );
            final QuantizationState quantizationState = train(field.getFieldInfo(), knnVectorValuesSupplier, totalLiveDocs);
            if (quantizationState == null && shouldSkipBuildingVectorDataStructure(totalLiveDocs)) {
                log.debug(
                    "Skip building vector data structure for field: {}, as liveDoc: {} is less than the threshold {} during flush",
                    fieldInfo.name,
                    totalLiveDocs,
                    approximateThreshold
                );
                continue;
            }
            final NativeIndexWriter writer = NativeIndexWriter.getWriter(
                fieldInfo,
                segmentWriteState,
                quantizationState,
                nativeIndexBuildStrategyFactory
            );

            StopWatch stopWatch = new StopWatch().start();
            writer.flushIndex(knnVectorValuesSupplier, totalLiveDocs);
            long time_in_millis = stopWatch.stop().totalTime().millis();
            KNNGraphValue.REFRESH_TOTAL_TIME_IN_MILLIS.incrementBy(time_in_millis);
            log.debug("Flush took {} ms for vector field [{}]", time_in_millis, fieldInfo.getName());

            // For merge-aware strategy, cluster at flush and write .kcs for future merges
            if (reorderStrategy instanceof MergeAwareReorderStrategy mergeAware && vectorDataType == VectorDataType.FLOAT) {
                writeFlushClusterSummary(mergeAware, field, fieldInfo, totalLiveDocs);
            }
        }
    }

    @Override
    public void mergeOneField(final FieldInfo fieldInfo, final MergeState mergeState) throws IOException {
        this.mergeState = mergeState;
        final VectorDataType vectorDataType = extractVectorDataType(fieldInfo);

        // Replacement-free path: write .vec in reordered order directly, build .faiss in reordered order.
        // Skips flatVectorsWriter.mergeOneField() and the post-hoc rewrite in finish().
        if (replacementFree && reorderStrategy != null && vectorDataType == VectorDataType.FLOAT) {
            final MergeOrdMappingBuilder.MergeOrdMapping mapping = MergeOrdMappingBuilder.build(mergeState, fieldInfo);
            if (mapping.totalLiveDocs() >= SegmentReorderService.MIN_VECTORS_FOR_REORDER) {
                mergeOneFieldReplacementFree(fieldInfo, mergeState, mapping);
                return;
            }
            // Below threshold — fall through to standard path
        }

        // Standard path: delegate writes .vec, optionally mark for post-hoc reorder in finish()
        flatVectorsWriter.mergeOneField(fieldInfo, mergeState);

        final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier = getKNNVectorValuesSupplierForMerge(
            vectorDataType,
            fieldInfo,
            mergeState
        );
        int totalLiveDocs = getLiveDocs(knnVectorValuesSupplier.get());
        if (totalLiveDocs == 0) {
            log.debug("[Merge] No live docs for field {}", fieldInfo.getName());
            return;
        }

        final QuantizationState quantizationState = train(fieldInfo, knnVectorValuesSupplier, totalLiveDocs);
        if (quantizationState == null && shouldSkipBuildingVectorDataStructure(totalLiveDocs)) {
            log.debug(
                "Skip building vector data structure for field: {}, as liveDoc: {} is less than the threshold {} during merge",
                fieldInfo.name,
                totalLiveDocs,
                approximateThreshold
            );
            return;
        }
        final NativeIndexWriter writer = NativeIndexWriter.getWriter(
            fieldInfo,
            segmentWriteState,
            quantizationState,
            nativeIndexBuildStrategyFactory
        );

        StopWatch stopWatch = new StopWatch().start();
        writer.mergeIndex(knnVectorValuesSupplier, totalLiveDocs);
        long time_in_millis = stopWatch.stop().totalTime().millis();
        KNNGraphValue.MERGE_TOTAL_TIME_IN_MILLIS.incrementBy(time_in_millis);
        log.debug("Merge took {} ms for vector field [{}]", time_in_millis, fieldInfo.getName());

        // Mark this field for reordering. The actual reorder happens in finish() after
        // flatVectorsWriter.finish() writes the .vec/.vemf footers, so we can read them.
        if (reorderStrategy != null && totalLiveDocs >= SegmentReorderService.MIN_VECTORS_FOR_REORDER) {
            fieldsToReorder.add(fieldInfo);

            // For merge-aware strategy, load source .kcs files now while we have MergeState.
            // All segments in a shard share the same Directory.
            if (reorderStrategy instanceof MergeAwareReorderStrategy) {
                List<ClusterSummary> summaries = loadSourceClusterSummaries(fieldInfo, mergeState);
                sourceClusterSummaries.put(fieldInfo.name, summaries);
            }
        }
    }

    /**
     * Replacement-free merge path: skips the delegate's mergeOneField, writes .vec in permuted
     * order and reordered .vemf metadata directly into the delegate's IndexOutput streams.
     * Builds .faiss with vectors already in reordered order. True single-pass — no re-reading.
     * Uses standard codec headers; the reader dispatches per-field via knn_reordered attribute.
     */
    private void mergeOneFieldReplacementFree(final FieldInfo fieldInfo, final MergeState mergeState,
                                              final MergeOrdMappingBuilder.MergeOrdMapping mapping) throws IOException {

        // Mark field so the unified reader knows to parse reordered metadata format
        fieldInfo.putAttribute("knn_reordered", "true");

        StopWatch sw = new StopWatch().start();

        // Build random-access composite over source segment mmap readers
        final MergedRandomAccessFloatVectorValues mergedRA = buildMergedRandomAccess(mergeState, fieldInfo, mapping);

        long buildRA_ms = sw.stop().totalTime().millis();
        sw = new StopWatch().start();

        // Compute permutation — use quantized vectors if strategy supports it and source .faiss files
        // have byte vectors (binary quantization). This reduces the working set from ~6GB to ~192MB
        // for 2M x 768-dim 32x BQ, making permutation computation fit in page cache.
        final int[] permutation;
        if (reorderStrategy instanceof org.opensearch.knn.memoryoptsearch.faiss.reorder.bpreorder.QuantizedBipartiteReorderStrategy quantizedStrategy) {
            final org.apache.lucene.index.ByteVectorValues quantizedVectors = buildMergedQuantizedByteVectors(mergeState, fieldInfo, mapping);
            if (quantizedVectors != null) {
                permutation = quantizedStrategy.computePermutationFromQuantized(
                    quantizedVectors, SegmentReorderService.DEFAULT_REORDER_THREADS
                );
            } else {
                // No quantized vectors available (uncompressed index) — fall back to float BP
                permutation = reorderStrategy.computePermutation(
                    mergedRA, SegmentReorderService.DEFAULT_REORDER_THREADS, fieldInfo.getVectorSimilarityFunction()
                );
            }
        } else {
            permutation = reorderStrategy.computePermutation(
                mergedRA, SegmentReorderService.DEFAULT_REORDER_THREADS, fieldInfo.getVectorSimilarityFunction()
            );
        }

        long permutation_ms = sw.stop().totalTime().millis();
        sw = new StopWatch().start();

        // Fused .vec write + .faiss build: the VecWritingReorderedKNNFloatVectorValues writes
        // each vector to .vec as it yields it to the FAISS graph builder. Single pass over the
        // permuted vectors — one set of random reads serves both .vec and .faiss.
        final ReorderAwareFlatVectorsWriter flatWriter = (ReorderAwareFlatVectorsWriter) flatVectorsWriter;
        final IndexOutput vectorData = flatWriter.getVectorDataOutput();
        final long vectorDataOffset = vectorData.alignFilePointer(Float.BYTES);

        // Train quantization (order-independent, uses standard supplier)
        final VectorDataType vectorDataType = extractVectorDataType(fieldInfo);
        final Supplier<KNNVectorValues<?>> standardSupplier = getKNNVectorValuesSupplierForMerge(
            vectorDataType, fieldInfo, mergeState
        );
        final QuantizationState quantizationState = train(fieldInfo, standardSupplier, mapping.totalLiveDocs());

        if (quantizationState == null && shouldSkipBuildingVectorDataStructure(mapping.totalLiveDocs())) {
            // No FAISS graph needed — fall back to separate .vec write loop
            final int dim = mergedRA.dimension();
            final java.nio.ByteBuffer buffer = java.nio.ByteBuffer.allocate(dim * Float.BYTES).order(java.nio.ByteOrder.LITTLE_ENDIAN);
            for (int newOrd = 0; newOrd < mapping.totalLiveDocs(); newOrd++) {
                float[] vec = mergedRA.vectorValue(permutation[newOrd]);
                buffer.asFloatBuffer().put(vec);
                vectorData.writeBytes(buffer.array(), buffer.array().length);
            }
            final long vectorDataLength = vectorData.getFilePointer() - vectorDataOffset;
            ReorderedFieldMetaWriter.writeReorderedMeta(
                flatWriter.getMetaOutput(), fieldInfo, vectorDataOffset, vectorDataLength,
                mapping.mergedOrdToDocId(), permutation
            );
            long writeVec_ms = sw.stop().totalTime().millis();
            log.info("[ReplacementFree] {} vectors field [{}]: buildRA={}ms permutation={}ms writeVec={}ms (skipped FAISS)",
                mapping.totalLiveDocs(), fieldInfo.getName(), buildRA_ms, permutation_ms, writeVec_ms);
            return;
        }

        // Create the fused supplier: writes .vec AND feeds FAISS in one pass
        final Supplier<KNNVectorValues<?>> fusedSupplier = () -> new VecWritingReorderedKNNFloatVectorValues(
            mergedRA, permutation, mapping.mergedOrdToDocId(), vectorData
        );
        final NativeIndexWriter writer = NativeIndexWriter.getWriter(
            fieldInfo, segmentWriteState, quantizationState, nativeIndexBuildStrategyFactory
        );
        writer.mergeIndex(fusedSupplier, mapping.totalLiveDocs());

        // After FAISS build exhausts the iterator, .vec is fully written
        final long vectorDataLength = vectorData.getFilePointer() - vectorDataOffset;

        // Write reordered .vemf metadata + skip list
        ReorderedFieldMetaWriter.writeReorderedMeta(
            flatWriter.getMetaOutput(), fieldInfo, vectorDataOffset, vectorDataLength,
            mapping.mergedOrdToDocId(), permutation
        );

        long fusedBuild_ms = sw.stop().totalTime().millis();
        KNNGraphValue.MERGE_TOTAL_TIME_IN_MILLIS.incrementBy(fusedBuild_ms);
        log.info("[ReplacementFree] {} vectors field [{}]: buildRA={}ms permutation={}ms fusedVecFaiss={}ms total={}ms",
            mapping.totalLiveDocs(), fieldInfo.getName(), buildRA_ms, permutation_ms, fusedBuild_ms,
            buildRA_ms + permutation_ms + fusedBuild_ms);
    }

    private MergedRandomAccessFloatVectorValues buildMergedRandomAccess(
        MergeState mergeState, FieldInfo fieldInfo, MergeOrdMappingBuilder.MergeOrdMapping mapping
    ) throws IOException {
        final int numSegments = mergeState.knnVectorsReaders.length;
        final FloatVectorValues[] segmentValues = new FloatVectorValues[numSegments];
        for (int seg = 0; seg < numSegments; seg++) {
            if (mergeState.knnVectorsReaders[seg] != null) {
                segmentValues[seg] = mergeState.knnVectorsReaders[seg].getFloatVectorValues(fieldInfo.name);
            }
        }
        return new MergedRandomAccessFloatVectorValues(segmentValues, mapping.segmentStarts(), mapping.liveLocalOrds());
    }

    /**
     * Build a merged random-access ByteVectorValues over source segments' quantized vectors
     * from their .faiss flat storage. Returns null if any source segment doesn't have byte
     * vectors (e.g., uncompressed index without binary quantization).
     */
    private org.apache.lucene.index.ByteVectorValues buildMergedQuantizedByteVectors(
        MergeState mergeState, FieldInfo fieldInfo, MergeOrdMappingBuilder.MergeOrdMapping mapping
    ) throws IOException {
        final int numSegments = mergeState.knnVectorsReaders.length;
        final org.apache.lucene.index.ByteVectorValues[] segmentValues = new org.apache.lucene.index.ByteVectorValues[numSegments];
        for (int seg = 0; seg < numSegments; seg++) {
            if (mergeState.knnVectorsReaders[seg] != null) {
                try {
                    segmentValues[seg] = mergeState.knnVectorsReaders[seg].getByteVectorValues(fieldInfo.name);
                } catch (UnsupportedOperationException | IllegalStateException e) {
                    // This segment doesn't have byte vectors — fall back to float path
                    log.debug("[QuantizedBP] Segment {} has no byte vectors for field {}, falling back to float BP",
                        seg, fieldInfo.name);
                    return null;
                }
                if (segmentValues[seg] == null) {
                    return null;
                }
            }
        }
        return new org.opensearch.knn.memoryoptsearch.faiss.reorder.MergedRandomAccessByteVectorValues(
            segmentValues, mapping.segmentStarts(), mapping.liveLocalOrds()
        );
    }

    @Override
    public void finish() throws IOException {
        if (finished) {
            throw new IllegalStateException("NativeEnginesKNNVectorsWriter is already finished");
        }
        finished = true;
        if (quantizationStateWriter != null) {
            quantizationStateWriter.writeFooter();
        }
        // This writes the .vec/.vemf codec footers.
        flatVectorsWriter.finish();
        // Close to flush the IndexOutput buffers to disk so we can read the files.
        flatVectorsWriter.close();

        // Now that .vec/.vemf are finalized and flushed to disk, reorder any fields
        // that were marked during mergeOneField(). Rewrites .vec, .vemf, and .faiss.
        for (FieldInfo fieldInfo : fieldsToReorder) {
            try {
                StopWatch reorderWatch = new StopWatch().start();

                if (reorderStrategy instanceof MergeAwareReorderStrategy mergeAware && mergeState != null) {
                    // Merge-aware path: load source .kcs, merge centroids, assign-only
                    reorderWithMergedSummaries(mergeAware, fieldInfo);
                } else {
                    // Standard path: full recluster
                    SegmentReorderService reorderService = new SegmentReorderService(
                        segmentWriteState, fieldInfo, reorderStrategy
                    );
                    reorderService.reorderSegmentFiles();
                }

                long reorderMs = reorderWatch.stop().totalTime().millis();
                log.info("Reorder took {} ms for field [{}]", reorderMs, fieldInfo.getName());
            } catch (Exception e) {
                log.error("Failed to reorder field [{}], continuing without reorder", fieldInfo.getName(), e);
            }
        }
    }

    @SuppressWarnings("unchecked")
    private void writeFlushClusterSummary(
        MergeAwareReorderStrategy mergeAware,
        NativeEngineFieldVectorsWriter<?> field,
        FieldInfo fieldInfo,
        int totalLiveDocs
    ) {
        try {
            // Materialize in-memory vectors into float[][]
            var vectors = (java.util.Map<Integer, float[]>) field.getVectors();
            float[][] floatVecs = new float[totalLiveDocs][];
            int idx = 0;
            for (float[] v : vectors.values()) {
                floatVecs[idx++] = v;
            }
            FloatVectorValues fvv = FloatVectorValues.fromFloats(java.util.Arrays.asList(floatVecs), fieldInfo.getVectorDimension());

            ClusterResult result = mergeAware.computePermutationWithSummary(fvv, 1, fieldInfo.getVectorSimilarityFunction());
            ClusterSummaryWriter.writeAndGetFileName(segmentWriteState, fieldInfo, result.summary());
            log.info("Wrote .kcs at flush for field [{}], k={}, n={}", fieldInfo.name, result.summary().k, totalLiveDocs);
        } catch (Exception e) {
            log.error("Failed to write .kcs at flush for field [{}]", fieldInfo.name, e);
        }
    }

    private void reorderWithMergedSummaries(MergeAwareReorderStrategy mergeAware, FieldInfo fieldInfo) throws IOException {
        List<ClusterSummary> sourceSummaries = sourceClusterSummaries.get(fieldInfo.name);
        if (sourceSummaries == null || sourceSummaries.isEmpty()) {
            throw new IOException("No source cluster summaries for field " + fieldInfo.name
                + ". All source segments must have .kcs files when kmeans_merge_aware is enabled.");
        }

        SegmentReorderService reorderService = new SegmentReorderService(
            segmentWriteState, fieldInfo, mergeAware
        );
        reorderService.reorderSegmentFilesWithSummaries(sourceSummaries, mergeAware);
    }

    /**
     * Load .kcs files from source segments being merged.
     * Scans the directory for .kcs files excluding the target segment.
     * Logs a warning if the count doesn't match the expected number of source segments,
     * which can happen during concurrent merges when unrelated segments' .kcs files are present.
     * Extra centroids from unrelated segments only affect cluster quality (more centroids in the
     * pool), not correctness — the permutation is always a valid bijection over the merged vectors.
     */
    private List<ClusterSummary> loadSourceClusterSummaries(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        Directory dir = segmentWriteState.directory;
        String targetSegName = segmentWriteState.segmentInfo.name;
        String suffix = "_" + fieldInfo.name + ".kcs";
        String compoundSuffix = suffix + "c";
        int expectedSources = mergeState.maxDocs.length;
        List<ClusterSummary> summaries = new ArrayList<>();

        for (String file : dir.listAll()) {
            if (file.startsWith(targetSegName + "_")) continue;
            if (file.endsWith(compoundSuffix) || file.endsWith(suffix)) {
                summaries.add(ClusterSummaryReader.read(dir, file, fieldInfo.name));
            }
        }

        if (summaries.size() != expectedSources) {
            log.warn("Expected {} source .kcs files but found {} for field [{}] (concurrent merge may have extra .kcs files)",
                expectedSources, summaries.size(), fieldInfo.name);
        }
        return summaries;
    }

    @Override
    public void close() throws IOException {
        if (quantizationStateWriter != null) {
            quantizationStateWriter.closeOutput();
        }
        IOUtils.close(flatVectorsWriter);
    }

    @Override
    public long ramBytesUsed() {
        return SHALLOW_SIZE + flatVectorsWriter.ramBytesUsed() + fields.stream()
            .mapToLong(NativeEngineFieldVectorsWriter::ramBytesUsed)
            .sum();
    }

    private QuantizationState train(
        final FieldInfo fieldInfo,
        final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier,
        final int totalLiveDocs
    ) throws IOException {
        final QuantizationService quantizationService = QuantizationService.getInstance();
        final QuantizationParams quantizationParams = quantizationService.getQuantizationParams(fieldInfo);
        QuantizationState quantizationState = null;
        if (quantizationParams != null && totalLiveDocs > 0) {
            initQuantizationStateWriterIfNecessary();
            quantizationState = quantizationService.train(quantizationParams, knnVectorValuesSupplier, totalLiveDocs);
            quantizationStateWriter.writeState(fieldInfo.getFieldNumber(), quantizationState);
        }
        return quantizationState;
    }

    private int getLiveDocs(KNNVectorValues<?> vectorValues) throws IOException {
        int liveDocs = 0;
        while (vectorValues.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
            liveDocs++;
        }
        return liveDocs;
    }

    private void initQuantizationStateWriterIfNecessary() throws IOException {
        if (quantizationStateWriter == null) {
            quantizationStateWriter = new KNN990QuantizationStateWriter(segmentWriteState);
            quantizationStateWriter.writeHeader(segmentWriteState);
        }
    }

    private boolean shouldSkipBuildingVectorDataStructure(final long docCount) {
        if (approximateThreshold < 0) {
            return true;
        }
        return docCount < approximateThreshold;
    }
}
