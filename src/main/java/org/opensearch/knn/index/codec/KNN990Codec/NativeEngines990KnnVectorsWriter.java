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
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.opensearch.common.StopWatch;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.AbstractNativeEnginesKnnVectorsWriter;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexWriter;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.ReorderedKNNFloatVectorValues;
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
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import static org.opensearch.knn.common.FieldInfoExtractor.extractVectorDataType;
import static org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory.getKNNVectorValuesSupplierForMerge;
import static org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory.getVectorValuesSupplier;

/**
 * A KNNVectorsWriter class for writing the vector data structures and flat vectors for Native Engines.
 */
@Log4j2
public class NativeEngines990KnnVectorsWriter extends AbstractNativeEnginesKnnVectorsWriter {
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
    private final Map<String, List<ClusterSummary>> sourceClusterSummaries = new HashMap<>();

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

    /**
     * Add new field for indexing.
     *
     * @param fieldInfo {@link FieldInfo}
     */
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

    /**
     * Flush all buffered data on disk. This is not fsync. This is lucene flush.
     *
     * @param maxDoc  int
     * @param sortMap {@link Sorter.DocMap}
     */
    @Override
    public void flush(int maxDoc, final Sorter.DocMap sortMap) throws IOException {
        flatVectorsWriter.flush(maxDoc, sortMap);

        for (final NativeEngineFieldVectorsWriter<?> field : fields) {
            final FieldInfo fieldInfo = field.getFieldInfo();
            doFlush(
                fieldInfo,
                field.getFlatFieldVectorsWriter(),
                field.getVectors(),
                this::train,
                approximateThreshold,
                segmentWriteState,
                nativeIndexBuildStrategyFactory,
                null
            );

            // For merge-aware strategy, cluster at flush and write .kcs for future merges
            final VectorDataType vectorDataType = extractVectorDataType(fieldInfo);
            if (reorderStrategy instanceof MergeAwareReorderStrategy mergeAware && vectorDataType == VectorDataType.FLOAT) {
                int totalLiveDocs = field.getVectors().size();
                if (totalLiveDocs > 0) {
                    writeFlushClusterSummary(mergeAware, field, fieldInfo, totalLiveDocs);
                }
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
        final int totalLiveDocs = getLiveDocs(knnVectorValuesSupplier.get());
        if (totalLiveDocs == 0) {
            log.debug("[Merge] No live docs for field {}", fieldInfo.getName());
            return;
        }

        final QuantizationState quantizationState = train(fieldInfo, knnVectorValuesSupplier, totalLiveDocs);
        if (quantizationState == null && shouldSkipBuildingVectorDataStructure(totalLiveDocs, approximateThreshold)) {
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

        final StopWatch stopWatch = new StopWatch().start();
        writer.mergeIndex(knnVectorValuesSupplier, totalLiveDocs);
        final long time_in_millis = stopWatch.stop().totalTime().millis();
        KNNGraphValue.MERGE_TOTAL_TIME_IN_MILLIS.incrementBy(time_in_millis);
        log.debug("Merge took {} ms for vector field [{}]", time_in_millis, fieldInfo.getName());

        // Mark this field for reordering. The actual reorder happens in finish() after
        // flatVectorsWriter.finish() writes the .vec/.vemf footers, so we can read them.
        if (reorderStrategy != null && totalLiveDocs >= SegmentReorderService.MIN_VECTORS_FOR_REORDER) {
            fieldsToReorder.add(fieldInfo);

            // For merge-aware strategy, load source .kcs files now while we have MergeState.
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

        // Compute permutation from source mmap
        final int[] permutation = reorderStrategy.computePermutation(
            mergedRA, SegmentReorderService.DEFAULT_REORDER_THREADS, fieldInfo.getVectorSimilarityFunction()
        );

        long permutation_ms = sw.stop().totalTime().millis();
        sw = new StopWatch().start();

        // Write .vec in permuted order directly into delegate's stream
        final ReorderAwareFlatVectorsWriter flatWriter = (ReorderAwareFlatVectorsWriter) flatVectorsWriter;
        final IndexOutput vectorData = flatWriter.getVectorDataOutput();
        final long vectorDataOffset = vectorData.alignFilePointer(Float.BYTES);
        final int dim = mergedRA.dimension();
        final java.nio.ByteBuffer buffer = java.nio.ByteBuffer.allocate(dim * Float.BYTES).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (int newOrd = 0; newOrd < mapping.totalLiveDocs(); newOrd++) {
            float[] vec = mergedRA.vectorValue(permutation[newOrd]);
            buffer.asFloatBuffer().put(vec);
            vectorData.writeBytes(buffer.array(), buffer.array().length);
        }
        final long vectorDataLength = vectorData.getFilePointer() - vectorDataOffset;

        // Write reordered .vemf metadata + skip list directly into delegate's stream
        ReorderedFieldMetaWriter.writeReorderedMeta(
            flatWriter.getMetaOutput(), fieldInfo, vectorDataOffset, vectorDataLength,
            mapping.mergedOrdToDocId(), permutation
        );

        long writeVec_ms = sw.stop().totalTime().millis();
        sw = new StopWatch().start();

        // Build .faiss with reordered supplier — graph built in permuted order
        final Supplier<KNNVectorValues<?>> reorderedSupplier = () -> new ReorderedKNNFloatVectorValues(
            mergedRA, permutation, mapping.mergedOrdToDocId()
        );
        final VectorDataType vectorDataType = extractVectorDataType(fieldInfo);
        final Supplier<KNNVectorValues<?>> standardSupplier = getKNNVectorValuesSupplierForMerge(
            vectorDataType, fieldInfo, mergeState
        );
        final QuantizationState quantizationState = train(fieldInfo, standardSupplier, mapping.totalLiveDocs());
        if (quantizationState == null && shouldSkipBuildingVectorDataStructure(mapping.totalLiveDocs(), approximateThreshold)) {
            log.info("[ReplacementFree] {} vectors field [{}]: buildRA={}ms permutation={}ms writeVec={}ms (skipped FAISS)",
                mapping.totalLiveDocs(), fieldInfo.getName(), buildRA_ms, permutation_ms, writeVec_ms);
            return;
        }
        final NativeIndexWriter writer = NativeIndexWriter.getWriter(
            fieldInfo, segmentWriteState, quantizationState, nativeIndexBuildStrategyFactory
        );
        writer.mergeIndex(reorderedSupplier, mapping.totalLiveDocs());

        long faiss_ms = sw.stop().totalTime().millis();
        KNNGraphValue.MERGE_TOTAL_TIME_IN_MILLIS.incrementBy(faiss_ms);
        log.info("[ReplacementFree] {} vectors field [{}]: buildRA={}ms permutation={}ms writeVec={}ms faiss={}ms total={}ms",
            mapping.totalLiveDocs(), fieldInfo.getName(), buildRA_ms, permutation_ms, writeVec_ms, faiss_ms,
            buildRA_ms + permutation_ms + writeVec_ms + faiss_ms);
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
            var vectors = (Map<Integer, float[]>) field.getVectors();
            float[][] floatVecs = new float[totalLiveDocs][];
            int idx = 0;
            for (float[] v : vectors.values()) {
                floatVecs[idx++] = v;
            }
            FloatVectorValues fvv = FloatVectorValues.fromFloats(Arrays.asList(floatVecs), fieldInfo.getVectorDimension());

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
    ) {
        final QuantizationService quantizationService = QuantizationService.getInstance();
        final QuantizationParams quantizationParams = quantizationService.getQuantizationParams(fieldInfo);
        QuantizationState quantizationState = null;
        if (quantizationParams != null && totalLiveDocs > 0) {
            try {
                initQuantizationStateWriterIfNecessary();
                quantizationState = quantizationService.train(quantizationParams, knnVectorValuesSupplier, totalLiveDocs);
                quantizationStateWriter.writeState(fieldInfo.getFieldNumber(), quantizationState);
            } catch (IOException e) {
                log.error("Failed to train quantization parameters for field: {}", fieldInfo.name, e);
                throw new RuntimeException(e);
            }
        }
        return quantizationState;
    }

    private void initQuantizationStateWriterIfNecessary() throws IOException {
        if (quantizationStateWriter == null) {
            quantizationStateWriter = new KNN990QuantizationStateWriter(segmentWriteState);
            quantizationStateWriter.writeHeader(segmentWriteState);
        }
    }
}
