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
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.opensearch.common.StopWatch;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexWriter;
import org.opensearch.knn.index.query.clumping.ClumpingContext;
import org.opensearch.knn.index.query.clumping.ClumpingIndexWriter;
import org.opensearch.knn.index.quantizationservice.QuantizationService;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.plugin.stats.KNNGraphValue;
import org.opensearch.knn.quantization.models.quantizationParams.QuantizationParams;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import static org.opensearch.knn.common.FieldInfoExtractor.extractClumpingFactor;
import static org.opensearch.knn.common.FieldInfoExtractor.extractVectorDataType;
import static org.opensearch.knn.common.FieldInfoExtractor.getSpaceType;
import static org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory.getKNNVectorValuesSupplierForMerge;
import static org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory.getVectorValuesSupplier;
import static org.opensearch.knn.indices.ModelDao.OpenSearchKNNModelDao;

/**
 * A KNNVectorsWriter class for writing the vector data strcutures and flat vectors for Native Engines.
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

    public NativeEngines990KnnVectorsWriter(
        SegmentWriteState segmentWriteState,
        FlatVectorsWriter flatVectorsWriter,
        Integer approximateThreshold,
        NativeIndexBuildStrategyFactory nativeIndexBuildStrategyFactory
    ) {
        this.segmentWriteState = segmentWriteState;
        this.flatVectorsWriter = flatVectorsWriter;
        this.approximateThreshold = approximateThreshold;
        this.nativeIndexBuildStrategyFactory = nativeIndexBuildStrategyFactory;
    }

    /**
     * Add new field for indexing.
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
     * @param maxDoc int
     * @param sortMap {@link Sorter.DocMap}
     */
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
            
            // Check if clumping is enabled for this field
            int clumpingFactor = extractClumpingFactor(fieldInfo);
            boolean clumpingEnabled = clumpingFactor > 1;
            
            Supplier<KNNVectorValues<?>> knnVectorValuesSupplier;
            int docsToIndex = totalLiveDocs;
            
            if (clumpingEnabled) {
                // Process vectors with clumping - select markers and write hidden vectors
                Map<Integer, float[]> processedVectors = processVectorsWithClumping(
                    field,
                    fieldInfo,
                    vectorDataType,
                    clumpingFactor
                );
                docsToIndex = processedVectors.size();
                
                if (docsToIndex == 0) {
                    log.debug("[Flush] No marker vectors after clumping for field {}", fieldInfo.getName());
                    continue;
                }
                
                // Create supplier for marker vectors only
                knnVectorValuesSupplier = () -> KNNVectorValuesFactory.getVectorValues(
                    vectorDataType,
                    field.getFlatFieldVectorsWriter().getDocsWithFieldSet(),
                    processedVectors
                );
                
                log.info(
                    "[Flush] Clumping enabled for field {}: {} total vectors -> {} markers indexed",
                    fieldInfo.getName(),
                    totalLiveDocs,
                    docsToIndex
                );
            } else {
                knnVectorValuesSupplier = getVectorValuesSupplier(
                    vectorDataType,
                    field.getFlatFieldVectorsWriter().getDocsWithFieldSet(),
                    field.getVectors()
                );
            }
            
            final QuantizationState quantizationState = train(fieldInfo, knnVectorValuesSupplier, docsToIndex);
            // should skip graph building only for non quantization use case and if threshold is met
            if (quantizationState == null && shouldSkipBuildingVectorDataStructure(docsToIndex)) {
                log.debug(
                    "Skip building vector data structure for field: {}, as liveDoc: {} is less than the threshold {} during flush",
                    fieldInfo.name,
                    docsToIndex,
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
            writer.flushIndex(knnVectorValuesSupplier, docsToIndex);
            long time_in_millis = stopWatch.stop().totalTime().millis();
            KNNGraphValue.REFRESH_TOTAL_TIME_IN_MILLIS.incrementBy(time_in_millis);
            log.debug("Flush took {} ms for vector field [{}]", time_in_millis, fieldInfo.getName());
        }
    }

    @Override
    public void mergeOneField(final FieldInfo fieldInfo, final MergeState mergeState) throws IOException {
        // This will ensure that we are merging the FlatIndex during force merge.
        flatVectorsWriter.mergeOneField(fieldInfo, mergeState);

        final VectorDataType vectorDataType = extractVectorDataType(fieldInfo);
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

        // Check if clumping is enabled for this field
        int clumpingFactor = extractClumpingFactor(fieldInfo);
        boolean clumpingEnabled = clumpingFactor > 1;
        
        Supplier<KNNVectorValues<?>> finalVectorValuesSupplier;
        int docsToIndex = totalLiveDocs;
        
        if (clumpingEnabled) {
            // For merge, we need to collect all vectors first, then process with clumping
            Map<Integer, float[]> allVectors = collectVectorsFromSupplier(knnVectorValuesSupplier, vectorDataType);
            
            // Process vectors with clumping
            Map<Integer, float[]> markerVectors = processVectorsWithClumpingForMerge(
                allVectors,
                fieldInfo,
                vectorDataType,
                clumpingFactor
            );
            docsToIndex = markerVectors.size();
            
            if (docsToIndex == 0) {
                log.debug("[Merge] No marker vectors after clumping for field {}", fieldInfo.getName());
                return;
            }
            
            // Create supplier for marker vectors only
            finalVectorValuesSupplier = () -> KNNVectorValuesFactory.getVectorValues(vectorDataType, markerVectors);
            
            log.info(
                "[Merge] Clumping enabled for field {}: {} total vectors -> {} markers indexed",
                fieldInfo.getName(),
                totalLiveDocs,
                docsToIndex
            );
        } else {
            finalVectorValuesSupplier = getKNNVectorValuesSupplierForMerge(vectorDataType, fieldInfo, mergeState);
        }

        final QuantizationState quantizationState = train(fieldInfo, finalVectorValuesSupplier, docsToIndex);
        // should skip graph building only for non quantization use case and if threshold is met
        if (quantizationState == null && shouldSkipBuildingVectorDataStructure(docsToIndex)) {
            log.debug(
                "Skip building vector data structure for field: {}, as liveDoc: {} is less than the threshold {} during merge",
                fieldInfo.name,
                docsToIndex,
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

        writer.mergeIndex(finalVectorValuesSupplier, docsToIndex);

        long time_in_millis = stopWatch.stop().totalTime().millis();
        KNNGraphValue.MERGE_TOTAL_TIME_IN_MILLIS.incrementBy(time_in_millis);
        log.debug("Merge took {} ms for vector field [{}]", time_in_millis, fieldInfo.getName());
    }

    /**
     * Called once at the end before close
     */
    @Override
    public void finish() throws IOException {
        if (finished) {
            throw new IllegalStateException("NativeEnginesKNNVectorsWriter is already finished");
        }
        finished = true;
        if (quantizationStateWriter != null) {
            quantizationStateWriter.writeFooter();
        }
        flatVectorsWriter.finish();
    }

    /**
     * Closes this stream and releases any system resources associated
     * with it. If the stream is already closed then invoking this
     * method has no effect.
     *
     * <p> As noted in {@link AutoCloseable#close()}, cases where the
     * close may fail require careful attention. It is strongly advised
     * to relinquish the underlying resources and to internally
     * <em>mark</em> the {@code Closeable} as closed, prior to throwing
     * the {@code IOException}.
     *
     * @throws IOException if an I/O error occurs
     */
    @Override
    public void close() throws IOException {
        if (quantizationStateWriter != null) {
            quantizationStateWriter.closeOutput();
        }
        IOUtils.close(flatVectorsWriter);
    }

    /**
     * Return the memory usage of this object in bytes. Negative values are illegal.
     */
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

    /**
     * The {@link KNNVectorValues} will be exhausted after this function run. So make sure that you are not sending the
     * vectorsValues object which you plan to use later
     */
    private int getLiveDocs(KNNVectorValues<?> vectorValues) throws IOException {
        // Count all the live docs as there vectorValues.totalLiveDocs() just gives the cost for the FloatVectorValues,
        // and doesn't tell the correct number of docs, if there are deleted docs in the segment. So we are counting
        // the total live docs here.
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

    /**
     * Processes vectors with clumping during flush.
     * Selects marker vectors and writes hidden vectors to disk.
     * 
     * @param field the field vectors writer containing all vectors
     * @param fieldInfo the field info
     * @param vectorDataType the vector data type
     * @param clumpingFactor the clumping factor
     * @return map of marker doc IDs to their vectors
     */
    private Map<Integer, float[]> processVectorsWithClumping(
        NativeEngineFieldVectorsWriter<?> field,
        FieldInfo fieldInfo,
        VectorDataType vectorDataType,
        int clumpingFactor
    ) throws IOException {
        // Convert vectors to float[] map
        Map<Integer, float[]> docIdToVector = new HashMap<>();
        Map<Integer, ?> vectors = field.getVectors();
        
        for (Map.Entry<Integer, ?> entry : vectors.entrySet()) {
            int docId = entry.getKey();
            Object vector = entry.getValue();
            
            if (vector instanceof float[]) {
                docIdToVector.put(docId, (float[]) vector);
            } else if (vector instanceof byte[]) {
                // Convert byte[] to float[] for clumping processing
                byte[] byteVector = (byte[]) vector;
                float[] floatVector = new float[byteVector.length];
                for (int i = 0; i < byteVector.length; i++) {
                    floatVector[i] = byteVector[i];
                }
                docIdToVector.put(docId, floatVector);
            }
        }
        
        if (docIdToVector.isEmpty()) {
            return docIdToVector;
        }
        
        // Get space type for distance calculation
        SpaceType spaceType = getSpaceTypeFromFieldInfo(fieldInfo);
        
        // Create clumping context and index writer
        ClumpingContext clumpingContext = ClumpingContext.withFactor(clumpingFactor);
        ClumpingIndexWriter clumpingWriter = new ClumpingIndexWriter(
            clumpingContext,
            segmentWriteState.directory,
            segmentWriteState.segmentInfo.name,
            fieldInfo.name
        );
        
        // Process vectors - this selects markers and writes hidden vectors to disk
        return clumpingWriter.processVectors(
            docIdToVector,
            spaceType,
            vectorDataType,
            fieldInfo.getVectorDimension()
        );
    }

    /**
     * Processes vectors with clumping during merge.
     * 
     * @param allVectors all vectors collected from the merge
     * @param fieldInfo the field info
     * @param vectorDataType the vector data type
     * @param clumpingFactor the clumping factor
     * @return map of marker doc IDs to their vectors
     */
    private Map<Integer, float[]> processVectorsWithClumpingForMerge(
        Map<Integer, float[]> allVectors,
        FieldInfo fieldInfo,
        VectorDataType vectorDataType,
        int clumpingFactor
    ) throws IOException {
        if (allVectors.isEmpty()) {
            return allVectors;
        }
        
        // Get space type for distance calculation
        SpaceType spaceType = getSpaceTypeFromFieldInfo(fieldInfo);
        
        // Create clumping context and index writer
        ClumpingContext clumpingContext = ClumpingContext.withFactor(clumpingFactor);
        ClumpingIndexWriter clumpingWriter = new ClumpingIndexWriter(
            clumpingContext,
            segmentWriteState.directory,
            segmentWriteState.segmentInfo.name,
            fieldInfo.name
        );
        
        // Process vectors - this selects markers and writes hidden vectors to disk
        return clumpingWriter.processVectors(
            allVectors,
            spaceType,
            vectorDataType,
            fieldInfo.getVectorDimension()
        );
    }

    /**
     * Collects all vectors from a supplier into a map.
     * 
     * @param supplier the vector values supplier
     * @param vectorDataType the vector data type
     * @return map of doc IDs to vectors
     */
    private Map<Integer, float[]> collectVectorsFromSupplier(
        Supplier<KNNVectorValues<?>> supplier,
        VectorDataType vectorDataType
    ) throws IOException {
        Map<Integer, float[]> result = new HashMap<>();
        KNNVectorValues<?> vectorValues = supplier.get();
        
        while (vectorValues.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
            int docId = vectorValues.docId();
            Object vector = vectorValues.getVector();
            
            if (vector instanceof float[]) {
                result.put(docId, ((float[]) vector).clone());
            } else if (vector instanceof byte[]) {
                byte[] byteVector = (byte[]) vector;
                float[] floatVector = new float[byteVector.length];
                for (int i = 0; i < byteVector.length; i++) {
                    floatVector[i] = byteVector[i];
                }
                result.put(docId, floatVector);
            }
        }
        
        return result;
    }

    /**
     * Gets the space type from field info.
     */
    private SpaceType getSpaceTypeFromFieldInfo(FieldInfo fieldInfo) {
        try {
            return getSpaceType(OpenSearchKNNModelDao.getInstance(), fieldInfo);
        } catch (Exception e) {
            // Default to L2 if space type cannot be determined
            log.warn("Could not determine space type for field {}, defaulting to L2", fieldInfo.name);
            return SpaceType.L2;
        }
    }
}
