/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.opensearch.common.ValidationException;
import org.opensearch.core.ParseField;
import org.opensearch.core.common.Strings;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.query.AbstractQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryRewriteContext;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.index.query.WithFieldName;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.VectorQueryType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNLibrarySearchContext;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MemoryOptimizedSearchSupportSpec;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.model.QueryContext;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.mapper.KNNMappingConfig;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;
import org.opensearch.knn.index.query.clumping.ClumpingContext;
import org.opensearch.knn.index.query.parser.ClumpingParser;
import org.opensearch.knn.index.query.parser.KNNQueryBuilderParser;
import org.opensearch.knn.index.query.parser.RescoreParser;
import org.opensearch.knn.index.query.rescore.RescoreContext;
import org.opensearch.knn.index.util.IndexUtil;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.indices.ModelMetadata;
import org.opensearch.knn.indices.ModelUtil;

import java.io.IOException;
import java.util.Arrays;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

import static org.opensearch.knn.common.KNNConstants.EXPAND_NESTED;
import static org.opensearch.knn.common.KNNConstants.MAX_DISTANCE;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_NPROBES;
import static org.opensearch.knn.common.KNNConstants.MIN_SCORE;
import static org.opensearch.knn.common.KNNValidationUtil.validateByteVectorValue;
import static org.opensearch.knn.index.engine.KNNEngine.ENGINES_SUPPORTING_RADIAL_SEARCH;
import static org.opensearch.knn.index.engine.validation.ParameterValidator.validateParameters;
import static org.opensearch.knn.index.query.parser.MethodParametersParser.validateMethodParameters;
import static org.opensearch.knn.index.query.parser.RescoreParser.RESCORE_OVERSAMPLE_PARAMETER;
import static org.opensearch.knn.index.query.parser.RescoreParser.RESCORE_PARAMETER;

/**
 * Helper class to build the KNN query
 */
// The builder validates the member variables so access to the constructor is prohibited to not accidentally bypass validations
@AllArgsConstructor(access = AccessLevel.PRIVATE)
@Log4j2
public class KNNQueryBuilder extends AbstractQueryBuilder<KNNQueryBuilder> implements WithFieldName {
    private static ModelDao modelDao;

    public static final ParseField VECTOR_FIELD = new ParseField("vector");
    public static final ParseField K_FIELD = new ParseField("k");
    public static final ParseField FILTER_FIELD = new ParseField("filter");
    public static final ParseField IGNORE_UNMAPPED_FIELD = new ParseField("ignore_unmapped");
    public static final ParseField EXPAND_NESTED_FIELD = new ParseField(EXPAND_NESTED);
    public static final ParseField MAX_DISTANCE_FIELD = new ParseField(MAX_DISTANCE);
    public static final ParseField MIN_SCORE_FIELD = new ParseField(MIN_SCORE);
    public static final ParseField EF_SEARCH_FIELD = new ParseField(METHOD_PARAMETER_EF_SEARCH);
    public static final ParseField NPROBE_FIELD = new ParseField(METHOD_PARAMETER_NPROBES);
    public static final ParseField METHOD_PARAMS_FIELD = new ParseField(METHOD_PARAMETER);
    public static final ParseField RESCORE_FIELD = new ParseField(RESCORE_PARAMETER);
    public static final ParseField RESCORE_OVERSAMPLE_FIELD = new ParseField(RESCORE_OVERSAMPLE_PARAMETER);
    public static final ParseField CLUMPING_FIELD = new ParseField(ClumpingParser.CLUMPING_PARAMETER);

    public static final int K_MAX = 10000;
    /**
     * The name for the knn query
     */
    public static final String NAME = "knn";
    /**
     * The default mode terms are combined in a match query
     */
    private final String fieldName;
    private final float[] vector;
    @Getter
    @Setter
    private Integer k;
    @Getter
    private Float maxDistance;
    @Getter
    private Float minScore;
    @Getter
    private Map<String, ?> methodParameters;
    @Getter
    private QueryBuilder filter;
    @Getter
    private boolean ignoreUnmapped;
    @Getter
    private RescoreContext rescoreContext;
    @Getter
    private ClumpingContext clumpingContext;
    @Getter
    private Boolean expandNested;

    /**
     * Constructs a new query with the given field name and vector
     *
     * @param fieldName Name of the field
     * @param vector    Array of floating points
     * @deprecated Use {@code {@link KNNQueryBuilder.Builder}} instead
     */
    @Deprecated
    public KNNQueryBuilder(String fieldName, float[] vector) {
        if (Strings.isNullOrEmpty(fieldName)) {
            throw new IllegalArgumentException(String.format("[%s] requires fieldName", NAME));
        }
        if (vector == null) {
            throw new IllegalArgumentException(String.format("[%s] requires query vector", NAME));
        }
        if (vector.length == 0) {
            throw new IllegalArgumentException(String.format("[%s] query vector is empty", NAME));
        }
        this.fieldName = fieldName;
        this.vector = vector;
    }

    /**
     * lombok SuperBuilder annotation requires a builder annotation on parent class to work well
     * {@link AbstractQueryBuilder#boost()} and {@link AbstractQueryBuilder#queryName()} both need to be called
     * A custom builder helps with the calls to the parent class, simultaneously addressing the problem of telescoping
     * constructors in this class.
     */
    public static class Builder {
        private String fieldName;
        private float[] vector;
        private Integer k;
        private Map<String, ?> methodParameters;
        private Float maxDistance;
        private Float minScore;
        private QueryBuilder filter;
        private boolean ignoreUnmapped;
        private String queryName;
        private float boost = DEFAULT_BOOST;
        private RescoreContext rescoreContext;
        private ClumpingContext clumpingContext;
        private Boolean expandNested;

        public Builder() {}

        public Builder fieldName(String fieldName) {
            this.fieldName = fieldName;
            return this;
        }

        public Builder vector(float[] vector) {
            this.vector = vector;
            return this;
        }

        public Builder k(Integer k) {
            this.k = k;
            return this;
        }

        public Builder methodParameters(Map<String, ?> methodParameters) {
            this.methodParameters = methodParameters;
            return this;
        }

        public Builder maxDistance(Float maxDistance) {
            this.maxDistance = maxDistance;
            return this;
        }

        public Builder minScore(Float minScore) {
            this.minScore = minScore;
            return this;
        }

        public Builder ignoreUnmapped(boolean ignoreUnmapped) {
            this.ignoreUnmapped = ignoreUnmapped;
            return this;
        }

        public Builder filter(QueryBuilder filter) {
            this.filter = filter;
            return this;
        }

        public Builder queryName(String queryName) {
            this.queryName = queryName;
            return this;
        }

        public Builder boost(float boost) {
            this.boost = boost;
            return this;
        }

        public Builder rescoreContext(RescoreContext rescoreContext) {
            this.rescoreContext = rescoreContext;
            return this;
        }

        public Builder clumpingContext(ClumpingContext clumpingContext) {
            this.clumpingContext = clumpingContext;
            return this;
        }

        public Builder expandNested(Boolean expandNested) {
            this.expandNested = expandNested;
            return this;
        }

        public KNNQueryBuilder build() {
            validate();
            return new KNNQueryBuilder(
                fieldName,
                vector,
                k,
                maxDistance,
                minScore,
                methodParameters,
                filter,
                ignoreUnmapped,
                rescoreContext,
                clumpingContext,
                expandNested
            ).boost(boost).queryName(queryName);
        }

        private void validate() {
            if (Strings.isNullOrEmpty(fieldName)) {
                throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] requires fieldName", NAME));
            }

            if (vector == null) {
                throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] requires query vector", NAME));
            } else if (vector.length == 0) {
                throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] query vector is empty", NAME));
            }

            if (k == null && minScore == null && maxDistance == null) {
                throw new IllegalArgumentException(
                    String.format(Locale.ROOT, "[%s] requires exactly one of k, distance or score to be set", NAME)
                );
            }

            if ((k != null && maxDistance != null) || (maxDistance != null && minScore != null) || (k != null && minScore != null)) {
                throw new IllegalArgumentException(
                    String.format(Locale.ROOT, "[%s] requires exactly one of k, distance or score to be set", NAME)
                );
            }

            if (k != null) {
                if (k <= 0 || k > K_MAX) {
                    final String errorMessage = "[" + NAME + "] requires k to be in the range (0, " + K_MAX + "]";
                    throw new IllegalArgumentException(errorMessage);
                }
            }

            if (minScore != null) {
                if (minScore <= 0) {
                    throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] requires minScore to be greater than 0", NAME));
                }
            }

            if (methodParameters != null) {
                ValidationException validationException = validateMethodParameters(methodParameters);
                if (validationException != null) {
                    throw new IllegalArgumentException(
                        String.format(Locale.ROOT, "[%s] errors in method parameter [%s]", NAME, validationException.getMessage())
                    );
                }
            }

            if (rescoreContext != null) {
                ValidationException validationException = RescoreParser.validate(rescoreContext);
                if (validationException != null) {
                    throw new IllegalArgumentException(
                        String.format(Locale.ROOT, "[%s] errors in rescore parameter [%s]", NAME, validationException.getMessage())
                    );
                }
            }
        }
    }

    public static KNNQueryBuilder.Builder builder() {
        return new KNNQueryBuilder.Builder();
    }

    /**
     * Constructs a new query for top k search
     *
     * @param fieldName Name of the filed
     * @param vector    Array of floating points
     * @param k         K nearest neighbours for the given vector
     */
    @Deprecated
    public KNNQueryBuilder(String fieldName, float[] vector, int k) {
        this(fieldName, vector, k, null);
    }

    @Deprecated
    public KNNQueryBuilder(String fieldName, float[] vector, int k, QueryBuilder filter) {
        if (Strings.isNullOrEmpty(fieldName)) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] requires fieldName", NAME));
        }
        if (vector == null) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] requires query vector", NAME));
        }
        if (vector.length == 0) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] query vector is empty", NAME));
        }
        if (k <= 0) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] requires k > 0", NAME));
        }
        if (k > K_MAX) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] requires k <= %d", NAME, K_MAX));
        }

        this.fieldName = fieldName;
        this.vector = vector;
        this.k = k;
        this.filter = filter;
        this.ignoreUnmapped = false;
        this.maxDistance = null;
        this.minScore = null;
        this.rescoreContext = null;
        this.expandNested = null;
    }

    public static void initialize(ModelDao modelDao) {
        KNNQueryBuilder.modelDao = modelDao;
    }

    /**
     * @param in Reads from stream
     * @throws IOException Throws IO Exception
     */
    public KNNQueryBuilder(StreamInput in) throws IOException {
        super(in);
        KNNQueryBuilder.Builder builder = KNNQueryBuilderParser.streamInput(in, IndexUtil::isClusterOnOrAfterMinRequiredVersion);
        fieldName = builder.fieldName;
        vector = builder.vector;
        k = builder.k;
        filter = builder.filter;
        ignoreUnmapped = builder.ignoreUnmapped;
        maxDistance = builder.maxDistance;
        minScore = builder.minScore;
        methodParameters = builder.methodParameters;
        rescoreContext = builder.rescoreContext;
        clumpingContext = builder.clumpingContext;
        expandNested = builder.expandNested;
    }

    @Override
    protected void doWriteTo(StreamOutput out) throws IOException {
        KNNQueryBuilderParser.streamOutput(out, this, IndexUtil::isClusterOnOrAfterMinRequiredVersion);
    }

    /**
     * @return The field name used in this query
     */
    @Override
    public String fieldName() {
        return this.fieldName;
    }

    /**
     * @return Returns the vector used in this query.
     */
    public Object vector() {
        return this.vector;
    }

    @Override
    public void doXContent(XContentBuilder builder, Params params) throws IOException {
        KNNQueryBuilderParser.toXContent(builder, params, this);
    }

    /**
     * Add a filter to Neural Query Builder
     * @param filterToBeAdded fiXlter to be added
     * @return return itself with underlying filter combined with passed in filter
     */
    @Override
    public QueryBuilder filter(QueryBuilder filterToBeAdded) {
        if (validateFilterParams(filterToBeAdded) == false) {
            return this;
        }

        if (this.filter == null) {
            return KNNQueryBuilder.builder()
                .fieldName(fieldName)
                .vector(vector)
                .k(k)
                .maxDistance(maxDistance)
                .minScore(minScore)
                .methodParameters(methodParameters)
                .filter(filterToBeAdded)
                .ignoreUnmapped(ignoreUnmapped)
                .rescoreContext(rescoreContext)
                .clumpingContext(clumpingContext)
                .expandNested(expandNested)
                .build();
        }

        return KNNQueryBuilder.builder()
            .fieldName(fieldName)
            .vector(vector)
            .k(k)
            .maxDistance(maxDistance)
            .minScore(minScore)
            .methodParameters(methodParameters)
            .filter(filter.filter(filterToBeAdded))
            .ignoreUnmapped(ignoreUnmapped)
            .rescoreContext(rescoreContext)
            .clumpingContext(clumpingContext)
            .expandNested(expandNested)
            .build();
    }

    @Override
    protected Query doToQuery(QueryShardContext context) {
        MappedFieldType mappedFieldType = context.fieldMapper(this.fieldName);

        if (mappedFieldType == null && ignoreUnmapped) {
            return new MatchNoDocsQuery();
        }

        if (!(mappedFieldType instanceof KNNVectorFieldType)) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Field '%s' is not knn_vector type.", this.fieldName));
        }
        KNNVectorFieldType knnVectorFieldType = (KNNVectorFieldType) mappedFieldType;
        KNNMappingConfig knnMappingConfig = knnVectorFieldType.getKnnMappingConfig();
        QueryConfigFromMapping queryConfigFromMapping = getQueryConfig(knnMappingConfig, knnVectorFieldType);

        KNNEngine knnEngine = queryConfigFromMapping.getKnnEngine();
        MethodComponentContext methodComponentContext = queryConfigFromMapping.getMethodComponentContext();
        SpaceType spaceType = queryConfigFromMapping.getSpaceType();
        VectorDataType vectorDataType = queryConfigFromMapping.getVectorDataType();
        RescoreContext processedRescoreContext = knnVectorFieldType.resolveRescoreContext(rescoreContext);
        // Transform the query vector if it's required. It will return `vector` itself if transform is not needed.
        // Otherwise, it will return a new transformed vector.
        final float[] transformedQueryVector = knnVectorFieldType.transformQueryVector(vector);

        VectorQueryType vectorQueryType = getVectorQueryType(k, maxDistance, minScore);
        final String indexName = context.index().getName();
        final boolean memoryOptimizedSearchEnabled = MemoryOptimizedSearchSupportSpec.isSupportedFieldType(knnVectorFieldType, indexName);
        updateQueryStats(vectorQueryType);

        // This could be null in the case of when a model did not have serialized methodComponent information
        final String method = methodComponentContext != null ? methodComponentContext.getName() : null;
        if (method != null && !method.isBlank()) {
            final KNNLibrarySearchContext engineSpecificMethodContext = knnEngine.getKNNLibrarySearchContext(method);
            QueryContext queryContext = new QueryContext(vectorQueryType);
            ValidationException validationException = validateParameters(
                engineSpecificMethodContext.supportedMethodParameters(queryContext),
                (Map<String, Object>) methodParameters,
                KNNMethodConfigContext.EMPTY
            );
            if (validationException != null) {
                throw new IllegalArgumentException(
                    String.format(
                        Locale.ROOT,
                        "Parameters not valid for [%s]:[%s]:[%s] combination: [%s]",
                        knnEngine,
                        method,
                        vectorQueryType.getQueryTypeName(),
                        validationException.getMessage()
                    )
                );
            }
        }

        if (this.maxDistance != null || this.minScore != null) {
            if (!ENGINES_SUPPORTING_RADIAL_SEARCH.contains(knnEngine)) {
                throw new UnsupportedOperationException(
                    String.format(Locale.ROOT, "Engine [%s] does not support radial search", knnEngine)
                );
            }
            if (vectorDataType == VectorDataType.BINARY) {
                throw new UnsupportedOperationException(String.format(Locale.ROOT, "Binary data type does not support radial search"));
            }

            if (knnMappingConfig.getQuantizationConfig() != QuantizationConfig.EMPTY) {
                throw new UnsupportedOperationException("Radial search is not supported for indices which have quantization enabled");
            }
        }

        // Currently, k-NN supports distance and score types radial search
        // We need transform distance/score to right type of engine required radius.
        Float radius = null;
        if (this.maxDistance != null) {
            if (this.maxDistance < 0 && SpaceType.INNER_PRODUCT.equals(spaceType) == false) {
                throw new IllegalArgumentException(
                    String.format("[" + NAME + "] requires distance to be non-negative for space type: %s", spaceType)
                );
            }
            if (memoryOptimizedSearchEnabled) {
                radius = MemoryOptimizedSearchScoreConverter.distanceToRadialThreshold(this.maxDistance, spaceType);
            } else {
                radius = knnEngine.distanceToRadialThreshold(this.maxDistance, spaceType);
            }
        }

        if (this.minScore != null) {
            if (this.minScore > 1 && SpaceType.INNER_PRODUCT.equals(spaceType) == false) {
                throw new IllegalArgumentException(
                    String.format("[" + NAME + "] requires score to be in the range [0, 1] for space type: %s", spaceType)
                );
            }
            if (memoryOptimizedSearchEnabled) {
                radius = MemoryOptimizedSearchScoreConverter.scoreToRadialThreshold(this.minScore, spaceType);
            } else {
                radius = knnEngine.scoreToRadialThreshold(this.minScore, spaceType);
            }
        }

        final int vectorLength = VectorDataType.BINARY == vectorDataType ? vector.length * Byte.SIZE : vector.length;
        if (knnMappingConfig.getDimension() != vectorLength) {
            throw new IllegalArgumentException(
                String.format(
                    "Query vector has invalid dimension: %d. Dimension should be: %d",
                    vectorLength,
                    knnMappingConfig.getDimension()
                )
            );
        }

        byte[] byteVector = new byte[0];
        switch (vectorDataType) {
            case BINARY:
                byteVector = new byte[vector.length];
                for (int i = 0; i < vector.length; i++) {
                    validateByteVectorValue(vector[i], knnVectorFieldType.getVectorDataType());
                    byteVector[i] = (byte) vector[i];
                }
                spaceType.validateVector(byteVector);
                break;
            case BYTE:
                if (isUsingLuceneQuery(knnEngine, memoryOptimizedSearchEnabled)) {
                    byteVector = new byte[vector.length];
                    for (int i = 0; i < vector.length; i++) {
                        validateByteVectorValue(vector[i], knnVectorFieldType.getVectorDataType());
                        byteVector[i] = (byte) vector[i];
                    }
                    spaceType.validateVector(byteVector);
                } else {
                    for (float v : vector) {
                        validateByteVectorValue(v, knnVectorFieldType.getVectorDataType());
                    }
                    spaceType.validateVector(vector);
                }
                break;
            default:
                spaceType.validateVector(vector);
        }

        if (KNNEngine.getEnginesThatCreateCustomSegmentFiles().contains(knnEngine)
            && filter != null
            && !KNNEngine.getEnginesThatSupportsFilters().contains(knnEngine)) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Engine [%s] does not support filters", knnEngine));
        }

        if (k != null && k != 0) {
            // Requirement 9.3: When both oversampling and clumping are enabled, oversampling is applied
            // to the marker search phase. When only clumping is enabled, the clumping expansion factor
            // is applied to the marker search to ensure sufficient candidates after expansion.
            // 
            // The first-pass k for marker search is determined as follows:
            // 1. If oversampling (rescore) is enabled: oversampling handles the first-pass k
            // 2. If only clumping is enabled: apply clumping expansion factor to k
            // 3. If neither is enabled: use k as-is
            int markerSearchK = calculateMarkerSearchK(
                this.k,
                processedRescoreContext,
                clumpingContext,
                hasClumpingEnabled(knnMappingConfig)
            );

            KNNQueryFactory.CreateQueryRequest createQueryRequest = KNNQueryFactory.CreateQueryRequest.builder()
                .knnEngine(knnEngine)
                .indexName(indexName)
                .fieldName(this.fieldName)
                .vector(getFloatVectorForCreatingQueryRequest(transformedQueryVector, vectorDataType, knnEngine))
                .originalVector(vector)
                .byteVector(getByteVectorForCreatingQueryRequest(vectorDataType, knnEngine, byteVector, memoryOptimizedSearchEnabled))
                .vectorDataType(vectorDataType)
                .k(markerSearchK)
                .methodParameters(this.methodParameters)
                .filter(this.filter)
                .context(context)
                .rescoreContext(processedRescoreContext)
                .expandNested(expandNested == null ? false : expandNested)
                .memoryOptimizedSearchEnabled(memoryOptimizedSearchEnabled)
                .build();
            Query baseQuery = KNNQueryFactory.create(createQueryRequest);

            // Wrap with ClumpingKNNVectorQuery if clumping should be applied
            // Requirements: 5.2, 5.3, 5.4
            return wrapWithClumpingIfNeeded(baseQuery, knnMappingConfig, spaceType, context.getShardId(), context);
        }
        if (radius != null) {
            RNNQueryFactory.CreateQueryRequest createQueryRequest = RNNQueryFactory.CreateQueryRequest.builder()
                .knnEngine(knnEngine)
                .indexName(indexName)
                .fieldName(this.fieldName)
                .vector(getFloatVectorForCreatingQueryRequest(transformedQueryVector, vectorDataType, knnEngine))
                .originalVector(vector)
                .byteVector(getByteVectorForCreatingQueryRequest(vectorDataType, knnEngine, byteVector, memoryOptimizedSearchEnabled))
                .vectorDataType(vectorDataType)
                .radius(radius)
                .methodParameters(this.methodParameters)
                .filter(this.filter)
                .context(context)
                .memoryOptimizedSearchEnabled(memoryOptimizedSearchEnabled)
                .build();
            return RNNQueryFactory.create(createQueryRequest);
        }
        throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] requires k or distance or score to be set", NAME));
    }

    private QueryConfigFromMapping getQueryConfig(final KNNMappingConfig knnMappingConfig, final KNNVectorFieldType knnVectorFieldType) {

        if (knnMappingConfig.getKnnMethodContext().isPresent()) {
            KNNMethodContext knnMethodContext = knnMappingConfig.getKnnMethodContext().get();
            return new QueryConfigFromMapping(
                knnMethodContext.getKnnEngine(),
                knnMethodContext.getMethodComponentContext(),
                knnMethodContext.getSpaceType(),
                knnVectorFieldType.getVectorDataType()
            );
        }

        if (knnMappingConfig.getModelId().isPresent()) {
            ModelMetadata modelMetadata = getModelMetadataForField(knnMappingConfig.getModelId().get());
            return new QueryConfigFromMapping(
                modelMetadata.getKnnEngine(),
                modelMetadata.getMethodComponentContext(),
                modelMetadata.getSpaceType(),
                modelMetadata.getVectorDataType()
            );
        }

        throw new IllegalArgumentException(String.format(Locale.ROOT, "Field '%s' is not built for ANN search.", this.fieldName));
    }

    /**
     * Wraps the base query with ClumpingKNNVectorQuery if clumping should be applied.
     * 
     * Clumping is applied when:
     * 1. The query has clumping enabled (clumpingContext != null and enabled)
     * 2. The index has clumping enabled (has clumping configuration in mapping)
     * 
     * If the query enables clumping but the index doesn't support it, a warning is logged
     * and the base query is returned without clumping.
     * 
     * Requirements: 5.2, 5.3, 5.4
     *
     * @param baseQuery       The base k-NN query to potentially wrap
     * @param knnMappingConfig The mapping configuration for the k-NN field
     * @param spaceType       The space type for distance calculation
     * @param shardId         The shard ID (for logging purposes)
     * @param context         The query shard context (for filter conversion)
     * @return The wrapped query if clumping should be applied, otherwise the base query
     */
    private Query wrapWithClumpingIfNeeded(
        Query baseQuery,
        KNNMappingConfig knnMappingConfig,
        SpaceType spaceType,
        int shardId,
        QueryShardContext context
    ) {
        // Requirement 5.2: If clumping is explicitly disabled in the query, skip clumping
        if (clumpingContext != null && !clumpingContext.isEnabled()) {
            log.debug("Clumping explicitly disabled in query for field {}", fieldName);
            return baseQuery;
        }

        // Check if the index has clumping enabled
        boolean indexHasClumping = hasClumpingEnabled(knnMappingConfig);

        // Requirement 5.4: If query specifies clumping but index doesn't have it, log warning and proceed without clumping
        if (clumpingContext != null && clumpingContext.isEnabled() && !indexHasClumping) {
            log.warn(
                "Clumping requested in query for field {} but index does not have clumping enabled. Proceeding without clumping.",
                fieldName
            );
            return baseQuery;
        }

        // Requirement 5.3: If clumping is not specified but index has clumping, use default clumping behavior
        // This means we apply clumping when the index has it, unless explicitly disabled
        if (!indexHasClumping) {
            return baseQuery;
        }

        // Resolve the clumping context - use provided context or default
        ClumpingContext resolvedClumpingContext = clumpingContext != null ? clumpingContext : ClumpingContext.getDefault();

        log.debug(
            "Wrapping query with ClumpingKNNVectorQuery for field {}, k={}, expansionFactor={}",
            fieldName,
            k,
            resolvedClumpingContext.getExpansionFactor()
        );

        // Get the filter query to pass to ClumpingKNNVectorQuery
        // Requirement 9.1: Apply filter to both marker search and hidden vector expansion
        // The filter is already applied to the baseQuery (marker search), but we also need
        // to pass it to ClumpingKNNVectorQuery so it can filter hidden vectors during expansion
        Query filterQueryForClumping = null;
        if (filter != null) {
            try {
                filterQueryForClumping = filter.toQuery(context);
            } catch (IOException e) {
                log.warn("Failed to convert filter to query for clumping, proceeding without filter on hidden vectors", e);
            }
        }

        // Requirement 9.2: Get the parent filter for nested field support
        // The parent filter is used to maintain parent-child relationships in nested documents
        org.apache.lucene.search.join.BitSetProducer parentFilter = context.getParentFilter();

        return new ClumpingKNNVectorQuery(
            baseQuery,
            fieldName,
            k,
            vector,
            shardId,
            resolvedClumpingContext,
            spaceType,
            filterQueryForClumping,
            parentFilter
        );
    }

    /**
     * Checks if the index has clumping enabled for this field.
     * 
     * Clumping is enabled if the field mapping contains a clumping factor configuration.
     *
     * @param knnMappingConfig The mapping configuration for the k-NN field
     * @return true if clumping is enabled for this field, false otherwise
     */
    private boolean hasClumpingEnabled(KNNMappingConfig knnMappingConfig) {
        // Check if the mapping has clumping factor configured
        // The clumping factor is stored in OriginalMappingParameters during mapping creation
        // and is written to field attributes during indexing
        // At query time, we check if the mapping configuration indicates clumping is enabled
        return knnMappingConfig.getClumpingFactor().isPresent();
    }

    /**
     * Calculates the k value to use for marker search when clumping is enabled.
     * 
     * Requirement 9.3: When both oversampling and clumping are enabled, oversampling is applied
     * to the marker search phase before clumping expansion. When only clumping is enabled,
     * the clumping expansion factor is applied to ensure sufficient candidates after expansion.
     * 
     * The logic is:
     * 1. If oversampling (rescore) is enabled: return the original k, as oversampling will handle
     *    the first-pass k calculation internally (via RescoreKNNVectorQuery or NativeEngineKnnVectorQuery)
     * 2. If only clumping is enabled (no oversampling): apply clumping expansion factor to k
     * 3. If neither is enabled: return the original k
     *
     * @param finalK              The final number of results desired
     * @param rescoreContext      The rescore context (may be null)
     * @param clumpingContext     The clumping context (may be null)
     * @param indexHasClumping    Whether the index has clumping enabled
     * @return The k value to use for marker search
     */
    private int calculateMarkerSearchK(
        int finalK,
        RescoreContext rescoreContext,
        ClumpingContext clumpingContext,
        boolean indexHasClumping
    ) {
        // Check if oversampling (rescore) is enabled
        boolean rescoreEnabled = rescoreContext != null && rescoreContext.isRescoreEnabled();

        // Check if clumping is enabled (either from query or index default)
        boolean clumpingEnabled = indexHasClumping && (clumpingContext == null || clumpingContext.isEnabled());

        // If oversampling is enabled, it will handle the first-pass k internally
        // The RescoreKNNVectorQuery or NativeEngineKnnVectorQuery will apply the oversample factor
        if (rescoreEnabled) {
            log.debug(
                "Oversampling is enabled, using original k={} for marker search (oversampling will be applied internally)",
                finalK
            );
            return finalK;
        }

        // If only clumping is enabled (no oversampling), apply clumping expansion factor
        // Requirement 6.2: Request k * expansion_factor results to ensure sufficient candidates
        if (clumpingEnabled) {
            ClumpingContext resolvedClumpingContext = clumpingContext != null ? clumpingContext : ClumpingContext.getDefault();
            int markerSearchK = resolvedClumpingContext.getFirstPassK(finalK);
            // Cap at K_MAX to avoid excessive results
            markerSearchK = Math.min(markerSearchK, K_MAX);
            log.debug(
                "Clumping enabled without oversampling, using markerSearchK={} (finalK={}, expansionFactor={})",
                markerSearchK,
                finalK,
                resolvedClumpingContext.getExpansionFactor()
            );
            return markerSearchK;
        }

        // Neither oversampling nor clumping is enabled, use original k
        return finalK;
    }

    /**
     * Determine whether the query will be using Lucene query to perform vector search.
     * Currently, if memory optimized search is enabled, it fallbacks to Lucene and delegate its HNSW graph searcher to perform ANN search
     * on FAISS index. Hence, if it is true, then we need to use Lucene query.
     *
     * @param engine Engine type
     * @param memoryOptimizedSearchEnabled A bool flag whether memory optimized search is enabled.
     * @return True when it should use Lucene query False otherwise.
     */
    private static boolean isUsingLuceneQuery(final KNNEngine engine, final boolean memoryOptimizedSearchEnabled) {
        return memoryOptimizedSearchEnabled || engine == KNNEngine.LUCENE;
    }

    private ModelMetadata getModelMetadataForField(String modelId) {
        ModelMetadata modelMetadata = modelDao.getMetadata(modelId);
        if (!ModelUtil.isModelCreated(modelMetadata)) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Model ID '%s' is not created.", modelId));
        }
        return modelMetadata;
    }

    /**
     * Function to get the vector query type based on the valid query parameter.
     *
     * @param k K nearest neighbours for the given vector, if k is set, then the query type is K
     * @param maxDistance Maximum distance for the given vector, if maxDistance is set, then the query type is MAX_DISTANCE
     * @param minScore Minimum score for the given vector, if minScore is set, then the query type is MIN_SCORE
     */
    private VectorQueryType getVectorQueryType(Integer k, Float maxDistance, Float minScore) {
        if (maxDistance != null) {
            return VectorQueryType.MAX_DISTANCE;
        }
        if (minScore != null) {
            return VectorQueryType.MIN_SCORE;
        }
        if (k != null && k != 0) {
            return VectorQueryType.K;
        }
        throw new IllegalArgumentException(String.format(Locale.ROOT, "[%s] requires exactly one of k, distance or score to be set", NAME));
    }

    /**
     * Function to update query stats.
     *
     * @param vectorQueryType The type of query to be executed
     */
    private void updateQueryStats(VectorQueryType vectorQueryType) {
        vectorQueryType.getQueryStatCounter().increment();
        if (filter != null) {
            vectorQueryType.getQueryWithFilterStatCounter().increment();
        }
    }

    private float[] getFloatVectorForCreatingQueryRequest(
        final float[] transformedVector,
        VectorDataType vectorDataType,
        KNNEngine knnEngine
    ) {

        if ((VectorDataType.FLOAT == vectorDataType) || (VectorDataType.BYTE == vectorDataType && KNNEngine.FAISS == knnEngine)) {
            return transformedVector;
        }
        return null;
    }

    private byte[] getByteVectorForCreatingQueryRequest(
        VectorDataType vectorDataType,
        KNNEngine knnEngine,
        byte[] byteVector,
        boolean memoryOptimizedSearchEnabled
    ) {

        if (VectorDataType.BINARY == vectorDataType
            || (VectorDataType.BYTE == vectorDataType && isUsingLuceneQuery(knnEngine, memoryOptimizedSearchEnabled))) {
            return byteVector;
        }
        return null;
    }

    @Override
    protected boolean doEquals(KNNQueryBuilder other) {
        return Objects.equals(fieldName, other.fieldName)
            && Arrays.equals(vector, other.vector)
            && Objects.equals(k, other.k)
            && Objects.equals(minScore, other.minScore)
            && Objects.equals(maxDistance, other.maxDistance)
            && Objects.equals(methodParameters, other.methodParameters)
            && Objects.equals(filter, other.filter)
            && Objects.equals(ignoreUnmapped, other.ignoreUnmapped)
            && Objects.equals(rescoreContext, other.rescoreContext)
            && Objects.equals(clumpingContext, other.clumpingContext)
            && Objects.equals(expandNested, other.expandNested);
    }

    @Override
    protected int doHashCode() {
        return Objects.hash(
            fieldName,
            Arrays.hashCode(vector),
            k,
            methodParameters,
            filter,
            ignoreUnmapped,
            maxDistance,
            minScore,
            rescoreContext,
            clumpingContext,
            expandNested
        );
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }

    @Override
    protected QueryBuilder doRewrite(QueryRewriteContext queryShardContext) throws IOException {
        QueryBuilder rewrittenFilter;
        if (Objects.nonNull(filter)) {
            rewrittenFilter = filter.rewrite(queryShardContext);
            if (rewrittenFilter != filter) {
                KNNQueryBuilder rewrittenQueryBuilder = KNNQueryBuilder.builder()
                    .fieldName(this.fieldName)
                    .vector(this.vector)
                    .k(this.k)
                    .maxDistance(this.maxDistance)
                    .minScore(this.minScore)
                    .methodParameters(this.methodParameters)
                    .filter(rewrittenFilter)
                    .ignoreUnmapped(this.ignoreUnmapped)
                    .rescoreContext(this.rescoreContext)
                    .clumpingContext(this.clumpingContext)
                    .expandNested(this.expandNested)
                    .build();
                return rewrittenQueryBuilder;
            }
        }
        return super.doRewrite(queryShardContext);
    }

    @Getter
    @AllArgsConstructor
    private static class QueryConfigFromMapping {
        private final KNNEngine knnEngine;
        private final MethodComponentContext methodComponentContext;
        private final SpaceType spaceType;
        private final VectorDataType vectorDataType;
    }
}
