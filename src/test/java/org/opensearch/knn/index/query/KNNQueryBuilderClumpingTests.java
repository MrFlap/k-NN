/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import lombok.SneakyThrows;
import org.opensearch.cluster.ClusterModule;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.core.common.io.stream.NamedWriteableAwareStreamInput;
import org.opensearch.core.common.io.stream.NamedWriteableRegistry;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.TermQueryBuilder;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.query.clumping.ClumpingContext;

import java.io.IOException;
import java.util.List;

/**
 * Unit tests for {@link KNNQueryBuilder} clumping integration.
 * 
 * Validates: Requirements 5.1, 5.2, 11.3, 11.5
 */
public class KNNQueryBuilderClumpingTests extends KNNTestCase {

    private static final String FIELD_NAME = "myvector";
    private static final int K = 10;
    private static final float[] QUERY_VECTOR = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };

    /**
     * Test that KNNQueryBuilder can be created with clumping context.
     * Validates: Requirement 5.1
     */
    public void testBuilderWithClumpingContext() {
        ClumpingContext clumpingContext = ClumpingContext.builder()
            .clumpingFactor(8)
            .expansionFactor(2.0f)
            .enabled(true)
            .build();

        KNNQueryBuilder queryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .k(K)
            .clumpingContext(clumpingContext)
            .build();

        assertNotNull(queryBuilder);
        assertEquals(FIELD_NAME, queryBuilder.fieldName());
        assertEquals(K, queryBuilder.getK());
        assertArrayEquals(QUERY_VECTOR, queryBuilder.vector(), 0.0001f);
        assertEquals(clumpingContext, queryBuilder.getClumpingContext());
    }

    /**
     * Test that KNNQueryBuilder can be created without clumping context.
     * Validates: Requirement 5.2 (disabled clumping behavior)
     */
    public void testBuilderWithoutClumpingContext() {
        KNNQueryBuilder queryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .k(K)
            .build();

        assertNotNull(queryBuilder);
        assertNull(queryBuilder.getClumpingContext());
    }

    /**
     * Test that clumping context with enabled=false is handled correctly.
     * Validates: Requirement 5.2 (disabled clumping behavior)
     */
    public void testBuilderWithDisabledClumpingContext() {
        ClumpingContext disabledContext = ClumpingContext.builder()
            .clumpingFactor(8)
            .expansionFactor(2.0f)
            .enabled(false)
            .build();

        KNNQueryBuilder queryBuilder = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .k(K)
            .clumpingContext(disabledContext)
            .build();

        assertNotNull(queryBuilder);
        assertEquals(disabledContext, queryBuilder.getClumpingContext());
        assertFalse(queryBuilder.getClumpingContext().isEnabled());
    }

    /**
     * Test stream serialization round-trip with clumping context.
     * Validates: Requirement 5.1 (query parsing correctness)
     */
    @SneakyThrows
    public void testStreamSerializationWithClumpingContext() {
        ClumpingContext clumpingContext = ClumpingContext.builder()
            .clumpingFactor(16)
            .expansionFactor(3.0f)
            .enabled(true)
            .build();

        KNNQueryBuilder original = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .k(K)
            .clumpingContext(clumpingContext)
            .build();

        KNNQueryBuilder deserialized = roundTripSerialization(original);

        assertEquals(original.fieldName(), deserialized.fieldName());
        assertEquals(original.getK(), deserialized.getK());
        assertArrayEquals(original.vector(), deserialized.vector(), 0.0001f);
        assertEquals(original.getClumpingContext(), deserialized.getClumpingContext());
    }

    /**
     * Test stream serialization round-trip without clumping context.
     */
    @SneakyThrows
    public void testStreamSerializationWithoutClumpingContext() {
        KNNQueryBuilder original = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .k(K)
            .build();

        KNNQueryBuilder deserialized = roundTripSerialization(original);

        assertEquals(original.fieldName(), deserialized.fieldName());
        assertEquals(original.getK(), deserialized.getK());
        assertArrayEquals(original.vector(), deserialized.vector(), 0.0001f);
        assertNull(deserialized.getClumpingContext());
    }

    /**
     * Test equals and hashCode with clumping context.
     */
    public void testEqualsAndHashCodeWithClumpingContext() {
        ClumpingContext context1 = ClumpingContext.builder()
            .clumpingFactor(8)
            .expansionFactor(2.0f)
            .enabled(true)
            .build();

        ClumpingContext context2 = ClumpingContext.builder()
            .clumpingFactor(8)
            .expansionFactor(2.0f)
            .enabled(true)
            .build();

        ClumpingContext differentContext = ClumpingContext.builder()
            .clumpingFactor(16)
            .expansionFactor(3.0f)
            .enabled(true)
            .build();

        KNNQueryBuilder query1 = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .k(K)
            .clumpingContext(context1)
            .build();

        KNNQueryBuilder query2 = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .k(K)
            .clumpingContext(context2)
            .build();

        KNNQueryBuilder query3 = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .k(K)
            .clumpingContext(differentContext)
            .build();

        // Same clumping context should be equal
        assertEquals(query1, query2);
        assertEquals(query1.hashCode(), query2.hashCode());

        // Different clumping context should not be equal
        assertNotEquals(query1, query3);
    }

    /**
     * Test equals and hashCode with null vs non-null clumping context.
     */
    public void testEqualsWithNullVsNonNullClumpingContext() {
        ClumpingContext context = ClumpingContext.getDefault();

        KNNQueryBuilder queryWithClumping = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .k(K)
            .clumpingContext(context)
            .build();

        KNNQueryBuilder queryWithoutClumping = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .k(K)
            .build();

        assertNotEquals(queryWithClumping, queryWithoutClumping);
    }

    /**
     * Test that different clumping factors produce different queries.
     */
    public void testDifferentClumpingFactorsProduceDifferentQueries() {
        ClumpingContext context8 = ClumpingContext.builder()
            .clumpingFactor(8)
            .build();

        ClumpingContext context16 = ClumpingContext.builder()
            .clumpingFactor(16)
            .build();

        KNNQueryBuilder query8 = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .k(K)
            .clumpingContext(context8)
            .build();

        KNNQueryBuilder query16 = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .k(K)
            .clumpingContext(context16)
            .build();

        assertNotEquals(query8, query16);
    }

    /**
     * Test that different expansion factors produce different queries.
     */
    public void testDifferentExpansionFactorsProduceDifferentQueries() {
        ClumpingContext context2x = ClumpingContext.builder()
            .clumpingFactor(8)
            .expansionFactor(2.0f)
            .build();

        ClumpingContext context3x = ClumpingContext.builder()
            .clumpingFactor(8)
            .expansionFactor(3.0f)
            .build();

        KNNQueryBuilder query2x = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .k(K)
            .clumpingContext(context2x)
            .build();

        KNNQueryBuilder query3x = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .k(K)
            .clumpingContext(context3x)
            .build();

        assertNotEquals(query2x, query3x);
    }

    /**
     * Test clumping context with various valid configurations.
     */
    public void testClumpingContextWithVariousConfigurations() {
        // Minimum clumping factor
        ClumpingContext minContext = ClumpingContext.builder()
            .clumpingFactor(ClumpingContext.MIN_CLUMPING_FACTOR)
            .build();
        
        KNNQueryBuilder minQuery = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .k(K)
            .clumpingContext(minContext)
            .build();
        assertNotNull(minQuery);
        assertEquals(ClumpingContext.MIN_CLUMPING_FACTOR, minQuery.getClumpingContext().getClumpingFactor());

        // Maximum clumping factor
        ClumpingContext maxContext = ClumpingContext.builder()
            .clumpingFactor(ClumpingContext.MAX_CLUMPING_FACTOR)
            .build();
        
        KNNQueryBuilder maxQuery = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(QUERY_VECTOR)
            .k(K)
            .clumpingContext(maxContext)
            .build();
        assertNotNull(maxQuery);
        assertEquals(ClumpingContext.MAX_CLUMPING_FACTOR, maxQuery.getClumpingContext().getClumpingFactor());
    }

    // Helper methods

    @Override
    protected NamedWriteableRegistry writableRegistry() {
        final List<NamedWriteableRegistry.Entry> entries = ClusterModule.getNamedWriteables();
        entries.add(new NamedWriteableRegistry.Entry(QueryBuilder.class, KNNQueryBuilder.NAME, KNNQueryBuilder::new));
        entries.add(new NamedWriteableRegistry.Entry(QueryBuilder.class, TermQueryBuilder.NAME, TermQueryBuilder::new));
        return new NamedWriteableRegistry(entries);
    }

    private KNNQueryBuilder roundTripSerialization(KNNQueryBuilder original) throws IOException {
        try (BytesStreamOutput output = new BytesStreamOutput()) {
            original.writeTo(output);

            try (StreamInput in = new NamedWriteableAwareStreamInput(output.bytes().streamInput(), writableRegistry())) {
                return new KNNQueryBuilder(in);
            }
        }
    }
}
