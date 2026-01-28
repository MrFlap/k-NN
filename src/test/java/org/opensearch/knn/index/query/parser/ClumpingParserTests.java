/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.parser;

import lombok.SneakyThrows;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.common.io.stream.NamedWriteableAwareStreamInput;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.query.clumping.ClumpingContext;

import java.io.IOException;

import static org.opensearch.knn.index.query.parser.ClumpingParser.CLUMPING_ENABLED_PARAMETER;
import static org.opensearch.knn.index.query.parser.ClumpingParser.CLUMPING_EXPANSION_FACTOR_PARAMETER;
import static org.opensearch.knn.index.query.parser.ClumpingParser.CLUMPING_PARAMETER;

/**
 * Unit tests for {@link ClumpingParser}.
 * 
 * Validates: Requirement 5.1
 */
public class ClumpingParserTests extends KNNTestCase {

    @SneakyThrows
    public void testStreams() {
        // Test with default values
        ClumpingContext clumpingContext = ClumpingContext.builder()
            .enabled(true)
            .expansionFactor(ClumpingContext.DEFAULT_EXPANSION_FACTOR)
            .build();
        validateStreams(clumpingContext);
        
        // Test with custom values
        ClumpingContext customContext = ClumpingContext.builder()
            .enabled(false)
            .expansionFactor(3.5f)
            .build();
        validateStreams(customContext);
        
        // Test with null
        validateStreams(null);
    }

    private void validateStreams(ClumpingContext clumpingContext) throws IOException {
        try (BytesStreamOutput output = new BytesStreamOutput()) {
            ClumpingParser.streamOutput(output, clumpingContext);

            try (StreamInput in = new NamedWriteableAwareStreamInput(output.bytes().streamInput(), writableRegistry())) {
                ClumpingContext parsedClumpingContext = ClumpingParser.streamInput(in);
                assertEquals(clumpingContext, parsedClumpingContext);
            }
        }
    }

    @SneakyThrows
    public void testDoXContent() {
        float expansionFactor = 2.5f;
        boolean enabled = true;
        
        XContentBuilder expectedBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(CLUMPING_PARAMETER)
            .field(CLUMPING_ENABLED_PARAMETER, enabled)
            .field(CLUMPING_EXPANSION_FACTOR_PARAMETER, expansionFactor)
            .endObject()
            .endObject();

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject();
        ClumpingParser.doXContent(builder, ClumpingContext.builder()
            .enabled(enabled)
            .expansionFactor(expansionFactor)
            .build());
        builder.endObject();
        
        assertEquals(expectedBuilder.toString(), builder.toString());
    }

    @SneakyThrows
    public void testDoXContent_withDisabled() {
        float expansionFactor = 1.5f;
        boolean enabled = false;
        
        XContentBuilder expectedBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(CLUMPING_PARAMETER)
            .field(CLUMPING_ENABLED_PARAMETER, enabled)
            .field(CLUMPING_EXPANSION_FACTOR_PARAMETER, expansionFactor)
            .endObject()
            .endObject();

        XContentBuilder builder = XContentFactory.jsonBuilder().startObject();
        ClumpingParser.doXContent(builder, ClumpingContext.builder()
            .enabled(enabled)
            .expansionFactor(expansionFactor)
            .build());
        builder.endObject();
        
        assertEquals(expectedBuilder.toString(), builder.toString());
    }

    @SneakyThrows
    public void testFromXContent_whenValid_thenSucceed() {
        // Test with both parameters
        float expansionFactor = 3.0f;
        boolean enabled = true;
        XContentBuilder builder1 = XContentFactory.jsonBuilder()
            .startObject()
            .field(CLUMPING_ENABLED_PARAMETER, enabled)
            .field(CLUMPING_EXPANSION_FACTOR_PARAMETER, expansionFactor)
            .endObject();
        validateFromXContent(enabled, expansionFactor, builder1);
        
        // Test with only enabled parameter
        XContentBuilder builder2 = XContentFactory.jsonBuilder()
            .startObject()
            .field(CLUMPING_ENABLED_PARAMETER, false)
            .endObject();
        validateFromXContent(false, ClumpingContext.DEFAULT_EXPANSION_FACTOR, builder2);
        
        // Test with only expansion_factor parameter
        XContentBuilder builder3 = XContentFactory.jsonBuilder()
            .startObject()
            .field(CLUMPING_EXPANSION_FACTOR_PARAMETER, 4.0f)
            .endObject();
        validateFromXContent(ClumpingContext.builder().build().isEnabled(), 4.0f, builder3);
        
        // Test with empty object (defaults)
        XContentBuilder builder4 = XContentFactory.jsonBuilder().startObject().endObject();
        ClumpingContext defaultContext = ClumpingContext.builder().build();
        validateFromXContent(defaultContext.isEnabled(), defaultContext.getExpansionFactor(), builder4);
    }

    @SneakyThrows
    public void testFromXContent_whenInvalid_thenFail() {
        // Test with invalid parameter name
        XContentBuilder invalidParamBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field("invalid_param", 0)
            .endObject();
        expectValidationException(invalidParamBuilder);

        // Test with invalid enabled value type
        XContentBuilder invalidEnabledBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(CLUMPING_ENABLED_PARAMETER, "not_a_boolean")
            .endObject();
        expectValidationException(invalidEnabledBuilder);

        // Test with invalid expansion_factor value type
        XContentBuilder invalidExpansionBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(CLUMPING_EXPANSION_FACTOR_PARAMETER, "not_a_number")
            .endObject();
        expectValidationException(invalidExpansionBuilder);

        // Test with extra invalid parameter
        XContentBuilder extraParamBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .field(CLUMPING_ENABLED_PARAMETER, true)
            .field("extra_invalid", 123)
            .endObject();
        expectValidationException(extraParamBuilder);
    }

    @SneakyThrows
    public void testValidate_whenValid_thenNoException() {
        // Valid context with positive expansion factor
        ClumpingContext validContext = ClumpingContext.builder()
            .enabled(true)
            .expansionFactor(2.0f)
            .build();
        assertNull(ClumpingParser.validate(validContext));
        
        // Null context is valid (no clumping)
        assertNull(ClumpingParser.validate(null));
    }

    @SneakyThrows
    public void testValidate_whenInvalid_thenException() {
        // Invalid: zero expansion factor
        ClumpingContext zeroExpansion = ClumpingContext.builder()
            .enabled(true)
            .expansionFactor(0.0f)
            .build();
        assertNotNull(ClumpingParser.validate(zeroExpansion));
        
        // Invalid: negative expansion factor
        ClumpingContext negativeExpansion = ClumpingContext.builder()
            .enabled(true)
            .expansionFactor(-1.0f)
            .build();
        assertNotNull(ClumpingParser.validate(negativeExpansion));
    }

    private void validateFromXContent(boolean expectedEnabled, float expectedExpansionFactor, XContentBuilder builder) 
            throws IOException {
        XContentParser parser = createParser(builder);
        ClumpingContext clumpingContext = ClumpingParser.fromXContent(parser);
        assertEquals(expectedEnabled, clumpingContext.isEnabled());
        assertEquals(expectedExpansionFactor, clumpingContext.getExpansionFactor(), 0.0001f);
    }

    private void expectValidationException(XContentBuilder builder) throws IOException {
        XContentParser parser = createParser(builder);
        expectThrows(IllegalArgumentException.class, () -> ClumpingParser.fromXContent(parser));
    }
}
