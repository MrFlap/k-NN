/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.parser;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.opensearch.common.ValidationException;
import org.opensearch.core.ParseField;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.ObjectParser;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.knn.index.query.clumping.ClumpingContext;
import org.opensearch.knn.index.util.IndexUtil;

import java.io.IOException;
import java.util.Locale;

/**
 * Parser for clumping parameters in k-NN queries.
 * Clumping is an optimization technique that reduces the size of the main k-NN index
 * by only indexing a subset of vectors (marker vectors) while storing the remaining
 * vectors (hidden vectors) in a separate file on disk.
 * 
 * This parser follows the RescoreParser pattern for consistency.
 */
@Getter
@AllArgsConstructor
@Log4j2
public final class ClumpingParser {

    public static final String CLUMPING_PARAMETER = "clumping";
    public static final String CLUMPING_ENABLED_PARAMETER = "enabled";
    public static final String CLUMPING_EXPANSION_FACTOR_PARAMETER = "expansion_factor";

    public static final ParseField CLUMPING_ENABLED_FIELD = new ParseField(CLUMPING_ENABLED_PARAMETER);
    public static final ParseField CLUMPING_EXPANSION_FACTOR_FIELD = new ParseField(CLUMPING_EXPANSION_FACTOR_PARAMETER);

    private static final ObjectParser<ClumpingContext.ClumpingContextBuilder, Void> INTERNAL_PARSER = createInternalObjectParser();

    private static ObjectParser<ClumpingContext.ClumpingContextBuilder, Void> createInternalObjectParser() {
        ObjectParser<ClumpingContext.ClumpingContextBuilder, Void> internalParser = new ObjectParser<>(
            CLUMPING_PARAMETER,
            ClumpingContext::builder
        );
        internalParser.declareBoolean(ClumpingContext.ClumpingContextBuilder::enabled, CLUMPING_ENABLED_FIELD);
        internalParser.declareFloat(ClumpingContext.ClumpingContextBuilder::expansionFactor, CLUMPING_EXPANSION_FACTOR_FIELD);
        return internalParser;
    }

    /**
     * Validate the clumping context.
     *
     * @param clumpingContext The clumping context to validate
     * @return ValidationException if validation fails, null otherwise
     */
    public static ValidationException validate(ClumpingContext clumpingContext) {
        if (clumpingContext == null) {
            return null;
        }

        // Validate expansion factor is positive
        if (clumpingContext.getExpansionFactor() <= 0) {
            ValidationException validationException = new ValidationException();
            validationException.addValidationError(
                String.format(
                    Locale.ROOT,
                    "Expansion factor [%f] must be greater than 0",
                    clumpingContext.getExpansionFactor()
                )
            );
            return validationException;
        }

        return null;
    }

    /**
     * Read a ClumpingContext from a stream input.
     *
     * @param in stream input
     * @return ClumpingContext or null if not present
     * @throws IOException on stream failure
     */
    public static ClumpingContext streamInput(StreamInput in) throws IOException {
        if (!IndexUtil.isVersionOnOrAfterMinRequiredVersion(in.getVersion(), CLUMPING_PARAMETER)) {
            return null;
        }
        Boolean enabled = in.readOptionalBoolean();
        if (enabled == null) {
            return null;
        }
        Float expansionFactor = in.readOptionalFloat();
        return ClumpingContext.builder()
            .enabled(enabled)
            .expansionFactor(expansionFactor != null ? expansionFactor : ClumpingContext.DEFAULT_EXPANSION_FACTOR)
            .build();
    }

    /**
     * Write a ClumpingContext to a stream output.
     *
     * @param out stream output
     * @param clumpingContext ClumpingContext to write
     * @throws IOException on stream failure
     */
    public static void streamOutput(StreamOutput out, ClumpingContext clumpingContext) throws IOException {
        if (!IndexUtil.isVersionOnOrAfterMinRequiredVersion(out.getVersion(), CLUMPING_PARAMETER)) {
            return;
        }
        if (clumpingContext == null) {
            out.writeOptionalBoolean(null);
            out.writeOptionalFloat(null);
        } else {
            out.writeOptionalBoolean(clumpingContext.isEnabled());
            out.writeOptionalFloat(clumpingContext.getExpansionFactor());
        }
    }

    /**
     * Write a ClumpingContext to XContent.
     *
     * @param builder XContentBuilder
     * @param clumpingContext ClumpingContext to write
     * @throws IOException on XContent failure
     */
    public static void doXContent(final XContentBuilder builder, final ClumpingContext clumpingContext) throws IOException {
        builder.startObject(CLUMPING_PARAMETER);
        builder.field(CLUMPING_ENABLED_PARAMETER, clumpingContext.isEnabled());
        builder.field(CLUMPING_EXPANSION_FACTOR_PARAMETER, clumpingContext.getExpansionFactor());
        builder.endObject();
    }

    /**
     * Parse a ClumpingContext from XContent.
     *
     * @param parser input parser
     * @return ClumpingContext
     */
    public static ClumpingContext fromXContent(final XContentParser parser) {
        return INTERNAL_PARSER.apply(parser, null).build();
    }
}
