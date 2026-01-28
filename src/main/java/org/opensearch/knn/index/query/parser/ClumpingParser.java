/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.parser;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.opensearch.common.ValidationException;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.knn.index.query.clumping.ClumpingContext;

import java.io.IOException;
import java.util.Locale;

/**
 * Parser for clumping parameters in k-NN queries.
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class ClumpingParser {

    public static final String CLUMPING_PARAMETER = "clumping";
    public static final String CLUMPING_FACTOR_PARAMETER = "clumping_factor";
    public static final String CLUMPING_ENABLED_PARAMETER = "enabled";

    /**
     * Parses clumping context from XContent.
     * 
     * Expected format:
     * {
     *   "clumping": {
     *     "enabled": true,
     *     "clumping_factor": 8
     *   }
     * }
     * 
     * Or simplified:
     * {
     *   "clumping_factor": 8
     * }
     * 
     * @param parser the XContent parser
     * @return ClumpingContext or null if not specified
     * @throws IOException if parsing fails
     */
    public static ClumpingContext parseClumping(XContentParser parser) throws IOException {
        XContentParser.Token token = parser.currentToken();
        
        if (token == XContentParser.Token.VALUE_NUMBER) {
            // Simple format: just the clumping factor
            int factor = parser.intValue();
            return ClumpingContext.withFactor(factor);
        }
        
        if (token != XContentParser.Token.START_OBJECT) {
            throw new IllegalArgumentException(
                String.format(Locale.ROOT, "[%s] must be an object or integer", CLUMPING_PARAMETER)
            );
        }

        boolean enabled = true;
        int clumpingFactor = ClumpingContext.DEFAULT_CLUMPING_FACTOR;

        while ((token = parser.nextToken()) != XContentParser.Token.END_OBJECT) {
            if (token == XContentParser.Token.FIELD_NAME) {
                String fieldName = parser.currentName();
                parser.nextToken();

                if (CLUMPING_FACTOR_PARAMETER.equals(fieldName)) {
                    clumpingFactor = parser.intValue();
                } else if (CLUMPING_ENABLED_PARAMETER.equals(fieldName)) {
                    enabled = parser.booleanValue();
                } else {
                    throw new IllegalArgumentException(
                        String.format(Locale.ROOT, "Unknown field [%s] in clumping", fieldName)
                    );
                }
            }
        }

        if (!enabled) {
            return ClumpingContext.DISABLED;
        }

        return ClumpingContext.withFactor(clumpingFactor);
    }

    /**
     * Validates clumping context.
     * 
     * @param clumpingContext the context to validate
     * @return ValidationException if invalid, null if valid
     */
    public static ValidationException validate(ClumpingContext clumpingContext) {
        if (clumpingContext == null) {
            return null;
        }

        ValidationException exception = null;

        int factor = clumpingContext.getClumpingFactor();
        if (factor < ClumpingContext.MIN_CLUMPING_FACTOR || factor > ClumpingContext.MAX_CLUMPING_FACTOR) {
            exception = new ValidationException();
            exception.addValidationError(
                String.format(
                    Locale.ROOT,
                    "clumping_factor must be between %d and %d, got %d",
                    ClumpingContext.MIN_CLUMPING_FACTOR,
                    ClumpingContext.MAX_CLUMPING_FACTOR,
                    factor
                )
            );
        }

        return exception;
    }
}
