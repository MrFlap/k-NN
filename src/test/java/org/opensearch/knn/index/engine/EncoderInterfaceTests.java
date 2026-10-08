/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.engine.faiss.FaissFlatEncoder;
import org.opensearch.knn.index.engine.faiss.FaissSQEncoder;
import org.opensearch.knn.index.engine.faiss.QFrameBitEncoder;
import org.opensearch.knn.index.engine.lucene.LuceneSQEncoder;
import org.opensearch.knn.index.mapper.CompressionLevel;

public class EncoderInterfaceTests extends KNNTestCase {

    public void testQuantizationBits_fromCompressionLevel() {
        // Bijective mapping: each CompressionLevel round-trips to its matching QuantizationBits.
        assertEquals(Encoder.QuantizationBits.ONE, Encoder.QuantizationBits.fromCompressionLevel(CompressionLevel.x32));
        assertEquals(Encoder.QuantizationBits.TWO, Encoder.QuantizationBits.fromCompressionLevel(CompressionLevel.x16));
        assertEquals(Encoder.QuantizationBits.FOUR, Encoder.QuantizationBits.fromCompressionLevel(CompressionLevel.x8));
        assertEquals(Encoder.QuantizationBits.SEVEN, Encoder.QuantizationBits.fromCompressionLevel(CompressionLevel.x4));
        assertEquals(Encoder.QuantizationBits.SIXTEEN, Encoder.QuantizationBits.fromCompressionLevel(CompressionLevel.x2));
        assertEquals(Encoder.QuantizationBits.FULL_PRECISION, Encoder.QuantizationBits.fromCompressionLevel(CompressionLevel.x1));
        // NOT_CONFIGURED and unmapped values fall back to FULL_PRECISION (matches fromValue's fallback).
        assertEquals(
            Encoder.QuantizationBits.FULL_PRECISION,
            Encoder.QuantizationBits.fromCompressionLevel(CompressionLevel.NOT_CONFIGURED)
        );
    }

    public void testQuantizationBits_fromCompressionLevel_isInverseOfGetCompressionLevel() {
        // For every enum value, fromCompressionLevel(getCompressionLevel(bits)) must return bits. EIGHT (int8) shares x4 with
        // the legacy SEVEN, and fromCompressionLevel resolves x4 to SEVEN by design: int8 is always picked explicitly.
        for (Encoder.QuantizationBits bits : Encoder.QuantizationBits.values()) {
            if (bits == Encoder.QuantizationBits.EIGHT) {
                continue;
            }
            assertEquals("round-trip failed for " + bits, bits, Encoder.QuantizationBits.fromCompressionLevel(bits.getCompressionLevel()));
        }
        assertEquals(CompressionLevel.x4, Encoder.QuantizationBits.EIGHT.getCompressionLevel());
        assertEquals(Encoder.QuantizationBits.SEVEN, Encoder.QuantizationBits.fromCompressionLevel(CompressionLevel.x4));
    }

    public void testFaissFlatEncoderType() {
        FaissFlatEncoder encoder = new FaissFlatEncoder();
        assertEquals(Encoder.EncoderType.FLAT, encoder.getEncoderType());
        assertTrue(encoder.getSupportedBits().contains(Encoder.QuantizationBits.FULL_PRECISION));
        assertEquals(1, encoder.getSupportedBits().size());
    }

    public void testFaissSQEncoderType() {
        FaissSQEncoder encoder = new FaissSQEncoder();
        assertEquals(Encoder.EncoderType.SQ, encoder.getEncoderType());
        // Multi-bit MOS (bits ∈ {1, 2, 4, 8}) and legacy fp16 (bits=16) are supported.
        assertTrue(encoder.getSupportedBits().contains(Encoder.QuantizationBits.ONE));
        assertTrue(encoder.getSupportedBits().contains(Encoder.QuantizationBits.TWO));
        assertTrue(encoder.getSupportedBits().contains(Encoder.QuantizationBits.FOUR));
        assertTrue(encoder.getSupportedBits().contains(Encoder.QuantizationBits.EIGHT));
        assertTrue(encoder.getSupportedBits().contains(Encoder.QuantizationBits.SIXTEEN));
        assertEquals(5, encoder.getSupportedBits().size());
    }

    public void testLuceneSQEncoderType() {
        LuceneSQEncoder encoder = new LuceneSQEncoder();
        assertEquals(Encoder.EncoderType.SQ, encoder.getEncoderType());
        assertTrue(encoder.getSupportedBits().contains(Encoder.QuantizationBits.ONE));
        assertTrue(encoder.getSupportedBits().contains(Encoder.QuantizationBits.TWO));
        assertTrue(encoder.getSupportedBits().contains(Encoder.QuantizationBits.FOUR));
        assertTrue(encoder.getSupportedBits().contains(Encoder.QuantizationBits.SEVEN));
        assertTrue(encoder.getSupportedBits().contains(Encoder.QuantizationBits.EIGHT));
        assertEquals(5, encoder.getSupportedBits().size());
    }

    public void testQFrameBitEncoderType() {
        QFrameBitEncoder encoder = new QFrameBitEncoder();
        assertEquals(Encoder.EncoderType.BQ, encoder.getEncoderType());
        assertTrue(encoder.getSupportedBits().contains(Encoder.QuantizationBits.ONE));
        assertTrue(encoder.getSupportedBits().contains(Encoder.QuantizationBits.TWO));
        assertTrue(encoder.getSupportedBits().contains(Encoder.QuantizationBits.FOUR));
        assertEquals(3, encoder.getSupportedBits().size());
    }

    public void testFaissHNSWPQEncoderType() {
        Encoder encoder = new org.opensearch.knn.index.engine.faiss.FaissHNSWPQEncoder();
        assertEquals(Encoder.EncoderType.PQ, encoder.getEncoderType());
        assertEquals(Encoder.QuantizationBits.FULL_PRECISION, encoder.getQuantizationBits());
        assertTrue(encoder.getSupportedBits().isEmpty());
    }
}
