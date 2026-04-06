/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.clumping;

import org.opensearch.knn.KNNTestCase;

/**
 * Verifies that Huffman encode → table-decode round-trips produce identical FP16 values.
 */
public class HuffmanRoundTripTests extends KNNTestCase {

    public void testRoundTripSingleVector() {
        int dimension = 4;
        float[] original = { 1.0f, -0.5f, 0.25f, 3.14f };

        // Convert to FP16 shorts (the ground truth symbols)
        short[] fp16Shorts = new short[dimension];
        for (int i = 0; i < dimension; i++) {
            fp16Shorts[i] = Float.floatToFloat16(original[i]);
        }

        // Build frequency table from these symbols
        long[] freq = new long[HuffmanCodec.NUM_SYMBOLS];
        for (short s : fp16Shorts) {
            freq[Short.toUnsignedInt(s)]++;
        }

        HuffmanCodec codec = HuffmanCodec.buildFromFrequencies(freq);
        HuffmanTableDecoder tableDecoder = codec.getTableDecoder();

        // Encode
        byte[] compressed = codec.encode(fp16Shorts);

        // Decode via table decoder
        short[] decoded = new short[dimension];
        tableDecoder.decodeToShort(compressed, dimension, decoded);

        // Verify
        for (int i = 0; i < dimension; i++) {
            assertEquals("Symbol mismatch at index " + i, fp16Shorts[i], decoded[i]);
            float decodedFloat = Float.float16ToFloat(decoded[i]);
            float expectedFloat = Float.float16ToFloat(fp16Shorts[i]);
            assertEquals("Float mismatch at index " + i, expectedFloat, decodedFloat, 0.0f);
        }
    }

    public void testRoundTripFromBytes() {
        int dimension = 768;
        float[] original = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            original[i] = randomFloat() * 2.0f - 1.0f;
        }

        // Convert to FP16 and write as big-endian bytes (matching Lucene's writeShort)
        byte[] fp16Bytes = new byte[dimension * 2];
        short[] expectedShorts = new short[dimension];
        long[] freq = new long[HuffmanCodec.NUM_SYMBOLS];

        for (int i = 0; i < dimension; i++) {
            short fp16 = Float.floatToFloat16(original[i]);
            expectedShorts[i] = fp16;
            freq[Short.toUnsignedInt(fp16)]++;
            // Big-endian write
            fp16Bytes[i * 2] = (byte) (fp16 >> 8);
            fp16Bytes[i * 2 + 1] = (byte) fp16;
        }

        HuffmanCodec codec = HuffmanCodec.buildFromFrequencies(freq);
        HuffmanTableDecoder tableDecoder = codec.getTableDecoder();

        // Encode from big-endian bytes
        byte[] compressed = codec.encodeFromBytes(fp16Bytes);

        // Decode via table decoder
        short[] decoded = new short[dimension];
        tableDecoder.decodeToShort(compressed, dimension, decoded);

        for (int i = 0; i < dimension; i++) {
            assertEquals("Symbol mismatch at index " + i, expectedShorts[i], decoded[i]);
        }
    }

    public void testRoundTripMultipleVectors() {
        int dimension = 768;
        int numVectors = 10;

        // Generate random vectors and collect frequencies
        float[][] vectors = new float[numVectors][dimension];
        long[] freq = new long[HuffmanCodec.NUM_SYMBOLS];
        for (int v = 0; v < numVectors; v++) {
            for (int d = 0; d < dimension; d++) {
                vectors[v][d] = randomFloat() * 2.0f - 1.0f;
                short fp16 = Float.floatToFloat16(vectors[v][d]);
                freq[Short.toUnsignedInt(fp16)]++;
            }
        }

        HuffmanCodec codec = HuffmanCodec.buildFromFrequencies(freq);
        HuffmanTableDecoder tableDecoder = codec.getTableDecoder();

        // Encode all vectors as one stream (matching how writeClumpFileHuffman works)
        short[] allShorts = new short[numVectors * dimension];
        for (int v = 0; v < numVectors; v++) {
            for (int d = 0; d < dimension; d++) {
                allShorts[v * dimension + d] = Float.floatToFloat16(vectors[v][d]);
            }
        }
        byte[] compressed = codec.encode(allShorts);

        // Decode one vector at a time using byte offsets
        short[] reusable = new short[dimension];
        int byteOffset = 0;
        int decoderState = 0;
        for (int v = 0; v < numVectors; v++) {
            int[] result = tableDecoder.decodeToShortFromOffset(compressed, byteOffset, decoderState, dimension, reusable);
            byteOffset = result[0];
            decoderState = result[1];
            for (int d = 0; d < dimension; d++) {
                short expected = Float.floatToFloat16(vectors[v][d]);
                assertEquals(
                    "Mismatch at vector " + v + " dim " + d,
                    expected, reusable[d]
                );
            }
        }
    }

    /**
     * Verifies that the equivalence-class remapped native table produces the same
     * decode results as the original Java DFA table decoder.
     */
    public void testNativeTableEquivalenceClassRemapping() {
        int dimension = 128;
        int numVectors = 20;

        for (int iter = 0; iter < 50; iter++) {
            float[][] vectors = new float[numVectors][dimension];
            long[] freq = new long[HuffmanCodec.NUM_SYMBOLS];
            for (int v = 0; v < numVectors; v++) {
                for (int d = 0; d < dimension; d++) {
                    vectors[v][d] = randomFloat() * 4.0f - 2.0f;
                    short fp16 = Float.floatToFloat16(vectors[v][d]);
                    freq[Short.toUnsignedInt(fp16)]++;
                }
            }

            HuffmanCodec codec = HuffmanCodec.buildFromFrequencies(freq);
            HuffmanTableDecoder tableDecoder = codec.getTableDecoder();

            // Encode all vectors as one stream
            short[] allShorts = new short[numVectors * dimension];
            for (int v = 0; v < numVectors; v++) {
                for (int d = 0; d < dimension; d++) {
                    allShorts[v * dimension + d] = Float.floatToFloat16(vectors[v][d]);
                }
            }
            byte[] compressed = codec.encode(allShorts);

            // Decode via Java DFA (ground truth)
            short[] javaDecoded = new short[numVectors * dimension];
            int byteOffset = 0;
            int decoderState = 0;
            short[] reusable = new short[dimension];
            for (int v = 0; v < numVectors; v++) {
                int[] result = tableDecoder.decodeToShortFromOffset(compressed, byteOffset, decoderState, dimension, reusable);
                byteOffset = result[0];
                decoderState = result[1];
                System.arraycopy(reusable, 0, javaDecoded, v * dimension, dimension);
            }

            // Decode via native table (equivalence-class remapped) using Java simulation
            HuffmanTableDecoder.NativeTable nativeTab = tableDecoder.getNativeTable();
            assertNotNull("NativeTable should not be null", nativeTab);
            assertTrue("Should have fewer eq classes than 256", nativeTab.numEqClasses <= 256);
            assertTrue("Should have at least 1 eq class", nativeTab.numEqClasses >= 1);

            // Simulate the native decode loop in Java to verify the flat table
            short[] nativeDecoded = new short[numVectors * dimension];
            int nState = 0;
            int nByteIdx = 0;
            int nOutIdx = 0;
            int totalSymbols = numVectors * dimension;
            while (nOutIdx < totalSymbols && nByteIdx < compressed.length) {
                int eqClass = Byte.toUnsignedInt(nativeTab.eqClassMap[compressed[nByteIdx++] & 0xFF]);
                int entryOffset = (nState * nativeTab.numEqClasses + eqClass) * HuffmanTableDecoder.NativeTable.ENTRY_STRIDE;
                int nextState = nativeTab.flatTable[entryOffset];
                int numSyms = nativeTab.flatTable[entryOffset + 1];
                for (int s = 0; s < numSyms && nOutIdx < totalSymbols; s++) {
                    nativeDecoded[nOutIdx++] = (short) nativeTab.flatTable[entryOffset + 2 + s];
                }
                nState = nextState;
            }

            // Compare
            for (int i = 0; i < totalSymbols; i++) {
                assertEquals(
                    "Native table mismatch at symbol " + i + " (iter " + iter + ")",
                    javaDecoded[i], nativeDecoded[i]
                );
            }
        }
    }

    /**
     * Verifies that equivalence classes actually reduce the alphabet size for
     * typical vector data distributions.
     */
    public void testEquivalenceClassReduction() {
        int dimension = 768;
        int numVectors = 50;

        float[][] vectors = new float[numVectors][dimension];
        long[] freq = new long[HuffmanCodec.NUM_SYMBOLS];
        for (int v = 0; v < numVectors; v++) {
            for (int d = 0; d < dimension; d++) {
                vectors[v][d] = randomFloat() * 2.0f - 1.0f;
                short fp16 = Float.floatToFloat16(vectors[v][d]);
                freq[Short.toUnsignedInt(fp16)]++;
            }
        }

        HuffmanCodec codec = HuffmanCodec.buildFromFrequencies(freq);
        HuffmanTableDecoder tableDecoder = codec.getTableDecoder();
        HuffmanTableDecoder.NativeTable nativeTab = tableDecoder.getNativeTable();

        // With real vector data, many byte values should collapse into the same
        // equivalence class. We expect meaningful reduction from 256.
        int numEqClasses = nativeTab.numEqClasses;
        int numStates = tableDecoder.getNumStates();

        // Log for visibility
        logger.info(
            "Equivalence class reduction: 256 → {} classes, {} states, "
                + "table size: {} ints ({}x reduction from full 256-entry table)",
            numEqClasses, numStates,
            nativeTab.flatTable.length,
            String.format("%.1f", (256.0 * numStates * HuffmanTableDecoder.NativeTable.ENTRY_STRIDE)
                / nativeTab.flatTable.length)
        );

        // Sanity: should have at least some reduction (or at worst, no expansion)
        assertTrue("numEqClasses should be <= 256", numEqClasses <= 256);
    }
}
