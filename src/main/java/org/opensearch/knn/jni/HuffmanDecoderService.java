/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.jni;

/**
 * JNI service for native DFA-based Huffman decoding with equivalence-class
 * alphabet remapping. Fuses decompression and FP16→float32 conversion in
 * a single native pass, avoiding per-bit branches and Java object overhead.
 * <p>
 * The native library is the same SIMD library used by {@link SimdVectorComputeService},
 * so no additional library loading is needed beyond what the clumping reader
 * already triggers.
 */
public class HuffmanDecoderService {

    static {
        KNNLibraryLoader.loadSimdLibrary();
    }

    /**
     * Batch-decodes multiple Huffman-compressed FP16 vectors to float32 in native code.
     * <p>
     * The flat DFA table and equivalence-class map are produced by
     * {@code HuffmanTableDecoder.buildNativeTable()}.
     *
     * @param compressed   Compressed byte buffer
     * @param byteOffset   Starting byte offset into compressed
     * @param numVectors   Number of vectors to decode
     * @param flatTable    Flat DFA table (int[] with equivalence-class remapping)
     * @param numEqClasses Number of equivalence classes
     * @param dimension    Number of FP16 symbols per vector
     * @param eqClassMap   Byte-to-equivalence-class mapping (byte[256])
     * @param outFloats    Output float array (length >= numVectors * dimension)
     * @return Byte offset after decoding all vectors
     */
    public static native int decodeBatchToFloat(
        byte[] compressed,
        int byteOffset,
        int numVectors,
        int[] flatTable,
        int numEqClasses,
        int dimension,
        byte[] eqClassMap,
        float[] outFloats
    );
}
