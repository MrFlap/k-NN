/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.clumping;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.PriorityQueue;

/**
 * Huffman codec for FP16 vector values stored in .clump files.
 * <p>
 * Operates on FP16 symbols (unsigned 16-bit values, 0..65535). The Huffman tree
 * is built from symbol frequencies observed across all hidden vectors in a segment,
 * serialized into the clump file header, and loaded into RAM at search time for
 * decompression.
 * <p>
 * The codec uses a canonical Huffman encoding: symbols are sorted by code length
 * then by symbol value, and codes are assigned sequentially. This allows the tree
 * to be serialized as just the code lengths per symbol (1 byte each), which is
 * compact and fast to reconstruct.
 */
public final class HuffmanCodec {

    /** Number of possible FP16 symbols (unsigned short range). */
    public static final int NUM_SYMBOLS = 65536;

    /** Maximum code length we allow. Longer codes are clamped during tree construction. */
    static final int MAX_CODE_LENGTH = 30;

    private final int[] codeLengths;  // code length per symbol (0 = symbol not present)
    private final long[] codes;       // canonical code per symbol
    private final int numSymbolsUsed; // count of symbols with non-zero frequency

    // Decode table: for each code length, the base code and the starting symbol index
    // in the sorted symbol list. Used for fast canonical decoding.
    private final int maxCodeLength;
    private final long[] baseCode;     // baseCode[len] = first canonical code of length len
    private final int[] baseIndex;     // baseIndex[len] = index into sortedSymbols for length len
    private final int[] sortedSymbols; // symbols sorted by (codeLength, symbolValue)
    private final boolean[] hasSymbols; // hasSymbols[len] = true if any symbol has this code length
    private final HuffmanTableDecoder tableDecoder; // DFA-based byte-at-a-time decoder

    private HuffmanCodec(int[] codeLengths, long[] codes, int maxCodeLength,
                         long[] baseCode, int[] baseIndex, int[] sortedSymbols, int numSymbolsUsed,
                         boolean[] hasSymbols) {
        this.codeLengths = codeLengths;
        this.codes = codes;
        this.maxCodeLength = maxCodeLength;
        this.baseCode = baseCode;
        this.baseIndex = baseIndex;
        this.sortedSymbols = sortedSymbols;
        this.numSymbolsUsed = numSymbolsUsed;
        this.hasSymbols = hasSymbols;
        this.tableDecoder = HuffmanTableDecoder.build(codeLengths, codes, maxCodeLength);
    }

    /**
     * Builds a HuffmanCodec from observed FP16 symbol frequencies.
     *
     * @param frequencies array of length {@link #NUM_SYMBOLS}, where frequencies[i]
     *                    is the count of FP16 symbol i across all hidden vectors
     * @return a HuffmanCodec ready for encoding and decoding
     */
    public static HuffmanCodec buildFromFrequencies(long[] frequencies) {
        if (frequencies.length != NUM_SYMBOLS) {
            throw new IllegalArgumentException("frequencies must have length " + NUM_SYMBOLS);
        }

        // Count symbols with non-zero frequency
        int numUsed = 0;
        for (long f : frequencies) {
            if (f > 0) numUsed++;
        }

        if (numUsed == 0) {
            throw new IllegalArgumentException("No symbols with non-zero frequency");
        }

        // Special case: only one unique symbol
        if (numUsed == 1) {
            int[] codeLens = new int[NUM_SYMBOLS];
            long[] cds = new long[NUM_SYMBOLS];
            int[] sorted = new int[1];
            for (int i = 0; i < NUM_SYMBOLS; i++) {
                if (frequencies[i] > 0) {
                    codeLens[i] = 1;
                    cds[i] = 0;
                    sorted[0] = i;
                    break;
                }
            }
            long[] bc = new long[3];
            int[] bi = new int[3];
            boolean[] hs = new boolean[3];
            bc[1] = 0;
            bi[1] = 0;
            hs[1] = true;
            return new HuffmanCodec(codeLens, cds, 1, bc, bi, sorted, 1, hs);
        }

        // Build Huffman tree via priority queue
        int[] codeLens = buildCodeLengths(frequencies, numUsed);

        // Build canonical codes from code lengths
        return buildCanonical(codeLens, numUsed);
    }

    /**
     * Builds code lengths via a standard Huffman tree construction.
     */
    private static int[] buildCodeLengths(long[] frequencies, int numUsed) {
        // Node: frequency, symbol (-1 for internal), left, right
        PriorityQueue<long[]> pq = new PriorityQueue<>((a, b) -> Long.compare(a[0], b[0]));

        // We'll track tree structure via parallel arrays to avoid object overhead
        // Each entry: [frequency, symbol_or_-1, leftIndex, rightIndex]
        // Leaf nodes have symbol >= 0, internal nodes have symbol = -1
        int nodeCount = 0;
        long[][] nodes = new long[2 * numUsed][4];

        for (int i = 0; i < NUM_SYMBOLS; i++) {
            if (frequencies[i] > 0) {
                nodes[nodeCount] = new long[]{frequencies[i], i, -1, -1};
                pq.add(nodes[nodeCount]);
                nodeCount++;
            }
        }

        // Build tree bottom-up
        while (pq.size() > 1) {
            long[] left = pq.poll();
            long[] right = pq.poll();
            long[] parent = new long[]{left[0] + right[0], -1, 0, 0};
            // Store references via identity — we'll traverse differently
            nodes[nodeCount] = parent;
            nodeCount++;
            pq.add(parent);

            // We need to track parent-child for depth calculation.
            // Instead, use a simpler approach: assign depths via recursive DFS
            // on a proper tree structure.
        }

        // Simpler approach: use the package-merge / length-limited approach
        // Actually, let's just compute depths directly from a proper tree.
        return buildCodeLengthsViaTree(frequencies, numUsed);
    }

    /**
     * Builds code lengths by constructing a proper Huffman tree and measuring depths.
     */
    private static int[] buildCodeLengthsViaTree(long[] frequencies, int numUsed) {
        int[] codeLens = new int[NUM_SYMBOLS];

        // Create leaf nodes
        int[][] leafSymbols = new int[numUsed][1]; // symbol for each leaf
        long[] leafFreqs = new long[numUsed];
        int idx = 0;
        for (int i = 0; i < NUM_SYMBOLS; i++) {
            if (frequencies[i] > 0) {
                leafSymbols[idx][0] = i;
                leafFreqs[idx] = frequencies[i];
                idx++;
            }
        }

        // Tree node: [freq, symbol (-1=internal), leftIdx, rightIdx]
        // We'll use arrays of size 2*numUsed - 1
        int totalNodes = 2 * numUsed - 1;
        long[] nodeFreq = new long[totalNodes];
        int[] nodeSymbol = new int[totalNodes];
        int[] nodeLeft = new int[totalNodes];
        int[] nodeRight = new int[totalNodes];
        java.util.Arrays.fill(nodeSymbol, -1);
        java.util.Arrays.fill(nodeLeft, -1);
        java.util.Arrays.fill(nodeRight, -1);

        // Initialize leaves
        for (int i = 0; i < numUsed; i++) {
            nodeFreq[i] = leafFreqs[i];
            nodeSymbol[i] = leafSymbols[i][0];
        }

        // Priority queue of node indices
        PriorityQueue<Integer> pq = new PriorityQueue<>((a, b) -> {
            int cmp = Long.compare(nodeFreq[a], nodeFreq[b]);
            if (cmp != 0) return cmp;
            // Tie-break: prefer leaves (higher symbol) to keep tree balanced
            return Integer.compare(nodeSymbol[b], nodeSymbol[a]);
        });
        for (int i = 0; i < numUsed; i++) {
            pq.add(i);
        }

        int nextNode = numUsed;
        while (pq.size() > 1) {
            int left = pq.poll();
            int right = pq.poll();
            nodeFreq[nextNode] = nodeFreq[left] + nodeFreq[right];
            nodeLeft[nextNode] = left;
            nodeRight[nextNode] = right;
            pq.add(nextNode);
            nextNode++;
        }

        // DFS to compute depths
        int root = pq.poll();
        int[] stack = new int[totalNodes];
        int[] depthStack = new int[totalNodes];
        int sp = 0;
        stack[sp] = root;
        depthStack[sp] = 0;
        sp++;

        while (sp > 0) {
            sp--;
            int node = stack[sp];
            int depth = depthStack[sp];

            if (nodeLeft[node] == -1 && nodeRight[node] == -1) {
                // Leaf
                int len = Math.min(depth, MAX_CODE_LENGTH);
                codeLens[nodeSymbol[node]] = len == 0 ? 1 : len; // minimum length 1
            } else {
                if (nodeRight[node] != -1) {
                    stack[sp] = nodeRight[node];
                    depthStack[sp] = depth + 1;
                    sp++;
                }
                if (nodeLeft[node] != -1) {
                    stack[sp] = nodeLeft[node];
                    depthStack[sp] = depth + 1;
                    sp++;
                }
            }
        }

        return codeLens;
    }

    /**
     * Builds canonical Huffman codes from code lengths.
     */
    private static HuffmanCodec buildCanonical(int[] codeLens, int numUsed) {
        // Find max code length
        int maxLen = 0;
        for (int len : codeLens) {
            if (len > maxLen) maxLen = len;
        }

        // Count symbols per length
        int[] lengthCount = new int[maxLen + 1];
        for (int sym = 0; sym < NUM_SYMBOLS; sym++) {
            if (codeLens[sym] > 0) {
                lengthCount[codeLens[sym]]++;
            }
        }

        // Sort symbols by (codeLength, symbolValue)
        int[] sorted = new int[numUsed];
        int si = 0;
        for (int len = 1; len <= maxLen; len++) {
            for (int sym = 0; sym < NUM_SYMBOLS; sym++) {
                if (codeLens[sym] == len) {
                    sorted[si++] = sym;
                }
            }
        }

        // Assign canonical codes
        long[] codes = new long[NUM_SYMBOLS];
        // For decoding: baseCode[len] = first canonical code of length len
        //               baseIndex[len] = index into sortedSymbols for the first symbol of length len
        // We size these to maxLen+2 so we can always safely access [len+1]
        long[] baseCode = new long[maxLen + 2];
        int[] baseIndex = new int[maxLen + 2];
        // Track which lengths actually have symbols
        boolean[] hasSymbols = new boolean[maxLen + 2];

        long code = 0;
        int sortedIdx = 0;
        for (int len = 1; len <= maxLen; len++) {
            baseCode[len] = code;
            baseIndex[len] = sortedIdx;
            hasSymbols[len] = lengthCount[len] > 0;

            for (int i = 0; i < lengthCount[len]; i++) {
                codes[sorted[sortedIdx]] = code;
                sortedIdx++;
                code++;
            }
            code <<= 1; // shift left for next length
        }
        // Sentinel: baseIndex[maxLen+1] = numUsed, so count for last length works
        baseIndex[maxLen + 1] = numUsed;

        return new HuffmanCodec(codeLens, codes, maxLen, baseCode, baseIndex, sorted, numUsed, hasSymbols);
    }

    // ---- Encoding ----

    /**
     * Returns the DFA-based table decoder for fast byte-at-a-time decoding.
     */
    public HuffmanTableDecoder getTableDecoder() {
        return tableDecoder;
    }

    /**
     * Encodes an array of FP16 short values into a compressed byte array.
     * Returns the compressed bytes. The caller is responsible for tracking
     * how many symbols were encoded (e.g., dimension * numVectors).
     */
    public byte[] encode(short[] fp16Values) {
        // Estimate output size (worst case: each symbol takes maxCodeLength bits)
        BitWriter writer = new BitWriter(fp16Values.length * 2 + 16);
        for (short val : fp16Values) {
            int sym = Short.toUnsignedInt(val);
            int len = codeLengths[sym];
            long c = codes[sym];
            writer.writeBits(c, len);
        }
        return writer.toByteArray();
    }

    /**
     * Encodes FP16 values from a raw byte buffer (big-endian shorts, matching Lucene's
     * DataOutput.writeShort format) into compressed bytes.
     */
    public byte[] encodeFromBytes(byte[] fp16Bytes) {
        int numShorts = fp16Bytes.length / Short.BYTES;
        BitWriter writer = new BitWriter(fp16Bytes.length + 16);
        java.nio.ByteBuffer buf = java.nio.ByteBuffer.wrap(fp16Bytes).order(java.nio.ByteOrder.BIG_ENDIAN);
        for (int i = 0; i < numShorts; i++) {
            int sym = Short.toUnsignedInt(buf.getShort());
            int len = codeLengths[sym];
            long c = codes[sym];
            writer.writeBits(c, len);
        }
        return writer.toByteArray();
    }

    // ---- Decoding ----

    /**
     * Decodes {@code numSymbols} FP16 values from compressed bytes into a float array.
     * Each FP16 symbol is converted to float32 via {@link Float#float16ToFloat(short)}.
     */
    public float[] decodeToFloat(byte[] compressed, int numSymbols) {
        float[] result = new float[numSymbols];
        BitReader reader = new BitReader(compressed);
        for (int i = 0; i < numSymbols; i++) {
            int sym = decodeOneSymbol(reader);
            result[i] = Float.float16ToFloat((short) sym);
        }
        return result;
    }

    /**
     * Decodes {@code numSymbols} FP16 values from compressed bytes starting at a
     * given bit offset. Returns the decoded floats and updates the bit offset.
     */
    public float[] decodeToFloat(byte[] compressed, int bitOffset, int numSymbols) {
        float[] result = new float[numSymbols];
        BitReader reader = new BitReader(compressed, bitOffset);
        for (int i = 0; i < numSymbols; i++) {
            int sym = decodeOneSymbol(reader);
            result[i] = Float.float16ToFloat((short) sym);
        }
        return result;
    }

    /**
     * Returns the current bit position after decoding. Useful for sequential
     * decoding of multiple vectors from the same compressed buffer.
     */
    public int decodeToFloatAndReturnBitPos(byte[] compressed, int bitOffset, int numSymbols, float[] out) {
        BitReader reader = new BitReader(compressed, bitOffset);
        for (int i = 0; i < numSymbols; i++) {
            int sym = decodeOneSymbol(reader);
            out[i] = Float.float16ToFloat((short) sym);
        }
        return reader.bitPosition();
    }

    private int decodeOneSymbol(BitReader reader) {
        long code = 0;
        for (int len = 1; len <= maxCodeLength; len++) {
            code = (code << 1) | reader.readBit();
            if (hasSymbols[len]) {
                long offset = code - baseCode[len];
                int count = baseIndex[len + 1] - baseIndex[len];
                if (offset >= 0 && offset < count) {
                    return sortedSymbols[baseIndex[len] + (int) offset];
                }
            }
        }
        throw new IllegalStateException("Invalid Huffman code in compressed data");
    }

    // ---- Serialization ----

    /**
     * Serializes the Huffman tree as code lengths (1 byte per symbol for symbols
     * that appear, preceded by a count of used symbols and their (symbol, length) pairs).
     * <p>
     * Format:
     * <pre>
     *   4 bytes: numSymbolsUsed (int)
     *   For each used symbol:
     *     2 bytes: symbol (unsigned short)
     *     1 byte:  codeLength
     * </pre>
     */
    public byte[] serialize() {
        // 4 bytes for count + 3 bytes per used symbol
        byte[] result = new byte[Integer.BYTES + numSymbolsUsed * 3];
        ByteBuffer buf = ByteBuffer.wrap(result).order(ByteOrder.LITTLE_ENDIAN);
        buf.putInt(numSymbolsUsed);
        for (int sym = 0; sym < NUM_SYMBOLS; sym++) {
            if (codeLengths[sym] > 0) {
                buf.putShort((short) sym);
                buf.put((byte) codeLengths[sym]);
            }
        }
        return result;
    }

    /**
     * Returns the serialized size in bytes.
     */
    public int serializedSizeBytes() {
        return Integer.BYTES + numSymbolsUsed * 3;
    }

    /**
     * Deserializes a HuffmanCodec from the format produced by {@link #serialize()}.
     */
    public static HuffmanCodec deserialize(byte[] data) {
        ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        int numUsed = buf.getInt();
        int[] codeLens = new int[NUM_SYMBOLS];
        for (int i = 0; i < numUsed; i++) {
            int sym = Short.toUnsignedInt(buf.getShort());
            int len = Byte.toUnsignedInt(buf.get());
            codeLens[sym] = len;
        }
        return buildCanonical(codeLens, numUsed);
    }

    /**
     * Deserializes a HuffmanCodec by reading from a Lucene IndexInput.
     */
    public static HuffmanCodec deserialize(org.apache.lucene.store.IndexInput input) throws IOException {
        int numUsed = input.readInt();
        int[] codeLens = new int[NUM_SYMBOLS];
        for (int i = 0; i < numUsed; i++) {
            int sym = Short.toUnsignedInt(input.readShort());
            int len = Byte.toUnsignedInt(input.readByte());
            codeLens[sym] = len;
        }
        return buildCanonical(codeLens, numUsed);
    }

    /**
     * Writes the serialized Huffman tree to a Lucene IndexOutput.
     */
    public void serialize(org.apache.lucene.store.IndexOutput output) throws IOException {
        output.writeInt(numSymbolsUsed);
        for (int sym = 0; sym < NUM_SYMBOLS; sym++) {
            if (codeLengths[sym] > 0) {
                output.writeShort((short) sym);
                output.writeByte((byte) codeLengths[sym]);
            }
        }
    }

    // ---- Bit I/O helpers ----

    /**
     * Writes bits MSB-first into a growable byte array.
     */
    static final class BitWriter {
        private byte[] data;
        private int bitPos;

        BitWriter(int estimatedBytes) {
            this.data = new byte[Math.max(estimatedBytes, 16)];
            this.bitPos = 0;
        }

        void writeBits(long value, int numBits) {
            for (int i = numBits - 1; i >= 0; i--) {
                int byteIdx = bitPos >> 3;
                if (byteIdx >= data.length) {
                    byte[] newData = new byte[data.length * 2];
                    System.arraycopy(data, 0, newData, 0, data.length);
                    data = newData;
                }
                int bitIdx = 7 - (bitPos & 7);
                if (((value >> i) & 1) == 1) {
                    data[byteIdx] |= (1 << bitIdx);
                }
                bitPos++;
            }
        }

        byte[] toByteArray() {
            int numBytes = (bitPos + 7) >> 3;
            byte[] result = new byte[numBytes];
            System.arraycopy(data, 0, result, 0, numBytes);
            return result;
        }

        int bitLength() {
            return bitPos;
        }
    }

    /**
     * Reads bits MSB-first from a byte array.
     */
    static final class BitReader {
        private final byte[] data;
        private int bitPos;

        BitReader(byte[] data) {
            this.data = data;
            this.bitPos = 0;
        }

        BitReader(byte[] data, int startBitPos) {
            this.data = data;
            this.bitPos = startBitPos;
        }

        int readBit() {
            int byteIdx = bitPos >> 3;
            int bitIdx = 7 - (bitPos & 7);
            bitPos++;
            return (data[byteIdx] >> bitIdx) & 1;
        }

        int bitPosition() {
            return bitPos;
        }
    }
}
