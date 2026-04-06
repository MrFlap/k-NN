/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex.clumping;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * DFA-based byte-at-a-time Huffman decoder.
 * <p>
 * Pre-computes a state machine where each state represents a partial Huffman code
 * (bits consumed but not yet resolved to a symbol). For each (state, inputByte) pair,
 * the table stores the output symbols and the next state.
 * <p>
 * Decoding processes one byte at a time with a single array lookup per byte,
 * producing zero or more FP16 symbols per byte. This is dramatically faster than
 * bit-by-bit decoding since it eliminates per-bit branches and leverages CPU
 * cache-friendly sequential byte access.
 * <p>
 * State 0 = symbol boundary (no leftover bits).
 * State -1 = terminal (should not be reached during normal decoding).
 */
public final class HuffmanTableDecoder {

    /**
     * Result of a single (state, byte) lookup.
     */
    public static final class Entry {
        /** Next state after consuming this byte. */
        public final int nextState;
        /** Decoded FP16 symbols (unsigned 16-bit values). May be empty. */
        public final int[] symbols;

        Entry(int nextState, int[] symbols) {
            this.nextState = nextState;
            this.symbols = symbols;
        }
    }

    /** tables[state][byteValue] = Entry(nextState, symbols) */
    private final Entry[][] tables;
    /** Number of states in the DFA. */
    private final int numStates;

    /** Cached native table for JNI decoding. Built lazily on first access. */
    private volatile NativeTable nativeTable;

    private HuffmanTableDecoder(Entry[][] tables) {
        this.tables = tables;
        this.numStates = tables.length;
    }

    /**
     * Builds the DFA table decoder from a HuffmanCodec.
     * <p>
     * Each state corresponds to a node in the Huffman tree (internal node = partial code).
     * State 0 is the root (symbol boundary). We recursively build tables by simulating
     * what happens when we feed each possible byte (0-255) starting from each state.
     *
     * @param codeLengths array of length 65536, codeLengths[sym] = bit length (0 = unused)
     * @param codes       array of length 65536, codes[sym] = canonical Huffman code
     * @param maxCodeLength maximum code length in the tree
     */
    public static HuffmanTableDecoder build(int[] codeLengths, long[] codes, int maxCodeLength) {
        // Build a trie from the Huffman codes. Each node is identified by an integer ID.
        // Node 0 = root. Children are allocated incrementally.
        // leafSymbol[node] = symbol if leaf, -1 if internal.
        int nodeCapacity = 65536 * 2;
        int[] leftChild = new int[nodeCapacity];   // child for bit 0
        int[] rightChild = new int[nodeCapacity];  // child for bit 1
        int[] leafSymbol = new int[nodeCapacity];
        java.util.Arrays.fill(leftChild, -1);
        java.util.Arrays.fill(rightChild, -1);
        java.util.Arrays.fill(leafSymbol, -1);
        int nextNodeId = 1; // 0 is root

        // Insert each symbol's code into the trie
        for (int sym = 0; sym < HuffmanCodec.NUM_SYMBOLS; sym++) {
            int len = codeLengths[sym];
            if (len == 0) continue;
            long code = codes[sym];

            int node = 0;
            for (int bit = len - 1; bit >= 0; bit--) {
                int b = (int) ((code >> bit) & 1);
                if (b == 0) {
                    if (leftChild[node] == -1) {
                        leftChild[node] = nextNodeId++;
                    }
                    node = leftChild[node];
                } else {
                    if (rightChild[node] == -1) {
                        rightChild[node] = nextNodeId++;
                    }
                    node = rightChild[node];
                }
            }
            leafSymbol[node] = sym;
        }

        // totalTrieNodes not needed after trie construction; state count is determined dynamically

        // Now build the DFA tables. Each trie internal node is a DFA state.
        // State 0 = root. For each (state, byte), simulate feeding 8 bits MSB-first
        // through the trie, collecting symbols when we hit leaves (and restarting from root).
        // The final trie node position after the 8 bits becomes the next state.

        // Map trie node ID -> DFA state ID. Only internal nodes become states.
        // Leaf nodes are never "states" because we immediately restart from root.
        Map<Integer, Integer> trieNodeToState = new HashMap<>();
        trieNodeToState.put(0, 0); // root = state 0

        List<Entry[]> tableList = new ArrayList<>();
        tableList.add(null); // placeholder for state 0

        // BFS/DFS: process states as we discover them
        List<Integer> stateTrieNodes = new ArrayList<>();
        stateTrieNodes.add(0); // state 0 = trie root

        int stateIdx = 0;
        while (stateIdx < stateTrieNodes.size()) {
            int trieNode = stateTrieNodes.get(stateIdx);
            Entry[] table = new Entry[256];

            for (int byteVal = 0; byteVal < 256; byteVal++) {
                // Simulate feeding 8 bits of byteVal (MSB first) starting from trieNode
                List<Integer> outputSymbols = new ArrayList<>();
                int currentNode = trieNode;

                for (int bitIdx = 7; bitIdx >= 0; bitIdx--) {
                    int bit = (byteVal >> bitIdx) & 1;
                    if (bit == 0) {
                        currentNode = leftChild[currentNode];
                    } else {
                        currentNode = rightChild[currentNode];
                    }

                    if (currentNode == -1) {
                        // Invalid path — shouldn't happen with valid Huffman codes
                        // but handle gracefully
                        currentNode = 0;
                        break;
                    }

                    if (leafSymbol[currentNode] != -1) {
                        // Hit a leaf — emit symbol and restart from root
                        outputSymbols.add(leafSymbol[currentNode]);
                        currentNode = 0; // back to root
                    }
                }

                // currentNode is now the trie node we ended up at after 8 bits.
                // Map it to a DFA state.
                int nextState;
                if (currentNode == 0 && outputSymbols.isEmpty() == false) {
                    // Ended at root with symbols emitted — clean boundary
                    nextState = 0;
                } else if (currentNode == 0) {
                    nextState = 0;
                } else {
                    // Ended at an internal node — need a state for it
                    Integer existing = trieNodeToState.get(currentNode);
                    if (existing != null) {
                        nextState = existing;
                    } else {
                        nextState = stateTrieNodes.size();
                        trieNodeToState.put(currentNode, nextState);
                        stateTrieNodes.add(currentNode);
                        tableList.add(null); // placeholder
                    }
                }

                int[] syms = outputSymbols.stream().mapToInt(Integer::intValue).toArray();
                table[byteVal] = new Entry(nextState, syms);
            }

            tableList.set(stateIdx, table);
            stateIdx++;
        }

        return new HuffmanTableDecoder(tableList.toArray(new Entry[0][]));
    }

    /**
     * Decodes compressed bytes into raw FP16 symbols (unsigned shorts).
     * The caller is responsible for converting to float32 for scoring.
     *
     * @param compressed  the compressed byte array
     * @param numSymbols  the number of FP16 symbols to decode
     * @param out         output short array of length >= numSymbols
     */
    public void decodeToShort(byte[] compressed, int numSymbols, short[] out) {
        int state = 0;
        int outIdx = 0;
        int byteIdx = 0;

        while (outIdx < numSymbols && byteIdx < compressed.length) {
            int b = compressed[byteIdx++] & 0xFF;
            Entry entry = tables[state][b];
            int[] syms = entry.symbols;
            for (int i = 0; i < syms.length && outIdx < numSymbols; i++) {
                out[outIdx++] = (short) syms[i];
            }
            state = entry.nextState;
        }
    }

    /**
     * Decodes compressed bytes into FP16 symbols, converting each to float32.
     *
     * @param compressed  the compressed byte array
     * @param numSymbols  the number of FP16 symbols to decode
     * @param out         output float array of length >= numSymbols
     */
    public void decodeToFloat(byte[] compressed, int numSymbols, float[] out) {
        int state = 0;
        int outIdx = 0;
        int byteIdx = 0;

        while (outIdx < numSymbols && byteIdx < compressed.length) {
            int b = compressed[byteIdx++] & 0xFF;
            Entry entry = tables[state][b];
            int[] syms = entry.symbols;
            for (int i = 0; i < syms.length && outIdx < numSymbols; i++) {
                out[outIdx++] = Float.float16ToFloat((short) syms[i]);
            }
            state = entry.nextState;
        }
    }

    /**
     * Decodes compressed bytes starting at a byte offset into raw FP16 shorts.
     * Carries decoder state across calls for sequential vector decoding.
     *
     * @param compressed   the compressed byte array
     * @param byteOffset   starting byte offset
     * @param state        decoder state from previous call (0 for first call)
     * @param numSymbols   number of FP16 symbols to decode
     * @param out          output short array
     * @return array of [nextByteOffset, nextState]
     */
    public int[] decodeToShortFromOffset(byte[] compressed, int byteOffset, int state, int numSymbols, short[] out) {
        int outIdx = 0;
        int byteIdx = byteOffset;

        while (outIdx < numSymbols && byteIdx < compressed.length) {
            int b = compressed[byteIdx++] & 0xFF;
            Entry entry = tables[state][b];
            int[] syms = entry.symbols;
            for (int i = 0; i < syms.length && outIdx < numSymbols; i++) {
                out[outIdx++] = (short) syms[i];
            }
            state = entry.nextState;
        }

        return new int[] { byteIdx, state };
    }

    /**
     * Decodes compressed bytes starting at a byte offset, returning decoded floats
     * and the byte position after decoding.
     * <p>
     * Note: unlike the bit-level decoder, this operates on byte boundaries.
     * The caller must ensure compressed data for each marker's vector block
     * starts at a byte boundary (which it does, since each marker's compressed
     * block is independently encoded).
     *
     * @param compressed   the compressed byte array
     * @param byteOffset   starting byte offset
     * @param numSymbols   number of FP16 symbols to decode
     * @param out          output float array
     * @return byte offset after decoding
     */
    public int decodeToFloatFromOffset(byte[] compressed, int byteOffset, int numSymbols, float[] out) {
        int state = 0;
        int outIdx = 0;
        int byteIdx = byteOffset;

        while (outIdx < numSymbols && byteIdx < compressed.length) {
            int b = compressed[byteIdx++] & 0xFF;
            Entry entry = tables[state][b];
            int[] syms = entry.symbols;
            for (int i = 0; i < syms.length && outIdx < numSymbols; i++) {
                out[outIdx++] = Float.float16ToFloat((short) syms[i]);
            }
            state = entry.nextState;
        }

        return byteIdx;
    }

    /**
     * Returns the number of states in the DFA. Useful for estimating memory usage.
     * Total memory ~ numStates * 256 * (4 + symbolsArrayOverhead) bytes.
     */
    /**
     * Returns the number of states in the DFA. Useful for estimating memory usage.
     * Total memory ~ numStates * 256 * (4 + symbolsArrayOverhead) bytes.
     */
    public int getNumStates() {
        return numStates;
    }

    /**
     * Pre-computed native-friendly DFA table with equivalence-class alphabet remapping.
     * <p>
     * Equivalence classes collapse byte values that produce identical (nextState, symbols)
     * transitions from every DFA state. This shrinks the table from numStates*256 entries
     * to numStates*numEqClasses entries, improving cache utilization in the native decoder.
     * <p>
     * The flat table is a single int[] where each entry occupies ENTRY_STRIDE ints:
     * [nextState, numSymbols, sym0, sym1, sym2, sym3]. Entry for (state, eqClass) is at
     * offset (state * numEqClasses + eqClass) * ENTRY_STRIDE.
     */
    public static final class NativeTable {
        /** Maximum symbols a single byte can decode to. Must match C++ MAX_SYMBOLS_PER_ENTRY. */
        public static final int MAX_SYMBOLS_PER_ENTRY = 4;
        /** Each entry: nextState(1) + numSymbols(1) + symbols(MAX). Must match C++ ENTRY_STRIDE. */
        public static final int ENTRY_STRIDE = 2 + MAX_SYMBOLS_PER_ENTRY;

        /** Flat DFA table: int[numStates * numEqClasses * ENTRY_STRIDE]. */
        public final int[] flatTable;
        /** Byte-to-equivalence-class mapping: byte[256]. */
        public final byte[] eqClassMap;
        /** Number of equivalence classes (reduced alphabet size). */
        public final int numEqClasses;

        NativeTable(int[] flatTable, byte[] eqClassMap, int numEqClasses) {
            this.flatTable = flatTable;
            this.eqClassMap = eqClassMap;
            this.numEqClasses = numEqClasses;
        }
    }

    /**
     * Returns the native-friendly DFA table with equivalence-class remapping.
     * Built lazily on first access and cached for the lifetime of this decoder.
     */
    public NativeTable getNativeTable() {
        NativeTable result = nativeTable;
        if (result == null) {
            synchronized (this) {
                result = nativeTable;
                if (result == null) {
                    result = buildNativeTable();
                    nativeTable = result;
                }
            }
        }
        return result;
    }

    /**
     * Builds the equivalence-class mapping and flat DFA table.
     * <p>
     * Two byte values are in the same equivalence class if, for every DFA state,
     * they produce the same (nextState, symbols) tuple. We compute a signature
     * string per byte value across all states and group identical signatures.
     */
    private NativeTable buildNativeTable() {
        // Step 1: Compute a signature for each byte value (0-255) across all states.
        // Signature = concatenation of (nextState, numSyms, sym0..symN) for each state.
        Map<String, Integer> signatureToClass = new HashMap<>();
        byte[] eqClassMap = new byte[256];
        int nextClassId = 0;

        for (int b = 0; b < 256; b++) {
            StringBuilder sig = new StringBuilder(numStates * 8);
            for (int s = 0; s < numStates; s++) {
                Entry entry = tables[s][b];
                sig.append(entry.nextState).append(':');
                sig.append(entry.symbols.length);
                for (int sym : entry.symbols) {
                    sig.append(':').append(sym);
                }
                sig.append(';');
            }
            String key = sig.toString();
            Integer classId = signatureToClass.get(key);
            if (classId == null) {
                classId = nextClassId++;
                signatureToClass.put(key, classId);
            }
            eqClassMap[b] = (byte) classId.intValue();
        }

        int numEqClasses = nextClassId;

        // Step 2: Build the flat table indexed by (state, eqClass).
        // For each equivalence class, pick any representative byte value and use its entries.
        // Find a representative byte for each equivalence class.
        int[] classRepresentative = new int[numEqClasses];
        for (int b = 0; b < 256; b++) {
            int cls = Byte.toUnsignedInt(eqClassMap[b]);
            classRepresentative[cls] = b; // last one wins, all are equivalent
        }

        int[] flatTable = new int[numStates * numEqClasses * NativeTable.ENTRY_STRIDE];

        for (int s = 0; s < numStates; s++) {
            for (int c = 0; c < numEqClasses; c++) {
                int rep = classRepresentative[c];
                Entry entry = tables[s][rep];
                int offset = (s * numEqClasses + c) * NativeTable.ENTRY_STRIDE;
                flatTable[offset] = entry.nextState;
                flatTable[offset + 1] = Math.min(entry.symbols.length, NativeTable.MAX_SYMBOLS_PER_ENTRY);
                for (int i = 0; i < flatTable[offset + 1]; i++) {
                    flatTable[offset + 2 + i] = entry.symbols[i];
                }
            }
        }

        return new NativeTable(flatTable, eqClassMap, numEqClasses);
    }
}
