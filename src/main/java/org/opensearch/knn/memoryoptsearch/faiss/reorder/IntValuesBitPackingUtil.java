package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import lombok.experimental.UtilityClass;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.packed.PackedInts;
import org.opensearch.common.lucene.store.ByteArrayIndexInput;

import java.io.IOException;

@UtilityClass
public class IntValuesBitPackingUtil {
    public static int bitsRequired(int[] values) {
        int maxValue = Integer.MIN_VALUE;
        for (int val : values) {
            if (val > maxValue) {
                maxValue = val;
            }
        }
        return PackedInts.bitsRequired(maxValue);
    }

    public static byte[] allocateBuffer(final int blockSize) {
        return new byte[minRequiredBufferSize(blockSize)];
    }

    private static int minRequiredBufferSize(final int blockSize) {
        return (Integer.BYTES + 1) * blockSize;
    }

    public static void writePackedInts(final int[] values, final byte[] buffer, final IndexOutput indexOutput) throws IOException {
        final int bitsPerValue = IntValuesBitPackingUtil.bitsRequired(values);
        indexOutput.writeByte((byte) bitsPerValue);
        final int packedNumBytes = IntValuesBitPackingUtil.pack(values, bitsPerValue, buffer);
        indexOutput.writeBytes(buffer, 0, packedNumBytes);
    }

    public static int getValue(final ByteArrayIndexInput indexInput, final int index, final byte[] underlyingBytes) {
        try {
            final int bitsPerValue = indexInput.readByte();
            return IntValuesBitPackingUtil.getValue(underlyingBytes, (int) indexInput.getFilePointer(), index, bitsPerValue);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static int getValue(final byte[] packed, final int offset, final int index, final int bitsPerValue) {
        // Since which bit the value begins?
        final long bitPos = (long) index * bitsPerValue;
        // Which byte need to look?
        final int bytePos = offset + (int) (bitPos >>> 3);
        // Starting bit in the byte block.
        // Ex: bitOffset = 3, 1 0 1 1 1 0 1 0
        // ^--------- this is the start offset
        final int bitOffset = (int) (bitPos & 7);

        // load 5 bytes
        long word = 0;
        word |= (packed[bytePos] & 0xFFL);
        word |= (packed[bytePos + 1] & 0xFFL) << 8;
        word |= (packed[bytePos + 2] & 0xFFL) << 16;
        word |= (packed[bytePos + 3] & 0xFFL) << 24;
        word |= (packed[bytePos + 4] & 0xFFL) << 32;

        return (int) ((word >>> bitOffset) & ((1L << bitsPerValue) - 1));
    }

    public static int pack(int[] values, final int bitsPerValue, byte[] dest) {
        long bitPos = 0;

        for (int v : values) {
            int bytePos = (int) (bitPos >>> 3);
            int bitOffset = (int) (bitPos & 7);

            // convert value to long without sign-extension
            long val = ((long) v) & ((1L << bitsPerValue) - 1);
            val <<= bitOffset;

            int bytes = (bitsPerValue + bitOffset + 7) >>> 3; // number of bytes affected
            for (int i = 0; i < bytes; i++) {
                // compute mask for this byte
                int bitsInByte = Math.min(8, bitsPerValue + bitOffset - i * 8);
                int mask = (1 << bitsInByte) - 1;

                // clear the bits we are going to write
                dest[bytePos + i] &= ~(mask << (i == 0 ? bitOffset : 0));

                // write the new bits
                dest[bytePos + i] |= (val & 0xFF);
                val >>>= 8;
            }

            bitPos += bitsPerValue;
        }

        return (int) ((bitPos + 7) >>> 3);
    }
}
