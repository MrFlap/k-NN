package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import java.io.IOException;
import java.util.concurrent.ThreadLocalRandom;

public class PackedIntsWriterTest {
    public static void main(String... args) throws IOException {
        final int[] ords = new int[256];
        for (int i = 0; i < ords.length; ++i) {
            ords[i] = 1000000 + Math.abs(ThreadLocalRandom.current().nextInt() % 10000);
        }
        ords[0] = 862947;

        final int bpv = IntValuesBitPackingUtil.bitsRequired(ords);
        // public static int pack(int[] values, final int bitsPerValue, byte[] dest) {
        final byte[] buffer = IntValuesBitPackingUtil.allocateBuffer(ords.length);
        final int packedNumBytes = IntValuesBitPackingUtil.pack(ords, bpv, buffer);

        byte[] buffer2 = new byte[777 + packedNumBytes + 100];
        System.arraycopy(buffer, 0, buffer2, 777, packedNumBytes);

        for (int i = 0; i < ords.length; ++i) {
            System.out.println("i=" + i);
            int got = IntValuesBitPackingUtil.getValue(buffer, 0, i, bpv);
            if (got != ords[i]) {
                throw new AssertionError("mismatch at " + i + ": expected=" + ords[i] + ", got=" + got);
            }
        }
    }
}
