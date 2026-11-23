/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec.rabitq;

import java.util.Arrays;

/**
 * Extended RaBitQ (Gao et al., VLDB 2024): B-bit codes of a vector's rotated unit residual, with an unbiased estimator
 * of its inner product with a query. Ported unchanged from the ClusterANN tiered RaBitQ POC
 * (SearchServicesKnnVectorFormats {@code quantization/ExtendedRaBitQ}).
 *
 * <p>The residual {@code o = (x - c) / |x - c|} is quantized to unsigned codes {@code u in [0, 2^B - 1]} that maximize
 * the cosine between {@code o} and the grid vector {@code g = u - half}, by sweeping the scales at which
 * {@code round(t * o_i)} changes. The top {@code t} bits of a B-bit code are themselves a valid t-bit code
 * ({@link #truncate}), so one stored code gives a cheap coarse tier and a full-precision refine tier.
 */
public final class ExtendedRaBitQ {

    private ExtendedRaBitQ() {}

    /** Grid half-width for B bits: grid values are {@code u - half}, {@code u in [0, 2^B - 1]}. */
    public static float half(final int bits) {
        return ((1 << bits) - 1) / 2.0f;
    }

    /** Reusable encoder scratch; one per writing thread. */
    public static final class Scratch {
        private static final int BINS = 1 << 14;
        private float[] crit = new float[0];
        private float[] binned = new float[0];
        private final int[] start = new int[BINS + 1];
        private float[] best = new float[0];
        private float[] g = new float[0];

        void ensure(final int critLen, final int dim) {
            if (crit.length < critLen) {
                crit = new float[critLen];
                binned = new float[critLen];
            }
            if (best.length < dim) {
                best = new float[dim];
                g = new float[dim];
            }
        }
    }

    /**
     * Quantizes a unit residual to B-bit unsigned codes.
     *
     * @param o the unit residual
     * @param bits B, in [1, 8]
     * @param out receives the codes at {@code offset}, one byte per dimension
     * @return the norm of the grid vector {@code g = code - half}
     */
    public static double quantize(final float[] o, final int bits, final byte[] out, final int offset, final Scratch scratch) {
        final int dim = o.length;
        final double h = half(bits);
        final int maxK = (int) Math.floor(h);
        scratch.ensure(dim * Math.max(1, maxK), dim);
        final float[] crit = scratch.crit;
        int ci = 0;
        for (int i = 0; i < dim; i++) {
            final double oi = Math.abs(o[i]);
            if (oi < 1e-12) {
                continue;
            }
            for (int k = 0; k < maxK; k++) {
                crit[ci++] = (float) ((k + 0.5) / oi);
            }
        }
        if (ci == 0) {
            final int center = Math.round((float) h);
            Arrays.fill(out, offset, offset + dim, (byte) center);
            return 0.0;
        }
        // Evaluating the cosine at every critical scale is O(dim^2 * maxK) per vector; the prototype sweeps an evenly
        // spaced subset of the sorted scales. Only those ranks are needed, so they are selected exactly by bucketing
        // the (positive, so bit-order-monotonic) floats and sorting just the buckets that hold a needed rank.
        final int cap = Math.min(ci, 96 + 32 * bits);
        final int stride = Math.max(1, ci / cap);
        final float[] binned = bucket(crit, ci, scratch);
        final int[] start = scratch.start;
        double bestCos = Double.NEGATIVE_INFINITY;
        final float[] best = scratch.best;
        final float[] g = scratch.g;
        boolean found = false;
        int bin = 0;
        int sortedBin = -1;
        for (int s = 0; s < ci; s += stride) {
            while (start[bin + 1] <= s) {
                bin++;
            }
            if (bin != sortedBin) {
                Arrays.sort(binned, start[bin], start[bin + 1]);
                sortedBin = bin;
            }
            final double t = binned[s];
            double gn = 0;
            double dot = 0;
            for (int i = 0; i < dim; i++) {
                double gi = Math.rint(t * o[i]);
                if (gi > h) {
                    gi = h;
                } else if (gi < -h) {
                    gi = -h;
                }
                g[i] = (float) gi;
                gn += gi * gi;
                dot += gi * o[i];
            }
            if (gn == 0) {
                continue;
            }
            final double cos = dot / Math.sqrt(gn);
            if (cos > bestCos) {
                bestCos = cos;
                found = true;
                System.arraycopy(g, 0, best, 0, dim);
            }
        }
        if (!found) {
            Arrays.fill(best, 0, dim, 0f);
        }
        double gnorm = 0;
        for (int i = 0; i < dim; i++) {
            out[offset + i] = (byte) (int) (best[i] + h);
            gnorm += (double) best[i] * best[i];
        }
        return Math.sqrt(gnorm);
    }

    /** Counting-sorts positive floats into ordered buckets by their bit patterns; fills {@code scratch.start}. */
    private static float[] bucket(final float[] values, final int n, final Scratch scratch) {
        int minBits = Integer.MAX_VALUE;
        int maxBits = Integer.MIN_VALUE;
        for (int i = 0; i < n; i++) {
            final int b = Float.floatToRawIntBits(values[i]);
            minBits = Math.min(minBits, b);
            maxBits = Math.max(maxBits, b);
        }
        final int bins = Scratch.BINS;
        int shift = 0;
        while (((maxBits - minBits) >>> shift) >= bins) {
            shift++;
        }
        final int[] start = scratch.start;
        Arrays.fill(start, 0);
        for (int i = 0; i < n; i++) {
            start[((Float.floatToRawIntBits(values[i]) - minBits) >>> shift) + 1]++;
        }
        for (int b = 0; b < bins; b++) {
            start[b + 1] += start[b];
        }
        final float[] out = scratch.binned;
        final int[] fill = new int[bins];
        System.arraycopy(start, 0, fill, 0, bins);
        for (int i = 0; i < n; i++) {
            out[fill[(Float.floatToRawIntBits(values[i]) - minBits) >>> shift]++] = values[i];
        }
        return out;
    }

    /** The top {@code keepBits} of a {@code fullBits} code. */
    public static int truncate(final int codeByte, final int fullBits, final int keepBits) {
        if (keepBits >= fullBits) {
            return codeByte & 0xFF;
        }
        return (codeByte & 0xFF) >>> (fullBits - keepBits);
    }

    /** Norm of the grid vector of the code truncated to {@code keepBits}. */
    public static double gnorm(final byte[] code, final int offset, final int dim, final int fullBits, final int keepBits) {
        final double h = half(keepBits);
        double s = 0;
        for (int i = 0; i < dim; i++) {
            final double g = truncate(code[offset + i], fullBits, keepBits) - h;
            s += g * g;
        }
        return Math.sqrt(s);
    }

    /** The estimator denominator {@code <g / |g|, o>} for the code truncated to {@code keepBits}. */
    public static double odob(
        final byte[] code,
        final int offset,
        final float[] o,
        final int fullBits,
        final int keepBits,
        final double gnorm
    ) {
        if (gnorm <= 0) {
            return 1.0;
        }
        final double h = half(keepBits);
        double dot = 0;
        for (int i = 0; i < o.length; i++) {
            dot += (truncate(code[offset + i], fullBits, keepBits) - h) * (double) o[i];
        }
        return dot / gnorm;
    }
}
