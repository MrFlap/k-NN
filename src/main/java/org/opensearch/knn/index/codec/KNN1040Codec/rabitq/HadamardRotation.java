/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec.rabitq;

import java.util.Random;

/**
 * A data-blind Hadamard rotation: a random sign flip, a random permutation, then a normalized Fast Walsh-Hadamard
 * Transform (FWHT) applied independently over each power-of-two block that composes the dimension. {@code R = F·P·S}.
 * Ported unchanged from SearchServicesKnnVectorFormats {@code clusterann/format/rotation/HadamardRotation}, which the
 * ClusterANN data-blind int8 (tiered RaBitQ) backend applies for every metric.
 *
 * <p>Orthonormal, so distances, dot products and the ranking survive it ({@code <Rq, Rd> = <q, d>}). Its purpose is to
 * flatten per-dimension variance so the RaBitQ codes are accurate for every dimension, at {@code O(d log d)} cost and
 * storing only a seed, from which the signs and permutation are regenerated.
 *
 * <p>Non-power-of-two dimensions are split into their set bits, largest first ({@code 768 -> [512][256]}), and an
 * independent normalized FWHT runs over each block. The sign flip and permutation span the whole dimension before the
 * per-block transforms, so energy from any input coordinate can reach any block.
 *
 * <p>Thread-safe and immutable once constructed.
 */
final class HadamardRotation {

    /** The seed new fields are written with; recorded in the metadata. */
    static final long DEFAULT_SEED = 42L;

    private final int dimension;
    private final long seed;
    /** Per-coordinate sign flips (+-1), length {@code dimension}. */
    private final float[] signs;
    /** Permutation of {@code [0, dimension)} applied before the block transforms. */
    private final int[] permutation;
    /** Start offset of each power-of-two FWHT block. */
    private final int[] blockOffsets;
    /** Length of each FWHT block (a power of two). */
    private final int[] blockLengths;

    private HadamardRotation(int dimension, long seed, float[] signs, int[] permutation, int[] blockOffsets, int[] blockLengths) {
        this.dimension = dimension;
        this.seed = seed;
        this.signs = signs;
        this.permutation = permutation;
        this.blockOffsets = blockOffsets;
        this.blockLengths = blockLengths;
    }

    /** Build a Hadamard rotation for {@code dimension} from {@code seed}; both sides reconstruct from these two. */
    static HadamardRotation create(int dimension, long seed) {
        if (dimension <= 0) {
            throw new IllegalArgumentException("dimension must be positive, got " + dimension);
        }
        final Random random = new Random(seed);

        final float[] signs = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            signs[i] = random.nextBoolean() ? 1f : -1f;
        }

        final int[] permutation = new int[dimension];
        for (int i = 0; i < dimension; i++) {
            permutation[i] = i;
        }
        for (int i = dimension - 1; i > 0; i--) {
            final int j = random.nextInt(i + 1);
            final int tmp = permutation[i];
            permutation[i] = permutation[j];
            permutation[j] = tmp;
        }

        // Decompose dim into power-of-two blocks (its set bits), largest first.
        final int numBlocks = Integer.bitCount(dimension);
        final int[] blockOffsets = new int[numBlocks];
        final int[] blockLengths = new int[numBlocks];
        int b = 0;
        int offset = 0;
        for (int bit = Integer.highestOneBit(dimension); bit != 0; bit >>>= 1) {
            if ((dimension & bit) != 0) {
                blockOffsets[b] = offset;
                blockLengths[b] = bit;
                offset += bit;
                b++;
            }
        }
        return new HadamardRotation(dimension, seed, signs, permutation, blockOffsets, blockLengths);
    }

    long seed() {
        return seed;
    }

    void rotate(float[] src, float[] dest) {
        if (src.length != dimension || dest.length != dimension) {
            throw new IllegalArgumentException(
                "This rotation is " + dimension + "-dimensional, got src=" + src.length + " dest=" + dest.length
            );
        }
        if (src == dest) {
            throw new IllegalArgumentException("src and dest must be distinct arrays");
        }
        // S then P: dest[i] = signs[perm[i]] * src[perm[i]]
        for (int i = 0; i < dimension; i++) {
            final int p = permutation[i];
            dest[i] = signs[p] * src[p];
        }
        // F: normalized FWHT over each power-of-two block, in place on dest.
        for (int blk = 0; blk < blockOffsets.length; blk++) {
            fwht(dest, blockOffsets[blk], blockLengths[blk]);
        }
    }

    /** In-place normalized Fast Walsh-Hadamard Transform over {@code a[off, off+len)}, len a power of two. */
    private static void fwht(float[] a, int off, int len) {
        for (int step = 1; step < len; step <<= 1) {
            for (int i = 0; i < len; i += step << 1) {
                for (int j = i; j < i + step; j++) {
                    final int x = off + j;
                    final int y = off + j + step;
                    final float u = a[x];
                    final float v = a[y];
                    a[x] = u + v;
                    a[y] = u - v;
                }
            }
        }
        // Normalize so the transform is orthonormal (preserves norms and dot products).
        final float inv = (float) (1.0 / Math.sqrt(len));
        for (int i = off; i < off + len; i++) {
            a[i] *= inv;
        }
    }
}
