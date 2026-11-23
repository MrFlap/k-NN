/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec.rabitq;

import org.apache.lucene.store.DataOutput;

import java.io.IOException;
import java.util.Arrays;

/**
 * Tiered Extended RaBitQ records with the codes and estimator of the ClusterANN POC's
 * {@code TieredRaBitQClusterWriter}, laid out per ordinal so an HNSW traversal reads one fixed-size coarse row per
 * node and touches the full code only when it refines:
 *
 * <pre>
 *   coarse row : top code bit, packed (ceil(dim/8) bytes) | resNorm | gC | oC | bC (float) | popcount (int)
 *   refine row : code[dim] (B-bit unsigned) | gR | oR (float)
 * </pre>
 *
 * <p>The code quantizes the vector's unit residual against the segment centroid. {@code gC, oC} and {@code gR, oR}
 * are the estimator's corrections for the code truncated to {@link #COARSE_BITS} and {@link #REFINE_BITS}.
 * {@code bC} is the top bits' inner product with the centroid, so a Euclidean scan can form
 * {@code <top bits, q - c>} from a per-query encoding of {@code q}.
 */
final class TieredRaBitQRecords {

    /** Bits per dimension of the stored code. */
    static final int FULL_BITS = 7;
    /** Bits of the coarse tier. */
    static final int COARSE_BITS = 1;
    /** Bits of the refine tier. */
    static final int REFINE_BITS = 7;
    /** Coarse float columns: resNorm, gC, oC, bC. */
    static final int COARSE_FLOATS = 4;
    /** Floats after each refine code: gR, oR. */
    static final int REFINE_FLOATS = 2;

    private TieredRaBitQRecords() {}

    /** Bytes of one vector's packed top-bit plane. */
    static int bitBytes(final int dimension) {
        return (dimension + 7) >>> 3;
    }

    /** Bytes of one coarse row: bit plane, float columns, popcount. */
    static int coarseBytes(final int dimension) {
        return bitBytes(dimension) + COARSE_FLOATS * Float.BYTES + Integer.BYTES;
    }

    /** Bytes of one refine row. */
    static int refineBytes(final int dimension) {
        return dimension + REFINE_FLOATS * Float.BYTES;
    }

    /** Encodes vectors against one centroid; one per writing thread. */
    static final class Encoder {
        private final int dimension;
        private final float[] centroid;
        private final ExtendedRaBitQ.Scratch scratch = new ExtendedRaBitQ.Scratch();
        private final float[] o;
        private final byte[] code;
        private final byte[] bits;

        Encoder(final float[] centroid) {
            this.dimension = centroid.length;
            this.centroid = centroid;
            this.o = new float[dimension];
            this.code = new byte[dimension];
            this.bits = new byte[bitBytes(dimension)];
        }

        /** Writes {@code x}'s coarse row to {@code coarse} and its refine row to {@code refine}. */
        void write(final float[] x, final DataOutput coarse, final DataOutput refine) throws IOException {
            double normSq = 0;
            for (int d = 0; d < dimension; d++) {
                final float r = x[d] - centroid[d];
                o[d] = r;
                normSq += (double) r * r;
            }
            final float resNorm = (float) Math.sqrt(normSq);
            if (resNorm > 0) {
                for (int d = 0; d < dimension; d++) {
                    o[d] /= resNorm;
                }
            } else {
                Arrays.fill(o, 0f);
            }
            ExtendedRaBitQ.quantize(o, FULL_BITS, code, 0, scratch);
            final double gC = ExtendedRaBitQ.gnorm(code, 0, dimension, FULL_BITS, COARSE_BITS);
            final double gR = ExtendedRaBitQ.gnorm(code, 0, dimension, FULL_BITS, REFINE_BITS);
            Arrays.fill(bits, (byte) 0);
            final int shift = FULL_BITS - COARSE_BITS;
            double bC = 0;
            int p = 0;
            for (int d = 0; d < dimension; d++) {
                final int u = code[d] & 0xFF;
                if ((u >>> shift) != 0) {
                    bits[d >>> 3] |= (byte) (1 << (d & 7));
                    p++;
                    bC += centroid[d];
                }
            }
            coarse.writeBytes(bits, 0, bits.length);
            coarse.writeInt(Float.floatToIntBits(resNorm));
            coarse.writeInt(Float.floatToIntBits((float) gC));
            coarse.writeInt(Float.floatToIntBits((float) ExtendedRaBitQ.odob(code, 0, o, FULL_BITS, COARSE_BITS, gC)));
            coarse.writeInt(Float.floatToIntBits((float) bC));
            coarse.writeInt(p);
            refine.writeBytes(code, 0, dimension);
            refine.writeInt(Float.floatToIntBits((float) gR));
            refine.writeInt(Float.floatToIntBits((float) ExtendedRaBitQ.odob(code, 0, o, FULL_BITS, REFINE_BITS, gR)));
        }
    }
}
