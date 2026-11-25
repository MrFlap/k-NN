/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec.rabitq;

import org.apache.lucene.util.VectorUtil;
import org.opensearch.knn.KNNTestCase;

import java.util.Random;

/** The rotation is orthonormal at power-of-two and composite dimensions, and rebuilds identically from its seed. */
public class HadamardRotationTests extends KNNTestCase {

    public void testPreservesDotProductsAndNorms() {
        final Random random = new Random(random().nextLong());
        for (int dim : new int[] { 1, 7, 64, 100, 384, 768, 1536 }) {
            final HadamardRotation rotation = HadamardRotation.create(dim, HadamardRotation.DEFAULT_SEED);
            final float[] a = new float[dim];
            final float[] b = new float[dim];
            for (int d = 0; d < dim; d++) {
                a[d] = (float) random.nextGaussian();
                b[d] = (float) random.nextGaussian() + (d % 3 == 0 ? 5f : 0f); // uneven variance
            }
            final float[] ra = new float[dim];
            final float[] rb = new float[dim];
            rotation.rotate(a, ra);
            rotation.rotate(b, rb);
            final float scale = (float) Math.sqrt(VectorUtil.dotProduct(a, a) * VectorUtil.dotProduct(b, b));
            assertEquals("dim=" + dim, VectorUtil.dotProduct(a, b), VectorUtil.dotProduct(ra, rb), 1e-4f * scale);
            assertEquals("dim=" + dim, VectorUtil.dotProduct(b, b), VectorUtil.dotProduct(rb, rb), 1e-4f * VectorUtil.dotProduct(b, b));
        }
    }

    public void testSameSeedSameRotation() {
        final float[] x = new float[768];
        for (int d = 0; d < x.length; d++) {
            x[d] = d;
        }
        final float[] r1 = new float[768];
        final float[] r2 = new float[768];
        HadamardRotation.create(768, 7L).rotate(x, r1);
        HadamardRotation.create(768, 7L).rotate(x, r2);
        assertArrayEquals(r1, r2, 0f);
    }
}
