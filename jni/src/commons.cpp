/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */
#ifndef OPENSEARCH_KNN_COMMONS_H
#define OPENSEARCH_KNN_COMMONS_H
#include <jni.h>

#include <vector>

#include "jni_util.h"
#include "commons.h"

jlong knn_jni::commons::storeVectorData(knn_jni::JNIUtilInterface *jniUtil, JNIEnv *env, jlong memoryAddressJ,
                                        jobjectArray dataJ, jlong initialCapacityJ) {
    int dim = jniUtil->GetInnerDimensionOf2dJavaFloatArray(env, dataJ);
    int numVectors = env->GetArrayLength(dataJ);
    batch_list *vect;
    size_t num_batches = 1;
    if ((long) memoryAddressJ == 0) {
        vect = new batch_list();
        //num_batches = (size_t)sqrt((double)dim * numVectors * sizeof(float) / sizeof(std::vector<float>));
        batch_list * runner = vect;
        for(int i = 1; i < num_batches; i++) {
            runner->next = new batch_list();
            runner = runner->next;
        }
    } else {
        vect = reinterpret_cast<batch_list*>(memoryAddressJ);
        batch_list * runner = vect;
        size_t found = 0;
        while(runner != NULL) {
            num_batches++;
            found += runner->batch.size() / (size_t)dim;
            if(found == numVectors) {
                break;
            }
            runner = runner->next;
        }
    }
    jniUtil->Convert2dJavaObjectArrayAndStoreToBatches(env, dataJ, dim, vect, num_batches);
    printf("num_batches: %lu\n", num_batches);
    return (jlong) vect;
}

void knn_jni::commons::freeVectorData(jlong memoryAddressJ) {
    if (memoryAddressJ != 0) {
        auto *vect = reinterpret_cast<std::vector<float>*>(memoryAddressJ);
        delete vect;
    }
}
#endif //OPENSEARCH_KNN_COMMONS_H