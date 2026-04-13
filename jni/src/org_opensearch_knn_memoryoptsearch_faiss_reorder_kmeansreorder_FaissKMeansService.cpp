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

#include "org_opensearch_knn_memoryoptsearch_faiss_reorder_kmeansreorder_FaissKMeansService.h"

#include <jni.h>
#include <vector>

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

#include "jni_util.h"

static knn_jni::JNIUtil jniUtilKMeans;

JNIEXPORT jobject JNICALL Java_org_opensearch_knn_memoryoptsearch_faiss_reorder_kmeansreorder_FaissKMeansService_kmeansWithDistances(
    JNIEnv* env, jclass cls,
    jlong vectorsAddress, jint numVectors, jint dimension, jint numClusters, jint numIterations, jint metricType)
{
    try {
        // vectorsAddress is a pointer to std::vector<float>*
        std::vector<float>* vectorPtr = reinterpret_cast<std::vector<float>*>(vectorsAddress);
        float* vectors = vectorPtr->data();

        // Set up clustering parameters
        faiss::ClusteringParameters cp;
        cp.niter = numIterations;
        cp.verbose = false;

        faiss::Clustering clustering(dimension, numClusters, cp);

        // Create index based on metric type
        faiss::Index* index;
        if (metricType == 1) {  // INNER_PRODUCT
            index = new faiss::IndexFlatIP(dimension);
        } else {  // L2
            index = new faiss::IndexFlatL2(dimension);
        }

        // Train clustering
        clustering.train(numVectors, vectors, *index);

        // Search to get assignments and distances
        std::vector<faiss::idx_t> assignments(numVectors);
        std::vector<float> distances(numVectors);
        index->search(numVectors, vectors, 1, distances.data(), assignments.data());

        delete index;

        // Create Java arrays
        jintArray assignmentsArray = env->NewIntArray(numVectors);
        jfloatArray distancesArray = env->NewFloatArray(numVectors);

        // Copy assignments (convert from idx_t to int)
        std::vector<jint> intAssignments(assignments.begin(), assignments.end());
        env->SetIntArrayRegion(assignmentsArray, 0, numVectors, intAssignments.data());
        env->SetFloatArrayRegion(distancesArray, 0, numVectors, distances.data());

        // Create KMeansResult object
        jclass resultClass = env->FindClass("org/opensearch/knn/memoryoptsearch/faiss/reorder/kmeansreorder/KMeansResult");
        if (resultClass == nullptr) {
            throw std::runtime_error("Could not find KMeansResult class");
        }

        jmethodID constructor = env->GetMethodID(resultClass, "<init>", "([I[F)V");
        if (constructor == nullptr) {
            throw std::runtime_error("Could not find KMeansResult constructor");
        }

        return env->NewObject(resultClass, constructor, assignmentsArray, distancesArray);
    } catch (const std::exception& e) {
        jniUtilKMeans.ThrowJavaException(env, "java/lang/RuntimeException", e.what());
        return nullptr;
    } catch (...) {
        jniUtilKMeans.ThrowJavaException(env, "java/lang/RuntimeException", "Unknown error in k-means");
        return nullptr;
    }
}

JNIEXPORT jobject JNICALL Java_org_opensearch_knn_memoryoptsearch_faiss_reorder_kmeansreorder_FaissKMeansService_kmeansWithDistancesMMap(
    JNIEnv* env, jclass cls,
    jlong mmapAddress, jint numVectors, jint dimension, jint numClusters, jint numIterations, jint metricType)
{
    try {
        // mmapAddress points directly to float data (mmap'd), not a std::vector wrapper
        float* vectors = reinterpret_cast<float*>(mmapAddress);

        // Pre-fault: sequential touch at page stride to warm the page cache.
        // Without this, FAISS's random-access subsampling triggers ~100k+ random page faults.
        {
            volatile char sum = 0;
            const char* bytes = reinterpret_cast<const char*>(vectors);
            long nbytes = (long)numVectors * dimension * sizeof(float);
            for (long i = 0; i < nbytes; i += 4096) {
                sum += bytes[i];
            }
        }

        faiss::ClusteringParameters cp;
        cp.niter = numIterations;
        cp.verbose = false;

        faiss::Clustering clustering(dimension, numClusters, cp);

        faiss::Index* index;
        if (metricType == 1) {
            index = new faiss::IndexFlatIP(dimension);
        } else {
            index = new faiss::IndexFlatL2(dimension);
        }

        clustering.train(numVectors, vectors, *index);

        std::vector<faiss::idx_t> assignments(numVectors);
        std::vector<float> distances(numVectors);
        index->search(numVectors, vectors, 1, distances.data(), assignments.data());

        delete index;

        jintArray assignmentsArray = env->NewIntArray(numVectors);
        jfloatArray distancesArray = env->NewFloatArray(numVectors);

        std::vector<jint> intAssignments(assignments.begin(), assignments.end());
        env->SetIntArrayRegion(assignmentsArray, 0, numVectors, intAssignments.data());
        env->SetFloatArrayRegion(distancesArray, 0, numVectors, distances.data());

        jclass resultClass = env->FindClass("org/opensearch/knn/memoryoptsearch/faiss/reorder/kmeansreorder/KMeansResult");
        if (resultClass == nullptr) {
            throw std::runtime_error("Could not find KMeansResult class");
        }

        jmethodID constructor = env->GetMethodID(resultClass, "<init>", "([I[F)V");
        if (constructor == nullptr) {
            throw std::runtime_error("Could not find KMeansResult constructor");
        }

        return env->NewObject(resultClass, constructor, assignmentsArray, distancesArray);
    } catch (const std::exception& e) {
        jniUtilKMeans.ThrowJavaException(env, "java/lang/RuntimeException", e.what());
        return nullptr;
    } catch (...) {
        jniUtilKMeans.ThrowJavaException(env, "java/lang/RuntimeException", "Unknown error in mmap k-means");
        return nullptr;
    }
}
