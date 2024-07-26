// SPDX-License-Identifier: Apache-2.0
//
// The OpenSearch Contributors require contributions made to
// this file be licensed under the Apache-2.0 license or a
// compatible open source license.
//
// Modifications Copyright OpenSearch Contributors. See
// GitHub history for details.

#include "faiss_index_service.h"
#include "faiss_methods.h"
#include "faiss/index_factory.h"
#include "faiss/Index.h"
#include "faiss/IndexBinary.h"
#include "faiss/IndexHNSW.h"
#include "faiss/IndexBinaryHNSW.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexBinaryIVF.h"
#include "faiss/IndexIDMap.h"
#include "faiss/index_io.h"
#include <algorithm>
#include <string>
#include <vector>
#include <memory>
#include <type_traits>

namespace knn_jni {
namespace faiss_wrapper {

template<typename INDEX, typename IVF, typename HNSW>
void SetExtraParameters(knn_jni::JNIUtilInterface * jniUtil, JNIEnv *env,
                        const std::unordered_map<std::string, jobject>& parametersCpp, INDEX * index) {
    std::unordered_map<std::string,jobject>::const_iterator value;
    if (auto * indexIvf = dynamic_cast<IVF*>(index)) {
        if ((value = parametersCpp.find(knn_jni::NPROBES)) != parametersCpp.end()) {
            indexIvf->nprobe = jniUtil->ConvertJavaObjectToCppInteger(env, value->second);
        }

        if ((value = parametersCpp.find(knn_jni::COARSE_QUANTIZER)) != parametersCpp.end()
                && indexIvf->quantizer != nullptr) {
            auto subParametersCpp = jniUtil->ConvertJavaMapToCppMap(env, value->second);
            SetExtraParameters<INDEX, IVF, HNSW>(jniUtil, env, subParametersCpp, indexIvf->quantizer);
        }
    }

    if (auto * indexHnsw = dynamic_cast<HNSW*>(index)) {

        if ((value = parametersCpp.find(knn_jni::EF_CONSTRUCTION)) != parametersCpp.end()) {
            indexHnsw->hnsw.efConstruction = jniUtil->ConvertJavaObjectToCppInteger(env, value->second);
        }

        if ((value = parametersCpp.find(knn_jni::EF_SEARCH)) != parametersCpp.end()) {
            indexHnsw->hnsw.efSearch = jniUtil->ConvertJavaObjectToCppInteger(env, value->second);
        }
    }
}

IndexService::IndexService(std::unique_ptr<FaissMethods> faissMethods) : faissMethods(std::move(faissMethods)) {}

void IndexService::initIndex(
        knn_jni::JNIUtilInterface * jniUtil,
        JNIEnv * env,
        IndexInfo * indexInfo
    ) {
    // Create index using Faiss factory method
    std::unique_ptr<faiss::Index> indexWriter(faissMethods->indexFactory(indexInfo->dimension, indexInfo->indexDescription.c_str(), indexInfo->metric));

    // Set thread count if it is passed in as a parameter. Setting this variable will only impact the current thread
    if(indexInfo->threadCount != 0) {
        omp_set_num_threads(indexInfo->threadCount);
    }

    // Add extra parameters that cant be configured with the index factory
    SetExtraParameters<faiss::Index, faiss::IndexIVF, faiss::IndexHNSW>(jniUtil, env, indexInfo->parameters, indexWriter.get());

    // Check that the index does not need to be trained
    if(!indexWriter->is_trained) {
        throw std::runtime_error("Index is not trained");
    }

    // Add vectors
    std::unique_ptr<faiss::IndexIDMap> idMap (faissMethods->indexIdMap(indexWriter.get()));

    /*
     * NOTE: The process of memory allocation is currently only implemented for HNSW.
     * This technique of checking the types of the index and subindices should be generalized into
     * another function.
     */

    // Check to see if the current index is HNSW
    faiss::IndexHNSW * hnsw = dynamic_cast<faiss::IndexHNSW *>(idMap->index);

    if(hnsw != NULL) {
        // Check to see if the HNSW storage is IndexFlat
        faiss::IndexFlat * storage = dynamic_cast<faiss::IndexFlat *>(hnsw->storage);
        if(storage != NULL) {
            // Allocate enough memory for all of the vectors we plan on inserting
            // We do this to avoid unnecessary memory allocations during insert
            storage->codes.reserve(indexInfo->dimension * indexInfo->plannedDocs * 4);
        }
    }
    indexWriter.release();
    indexInfo->indexAddress = reinterpret_cast<jlong>(idMap.release());
}

void IndexService::insertToIndex(
        IndexInfo * indexInfo,
        int64_t vectorsAddress,
        std::vector<int64_t> & ids
    ) {
    // Read vectors from memory address (unique ptr since we want to remove from memory after use)
    std::vector<float> * inputVectors = reinterpret_cast<std::vector<float>*>(vectorsAddress);

    // The number of vectors can be int here because a lucene segment number of total docs never crosses INT_MAX value
    int numVectors = (int) (inputVectors->size() / (uint64_t) indexInfo->dimension);
    if(numVectors == 0) {
        throw std::runtime_error("Number of vectors cannot be 0");
    }

    // Set thread count if it is passed in as a parameter. Setting this variable will only impact the current thread
    if(indexInfo->threadCount != 0) {
        omp_set_num_threads(indexInfo->threadCount);
    }

    faiss::IndexIDMap * idMap = reinterpret_cast<faiss::IndexIDMap *> (indexInfo->indexAddress);

    // Add vectors
    idMap->add_with_ids(numVectors, inputVectors->data(), ids.data());
}

void IndexService::writeIndex(IndexInfo * indexInfo) {
    // Set thread count if it is passed in as a parameter. Setting this variable will only impact the current thread
    if(indexInfo->threadCount != 0) {
        omp_set_num_threads(indexInfo->threadCount);
    }

    std::unique_ptr<faiss::IndexIDMap> idMap (reinterpret_cast<faiss::IndexIDMap *> (indexInfo->indexAddress));

    // Write the index to disk
    faissMethods->writeIndex(idMap.get(), indexInfo->indexPath.c_str());

    // Free the memory used by the index
    delete idMap->index;
}

void IndexService::createIndex(
        knn_jni::JNIUtilInterface * jniUtil,
        JNIEnv * env,
        faiss::MetricType metric,
        std::string indexDescription,
        int dim,
        int numIds,
        int threadCount,
        int64_t vectorsAddress,
        std::vector<int64_t> ids,
        std::string indexPath,
        std::unordered_map<std::string, jobject> parameters
    ) {
    // Read vectors from memory address
    auto *inputVectors = reinterpret_cast<std::vector<float>*>(vectorsAddress);

    // The number of vectors can be int here because a lucene segment number of total docs never crosses INT_MAX value
    int numVectors = (int) (inputVectors->size() / (uint64_t) dim);
    if(numVectors == 0) {
        throw std::runtime_error("Number of vectors cannot be 0");
    }

    if (numIds != numVectors) {
        throw std::runtime_error("Number of IDs does not match number of vectors");
    }

    std::unique_ptr<faiss::Index> indexWriter(faissMethods->indexFactory(dim, indexDescription.c_str(), metric));

    // Set thread count if it is passed in as a parameter. Setting this variable will only impact the current thread
    if(threadCount != 0) {
        omp_set_num_threads(threadCount);
    }

    // Add extra parameters that cant be configured with the index factory
    SetExtraParameters<faiss::Index, faiss::IndexIVF, faiss::IndexHNSW>(jniUtil, env, parameters, indexWriter.get());

    // Check that the index does not need to be trained
    if(!indexWriter->is_trained) {
        throw std::runtime_error("Index is not trained");
    }

    // Add vectors
    std::unique_ptr<faiss::IndexIDMap> idMap(faissMethods->indexIdMap(indexWriter.get()));
    idMap->add_with_ids(numVectors, inputVectors->data(), ids.data());

    // Write the index to disk
    faissMethods->writeIndex(idMap.get(), indexPath.c_str());
    delete idMap->index;
}

BinaryIndexService::BinaryIndexService(std::unique_ptr<FaissMethods> faissMethods) : IndexService(std::move(faissMethods)) {}

void BinaryIndexService::initIndex(
        knn_jni::JNIUtilInterface * jniUtil,
        JNIEnv * env,
        IndexInfo * indexInfo
    ) {
    // Create index using Faiss factory method
    std::unique_ptr<faiss::IndexBinary> indexWriter(faissMethods->indexBinaryFactory(indexInfo->dimension, indexInfo->indexDescription.c_str()));

    // Set thread count if it is passed in as a parameter. Setting this variable will only impact the current thread
    if(indexInfo->threadCount != 0) {
        omp_set_num_threads(indexInfo->threadCount);
    }

    // Add extra parameters that cant be configured with the index factory
    SetExtraParameters<faiss::IndexBinary, faiss::IndexBinaryIVF, faiss::IndexBinaryHNSW>(jniUtil, env, indexInfo->parameters, indexWriter.get());

    // Check that the index does not need to be trained
    if(!indexWriter->is_trained) {
        throw std::runtime_error("Index is not trained");
    }

    // Add vectors
    std::unique_ptr<faiss::IndexBinaryIDMap> idMap(faissMethods->indexBinaryIdMap(indexWriter.get()));

    /*
     * NOTE: The process of memory allocation is currently only implemented for HNSW.
     * This technique of checking the types of the index and subindices should be generalized into
     * another function.
     */

    // Check to see if the current index is BinaryHNSW
    faiss::IndexBinaryHNSW * hnsw = dynamic_cast<faiss::IndexBinaryHNSW *>(idMap->index);

    if(hnsw != NULL) {
        // Check to see if the HNSW storage is IndexBinaryFlat
        faiss::IndexBinaryFlat * storage = dynamic_cast<faiss::IndexBinaryFlat *>(hnsw->storage);
        if(storage != NULL) {
            // Allocate enough memory for all of the vectors we plan on inserting
            // We do this to avoid unnecessary memory allocations during insert
            storage->xb.reserve(indexInfo->dimension / 8 * indexInfo->plannedDocs);
        }
    }
    indexWriter.release();
    indexInfo->indexAddress = reinterpret_cast<jlong>(idMap.release());
}

void BinaryIndexService::insertToIndex(
        IndexInfo * indexInfo,
        int64_t vectorsAddress,
        std::vector<int64_t> & ids
    ) {
    // Read vectors from memory address (unique ptr since we want to remove from memory after use)
    std::vector<uint8_t> * inputVectors = reinterpret_cast<std::vector<uint8_t>*>(vectorsAddress);

    // The number of vectors can be int here because a lucene segment number of total docs never crosses INT_MAX value
    int numVectors = (int) (inputVectors->size() / (uint64_t) (indexInfo->dimension / 8));
    if(numVectors == 0) {
        throw std::runtime_error("Number of vectors cannot be 0");
    }

    // Set thread count if it is passed in as a parameter. Setting this variable will only impact the current thread
    if(indexInfo->threadCount != 0) {
        omp_set_num_threads(indexInfo->threadCount);
    }

    faiss::IndexBinaryIDMap * idMap = reinterpret_cast<faiss::IndexBinaryIDMap *> (indexInfo->indexAddress);

    // Add vectors
    idMap->add_with_ids(numVectors, inputVectors->data(), ids.data());
}

void BinaryIndexService::writeIndex(
        IndexInfo * indexInfo
    ) {
    // Set thread count if it is passed in as a parameter. Setting this variable will only impact the current thread
    if(indexInfo->threadCount != 0) {
        omp_set_num_threads(indexInfo->threadCount);
    }

    std::unique_ptr<faiss::IndexBinaryIDMap> idMap (reinterpret_cast<faiss::IndexBinaryIDMap *> (indexInfo->indexAddress));

    // Write the index to disk
    faissMethods->writeIndexBinary(idMap.get(), indexInfo->indexPath.c_str());

    // Free the memory used by the index
    delete idMap->index;
}

void BinaryIndexService::createIndex(
        knn_jni::JNIUtilInterface * jniUtil,
        JNIEnv * env,
        faiss::MetricType metric,
        std::string indexDescription,
        int dim,
        int numIds,
        int threadCount,
        int64_t vectorsAddress,
        std::vector<int64_t> ids,
        std::string indexPath,
        std::unordered_map<std::string, jobject> parameters
    ) {
    // Read vectors from memory address
    auto *inputVectors = reinterpret_cast<std::vector<uint8_t>*>(vectorsAddress);

    if (dim % 8 != 0) {
        throw std::runtime_error("Dimensions should be multiply of 8");
    }
    // The number of vectors can be int here because a lucene segment number of total docs never crosses INT_MAX value
    int numVectors = (int) (inputVectors->size() / (uint64_t) (dim / 8));
    if(numVectors == 0) {
        throw std::runtime_error("Number of vectors cannot be 0");
    }

    if (numIds != numVectors) {
        throw std::runtime_error("Number of IDs does not match number of vectors");
    }

    std::unique_ptr<faiss::IndexBinary> indexWriter(faissMethods->indexBinaryFactory(dim, indexDescription.c_str()));

    // Set thread count if it is passed in as a parameter. Setting this variable will only impact the current thread
    if(threadCount != 0) {
        omp_set_num_threads(threadCount);
    }

    // Add extra parameters that cant be configured with the index factory
    SetExtraParameters<faiss::IndexBinary, faiss::IndexBinaryIVF, faiss::IndexBinaryHNSW>(jniUtil, env, parameters, indexWriter.get());

    // Check that the index does not need to be trained
    if(!indexWriter->is_trained) {
        throw std::runtime_error("Index is not trained");
    }

    // Add vectors
    std::unique_ptr<faiss::IndexBinaryIDMap> idMap(faissMethods->indexBinaryIdMap(indexWriter.get()));
    idMap->add_with_ids(numVectors, inputVectors->data(), ids.data());

    // Write the index to disk
    faissMethods->writeIndexBinary(idMap.get(), indexPath.c_str());
    delete idMap->index;
}

} // namespace faiss_wrapper
} // namesapce knn_jni
