/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstring>
#include <limits>
#include <algorithm>
#include "org_opensearch_knn_jni_SimdVectorComputeService.h"
#include "jni_util.h"
#include "simd/similarity_function/similarity_function.h"
#include "simd/similarity_function/clusterann_rabitq.h"

static knn_jni::JNIUtil JNI_UTIL;
static constexpr jint KNN_SIMD_COMPUTE_JNI_VERSION = JNI_VERSION_1_1;

using knn_jni::simd::similarity_function::SimilarityFunction;
using knn_jni::simd::similarity_function::SimdVectorSearchContext;

jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    // Obtain the JNIEnv from the VM and confirm JNI_VERSION
    JNIEnv* env;
    if (vm->GetEnv((void**)&env, KNN_SIMD_COMPUTE_JNI_VERSION) != JNI_OK) {
        return JNI_ERR;
    }

    JNI_UTIL.Initialize(env, vm);

    return KNN_SIMD_COMPUTE_JNI_VERSION;
}

void JNI_OnUnload(JavaVM *vm, void *reserved) {
    JNIEnv* env;
    vm->GetEnv((void**)&env, KNN_SIMD_COMPUTE_JNI_VERSION);
    JNI_UTIL.Uninitialize(env);
}

JNIEXPORT jfloat JNICALL Java_org_opensearch_knn_jni_SimdVectorComputeService_scoreSimilarityInBulk
  (JNIEnv *env, jclass clazz, jintArray internalVectorIds, jfloatArray jscores, const jint numVectors) {
    if (numVectors <= 0) {
      return std::numeric_limits<float>::min();
    }

    try {
      // Get search context
      SimdVectorSearchContext* srchContext = SimilarityFunction::getSearchContext();
      if (srchContext == nullptr || srchContext->similarityFunction == nullptr) {
          throw std::runtime_error("No search context has been initialized, SimdVectorSearchContext* was empty.");
      }

      // Get pointers of vectorIds and scores
      jint* vectorIds = static_cast<jint*>(JNI_UTIL.GetPrimitiveArrayCritical(env, internalVectorIds, nullptr));
      knn_jni::JNIReleaseElements releaseVectorIds {[=]{
        JNI_UTIL.ReleasePrimitiveArrayCritical(env, internalVectorIds, vectorIds, 0);
      }};

      jfloat* scores = static_cast<jfloat*>(JNI_UTIL.GetPrimitiveArrayCritical(env, jscores, nullptr));
      knn_jni::JNIReleaseElements releaseScores {[=]{
        JNI_UTIL.ReleasePrimitiveArrayCritical(env, jscores, scores, 0);
      }};

      // Bulk similarity calculation
      srchContext->similarityFunction->calculateSimilarityInBulk(
          srchContext,
          reinterpret_cast<int32_t*>(vectorIds),
          reinterpret_cast<float*>(scores),
          numVectors);

      jfloat maxScore = *std::max_element(scores, scores + numVectors);
      return maxScore;
    } catch (...) {
      JNI_UTIL.CatchCppExceptionAndThrowJava(env);
      return 0.0f;   // value ignored if exception pending
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_SimdVectorComputeService_saveSearchContext
  (JNIEnv *env, jclass clazz, jfloatArray query, jlongArray addressAndSize, const jint nativeFunctionTypeOrd) {
    try {
      // Get raw pointer of query vector + size
      const jsize queryVecSize = JNI_UTIL.GetJavaFloatArrayLength(env, query);
      jfloat* queryVecPtr = static_cast<jfloat*>(JNI_UTIL.GetPrimitiveArrayCritical(env, query, nullptr));

      // Get mmap address and size
      const jsize mmapAddressAndSizeLength = JNI_UTIL.GetJavaLongArrayLength(env, addressAndSize);
      jlong* mmapAddressAndSize = static_cast<jlong*>(JNI_UTIL.GetPrimitiveArrayCritical(env, addressAndSize, nullptr));

      // Save search context
      SimilarityFunction::saveSearchContext(
          (uint8_t*) queryVecPtr, sizeof(jfloat) * queryVecSize,
          queryVecSize,
          (int64_t*) mmapAddressAndSize, mmapAddressAndSizeLength,
          nativeFunctionTypeOrd, /*docBits=*/1 /* unused by FP16 / BF16 */);

      // Release query vector
      JNI_UTIL.ReleasePrimitiveArrayCritical(env, query, queryVecPtr, 0);
      JNI_UTIL.ReleasePrimitiveArrayCritical(env, addressAndSize, mmapAddressAndSize, 0);
    } catch (...) {
      JNI_UTIL.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT jfloat JNICALL Java_org_opensearch_knn_jni_SimdVectorComputeService_scoreSimilarity
  (JNIEnv *env, jclass clazz, const jint internalVectorId) {

    try {
      // Get search context
      SimdVectorSearchContext* srchContext = SimilarityFunction::getSearchContext();

      // Single vector similarity calculation.
      return srchContext->similarityFunction->calculateSimilarity(srchContext, internalVectorId);
    } catch (...) {
      JNI_UTIL.CatchCppExceptionAndThrowJava(env);
    }

    return 0;
}

JNIEXPORT jfloat JNICALL Java_org_opensearch_knn_jni_SimdVectorComputeService_scoreSimilarityInBulkFromFp16Bytes
  (JNIEnv *env, jclass clazz, jbyteArray fp16Vectors, const jint numVectors, jintArray internalVectorIds, jfloatArray jscores) {
    if (numVectors <= 0) {
      return -std::numeric_limits<float>::infinity();
    }

    try {
      // The query buffer and similarity function were already set up once by saveSearchContext,
      // called from the scorer's constructor. Only the vector chunk location needs to be refreshed
      // here, since (unlike a long-lived mmap region) this Java byte[] is only pinned for this call.
      SimdVectorSearchContext* srchContext = SimilarityFunction::getSearchContext();
      if (srchContext == nullptr || srchContext->similarityFunction == nullptr) {
          throw std::runtime_error("No search context has been initialized, SimdVectorSearchContext* was empty.");
      }

      jint* vectorIds = JNI_UTIL.GetIntArrayElements(env, internalVectorIds, nullptr);
      knn_jni::JNIReleaseElements releaseVectorIds {[=]{
        JNI_UTIL.ReleaseIntArrayElements(env, internalVectorIds, vectorIds, 0);
      }};

      jfloat* scores = JNI_UTIL.GetFloatArrayElements(env, jscores, nullptr);
      knn_jni::JNIReleaseElements releaseScores {[=]{
        JNI_UTIL.ReleaseFloatArrayElements(env, jscores, scores, 0);
      }};

      // Get raw pointer of `numVectors` FP16-encoded vectors packed contiguously in a plain Java
      // byte[]. SimdVectorSearchContext addresses vectors purely by (base pointer, byte size), so
      // this single pinned array works as a one-chunk region holding all `numVectors` vectors.
      const jsize vectorsByteSize = JNI_UTIL.GetJavaBytesArrayLength(env, fp16Vectors);
      jbyte* vectorsPtr = static_cast<jbyte*>(JNI_UTIL.GetPrimitiveArrayCritical(env, fp16Vectors, nullptr));
      knn_jni::JNIReleaseElements releaseVectors {[=]{
        JNI_UTIL.ReleasePrimitiveArrayCritical(env, fp16Vectors, vectorsPtr, JNI_ABORT);
      }};

      SimilarityFunction::updateVectorChunk(reinterpret_cast<uint8_t*>(vectorsPtr), vectorsByteSize);

      srchContext->similarityFunction->calculateSimilarityInBulk(
          srchContext,
          reinterpret_cast<int32_t*>(vectorIds),
          reinterpret_cast<float*>(scores),
          numVectors);

      return *std::max_element(scores, scores + numVectors);
    } catch (...) {
      JNI_UTIL.CatchCppExceptionAndThrowJava(env);
    }

    return 0.0f;
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_SimdVectorComputeService_saveSQSearchContext
  (JNIEnv *env, jclass clazz, jbyteArray quantizedQuery,
   jfloat lowerInterval, jfloat upperInterval, jfloat additionalCorrection,
   jint quantizedComponentSum, jlongArray addressAndSize,
   jint functionTypeOrd, jint dimension, jfloat centroidDp, jint docBits) {
    try {
      // Get quantized query bytes
      const jsize queryByteSize = JNI_UTIL.GetJavaBytesArrayLength(env, quantizedQuery);
      jbyte* queryPtr = static_cast<jbyte*>(JNI_UTIL.GetPrimitiveArrayCritical(env, quantizedQuery, nullptr));
      knn_jni::JNIReleaseElements queryRelease {[=]{
        JNI_UTIL.ReleasePrimitiveArrayCritical(env, quantizedQuery, queryPtr, 0);
      }};

      // Get mmap address and size
      const jsize mmapAddressAndSizeLength = JNI_UTIL.GetJavaLongArrayLength(env, addressAndSize);
      jlong* mmapAddressAndSize = static_cast<jlong*>(JNI_UTIL.GetPrimitiveArrayCritical(env, addressAndSize, nullptr));
      knn_jni::JNIReleaseElements mmapAddressAndSizeRelease {[=]{
        JNI_UTIL.ReleasePrimitiveArrayCritical(env, addressAndSize, mmapAddressAndSize, 0);
      }};

      // Store correction factors in tmpBuffer before calling saveSearchContext.
      // saveSearchContext will reset tmpBuffer at the beginning, so we need to call it first,
      // then write correction factors after.
      SimilarityFunction::saveSearchContext(
          reinterpret_cast<uint8_t*>(queryPtr), queryByteSize,
          dimension,
          reinterpret_cast<int64_t*>(mmapAddressAndSize), mmapAddressAndSizeLength,
          functionTypeOrd, docBits);

      // Now store correction factors in tmpBuffer (saveSearchContext clears it, then SQ_IP branch leaves it empty)
      SimdVectorSearchContext* ctx = SimilarityFunction::getSearchContext();
      ctx->tmpBuffer.resize(5 * sizeof(float));
      auto* correctionPtr = reinterpret_cast<float*>(ctx->tmpBuffer.data());
      correctionPtr[0] = lowerInterval;
      correctionPtr[1] = upperInterval;
      correctionPtr[2] = additionalCorrection;
      std::memcpy(&correctionPtr[3], &quantizedComponentSum, sizeof(int32_t));
      correctionPtr[4] = centroidDp;
    } catch (...) {
      JNI_UTIL.CatchCppExceptionAndThrowJava(env);
    }
}

// ClusterANN tiered RaBitQ: one call per cluster. All lengths are checked before any array is pinned; the kernels make
// no JNI calls, so the critical regions hold only the arithmetic.
JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_SimdVectorComputeService_rabitqCoarseDots
  (JNIEnv *env, jclass clazz, jbyteArray jplanes, jbyteArray jbits, const jint bitBytes, const jint n, jintArray jout) {
    if (n <= 0) {
      return;
    }
    try {
      if (bitBytes <= 0) {
          throw std::runtime_error("bitBytes must be positive, got " + std::to_string(bitBytes));
      }
      if (JNI_UTIL.GetJavaBytesArrayLength(env, jplanes) < 4LL * bitBytes) {
          throw std::runtime_error("query planes need " + std::to_string(4LL * bitBytes) + " bytes");
      }
      if (JNI_UTIL.GetJavaBytesArrayLength(env, jbits) < static_cast<int64_t>(n) * bitBytes) {
          throw std::runtime_error("bits need " + std::to_string(static_cast<int64_t>(n) * bitBytes) + " bytes");
      }
      if (JNI_UTIL.GetJavaIntArrayLength(env, jout) < n) {
          throw std::runtime_error("out needs " + std::to_string(n) + " ints");
      }
      auto* planes = static_cast<uint8_t*>(JNI_UTIL.GetPrimitiveArrayCritical(env, jplanes, nullptr));
      if (planes == nullptr) {
          return;
      }
      knn_jni::JNIReleaseElements releasePlanes {[=]{ JNI_UTIL.ReleasePrimitiveArrayCritical(env, jplanes, planes, JNI_ABORT); }};
      auto* bits = static_cast<uint8_t*>(JNI_UTIL.GetPrimitiveArrayCritical(env, jbits, nullptr));
      if (bits == nullptr) {
          return;
      }
      knn_jni::JNIReleaseElements releaseBits {[=]{ JNI_UTIL.ReleasePrimitiveArrayCritical(env, jbits, bits, JNI_ABORT); }};
      auto* out = static_cast<int32_t*>(JNI_UTIL.GetPrimitiveArrayCritical(env, jout, nullptr));
      if (out == nullptr) {
          return;
      }
      knn_jni::JNIReleaseElements releaseOut {[=]{ JNI_UTIL.ReleasePrimitiveArrayCritical(env, jout, out, 0); }};
      knn_jni::simd::clusterann::rabitqCoarseDots(planes, bits, bitBytes, n, out);
    } catch (...) {
      JNI_UTIL.CatchCppExceptionAndThrowJava(env);
    }
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_SimdVectorComputeService_rabitqRefineDots
  (JNIEnv *env, jclass clazz, jbyteArray jrecs, const jint stride, const jint m, const jint dim, jbyteArray jq,
   jintArray jout) {
    if (m <= 0) {
      return;
    }
    try {
      if (dim <= 0 || stride < dim) {
          throw std::runtime_error("need 0 < dim <= stride, got dim=" + std::to_string(dim)
                                   + " stride=" + std::to_string(stride));
      }
      if (JNI_UTIL.GetJavaBytesArrayLength(env, jrecs) < static_cast<int64_t>(m - 1) * stride + dim) {
          throw std::runtime_error("records too short for " + std::to_string(m) + " rows");
      }
      if (JNI_UTIL.GetJavaBytesArrayLength(env, jq) < dim) {
          throw std::runtime_error("query needs " + std::to_string(dim) + " bytes");
      }
      if (JNI_UTIL.GetJavaIntArrayLength(env, jout) < m) {
          throw std::runtime_error("out needs " + std::to_string(m) + " ints");
      }
      auto* recs = static_cast<int8_t*>(JNI_UTIL.GetPrimitiveArrayCritical(env, jrecs, nullptr));
      if (recs == nullptr) {
          return;
      }
      knn_jni::JNIReleaseElements releaseRecs {[=]{ JNI_UTIL.ReleasePrimitiveArrayCritical(env, jrecs, recs, JNI_ABORT); }};
      auto* q = static_cast<int8_t*>(JNI_UTIL.GetPrimitiveArrayCritical(env, jq, nullptr));
      if (q == nullptr) {
          return;
      }
      knn_jni::JNIReleaseElements releaseQ {[=]{ JNI_UTIL.ReleasePrimitiveArrayCritical(env, jq, q, JNI_ABORT); }};
      auto* out = static_cast<int32_t*>(JNI_UTIL.GetPrimitiveArrayCritical(env, jout, nullptr));
      if (out == nullptr) {
          return;
      }
      knn_jni::JNIReleaseElements releaseOut {[=]{ JNI_UTIL.ReleasePrimitiveArrayCritical(env, jout, out, 0); }};
      knn_jni::simd::clusterann::rabitqRefineDots(recs, stride, m, dim, q, out);
    } catch (...) {
      JNI_UTIL.CatchCppExceptionAndThrowJava(env);
    }
}
