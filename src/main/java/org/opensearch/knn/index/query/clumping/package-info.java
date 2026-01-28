/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Clumping-based search optimization for k-NN.
 * 
 * <h2>Overview</h2>
 * Clumping is a technique to reduce memory usage by storing only a fraction of vectors
 * (called "marker" vectors) in the main index. The remaining vectors ("hidden" vectors)
 * are stored separately on disk and are associated with their nearest marker vector.
 * 
 * <h2>How it works</h2>
 * <ol>
 *   <li><b>Indexing:</b> When vectors are indexed with clumping enabled:
 *     <ul>
 *       <li>1/clumpingFactor vectors are randomly selected as markers</li>
 *       <li>Each hidden vector is assigned to its nearest marker</li>
 *       <li>Markers go into the main index (FAISS/NMSLIB/Lucene)</li>
 *       <li>Hidden vectors are stored in a separate .clump file</li>
 *     </ul>
 *   </li>
 *   <li><b>Search:</b> When a k-NN query is executed:
 *     <ul>
 *       <li>k-NN search is performed on marker vectors only</li>
 *       <li>For each marker found, its associated hidden vectors are retrieved</li>
 *       <li>All vectors (markers + hidden) are scored against the query</li>
 *       <li>Top-k results are returned from the combined set</li>
 *     </ul>
 *   </li>
 * </ol>
 * 
 * <h2>Key Classes</h2>
 * <ul>
 *   <li>{@link org.opensearch.knn.index.query.clumping.ClumpingContext} - Configuration for clumping</li>
 *   <li>{@link org.opensearch.knn.index.query.clumping.ClumpingVectorStore} - Stores/retrieves hidden vectors</li>
 *   <li>{@link org.opensearch.knn.index.query.clumping.ClumpingSearchHandler} - Handles search expansion</li>
 *   <li>{@link org.opensearch.knn.index.query.clumping.MarkerVectorSelector} - Selects markers during indexing</li>
 *   <li>{@link org.opensearch.knn.index.query.clumping.HiddenVectorMapping} - Maps markers to hidden vectors</li>
 * </ul>
 * 
 * <h2>Usage Example</h2>
 * <pre>
 * // In a k-NN query:
 * {
 *   "knn": {
 *     "my_vector": {
 *       "vector": [1.0, 2.0, 3.0],
 *       "k": 10,
 *       "clumping": {
 *         "enabled": true,
 *         "clumping_factor": 8
 *       }
 *     }
 *   }
 * }
 * </pre>
 * 
 * <h2>Trade-offs</h2>
 * <ul>
 *   <li><b>Memory:</b> Reduces index memory by ~(1 - 1/clumpingFactor)</li>
 *   <li><b>Recall:</b> May reduce recall since hidden vectors aren't directly searchable</li>
 *   <li><b>Latency:</b> Adds disk I/O for hidden vector retrieval during search</li>
 * </ul>
 * 
 * @see org.opensearch.knn.index.query.rescore.RescoreContext for similar oversampling approach
 */
package org.opensearch.knn.index.query.clumping;
