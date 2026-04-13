#!/usr/bin/env python3
"""
End-to-end reorder test for k-NN plugin.

Tests two configurations:
  1. Standard FAISS HNSW (no compression)
  2. On-disk 32x compression (SQ 1-bit)

For each:
  - Creates an index with reorder_strategy=kmeans
  - Indexes 15k vectors (above MIN_VECTORS_FOR_REORDER=10k)
  - Force merges to 1 segment
  - Snapshots .vec file bytes
  - Queries and records results
  - Compares .vec bytes before/after merge (reorder happens during merge)
  - Verifies search results are correct

Usage:
  python3 test_reorder_e2e.py [--host localhost] [--port 9200]
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.request
import urllib.error

DIMENSION = 32
NUM_VECTORS = 15000
NUM_QUERIES = 5
K = 10

def req(method, url, body=None, expected=(200, 201)):
    """Simple HTTP request helper."""
    data = json.dumps(body).encode() if body else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Content-Type", "application/json")
    try:
        resp = urllib.request.urlopen(r)
        resp_body = resp.read().decode()
        return json.loads(resp_body) if resp_body else {}
    except urllib.error.HTTPError as e:
        resp_body = e.read().decode()
        if e.code not in expected:
            print(f"  ERROR {e.code}: {resp_body[:500]}")
            raise
        return json.loads(resp_body) if resp_body else {}

def get(url):
    resp = urllib.request.urlopen(url)
    return json.loads(resp.read().decode())

def wait_for_cluster(base):
    for _ in range(30):
        try:
            get(f"{base}/_cluster/health")
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("Cluster not ready")


def get_segment_files(base, index_name):
    """Get segment file info including .vec file sizes via _segments API."""
    resp = get(f"{base}/{index_name}/_segments")
    shards = resp["indices"][index_name]["shards"]
    segments = {}
    for shard_id, shard_list in shards.items():
        for shard in shard_list:
            for seg_name, seg_info in shard["segments"].items():
                segments[seg_name] = seg_info
    return segments


def get_vec_file_stats(base, index_name):
    """Get stats about the index files via _cat/segments."""
    resp = get(f"{base}/_cat/segments/{index_name}?format=json&h=segment,shard,size")
    return resp


def snapshot_vec_bytes(base, index_name):
    """
    Snapshot the .vec file content by reading the raw segment stats.
    We use the segment-level stats (size) as a proxy since we can't read
    raw files via the REST API. Instead, we'll use _cat/segments size
    and the node stats file_cache to detect changes.
    """
    # Use node stats to get the data path, then compute a hash of .vec files
    stats = get(f"{base}/_nodes/stats/fs")
    nodes = stats["nodes"]
    data_paths = []
    for node_id, node_info in nodes.items():
        for path_info in node_info["fs"]["data"]:
            data_paths.append(path_info["path"])
    return data_paths


def hash_vec_files_for_index(base, data_paths, index_name):
    """Hash all .vec and .cfs files for a specific index by resolving its UUID."""
    resp = get(f"{base}/{index_name}/_settings")
    uuid = resp[index_name]["settings"]["index"]["uuid"]

    hashes = {}
    for data_path in data_paths:
        index_dir = os.path.join(data_path, "indices", uuid, "0", "index")
        if not os.path.isdir(index_dir):
            continue
        for f in os.listdir(index_dir):
            # Hash .vec files (non-compound) or .cfs files (compound)
            if f.endswith(".vec") or f.endswith(".cfs"):
                fpath = os.path.join(index_dir, f)
                h = hashlib.sha256()
                with open(fpath, "rb") as fh:
                    h.update(fh.read())
                hashes[f] = h.hexdigest()
    return hashes


def create_index(base, index_name, use_sq_1bit=False, reorder_strategy="none"):
    """Create a k-NN index with the given configuration."""
    # Delete if exists
    try:
        req("DELETE", f"{base}/{index_name}", expected=(200, 404))
    except Exception:
        pass

    if use_sq_1bit:
        # On-disk 32x compression with SQ 1-bit
        mapping = {
            "settings": {
                "index": {
                    "knn": True,
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "knn.advanced.reorder_strategy": reorder_strategy,
                }
            },
            "mappings": {
                "properties": {
                    "vec": {
                        "type": "knn_vector",
                        "dimension": DIMENSION,
                        "mode": "on_disk",
                        "compression_level": "32x",
                        "method": {
                            "name": "hnsw",
                            "engine": "faiss",
                            "parameters": {
                                "m": 16,
                                "ef_construction": 100,
                                "encoder": {
                                    "name": "sq",
                                    "parameters": {"bits": 1}
                                }
                            }
                        }
                    }
                }
            }
        }
    else:
        # Standard FAISS HNSW (no compression)
        mapping = {
            "settings": {
                "index": {
                    "knn": True,
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "knn.advanced.reorder_strategy": reorder_strategy,
                }
            },
            "mappings": {
                "properties": {
                    "vec": {
                        "type": "knn_vector",
                        "dimension": DIMENSION,
                        "method": {
                            "name": "hnsw",
                            "engine": "faiss",
                            "parameters": {"m": 16, "ef_construction": 100}
                        }
                    }
                }
            }
        }

    resp = req("PUT", f"{base}/{index_name}", mapping)
    print(f"  Created index: {resp.get('acknowledged', False)}")


def bulk_index(base, index_name, vectors):
    """Bulk index vectors in batches."""
    batch_size = 1000
    for start in range(0, len(vectors), batch_size):
        end = min(start + batch_size, len(vectors))
        lines = []
        for i in range(start, end):
            lines.append(json.dumps({"index": {"_id": str(i)}}))
            lines.append(json.dumps({"vec": vectors[i]}))
        body = "\n".join(lines) + "\n"
        r = urllib.request.Request(
            f"{base}/{index_name}/_bulk",
            data=body.encode(),
            method="POST"
        )
        r.add_header("Content-Type", "application/x-ndjson")
        resp = urllib.request.urlopen(r)
        result = json.loads(resp.read().decode())
        if result.get("errors"):
            print(f"  Bulk errors at batch {start}-{end}")
            for item in result["items"]:
                if "error" in item.get("index", {}):
                    print(f"    {item['index']['error']}")
                    break
            return False
    return True


def force_merge(base, index_name):
    """Force merge to 1 segment."""
    req("POST", f"{base}/{index_name}/_forcemerge?max_num_segments=1&flush=true",
        expected=(200,))
    # Wait for merge to complete
    time.sleep(2)
    req("POST", f"{base}/{index_name}/_refresh", expected=(200,))


def search_knn(base, index_name, query_vec, k=K):
    """Run a k-NN search and return results."""
    body = {
        "size": k,
        "query": {
            "knn": {
                "vec": {
                    "vector": query_vec,
                    "k": k
                }
            }
        }
    }
    resp = req("POST", f"{base}/{index_name}/_search", body)
    hits = resp["hits"]["hits"]
    return [(h["_id"], h["_score"]) for h in hits]


def run_test(base, test_name, use_sq_1bit=False):
    """
    Run the reorder test for a given configuration.

    Creates two indices:
      - {test_name}_noreorder: baseline without reorder
      - {test_name}_reorder: with reorder_strategy=kmeans

    Both get the same vectors. After force merge (which triggers reorder),
    we verify:
      1. .vec file hashes differ between the two indices
      2. Search results are identical (same doc IDs, same scores)
    """
    idx_no = f"{test_name}_noreorder"
    idx_re = f"{test_name}_reorder"

    print(f"\n{'='*60}")
    print(f"TEST: {test_name} (sq_1bit={use_sq_1bit})")
    print(f"{'='*60}")

    # Generate deterministic vectors
    random.seed(42)
    vectors = [[random.gauss(0, 1) for _ in range(DIMENSION)] for _ in range(NUM_VECTORS)]

    # Generate query vectors
    random.seed(123)
    queries = [[random.gauss(0, 1) for _ in range(DIMENSION)] for _ in range(NUM_QUERIES)]

    # Get data paths for file hashing
    data_paths = snapshot_vec_bytes(base, idx_no)

    # --- Create and populate baseline (no reorder) ---
    print(f"\n[1] Creating baseline index: {idx_no}")
    create_index(base, idx_no, use_sq_1bit=use_sq_1bit, reorder_strategy="none")
    print(f"  Indexing {NUM_VECTORS} vectors...")
    bulk_index(base, idx_no, vectors)
    req("POST", f"{base}/{idx_no}/_refresh", expected=(200,))

    print(f"  Force merging to 1 segment...")
    force_merge(base, idx_no)

    # Hash .vec files for baseline
    hashes_no = hash_vec_files_for_index(base, data_paths, idx_no)
    print(f"  .vec file hashes (no reorder): {hashes_no}")

    # Search baseline
    print(f"  Running {NUM_QUERIES} queries...")
    results_no = []
    for i, q in enumerate(queries):
        results_no.append(search_knn(base, idx_no, q))

    # --- Create and populate reorder index ---
    print(f"\n[2] Creating reorder index: {idx_re}")
    create_index(base, idx_re, use_sq_1bit=use_sq_1bit, reorder_strategy="kmeans")
    print(f"  Indexing {NUM_VECTORS} vectors...")
    bulk_index(base, idx_re, vectors)
    req("POST", f"{base}/{idx_re}/_refresh", expected=(200,))

    print(f"  Force merging to 1 segment (triggers reorder)...")
    force_merge(base, idx_re)

    # Hash .vec files for reorder
    hashes_re = hash_vec_files_for_index(base, data_paths, idx_re)
    print(f"  .vec file hashes (reorder):    {hashes_re}")

    # Search reorder
    print(f"  Running {NUM_QUERIES} queries...")
    results_re = []
    for i, q in enumerate(queries):
        results_re.append(search_knn(base, idx_re, q))

    # --- Verify results ---
    print(f"\n[3] Verification")

    # Check .vec files differ
    vec_files_differ = False
    if hashes_no and hashes_re:
        no_values = set(hashes_no.values())
        re_values = set(hashes_re.values())
        vec_files_differ = no_values != re_values
    print(f"  .vec files differ: {vec_files_differ}")

    # Check search results match
    results_match = True
    for i in range(NUM_QUERIES):
        ids_no = set(r[0] for r in results_no[i])
        ids_re = set(r[0] for r in results_re[i])
        if ids_no != ids_re:
            results_match = False
            print(f"  Query {i}: IDs MISMATCH")
            print(f"    no_reorder: {sorted(ids_no)}")
            print(f"    reorder:    {sorted(ids_re)}")
        else:
            # Check scores match (within tolerance for float rounding)
            scores_no = sorted(results_no[i], key=lambda x: x[0])
            scores_re = sorted(results_re[i], key=lambda x: x[0])
            for (id_n, s_n), (id_r, s_r) in zip(scores_no, scores_re):
                if abs(s_n - s_r) > 1e-4:
                    results_match = False
                    print(f"  Query {i}: Score mismatch for doc {id_n}: {s_n} vs {s_r}")
                    break

    print(f"  Search results match: {results_match}")

    # Summary
    passed = vec_files_differ and results_match
    status = "PASS" if passed else "FAIL"
    print(f"\n  [{status}] {test_name}")
    if not vec_files_differ:
        print(f"    FAIL: .vec files should differ after reorder")
    if not results_match:
        print(f"    FAIL: Search results should match between reorder and no-reorder")

    # Cleanup
    try:
        req("DELETE", f"{base}/{idx_no}", expected=(200, 404))
        req("DELETE", f"{base}/{idx_re}", expected=(200, 404))
    except Exception:
        pass

    return passed


def main():
    parser = argparse.ArgumentParser(description="End-to-end reorder test")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=9200)
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    print(f"Connecting to {base}...")
    wait_for_cluster(base)
    print("Cluster ready.")

    results = {}

    # Test 1: Standard FAISS HNSW
    results["standard_faiss"] = run_test(base, "reorder_std", use_sq_1bit=False)

    # Test 2: On-disk 32x compression (SQ 1-bit)
    results["sq_32x"] = run_test(base, "reorder_sq32x", use_sq_1bit=True)

    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
