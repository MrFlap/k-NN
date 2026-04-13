#!/bin/bash
cd /Users/finnrobl/Documents/k-NN-2/k-NN

INDEX_BASE="/Users/finnrobl/Documents/k-NN-2/sift-binary/nodes/0/indices/80darGsMRXy1pVee2ywcvQ"
DOCIDS_BASE="/Users/finnrobl/Documents/k-NN-2/sift-binary/query_doc_ids"
DIMENSION=128

echo "=== Running ReorderDistanceAnalyzer (per shard) ==="

for shard_dir in "$INDEX_BASE"/*/index; do
    shard=$(basename $(dirname "$shard_dir"))
    perm_file="$shard_dir/permutation.txt"
    docids_file="$DOCIDS_BASE/exactsearcher_docids_shard_${shard}.txt"
    
    if [[ -f "$perm_file" && -f "$docids_file" ]]; then
        echo ""
        echo "========== SHARD $shard =========="
        java -cp build/classes/java/main:build/resources/main \
            org.opensearch.knn.memoryoptsearch.faiss.reorder.ReorderDistanceAnalyzer \
            "$perm_file" "$docids_file" "$DIMENSION"
    else
        echo "Shard $shard: missing permutation or docids file, skipping"
    fi
done
