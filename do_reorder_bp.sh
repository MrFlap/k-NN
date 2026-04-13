#!/bin/bash

java -cp \
        "/home/ec2-user/k-NN-gorder/build/testclusters/integTest-0/distro/3.5.0-ARCHIVE/lib/*:/home/ec2-user/k-NN-gorder/build/testclusters/integTest-0/distro/3.5.0-ARCHIVE/plugins/opensearch-knn/*" \
          org.opensearch.knn.memoryoptsearch.faiss.reorder.ReorderAllWithBP
