package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;

import java.io.IOException;
import java.nio.file.Path;

public class FixedBlockSkipListIndexReaderTest {
    public static void main(String... args) throws IOException {
        final String testDirPath = "/Users/kdooyong/workspace/opensearch-gorder/kdy";
        final String skipListIndex = "skipListIndex.bin";
        final String metaInfoFile = "metaInfo.txt";
        int maxDoc;
        int[] ords;

        try (final Directory directory = new MMapDirectory(Path.of(testDirPath))) {
            try (final IndexInput metaIn = directory.openInput(metaInfoFile, IOContext.DEFAULT)) {
                // #docs
                maxDoc = metaIn.readInt();

                // Ords for validation
                ords = new int[metaIn.readInt()];
                for (int i = 0; i < ords.length; ++i) {
                    ords[i] = metaIn.readInt();
                }
            }

            // Validation
            try (final IndexInput indexInput = directory.openInput(skipListIndex, IOContext.DEFAULT)) {
                final FixedBlockSkipListIndexReader reader = new FixedBlockSkipListIndexReader(indexInput, maxDoc);

                for (int i = 0; i <= maxDoc; ++i) {
                    int doc = reader.skipTo(i);
                    int ord = reader.getOrd();
                    if (ord != ords[doc]) {
                        System.out.println("Doc: " + doc);
                        System.out.println("Ord: " + ord + " vs " + ords[doc]);
                    }
                    assert ord == ords[doc];
                }
            }

            // Performance testing
            try (final IndexInput indexInput = directory.openInput(skipListIndex, IOContext.DEFAULT)) {
                for (int k = 0, loop = 1000; k < loop; ++k) {
                    indexInput.seek(0);
                    final FixedBlockSkipListIndexReader reader = new FixedBlockSkipListIndexReader(indexInput, maxDoc);

                    long s = System.nanoTime();
                    for (int i = 0; i <= maxDoc; i += 1000) {
                        int doc = reader.skipTo(i);
                        int ord = reader.getOrd();
                        assert ord == ords[doc];
                    }
                    long e = System.nanoTime();
                    long took = e - s;
                    if (k == (loop - 1)) {
                        System.out.println("Took: " + took + " nano, " + (took / 1e3) + "micro");
                    }
                }
            }
        }
    }
}
