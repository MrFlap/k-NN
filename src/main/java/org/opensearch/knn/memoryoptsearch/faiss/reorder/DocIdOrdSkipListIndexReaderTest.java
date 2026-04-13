package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;

import java.io.IOException;
import java.nio.file.Path;

public class DocIdOrdSkipListIndexReaderTest {
    public static void main(String... args) throws IOException {
        final boolean isDense = true;
        final int numLevel = 4;
        final int numDocsForGrouping = 256;
        final int groupFactor = 8;
        final String testDirPath = "/Users/kdooyong/workspace/opensearch-gorder/kdy";
        final String skipListIndex = "skipListIndex.bin";
        final String metaInfoFile = "metaInfo.txt";
        int maxDoc;
        long[] offsets;
        long skipListStartOffset;
        int[] ords;

        try (final Directory directory = new MMapDirectory(Path.of(testDirPath))) {
            try (final IndexInput metaIn = directory.openInput(metaInfoFile, IOContext.DEFAULT)) {
                // #docs
                maxDoc = metaIn.readInt() - 1;

                // Offsets
                offsets = new long[metaIn.readInt()];
                metaIn.readLongs(offsets, 0, offsets.length);

                // Skip list start offset
                skipListStartOffset = metaIn.readLong();

                // Ords for validation
                ords = new int[metaIn.readInt()];
                for (int i = 0; i < ords.length; ++i) {
                    ords[i] = metaIn.readInt();
                }
            }

            try (final IndexInput indexInput = directory.openInput(skipListIndex, IOContext.DEFAULT)) {
                final DocIdOrdSkipListIndex skipReader = new DocIdOrdSkipListIndex(
                    indexInput,
                    isDense,
                    numLevel,
                    numDocsForGrouping,
                    groupFactor,
                    offsets,
                    maxDoc
                );

                final DocIdOrdSkipListIndex.Reader skipper = skipReader.createReader();

                for (int i = 0; i <= maxDoc; ++i) {
                    int doc = skipper.skipTo(i);
                    int ord = skipper.getOrd();
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
                    final DocIdOrdSkipListIndex skipReader = new DocIdOrdSkipListIndex(
                        indexInput,
                        isDense,
                        numLevel,
                        numDocsForGrouping,
                        groupFactor,
                        offsets,
                        maxDoc
                    );

                    final DocIdOrdSkipListIndex.Reader skipper = skipReader.createReader();

                    long s = System.nanoTime();
                    for (int i = 0; i <= maxDoc; i += 1000) {
                        int doc = skipper.skipTo(i);
                        int ord = skipper.getOrd();
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
