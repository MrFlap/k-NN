/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;

import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.concurrent.ThreadLocalRandom;

public class DocIdOrdSkipListIndexBuilderTest {
    public static void main(String... args) throws IOException {
        final boolean isDense = true;
        final int numLevel = 4;
        final int numDocsForGrouping = 256;
        final int groupFactor = 8;
        final String testDirPath = "/Users/kdooyong/workspace/opensearch-gorder/kdy";
        final String skipListIndex = "skipListIndex.bin";
        final String metaInfoFile = "metaInfo.txt";

        cleanDirectory(testDirPath);

        try (final Directory directory = new MMapDirectory(Path.of(testDirPath))) {
            try (final IndexOutput indexOutput = directory.createOutput(skipListIndex, IOContext.DEFAULT)) {
                final DocIdOrdSkipListIndexBuilder builder = new DocIdOrdSkipListIndexBuilder(
                    isDense,
                    numLevel,
                    numDocsForGrouping,
                    groupFactor,
                    indexOutput
                );

                int N = 256 * 4000 * 5 + 77;
                int[] ords = new int[N];
                for (int i = 0; i < N; ++i) {
                    ords[i] = i;
                }

                // Shuffle using Fisher-Yates
                for (int i = N - 1; i > 0; i--) {
                    int j = ThreadLocalRandom.current().nextInt(i + 1);
                    int temp = ords[i];
                    ords[i] = ords[j];
                    ords[j] = temp;
                }

                for (int doc = 0; doc < N; ++doc) {
                    builder.add(doc, ords[doc]);
                }

                final long[] offsets = builder.finish();
                try (final IndexOutput metaOut = directory.createOutput(metaInfoFile, IOContext.DEFAULT)) {
                    // #docs
                    metaOut.writeInt(N);

                    // Offsets
                    metaOut.writeInt(offsets.length);
                    for (long offset : offsets) {
                        metaOut.writeLong(offset);
                    }

                    // Level-0 end offset
                    metaOut.writeLong(builder.getLevel0EndOffset());

                    // Ords
                    metaOut.writeInt(ords.length);
                    for (int ord : ords) {
                        metaOut.writeInt(ord);
                    }
                }
            }
        }
    }

    private static void cleanDirectory(final String testDirPath) throws IOException {
        Path root = Paths.get(testDirPath);

        if (!Files.exists(root)) {
            System.out.println("Path does not exist: " + root);
            return;
        }

        Files.walkFileTree(root, new SimpleFileVisitor<>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                Files.delete(file);
                return FileVisitResult.CONTINUE;
            }

            @Override
            public FileVisitResult postVisitDirectory(Path dir, IOException exc) throws IOException {
                if (!dir.equals(root)) {
                    Files.delete(dir);
                }
                return FileVisitResult.CONTINUE;
            }
        });
    }
}
