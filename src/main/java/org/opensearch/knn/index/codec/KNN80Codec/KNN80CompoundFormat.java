/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN80Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.backward_codecs.lucene50.Lucene50CompoundFormat;
import org.opensearch.knn.common.KNNConstants;
import org.apache.lucene.codecs.CompoundDirectory;
import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Class to encode/decode compound file.
 */
@Log4j2
public class KNN80CompoundFormat extends CompoundFormat {

    private final CompoundFormat delegate;

    public KNN80CompoundFormat() {
        this.delegate = new Lucene50CompoundFormat();
    }

    public KNN80CompoundFormat(CompoundFormat delegate) {
        this.delegate = delegate;
    }

    @Override
    public CompoundDirectory getCompoundReader(Directory dir, SegmentInfo si) throws IOException {
        return new KNN80CompoundDirectory(delegate.getCompoundReader(dir, si), dir);
    }

    @Override
    public void write(Directory dir, SegmentInfo si, IOContext context) throws IOException {
        for (KNNEngine knnEngine : KNNEngine.getEnginesThatCreateCustomSegmentFiles()) {
            writeEngineFiles(dir, si, context, knnEngine.getExtension());
        }
        // Handle .kcs (cluster summary) files same as engine files — extract from compound
        writeEngineFiles(dir, si, context, ".kcs");
        delegate.write(dir, si, context);
    }

    private void writeEngineFiles(Directory dir, SegmentInfo si, IOContext context, String engineExtension) throws IOException {
        // Find engine files both from si.files() and by scanning directory (flush may not register in si.files())
        Set<String> engineFiles = new HashSet<>();
        for (String file : si.files()) {
            if (file.endsWith(engineExtension)) engineFiles.add(file);
        }
        String prefix = si.name + "_";
        for (String file : dir.listAll()) {
            if (file.startsWith(prefix) && file.endsWith(engineExtension)) engineFiles.add(file);
        }
        Set<String> segmentFiles = new HashSet<>(si.files());

        if (!engineFiles.isEmpty()) {
            for (String engineFile : engineFiles) {
                String engineCompoundFile = engineFile + KNNConstants.COMPOUND_EXTENSION;
                dir.copyFrom(dir, engineFile, engineCompoundFile, context);
            }
            segmentFiles.removeAll(engineFiles);
            si.setFiles(segmentFiles);
        }
    }
}
