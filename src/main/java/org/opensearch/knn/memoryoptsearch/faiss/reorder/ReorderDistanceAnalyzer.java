/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.DoubleSummaryStatistics;
import java.util.List;

/**
 * Utility to analyze the distance between internal vector ordinals before and after reordering.
 */
public class ReorderDistanceAnalyzer {

    private static final int PAGE_SIZE_BYTES = 4096;  // 4KB page
    private static final int READ_AHEAD_BYTES = 262144;  // 256KB read-ahead

    public record Stats(double mean, double stdDev, long count, double min, double max) {
        @Override
        public String toString() {
            return String.format("count=%d, mean=%.2f, stdDev=%.2f, min=%.0f, max=%.0f", count, mean, stdDev, min, max);
        }
    }

    public static int extractShardFromPath(Path path) {
        String pathStr = path.toString();
        int indexPos = pathStr.lastIndexOf("/index/");
        if (indexPos > 0) {
            int slashBefore = pathStr.lastIndexOf('/', indexPos - 1);
            if (slashBefore >= 0) {
                try {
                    return Integer.parseInt(pathStr.substring(slashBefore + 1, indexPos));
                } catch (NumberFormatException e) {
                    return -1;
                }
            }
        }
        return -1;
    }

    public static void savePermutation(int[] newOrd2Old, Path outputPath, int shardId) throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(outputPath)) {
            writer.write("# shard=" + shardId + " timestamp=" + Instant.now().toString());
            writer.newLine();
            for (int i = 0; i < newOrd2Old.length; i++) {
                writer.write(Integer.toString(newOrd2Old[i]));
                writer.newLine();
            }
        }
    }

    public static int[] loadPermutation(Path inputPath) throws IOException {
        try (BufferedReader reader = Files.newBufferedReader(inputPath)) {
            return reader.lines()
                .filter(line -> !line.startsWith("#"))
                .mapToInt(Integer::parseInt)
                .toArray();
        }
    }

    public static Stats computeDisplacementStats(int[] newOrd2Old) {
        if (newOrd2Old == null || newOrd2Old.length == 0) {
            return new Stats(0, 0, 0, 0, 0);
        }
        double[] displacements = new double[newOrd2Old.length];
        for (int newOrd = 0; newOrd < newOrd2Old.length; newOrd++) {
            displacements[newOrd] = Math.abs(newOrd - newOrd2Old[newOrd]);
        }
        DoubleSummaryStatistics summary = java.util.Arrays.stream(displacements).summaryStatistics();
        double mean = summary.getAverage();
        double variance = java.util.Arrays.stream(displacements)
            .map(d -> (d - mean) * (d - mean))
            .average().orElse(0);
        return new Stats(mean, Math.sqrt(variance), summary.getCount(), summary.getMin(), summary.getMax());
    }

    public static Stats computeBaselineStats(int numVectors) {
        return new Stats(0, 0, numVectors, 0, 0);
    }

    public static String compareStats(Stats reordered, Stats baseline) {
        StringBuilder sb = new StringBuilder();
        sb.append("=== Reorder Distance Analysis ===\n");
        sb.append(String.format("Baseline (1:1):  %s%n", baseline));
        sb.append(String.format("Reordered:       %s%n", reordered));
        sb.append(String.format("Mean increase:   %.2f%n", reordered.mean - baseline.mean));
        sb.append(String.format("StdDev increase: %.2f%n", reordered.stdDev - baseline.stdDev));
        return sb.toString();
    }

    /**
     * Computes paired t-test: t = mean(diff) / (std(diff) / sqrt(n))
     */
    public static double computePairedTTest(double[] baseline, double[] reordered) {
        if (baseline.length != reordered.length || baseline.length == 0) return Double.NaN;
        int n = baseline.length;
        double[] diff = new double[n];
        double sumDiff = 0;
        for (int i = 0; i < n; i++) {
            diff[i] = reordered[i] - baseline[i];
            sumDiff += diff[i];
        }
        double meanDiff = sumDiff / n;
        double sumSqDev = 0;
        for (int i = 0; i < n; i++) {
            sumSqDev += (diff[i] - meanDiff) * (diff[i] - meanDiff);
        }
        double stdDiff = Math.sqrt(sumSqDev / (n - 1));
        return meanDiff / (stdDiff / Math.sqrt(n));
    }

    /**
     * Computes number of distinct pages touched by ordinals.
     * @param vectorBytes size of one vector in bytes (dimension * 4)
     */
    public static double computePageFaults(List<Integer> ordinals, int vectorBytes) {
        if (ordinals.isEmpty()) return 0;
        return ordinals.stream()
            .map(ord -> (ord * vectorBytes) / PAGE_SIZE_BYTES)
            .distinct()
            .count();
    }

    /**
     * Computes disk IOs needed with SSD read-ahead.
     * Sorts ordinals, then simulates sequential access where each IO reads readAheadBytes.
     * @param vectorBytes size of one vector in bytes (dimension * 4)
     */
    public static double computeDiskIOs(List<Integer> ordinals, int vectorBytes) {
        if (ordinals.isEmpty()) return 0;
        int[] sorted = ordinals.stream().mapToInt(Integer::intValue).sorted().toArray();
        int ios = 0;
        long cachedUpToByte = -1;
        for (int ord : sorted) {
            long byteOffset = (long) ord * vectorBytes;
            if (byteOffset > cachedUpToByte) {
                ios++;
                cachedUpToByte = byteOffset + READ_AHEAD_BYTES - 1;
            }
        }
        return ios;
    }

    /**
     * Maps docIds (which equal oldOrds) to newOrds using inverted permutation.
     * permutation[newOrd] = oldOrd, so we invert to get oldOrd → newOrd
     */
    public static List<Integer> mapToNewOrds(List<Integer> docIds, int[] newOrd2Old) {
        // Build oldOrd2New (inverse of newOrd2Old)
        int[] oldOrd2New = new int[newOrd2Old.length];
        for (int newOrd = 0; newOrd < newOrd2Old.length; newOrd++) {
            oldOrd2New[newOrd2Old[newOrd]] = newOrd;
        }
        List<Integer> newOrds = new ArrayList<>();
        for (int docId : docIds) {
            if (docId >= 0 && docId < oldOrd2New.length) {
                newOrds.add(oldOrd2New[docId]);
            }
        }
        return newOrds;
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 3) {
            System.out.println("Usage: ReorderDistanceAnalyzer <permutation-file> <docids-file> <dimension>");
            System.out.println("  dimension: vector dimensionality (e.g., 128 for SIFT)");
            return;
        }

        Path permPath = Path.of(args[0]);
        Path docIdsPath = Path.of(args[1]);
        int dimension = Integer.parseInt(args[2]);
        int vectorBytes = dimension * 4;  // 4 bytes per float
        int vectorsPerPage = PAGE_SIZE_BYTES / vectorBytes;

        int[] permutation = loadPermutation(permPath);

        Stats reordered = computeDisplacementStats(permutation);
        Stats baseline = computeBaselineStats(permutation.length);
        System.out.println(compareStats(reordered, baseline));

        // Parse docIds file - each query separated by "-------"
        List<List<Integer>> queries = new ArrayList<>();
        List<Integer> current = new ArrayList<>();
        try (BufferedReader reader = Files.newBufferedReader(docIdsPath)) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("#")) continue;
                if (line.equals("-------")) {
                    if (!current.isEmpty()) {
                        queries.add(current);
                        current = new ArrayList<>();
                    }
                } else {
                    current.add(Integer.parseInt(line.trim()));
                }
            }
        }

        // Compute page faults: baseline (docIds=oldOrds) vs reordered (newOrds)
        double[] baselineFaults = new double[queries.size()];
        double[] reorderedFaults = new double[queries.size()];
        double[] baselineIOs = new double[queries.size()];
        double[] reorderedIOs = new double[queries.size()];
        
        for (int i = 0; i < queries.size(); i++) {
            List<Integer> docIds = queries.get(i);  // docIds == oldOrds
            List<Integer> newOrds = mapToNewOrds(docIds, permutation);
            baselineFaults[i] = computePageFaults(docIds, vectorBytes);
            reorderedFaults[i] = computePageFaults(newOrds, vectorBytes);
            baselineIOs[i] = computeDiskIOs(docIds, vectorBytes);
            reorderedIOs[i] = computeDiskIOs(newOrds, vectorBytes);
        }

        double tStat = computePairedTTest(baselineFaults, reorderedFaults);
        double meanBaseline = java.util.Arrays.stream(baselineFaults).average().orElse(0);
        double meanReordered = java.util.Arrays.stream(reorderedFaults).average().orElse(0);

        System.out.println("\n=== Config ===");
        System.out.println("Dimension: " + dimension + ", Vector size: " + vectorBytes + " bytes");
        System.out.println("Page size: " + PAGE_SIZE_BYTES + " bytes (" + vectorsPerPage + " vectors/page)");
        System.out.println("Read-ahead: " + READ_AHEAD_BYTES + " bytes (" + (READ_AHEAD_BYTES / vectorBytes) + " vectors)");

        System.out.println("\n=== Paired T-Test (Page Faults) ===");
        System.out.println("Number of queries: " + queries.size());
        System.out.println("Mean page faults baseline (oldOrds):  " + String.format("%.2f", meanBaseline));
        System.out.println("Mean page faults reordered (newOrds): " + String.format("%.2f", meanReordered));
        System.out.println("t-statistic: " + String.format("%.4f", tStat));

        double tStatIO = computePairedTTest(baselineIOs, reorderedIOs);
        double meanBaselineIO = java.util.Arrays.stream(baselineIOs).average().orElse(0);
        double meanReorderedIO = java.util.Arrays.stream(reorderedIOs).average().orElse(0);

        System.out.println("\n=== Paired T-Test (Disk IOs with read-ahead) ===");
        System.out.println("Mean disk IOs baseline (oldOrds):  " + String.format("%.2f", meanBaselineIO));
        System.out.println("Mean disk IOs reordered (newOrds): " + String.format("%.2f", meanReorderedIO));
        System.out.println("t-statistic: " + String.format("%.4f", tStatIO));
        System.out.println("\n(negative t = reordering reduced IOs)");
    }
}
