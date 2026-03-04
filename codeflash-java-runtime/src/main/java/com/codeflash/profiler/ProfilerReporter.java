package com.codeflash.profiler;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

/**
 * Writes profiling results to a JSON file in the same format as the old source-injected profiler.
 *
 * <p>Output format (consumed by {@code JavaLineProfiler.parse_results()} in Python):
 * <pre>
 * {
 *   "/path/to/File.java:10": {
 *     "hits": 100,
 *     "time": 5000000,
 *     "file": "/path/to/File.java",
 *     "line": 10,
 *     "content": "int x = compute();"
 *   },
 *   ...
 * }
 * </pre>
 */
public final class ProfilerReporter {

    private ProfilerReporter() {}

    /**
     * Write profiling results to the output file. Called once from a JVM shutdown hook.
     */
    public static void writeResults(String outputFile, ProfilerConfig config) {
        if (outputFile == null || outputFile.isEmpty()) return;

        long[] globalHits = ProfilerData.getGlobalHitCounts();
        long[] globalTimes = ProfilerData.getGlobalSelfTimeNs();
        int maxId = ProfilerRegistry.getMaxId();
        Map<String, String> lineContents = config.getLineContents();

        StringBuilder json = new StringBuilder(Math.max(maxId * 128, 256));
        json.append("{\n");

        boolean first = true;
        for (int id = 0; id < maxId; id++) {
            long hits = (id < globalHits.length) ? globalHits[id] : 0;
            long timeNs = (id < globalTimes.length) ? globalTimes[id] : 0;
            if (hits == 0 && timeNs == 0) continue;

            String file = ProfilerRegistry.getFile(id);
            int line = ProfilerRegistry.getLine(id);
            if (file == null) continue;

            String key = file + ":" + line;
            String content = lineContents.getOrDefault(key, "");

            if (!first) json.append(",\n");
            first = false;

            json.append("  \"").append(escapeJson(key)).append("\": {\n");
            json.append("    \"hits\": ").append(hits).append(",\n");
            json.append("    \"time\": ").append(timeNs).append(",\n");
            json.append("    \"file\": \"").append(escapeJson(file)).append("\",\n");
            json.append("    \"line\": ").append(line).append(",\n");
            json.append("    \"content\": \"").append(escapeJson(content)).append("\"\n");
            json.append("  }");
        }

        json.append("\n}");

        try {
            Path path = Paths.get(outputFile);
            Path parent = path.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            Files.write(path, json.toString().getBytes(StandardCharsets.UTF_8));
        } catch (IOException e) {
            System.err.println("[codeflash-profiler] Failed to write results: " + e.getMessage());
        }
    }

    private static String escapeJson(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }
}
