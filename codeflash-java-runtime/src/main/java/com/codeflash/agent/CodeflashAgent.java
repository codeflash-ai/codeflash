package com.codeflash.agent;

import java.lang.instrument.Instrumentation;
import java.util.HashMap;
import java.util.Map;

/**
 * Java agent entry point for CodeFlash profiling.
 *
 * Attaches via -javaagent:codeflash-runtime.jar=key1=val1;key2=val2
 *
 * Supported agent arguments:
 * - packages: semicolon-separated list of package prefixes to instrument (comma-separated within)
 * - output: path to the output SQLite file
 * - sourceRoot: root directory for source file path resolution
 * - timeout: maximum profiling duration in seconds (default: 300)
 * - maxFunctionCount: max captured argument sets per function (default: 256)
 */
public final class CodeflashAgent {

    private CodeflashAgent() {}

    public static void premain(String agentArgs, Instrumentation inst) {
        Map<String, String> args = parseArgs(agentArgs);

        String packagesStr = args.getOrDefault("packages", "");
        String outputPath = args.getOrDefault("output", "codeflash_trace.sqlite");
        String sourceRoot = args.getOrDefault("sourceRoot", "");
        int maxFunctionCount = parseIntOrDefault(args.get("maxFunctionCount"), 256);

        if (packagesStr.isEmpty()) {
            System.err.println("[codeflash-agent] No packages specified, agent will not instrument any classes.");
            return;
        }

        String[] packagePrefixes = packagesStr.split(",");

        // Configure the call tracker
        CallTracker tracker = CallTracker.getInstance();
        tracker.setMaxFunctionCount(maxFunctionCount);
        tracker.markStart();

        // Register the transformer
        CodeflashTransformer transformer = new CodeflashTransformer(packagePrefixes, sourceRoot);
        inst.addTransformer(transformer);

        // Register shutdown hook to flush profiling data
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            try {
                PstatsWriter.flush(outputPath);
            } catch (Exception e) {
                System.err.println("[codeflash-agent] Error flushing trace data: " + e.getMessage());
            }
        }, "codeflash-agent-shutdown"));

        System.out.println("[codeflash-agent] Profiling active for packages: " + packagesStr);
        System.out.println("[codeflash-agent] Output: " + outputPath);
    }

    /**
     * Parse agent arguments in the format: key1=val1;key2=val2
     */
    private static Map<String, String> parseArgs(String agentArgs) {
        Map<String, String> result = new HashMap<>();
        if (agentArgs == null || agentArgs.isEmpty()) {
            return result;
        }

        for (String pair : agentArgs.split(";")) {
            int eq = pair.indexOf('=');
            if (eq > 0) {
                String key = pair.substring(0, eq).trim();
                String value = pair.substring(eq + 1).trim();
                result.put(key, value);
            }
        }
        return result;
    }

    private static int parseIntOrDefault(String value, int defaultValue) {
        if (value == null || value.isEmpty()) return defaultValue;
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }
}
