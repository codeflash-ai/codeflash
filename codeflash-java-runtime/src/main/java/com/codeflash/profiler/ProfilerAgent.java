package com.codeflash.profiler;

import java.lang.instrument.Instrumentation;

/**
 * Java agent entry point for the CodeFlash line profiler.
 *
 * <p>Loaded via {@code -javaagent:codeflash-profiler-agent.jar=config=/path/to/config.json}.
 *
 * <p>The agent:
 * <ol>
 *   <li>Parses the config file specifying which classes/methods to profile</li>
 *   <li>Registers a {@link LineProfilingTransformer} to instrument target classes at load time</li>
 *   <li>Registers a shutdown hook to write profiling results to JSON</li>
 * </ol>
 */
public class ProfilerAgent {

    /**
     * Called by the JVM before {@code main()} when the agent is loaded.
     *
     * @param agentArgs comma-separated key=value pairs (e.g., {@code config=/path/to/config.json})
     * @param inst      the JVM instrumentation interface
     */
    public static void premain(String agentArgs, Instrumentation inst) {
        ProfilerConfig config = ProfilerConfig.parse(agentArgs);

        if (config.getTargetClasses().isEmpty()) {
            System.err.println("[codeflash-profiler] No target classes configured, profiler inactive");
            return;
        }

        // Pre-allocate registry with estimated capacity
        ProfilerRegistry.initialize(config.getExpectedLineCount());

        // Register the bytecode transformer
        inst.addTransformer(new LineProfilingTransformer(config), true);

        // Register shutdown hook to write results on JVM exit
        String outputFile = config.getOutputFile();
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            ProfilerReporter.writeResults(outputFile, config);
        }, "codeflash-profiler-shutdown"));

        System.err.println("[codeflash-profiler] Agent loaded, profiling "
                + config.getTargetClasses().size() + " class(es)");
    }
}
