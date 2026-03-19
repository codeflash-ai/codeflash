package com.codeflash.tracer;

import java.lang.instrument.Instrumentation;

public class TracerAgent {

    public static void premain(String agentArgs, Instrumentation inst) {
        TracerConfig config = TracerConfig.parse(agentArgs);

        if (config.getPackages().isEmpty()) {
            System.err.println("[codeflash-tracer] Warning: no packages configured, will instrument all non-JDK classes");
        }

        // Register transformer BEFORE initializing TraceRecorder, to ensure
        // classes loaded during initialization (SQLite, Kryo) are visible.
        inst.addTransformer(new TracingTransformer(config), true);

        TraceRecorder.initialize(config);

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            TraceRecorder.getInstance().flush();
        }, "codeflash-tracer-shutdown"));

        System.err.println("[codeflash-tracer] Agent loaded, tracing packages: " + config.getPackages());
    }
}
