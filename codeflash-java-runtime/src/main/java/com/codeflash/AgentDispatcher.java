package com.codeflash;

import java.lang.instrument.Instrumentation;

/**
 * Premain dispatcher that routes to either the CodeFlash line profiler or the
 * JaCoCo coverage agent based on the agent arguments.
 *
 * <p>Detection logic:
 * <ul>
 *   <li>Args contain {@code config=} → line profiler mode → delegate to
 *       {@link com.codeflash.profiler.ProfilerAgent}</li>
 *   <li>Otherwise → JaCoCo mode → delegate to JaCoCo's PreMain</li>
 * </ul>
 *
 * <p>This is reliable because our profiler always receives
 * {@code config=/path/to/config.json} while JaCoCo always receives
 * {@code destfile=/path/to/jacoco.exec}.
 */
public class AgentDispatcher {

    static boolean isTracerMode(String agentArgs) {
        return agentArgs != null
                && (agentArgs.startsWith("trace=") || agentArgs.contains(",trace="));
    }

    static boolean isProfilerMode(String agentArgs) {
        return agentArgs != null
                && (agentArgs.startsWith("config=") || agentArgs.contains(",config="));
    }

    public static void premain(String agentArgs, Instrumentation inst) throws Exception {
        if (isTracerMode(agentArgs)) {
            com.codeflash.tracer.TracerAgent.premain(agentArgs, inst);
        } else if (isProfilerMode(agentArgs)) {
            com.codeflash.profiler.ProfilerAgent.premain(agentArgs, inst);
        } else {
            org.jacoco.agent.rt.internal_0e20598.PreMain.premain(agentArgs, inst);
        }
    }
}
