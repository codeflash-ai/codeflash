package com.codeflash;

/**
 * Context object for tracking benchmark timing.
 *
 * Created by {@link CodeFlash#startBenchmark(String)} and passed to
 * {@link CodeFlash#endBenchmark(BenchmarkContext)}.
 */
public final class BenchmarkContext {

    private final String methodId;
    private final long startTime;

    /**
     * Create a new benchmark context.
     *
     * @param methodId Method being benchmarked
     * @param startTime Start time in nanoseconds
     */
    BenchmarkContext(String methodId, long startTime) {
        this.methodId = methodId;
        this.startTime = startTime;
    }

    /**
     * Get the method ID.
     *
     * @return Method identifier
     */
    public String getMethodId() {
        return methodId;
    }

    /**
     * Get the start time.
     *
     * @return Start time in nanoseconds
     */
    public long getStartTime() {
        return startTime;
    }
}
