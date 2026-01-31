package com.codeflash;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Main API for CodeFlash runtime instrumentation.
 *
 * Provides methods for:
 * - Capturing function inputs/outputs for behavior verification
 * - Benchmarking with JMH-inspired best practices
 * - Preventing dead code elimination
 *
 * Usage:
 * <pre>
 * // Behavior capture
 * CodeFlash.captureInput("Calculator.add", a, b);
 * int result = a + b;
 * return CodeFlash.captureOutput("Calculator.add", result);
 *
 * // Benchmarking
 * BenchmarkContext ctx = CodeFlash.startBenchmark("Calculator.add");
 * // ... code to benchmark ...
 * CodeFlash.endBenchmark(ctx);
 * </pre>
 */
public final class CodeFlash {

    private static final AtomicLong callIdCounter = new AtomicLong(0);
    private static volatile ResultWriter resultWriter;
    private static volatile boolean initialized = false;
    private static volatile String outputFile;

    // Configuration from environment variables
    private static final int DEFAULT_WARMUP_ITERATIONS = 10;
    private static final int DEFAULT_MEASUREMENT_ITERATIONS = 20;

    static {
        // Register shutdown hook to flush results
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            if (resultWriter != null) {
                resultWriter.close();
            }
        }));
    }

    private CodeFlash() {
        // Utility class, no instantiation
    }

    /**
     * Initialize CodeFlash with output file path.
     * Called automatically if CODEFLASH_OUTPUT_FILE env var is set.
     *
     * @param outputPath Path to output file (SQLite database)
     */
    public static synchronized void initialize(String outputPath) {
        if (!initialized || !outputPath.equals(outputFile)) {
            outputFile = outputPath;
            Path path = Paths.get(outputPath);
            resultWriter = new ResultWriter(path);
            initialized = true;
        }
    }

    /**
     * Get or create the result writer, initializing from environment if needed.
     */
    private static ResultWriter getWriter() {
        if (!initialized) {
            String envPath = System.getenv("CODEFLASH_OUTPUT_FILE");
            if (envPath != null && !envPath.isEmpty()) {
                initialize(envPath);
            } else {
                // Default to temp file if no env var
                initialize(System.getProperty("java.io.tmpdir") + "/codeflash_results.db");
            }
        }
        return resultWriter;
    }

    /**
     * Capture function input arguments.
     *
     * @param methodId Unique identifier for the method (e.g., "Calculator.add")
     * @param args Input arguments
     */
    public static void captureInput(String methodId, Object... args) {
        long callId = callIdCounter.incrementAndGet();
        String argsJson = Serializer.toJson(args);
        getWriter().recordInput(callId, methodId, argsJson, System.nanoTime());
    }

    /**
     * Capture function output and return it (for chaining in return statements).
     *
     * @param methodId Unique identifier for the method
     * @param result The result value
     * @param <T> Type of the result
     * @return The same result (for chaining)
     */
    public static <T> T captureOutput(String methodId, T result) {
        long callId = callIdCounter.get(); // Use same callId as input
        String resultJson = Serializer.toJson(result);
        getWriter().recordOutput(callId, methodId, resultJson, System.nanoTime());
        return result;
    }

    /**
     * Capture an exception thrown by the function.
     *
     * @param methodId Unique identifier for the method
     * @param error The exception
     */
    public static void captureException(String methodId, Throwable error) {
        long callId = callIdCounter.get();
        String errorJson = Serializer.exceptionToJson(error);
        getWriter().recordError(callId, methodId, errorJson, System.nanoTime());
    }

    /**
     * Start a benchmark context for timing code execution.
     * Implements JMH-inspired warmup and measurement phases.
     *
     * @param methodId Unique identifier for the method being benchmarked
     * @return BenchmarkContext to pass to endBenchmark
     */
    public static BenchmarkContext startBenchmark(String methodId) {
        return new BenchmarkContext(methodId, System.nanoTime());
    }

    /**
     * End a benchmark and record the timing.
     *
     * @param ctx The benchmark context from startBenchmark
     */
    public static void endBenchmark(BenchmarkContext ctx) {
        long endTime = System.nanoTime();
        long duration = endTime - ctx.getStartTime();
        getWriter().recordBenchmark(ctx.getMethodId(), duration, endTime);
    }

    /**
     * Run a benchmark with proper JMH-style warmup and measurement.
     *
     * @param methodId Unique identifier for the method
     * @param runnable Code to benchmark
     * @return Benchmark result with statistics
     */
    public static BenchmarkResult runBenchmark(String methodId, Runnable runnable) {
        int warmupIterations = getWarmupIterations();
        int measurementIterations = getMeasurementIterations();

        // Warmup phase - results discarded
        for (int i = 0; i < warmupIterations; i++) {
            runnable.run();
        }

        // Suggest GC before measurement (hint only, not guaranteed)
        System.gc();

        // Measurement phase
        long[] measurements = new long[measurementIterations];
        for (int i = 0; i < measurementIterations; i++) {
            long start = System.nanoTime();
            runnable.run();
            measurements[i] = System.nanoTime() - start;
        }

        BenchmarkResult result = new BenchmarkResult(methodId, measurements);
        getWriter().recordBenchmarkResult(methodId, result);
        return result;
    }

    /**
     * Run a benchmark that returns a value (prevents dead code elimination).
     *
     * @param methodId Unique identifier for the method
     * @param supplier Code to benchmark that returns a value
     * @param <T> Return type
     * @return Benchmark result with statistics
     */
    public static <T> BenchmarkResult runBenchmarkWithResult(String methodId, java.util.function.Supplier<T> supplier) {
        int warmupIterations = getWarmupIterations();
        int measurementIterations = getMeasurementIterations();

        // Warmup phase - consume results to prevent dead code elimination
        for (int i = 0; i < warmupIterations; i++) {
            Blackhole.consume(supplier.get());
        }

        // Suggest GC before measurement
        System.gc();

        // Measurement phase
        long[] measurements = new long[measurementIterations];
        for (int i = 0; i < measurementIterations; i++) {
            long start = System.nanoTime();
            T result = supplier.get();
            measurements[i] = System.nanoTime() - start;
            Blackhole.consume(result); // Prevent dead code elimination
        }

        BenchmarkResult benchmarkResult = new BenchmarkResult(methodId, measurements);
        getWriter().recordBenchmarkResult(methodId, benchmarkResult);
        return benchmarkResult;
    }

    /**
     * Get warmup iterations from environment or use default.
     */
    private static int getWarmupIterations() {
        String env = System.getenv("CODEFLASH_WARMUP_ITERATIONS");
        if (env != null) {
            try {
                return Integer.parseInt(env);
            } catch (NumberFormatException e) {
                // Use default
            }
        }
        return DEFAULT_WARMUP_ITERATIONS;
    }

    /**
     * Get measurement iterations from environment or use default.
     */
    private static int getMeasurementIterations() {
        String env = System.getenv("CODEFLASH_MEASUREMENT_ITERATIONS");
        if (env != null) {
            try {
                return Integer.parseInt(env);
            } catch (NumberFormatException e) {
                // Use default
            }
        }
        return DEFAULT_MEASUREMENT_ITERATIONS;
    }

    /**
     * Get the current call ID (for correlation).
     *
     * @return Current call ID
     */
    public static long getCurrentCallId() {
        return callIdCounter.get();
    }

    /**
     * Reset the call ID counter (for testing).
     */
    public static void resetCallId() {
        callIdCounter.set(0);
    }

    /**
     * Force flush all pending writes.
     */
    public static void flush() {
        if (resultWriter != null) {
            resultWriter.flush();
        }
    }
}
