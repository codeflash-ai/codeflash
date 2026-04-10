package com.codeflash.tracer;

import com.codeflash.Serializer;

import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

public final class TraceRecorder {

    private static volatile TraceRecorder instance;

    private static final long SERIALIZATION_TIMEOUT_MS = 500;

    private final TracerConfig config;
    private final TraceWriter writer;
    private final ConcurrentHashMap<String, AtomicInteger> functionCounts = new ConcurrentHashMap<>();
    private final AtomicInteger droppedCaptures = new AtomicInteger(0);
    private final AtomicLong totalOnEntryNs = new AtomicLong(0);
    private final AtomicLong totalSerializationNs = new AtomicLong(0);
    private final int maxFunctionCount;
    private final ExecutorService serializerExecutor;

    // Reentrancy guard: prevent recursive tracing when serialization triggers class loading
    private static final ThreadLocal<Boolean> RECORDING = ThreadLocal.withInitial(() -> Boolean.FALSE);

    private TraceRecorder(TracerConfig config) {
        this.config = config;
        this.writer = new TraceWriter(config.getDbPath(), config.isInMemoryDb());
        this.maxFunctionCount = config.getMaxFunctionCount();
        this.serializerExecutor = Executors.newCachedThreadPool(r -> {
            Thread t = new Thread(r, "codeflash-serializer");
            t.setDaemon(true);
            return t;
        });
    }

    public static void initialize(TracerConfig config) {
        instance = new TraceRecorder(config);
    }

    public static TraceRecorder getInstance() {
        return instance;
    }

    public static boolean isRecording() {
        return Boolean.TRUE.equals(RECORDING.get());
    }

    public void onEntry(String className, String methodName, String descriptor,
                        int lineNumber, String sourceFile, Object[] args) {
        // Reentrancy guard
        if (RECORDING.get()) {
            return;
        }
        RECORDING.set(Boolean.TRUE);
        try {
            onEntryImpl(className, methodName, descriptor, lineNumber, sourceFile, args);
        } finally {
            RECORDING.set(Boolean.FALSE);
        }
    }

    private void onEntryImpl(String className, String methodName, String descriptor,
                             int lineNumber, String sourceFile, Object[] args) {
        long entryStart = System.nanoTime();

        String qualifiedName = className + "." + methodName + descriptor;

        // Check per-method count limit
        AtomicInteger count = functionCounts.computeIfAbsent(qualifiedName, k -> new AtomicInteger(0));
        if (count.get() >= maxFunctionCount) {
            return;
        }

        // Serialize args — try inline fast path first, fall back to async with timeout
        byte[] argsBlob;
        long serStart = System.nanoTime();
        argsBlob = Serializer.serializeFast(args);
        if (argsBlob == null) {
            // Slow path: async serialization with timeout for complex/unknown types
            Future<byte[]> future = serializerExecutor.submit(() -> Serializer.serialize(args));
            try {
                argsBlob = future.get(SERIALIZATION_TIMEOUT_MS, TimeUnit.MILLISECONDS);
            } catch (TimeoutException e) {
                future.cancel(true);
                droppedCaptures.incrementAndGet();
                System.err.println("[codeflash-tracer] Serialization timed out for " + className + "."
                        + methodName);
                return;
            } catch (Exception e) {
                Throwable cause = e.getCause() != null ? e.getCause() : e;
                droppedCaptures.incrementAndGet();
                System.err.println("[codeflash-tracer] Serialization failed for " + className + "."
                        + methodName + ": " + cause.getClass().getSimpleName() + ": " + cause.getMessage());
                return;
            }
        }
        totalSerializationNs.addAndGet(System.nanoTime() - serStart);

        long timeNs = System.nanoTime();
        count.incrementAndGet();

        writer.recordFunctionCall("call", methodName, className, sourceFile,
                lineNumber, descriptor, timeNs, argsBlob);

        totalOnEntryNs.addAndGet(System.nanoTime() - entryStart);
    }

    public void flush() {
        serializerExecutor.shutdownNow();
        // Write metadata
        Map<String, String> metadata = new LinkedHashMap<>();
        metadata.put("projectRoot", config.getProjectRoot());
        metadata.put("timestamp", Instant.now().toString());
        metadata.put("totalFunctions", String.valueOf(functionCounts.size()));

        int totalCaptures = 0;
        for (AtomicInteger count : functionCounts.values()) {
            totalCaptures += count.get();
        }
        metadata.put("totalCaptures", String.valueOf(totalCaptures));

        int dropped = droppedCaptures.get();
        metadata.put("droppedCaptures", String.valueOf(dropped));

        writer.writeMetadata(metadata);
        writer.flush();
        writer.close();

        System.err.println("[codeflash-tracer] Captured " + totalCaptures
                + " invocations across " + functionCounts.size() + " methods"
                + (dropped > 0 ? " (" + dropped + " dropped due to serialization timeout/failure)" : ""));

        // Timing summary
        long onEntryMs = totalOnEntryNs.get() / 1_000_000;
        long serMs = totalSerializationNs.get() / 1_000_000;
        String writerSummary = writer.getTimingSummary();
        System.err.println("[codeflash-tracer] Timing: onEntry=" + onEntryMs + "ms"
                + " (serialization=" + serMs + "ms)"
                + (totalCaptures > 0
                    ? " avg=" + String.format("%.2f", (double) onEntryMs / totalCaptures) + "ms/capture"
                    : "")
                + " " + writerSummary);
    }
}
