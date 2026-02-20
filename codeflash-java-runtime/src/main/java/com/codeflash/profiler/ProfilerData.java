package com.codeflash.profiler;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Zero-allocation, zero-contention per-line profiling data storage.
 *
 * <p>Each thread gets its own primitive {@code long[]} arrays for hit counts and self-time.
 * The hot path ({@link #hit(int)}) performs only an array-index increment and a single
 * {@link System#nanoTime()} call — no object allocations, no locks, no shared-state contention.
 *
 * <p>A per-thread call stack tracks method entry/exit to:
 * <ul>
 *   <li>Attribute time to the last line of a function (fixes the "last line 0ms" bug)</li>
 *   <li>Pause parent-line timing during callee execution (fixes cross-function timing)</li>
 *   <li>Handle recursion correctly (each stack frame is independent)</li>
 * </ul>
 */
public final class ProfilerData {

    private static final int INITIAL_CAPACITY = 4096;
    private static final int MAX_CALL_DEPTH = 256;

    // Thread-local arrays — each thread gets its own, no contention
    private static final ThreadLocal<long[]> hitCounts =
            ThreadLocal.withInitial(() -> registerArray(new long[INITIAL_CAPACITY]));
    private static final ThreadLocal<long[]> selfTimeNs =
            ThreadLocal.withInitial(() -> registerTimeArray(new long[INITIAL_CAPACITY]));

    // Per-thread "last line" tracking for time attribution
    // Using int[1] and long[1] to avoid boxing
    private static final ThreadLocal<int[]> lastLineId =
            ThreadLocal.withInitial(() -> new int[]{-1});
    private static final ThreadLocal<long[]> lastLineTime =
            ThreadLocal.withInitial(() -> new long[]{0L});

    // Per-thread call stack for method entry/exit
    private static final ThreadLocal<int[]> callStackLineIds =
            ThreadLocal.withInitial(() -> new int[MAX_CALL_DEPTH]);
    private static final ThreadLocal<int[]> callStackDepth =
            ThreadLocal.withInitial(() -> new int[]{0});

    // Global references to all thread-local arrays for harvesting at shutdown
    private static final List<long[]> allHitArrays = new CopyOnWriteArrayList<>();
    private static final List<long[]> allTimeArrays = new CopyOnWriteArrayList<>();

    // Warmup state: the method visitor injects a self-calling warmup loop,
    // warmupInProgress guards against recursive re-entry into the warmup block.
    private static volatile int warmupThreshold = 0;
    private static volatile boolean warmupComplete = false;
    private static volatile boolean warmupInProgress = false;

    private ProfilerData() {}

    private static long[] registerArray(long[] arr) {
        allHitArrays.add(arr);
        return arr;
    }

    private static long[] registerTimeArray(long[] arr) {
        allTimeArrays.add(arr);
        return arr;
    }

    /**
     * Set the number of self-call warmup iterations before measurement begins.
     * Called once from {@link ProfilerAgent#premain} before any classes are loaded.
     *
     * @param threshold number of warmup iterations (0 = no warmup)
     */
    public static void setWarmupThreshold(int threshold) {
        warmupThreshold = threshold;
        warmupComplete = (threshold <= 0);
    }

    /**
     * Check whether warmup is still needed. Called by injected bytecode at target method entry.
     * Returns {@code true} only on the very first call — subsequent calls (including recursive
     * warmup calls) return {@code false}.
     */
    public static boolean isWarmupNeeded() {
        return !warmupComplete && !warmupInProgress && warmupThreshold > 0;
    }

    /**
     * Enter warmup phase. Sets a guard flag so recursive warmup calls skip the warmup block.
     */
    public static void startWarmup() {
        warmupInProgress = true;
    }

    /**
     * Return the configured warmup iteration count.
     */
    public static int getWarmupThreshold() {
        return warmupThreshold;
    }

    /**
     * End warmup: zero all profiling counters, mark warmup complete, clear the guard flag.
     * The next execution of the method body is the clean measurement.
     */
    public static void finishWarmup() {
        resetAll();
        warmupComplete = true;
        warmupInProgress = false;
        System.err.println("[codeflash-profiler] Warmup complete after " + warmupThreshold
                + " iterations, measurement started");
    }

    /**
     * Reset all profiling counters across all threads.
     * Called once when warmup phase completes to discard warmup data.
     */
    private static void resetAll() {
        for (long[] arr : allHitArrays) {
            Arrays.fill(arr, 0L);
        }
        for (long[] arr : allTimeArrays) {
            Arrays.fill(arr, 0L);
        }
    }

    /**
     * Record a hit on a profiled line. This is the HOT PATH.
     *
     * <p>Called at every instrumented line number. Must not allocate after the initial
     * thread-local array expansion.
     *
     * @param globalId the line's registered ID from {@link ProfilerRegistry}
     */
    public static void hit(int globalId) {
        long now = System.nanoTime();

        long[] hits = hitCounts.get();
        if (globalId >= hits.length) {
            hits = ensureCapacity(hitCounts, allHitArrays, globalId);
        }
        hits[globalId]++;

        // Attribute elapsed time to the PREVIOUS line (the one that was executing)
        int[] lastId = lastLineId.get();
        long[] lastTime = lastLineTime.get();
        if (lastId[0] >= 0) {
            long[] times = selfTimeNs.get();
            if (lastId[0] >= times.length) {
                times = ensureCapacity(selfTimeNs, allTimeArrays, lastId[0]);
            }
            times[lastId[0]] += now - lastTime[0];
        }

        lastId[0] = globalId;
        lastTime[0] = now;
    }

    /**
     * Called at method entry to push a call-stack frame.
     *
     * <p>Attributes any pending time to the previous line (the call site), then
     * saves the caller's line state onto the stack so it can be restored in
     * {@link #exitMethod()}.
     *
     * @param entryLineId the globalId of the first line in the entering method (unused for stack,
     *                     but may be used for future total-time tracking)
     */
    public static void enterMethod(int entryLineId) {
        long now = System.nanoTime();

        // Flush pending time to the line that made the call
        int[] lastId = lastLineId.get();
        long[] lastTime = lastLineTime.get();
        if (lastId[0] >= 0) {
            long[] times = selfTimeNs.get();
            if (lastId[0] >= times.length) {
                times = ensureCapacity(selfTimeNs, allTimeArrays, lastId[0]);
            }
            times[lastId[0]] += now - lastTime[0];
        }

        // Push caller's line ID onto the stack
        int[] depth = callStackDepth.get();
        int[] stack = callStackLineIds.get();
        if (depth[0] < stack.length) {
            stack[depth[0]] = lastId[0];
        }
        depth[0]++;

        // Reset for the new method scope
        lastId[0] = -1;
        lastTime[0] = now;
    }

    /**
     * Called at method exit (before RETURN or ATHROW) to pop the call stack.
     *
     * <p>Attributes remaining time to the last line of the exiting method (fixes the
     * "last line always 0ms" bug), then restores the caller's timing state.
     */
    public static void exitMethod() {
        long now = System.nanoTime();

        // Attribute remaining time to the last line of the exiting method
        int[] lastId = lastLineId.get();
        long[] lastTime = lastLineTime.get();
        if (lastId[0] >= 0) {
            long[] times = selfTimeNs.get();
            if (lastId[0] >= times.length) {
                times = ensureCapacity(selfTimeNs, allTimeArrays, lastId[0]);
            }
            times[lastId[0]] += now - lastTime[0];
        }

        // Pop the call stack and restore parent's timing state
        int[] depth = callStackDepth.get();
        if (depth[0] > 0) {
            depth[0]--;
            int[] stack = callStackLineIds.get();
            int parentLineId = stack[depth[0]];

            lastId[0] = parentLineId;
            lastTime[0] = now; // Self-time: exclude callee duration
        } else {
            lastId[0] = -1;
            lastTime[0] = 0L;
        }
    }

    /**
     * Sum hit counts across all threads. Called once at shutdown for reporting.
     */
    public static long[] getGlobalHitCounts() {
        int maxId = ProfilerRegistry.getMaxId();
        long[] global = new long[maxId];
        for (long[] threadHits : allHitArrays) {
            int limit = Math.min(threadHits.length, maxId);
            for (int i = 0; i < limit; i++) {
                global[i] += threadHits[i];
            }
        }
        return global;
    }

    /**
     * Sum self-time across all threads. Called once at shutdown for reporting.
     */
    public static long[] getGlobalSelfTimeNs() {
        int maxId = ProfilerRegistry.getMaxId();
        long[] global = new long[maxId];
        for (long[] threadTimes : allTimeArrays) {
            int limit = Math.min(threadTimes.length, maxId);
            for (int i = 0; i < limit; i++) {
                global[i] += threadTimes[i];
            }
        }
        return global;
    }

    private static long[] ensureCapacity(ThreadLocal<long[]> tl, List<long[]> registry, int minIndex) {
        long[] old = tl.get();
        int newSize = Math.max((minIndex + 1) * 2, INITIAL_CAPACITY);
        long[] expanded = new long[newSize];
        System.arraycopy(old, 0, expanded, 0, old.length);

        // Update the registry: remove old, add new
        registry.remove(old);
        registry.add(expanded);

        tl.set(expanded);
        return expanded;
    }
}
