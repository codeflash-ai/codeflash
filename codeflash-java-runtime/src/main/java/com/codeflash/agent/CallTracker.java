package com.codeflash.agent;

import com.codeflash.Serializer;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Thread-safe singleton collecting profiling data during execution.
 *
 * Mirrors the Python tracer's call stack and timing logic from tracing_new_process.py.
 * Each thread maintains its own call stack via ThreadLocal. Timing stats and captured
 * arguments are stored in concurrent maps shared across threads.
 */
public final class CallTracker {

    private static final CallTracker INSTANCE = new CallTracker();

    private final ThreadLocal<Deque<CallFrame>> callStack =
        ThreadLocal.withInitial(ArrayDeque::new);

    private final ConcurrentHashMap<MethodKey, MethodStats> timings = new ConcurrentHashMap<>();

    private final ConcurrentHashMap<String, AtomicInteger> functionCallCount = new ConcurrentHashMap<>();

    private final ConcurrentHashMap<MethodKey, List<byte[]>> capturedArgs = new ConcurrentHashMap<>();

    // Set of MethodKeys currently on the stack for this thread (for recursion detection)
    private final ThreadLocal<Set<MethodKey>> onStack =
        ThreadLocal.withInitial(() -> Collections.newSetFromMap(new ConcurrentHashMap<>()));

    private final AtomicLong globalStartNs = new AtomicLong(0);
    private final AtomicLong globalEndNs = new AtomicLong(0);

    private volatile int maxFunctionCount = 256;

    private CallTracker() {}

    public static CallTracker getInstance() {
        return INSTANCE;
    }

    public void setMaxFunctionCount(int max) {
        this.maxFunctionCount = max;
    }

    public void markStart() {
        globalStartNs.compareAndSet(0, System.nanoTime());
    }

    public void markEnd() {
        globalEndNs.set(System.nanoTime());
    }

    /**
     * Called at method entry. Pushes a frame onto the thread-local call stack.
     *
     * @param className  fully qualified class name
     * @param methodName method name
     * @param fileName   source file path
     * @param lineNumber first line number of the method
     * @param args       method arguments (may be null)
     */
    public void enter(String className, String methodName, String fileName, int lineNumber, Object[] args) {
        MethodKey key = new MethodKey(fileName, lineNumber, methodName, className);
        Deque<CallFrame> stack = callStack.get();

        // Track recursion: if key is already on the stack, this is a recursive call
        Set<MethodKey> currentOnStack = onStack.get();
        boolean isRecursive = !currentOnStack.add(key);

        // Get or create stats and mark as nested if recursive
        MethodStats stats = timings.computeIfAbsent(key, k -> new MethodStats());
        if (isRecursive) {
            stats.incrementNested();
        }

        stack.push(new CallFrame(key, System.nanoTime(), isRecursive));

        // Capture arguments if under the limit
        if (args != null) {
            String qualifiedName = className + "." + methodName;
            AtomicInteger count = functionCallCount.computeIfAbsent(qualifiedName, k -> new AtomicInteger(0));
            if (count.get() < maxFunctionCount) {
                count.incrementAndGet();
                try {
                    byte[] serialized = Serializer.serialize(args);
                    if (serialized != null) {
                        capturedArgs.computeIfAbsent(key, k ->
                            Collections.synchronizedList(new ArrayList<>())
                        ).add(serialized);
                    }
                } catch (Exception e) {
                    // Serialization failure is non-fatal
                }
            }
        }
    }

    /**
     * Called at method exit. Pops from the call stack and updates timing stats.
     *
     * Mirrors the Python tracer's trace_dispatch_return logic:
     * - Own time (tt) is always accumulated
     * - Cumulative time (ct) and non-recursive call count (cc) are only
     *   updated when the outermost invocation of a recursive method returns
     * - Caller relationship is always tracked
     */
    public void exit() {
        Deque<CallFrame> stack = callStack.get();
        if (stack.isEmpty()) {
            return;
        }

        CallFrame frame = stack.pop();
        long elapsed = System.nanoTime() - frame.getStartTimeNs();
        MethodKey key = frame.getMethodKey();

        // Remove from on-stack set if this is the outermost call
        if (!frame.isRecursive()) {
            onStack.get().remove(key);
        }

        MethodStats stats = timings.get(key);
        if (stats == null) {
            return;
        }

        // Determine caller from parent frame
        MethodKey caller = null;
        if (!stack.isEmpty()) {
            caller = stack.peek().getMethodKey();
        }

        // The frame's own time is `elapsed` minus time spent in callees.
        // Since we're wrapping the entire method body, `elapsed` is the cumulative/frame total.
        // The Python tracer tracks rit (own internal time) and ret (time in callees).
        // Here, we pass elapsed as both ownTime and frameTotal when non-recursive;
        // for recursive calls, we track it the same way.
        boolean isRecursive = frame.isRecursive();
        stats.recordReturn(elapsed, elapsed, isRecursive, caller);

        if (isRecursive) {
            stats.decrementNested();
        }
    }

    public ConcurrentHashMap<MethodKey, MethodStats> getTimings() {
        return timings;
    }

    public ConcurrentHashMap<MethodKey, List<byte[]>> getCapturedArgs() {
        return capturedArgs;
    }

    public long getTotalTimeNs() {
        long start = globalStartNs.get();
        long end = globalEndNs.get();
        if (start == 0) return 0;
        if (end == 0) end = System.nanoTime();
        return end - start;
    }

    public void reset() {
        timings.clear();
        functionCallCount.clear();
        capturedArgs.clear();
        globalStartNs.set(0);
        globalEndNs.set(0);
    }

    /**
     * Internal frame on the thread-local call stack.
     */
    static final class CallFrame {
        private final MethodKey methodKey;
        private final long startTimeNs;
        private final boolean recursive;

        CallFrame(MethodKey methodKey, long startTimeNs, boolean recursive) {
            this.methodKey = methodKey;
            this.startTimeNs = startTimeNs;
            this.recursive = recursive;
        }

        MethodKey getMethodKey() {
            return methodKey;
        }

        long getStartTimeNs() {
            return startTimeNs;
        }

        boolean isRecursive() {
            return recursive;
        }
    }
}
