package com.codeflash.agent;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Mutable, thread-safe accumulator for method timing statistics.
 *
 * Matches the Python tracer's per-function stats tuple:
 * (cc, ns, tt, ct, callers) where:
 * - cc = call count (non-recursive)
 * - ns = nest/recursion count (number of active recursive invocations)
 * - tt = total time (own time only, excluding callees)
 * - ct = cumulative time (including callees, updated only for outermost call)
 * - callers = map of caller -> call count
 */
public final class MethodStats {

    private long callCount;
    private long nestedCount;
    private long totalTimeNs;
    private long cumulativeTimeNs;
    private final ConcurrentHashMap<MethodKey, AtomicLong> callers;

    public MethodStats() {
        this.callCount = 0;
        this.nestedCount = 0;
        this.totalTimeNs = 0;
        this.cumulativeTimeNs = 0;
        this.callers = new ConcurrentHashMap<>();
    }

    public synchronized void recordReturn(long ownTimeNs, long frameTotal, boolean isRecursive, MethodKey caller) {
        totalTimeNs += ownTimeNs;
        if (!isRecursive) {
            cumulativeTimeNs += frameTotal;
            callCount++;
        }
        if (caller != null) {
            callers.computeIfAbsent(caller, k -> new AtomicLong(0)).incrementAndGet();
        }
    }

    public synchronized void incrementNested() {
        nestedCount++;
    }

    public synchronized void decrementNested() {
        nestedCount--;
    }

    public synchronized boolean isRecursive() {
        return nestedCount > 0;
    }

    public synchronized long getCallCount() {
        return callCount;
    }

    public synchronized long getNestedCount() {
        return nestedCount;
    }

    public synchronized long getTotalTimeNs() {
        return totalTimeNs;
    }

    public synchronized long getCumulativeTimeNs() {
        return cumulativeTimeNs;
    }

    public ConcurrentHashMap<MethodKey, AtomicLong> getCallers() {
        return callers;
    }
}
