package com.codeflash.profiler;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Maps (sourceFile, lineNumber) pairs to compact integer IDs at class-load time.
 *
 * <p>Registration happens once per unique line during class transformation (not on the hot path).
 * The integer IDs are used as direct array indices in {@link ProfilerData} for zero-allocation
 * hit recording at runtime.
 */
public final class ProfilerRegistry {

    private static final AtomicInteger nextId = new AtomicInteger(0);
    private static final ConcurrentHashMap<Long, Integer> lineToId = new ConcurrentHashMap<>();

    private static volatile String[] idToFile;
    private static volatile int[] idToLine;
    private static volatile String[] idToClassName;
    private static volatile String[] idToMethodName;

    private static int capacity;
    private static final Object growLock = new Object();

    private ProfilerRegistry() {}

    /**
     * Pre-allocate reverse-lookup arrays with the given capacity.
     * Called once from {@link ProfilerAgent#premain} before any classes are loaded.
     */
    public static void initialize(int expectedLines) {
        capacity = Math.max(expectedLines * 2, 4096);
        idToFile = new String[capacity];
        idToLine = new int[capacity];
        idToClassName = new String[capacity];
        idToMethodName = new String[capacity];
    }

    /**
     * Register a source line and return its global ID.
     *
     * <p>Thread-safe. Called during class loading by the ASM visitor. If the same
     * (className, lineNumber) pair has already been registered, returns the existing ID.
     *
     * @param sourceFile  absolute path of the source file
     * @param className   dot-separated class name (e.g. "com.example.Calculator")
     * @param methodName  method name
     * @param lineNumber  1-indexed line number in the source file
     * @return compact integer ID usable as an array index
     */
    public static int register(String sourceFile, String className, String methodName, int lineNumber) {
        // Pack className hash + lineNumber into a 64-bit key for fast lookup
        long key = ((long) className.hashCode() << 32) | (lineNumber & 0xFFFFFFFFL);
        Integer existing = lineToId.get(key);
        if (existing != null) {
            return existing;
        }

        int id = nextId.getAndIncrement();
        if (id >= capacity) {
            grow(id + 1);
        }

        Integer winner = lineToId.putIfAbsent(key, id);
        if (winner != null) {
            // Another thread registered first — use its ID
            return winner;
        }

        idToFile[id] = sourceFile;
        idToLine[id] = lineNumber;
        idToClassName[id] = className;
        idToMethodName[id] = methodName;
        return id;
    }

    private static void grow(int minCapacity) {
        synchronized (growLock) {
            if (minCapacity <= capacity) return;

            int newCapacity = Math.max(minCapacity * 2, capacity * 2);
            String[] newFiles = new String[newCapacity];
            int[] newLines = new int[newCapacity];
            String[] newClasses = new String[newCapacity];
            String[] newMethods = new String[newCapacity];

            System.arraycopy(idToFile, 0, newFiles, 0, capacity);
            System.arraycopy(idToLine, 0, newLines, 0, capacity);
            System.arraycopy(idToClassName, 0, newClasses, 0, capacity);
            System.arraycopy(idToMethodName, 0, newMethods, 0, capacity);

            idToFile = newFiles;
            idToLine = newLines;
            idToClassName = newClasses;
            idToMethodName = newMethods;
            capacity = newCapacity;
        }
    }

    public static int getMaxId() {
        return nextId.get();
    }

    public static String getFile(int id) {
        return idToFile[id];
    }

    public static int getLine(int id) {
        return idToLine[id];
    }

    public static String getClassName(int id) {
        return idToClassName[id];
    }

    public static String getMethodName(int id) {
        return idToMethodName[id];
    }
}
