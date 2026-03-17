package com.codeflash;

/**
 * Utility class to prevent dead code elimination by the JIT compiler.
 *
 * Inspired by JMH's Blackhole class. When the JVM detects that a computed
 * value is never used, it may eliminate the computation entirely. By
 * "consuming" values through this class, we prevent such optimizations.
 *
 * Usage:
 * <pre>
 * int result = expensiveComputation();
 * Blackhole.consume(result);  // Prevents JIT from eliminating the computation
 * </pre>
 *
 * The implementation uses volatile writes which act as memory barriers,
 * preventing the JIT from optimizing away the computation.
 */
public final class Blackhole {

    // Volatile fields act as memory barriers, preventing optimization
    private static volatile int intSink;
    private static volatile long longSink;
    private static volatile double doubleSink;
    private static volatile Object objectSink;

    private Blackhole() {
        // Utility class, no instantiation
    }

    /**
     * Consume an int value to prevent dead code elimination.
     *
     * @param value Value to consume
     */
    public static void consume(int value) {
        intSink = value;
    }

    /**
     * Consume a long value to prevent dead code elimination.
     *
     * @param value Value to consume
     */
    public static void consume(long value) {
        longSink = value;
    }

    /**
     * Consume a double value to prevent dead code elimination.
     *
     * @param value Value to consume
     */
    public static void consume(double value) {
        doubleSink = value;
    }

    /**
     * Consume a float value to prevent dead code elimination.
     *
     * @param value Value to consume
     */
    public static void consume(float value) {
        doubleSink = value;
    }

    /**
     * Consume a boolean value to prevent dead code elimination.
     *
     * @param value Value to consume
     */
    public static void consume(boolean value) {
        intSink = value ? 1 : 0;
    }

    /**
     * Consume a byte value to prevent dead code elimination.
     *
     * @param value Value to consume
     */
    public static void consume(byte value) {
        intSink = value;
    }

    /**
     * Consume a short value to prevent dead code elimination.
     *
     * @param value Value to consume
     */
    public static void consume(short value) {
        intSink = value;
    }

    /**
     * Consume a char value to prevent dead code elimination.
     *
     * @param value Value to consume
     */
    public static void consume(char value) {
        intSink = value;
    }

    /**
     * Consume an Object to prevent dead code elimination.
     * Works for any reference type including arrays and collections.
     *
     * @param value Value to consume
     */
    public static void consume(Object value) {
        objectSink = value;
    }

    /**
     * Consume an int array to prevent dead code elimination.
     *
     * @param values Array to consume
     */
    public static void consume(int[] values) {
        objectSink = values;
        if (values != null && values.length > 0) {
            intSink = values[0];
        }
    }

    /**
     * Consume a long array to prevent dead code elimination.
     *
     * @param values Array to consume
     */
    public static void consume(long[] values) {
        objectSink = values;
        if (values != null && values.length > 0) {
            longSink = values[0];
        }
    }

    /**
     * Consume a double array to prevent dead code elimination.
     *
     * @param values Array to consume
     */
    public static void consume(double[] values) {
        objectSink = values;
        if (values != null && values.length > 0) {
            doubleSink = values[0];
        }
    }
}
