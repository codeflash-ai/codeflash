package com.codeflash;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the BenchmarkResult class.
 */
@DisplayName("BenchmarkResult Tests")
class BenchmarkResultTest {

    @Test
    @DisplayName("should calculate mean correctly")
    void testMean() {
        long[] measurements = {100, 200, 300, 400, 500};
        BenchmarkResult result = new BenchmarkResult("test", measurements);

        assertEquals(300, result.getMean());
    }

    @Test
    @DisplayName("should calculate min and max")
    void testMinMax() {
        long[] measurements = {100, 50, 200, 150, 75};
        BenchmarkResult result = new BenchmarkResult("test", measurements);

        assertEquals(50, result.getMin());
        assertEquals(200, result.getMax());
    }

    @Test
    @DisplayName("should calculate percentiles")
    void testPercentiles() {
        long[] measurements = new long[100];
        for (int i = 0; i < 100; i++) {
            measurements[i] = i + 1; // 1 to 100
        }
        BenchmarkResult result = new BenchmarkResult("test", measurements);

        assertEquals(50, result.getP50());
        assertEquals(90, result.getP90());
        assertEquals(99, result.getP99());
    }

    @Test
    @DisplayName("should calculate standard deviation")
    void testStdDev() {
        // All same values should have 0 std dev
        long[] sameValues = {100, 100, 100, 100, 100};
        BenchmarkResult sameResult = new BenchmarkResult("test", sameValues);
        assertEquals(0, sameResult.getStdDev());

        // Different values should have non-zero std dev
        long[] differentValues = {100, 200, 300, 400, 500};
        BenchmarkResult diffResult = new BenchmarkResult("test", differentValues);
        assertTrue(diffResult.getStdDev() > 0);
    }

    @Test
    @DisplayName("should calculate coefficient of variation")
    void testCoefficientOfVariation() {
        long[] measurements = {100, 100, 100, 100, 100};
        BenchmarkResult result = new BenchmarkResult("test", measurements);

        assertEquals(0.0, result.getCoefficientOfVariation(), 0.001);
    }

    @Test
    @DisplayName("should detect stable measurements")
    void testIsStable() {
        // Low variance - stable
        long[] stableMeasurements = {100, 101, 99, 100, 102};
        BenchmarkResult stableResult = new BenchmarkResult("test", stableMeasurements);
        assertTrue(stableResult.isStable());

        // High variance - unstable
        long[] unstableMeasurements = {100, 200, 50, 300, 25};
        BenchmarkResult unstableResult = new BenchmarkResult("test", unstableMeasurements);
        assertFalse(unstableResult.isStable());
    }

    @Test
    @DisplayName("should convert to milliseconds")
    void testMillisecondConversion() {
        long[] measurements = {1_000_000, 2_000_000, 3_000_000}; // 1ms, 2ms, 3ms
        BenchmarkResult result = new BenchmarkResult("test", measurements);

        assertEquals(2.0, result.getMeanMs(), 0.001);
    }

    @Test
    @DisplayName("should clone measurements array")
    void testMeasurementsCloned() {
        long[] original = {100, 200, 300};
        BenchmarkResult result = new BenchmarkResult("test", original);

        long[] retrieved = result.getMeasurements();
        retrieved[0] = 999;

        // Original should not be affected
        assertEquals(100, result.getMeasurements()[0]);
    }

    @Test
    @DisplayName("should return correct iteration count")
    void testIterationCount() {
        long[] measurements = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        BenchmarkResult result = new BenchmarkResult("test", measurements);

        assertEquals(10, result.getIterationCount());
    }

    @Test
    @DisplayName("should have meaningful toString")
    void testToString() {
        long[] measurements = {1_000_000, 2_000_000};
        BenchmarkResult result = new BenchmarkResult("Calculator.add", measurements);

        String str = result.toString();
        assertTrue(str.contains("Calculator.add"));
        assertTrue(str.contains("mean="));
        assertTrue(str.contains("ms"));
    }
}
