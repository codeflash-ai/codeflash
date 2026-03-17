package com.example;

import org.junit.jupiter.api.Test;
import java.util.Map;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Calculator statistics class.
 */
class CalculatorTest {

    @Test
    void testCalculateStats() {
        Map<String, Double> stats = Calculator.calculateStats(new double[]{1, 2, 3, 4, 5});

        assertEquals(15.0, stats.get("sum"));
        assertEquals(3.0, stats.get("average"));
        assertEquals(1.0, stats.get("min"));
        assertEquals(5.0, stats.get("max"));
        assertEquals(4.0, stats.get("range"));
    }

    @Test
    void testCalculateStatsEmpty() {
        Map<String, Double> stats = Calculator.calculateStats(new double[]{});

        assertEquals(0.0, stats.get("sum"));
        assertEquals(0.0, stats.get("average"));
        assertEquals(0.0, stats.get("min"));
        assertEquals(0.0, stats.get("max"));
        assertEquals(0.0, stats.get("range"));
    }

    @Test
    void testCalculateStatsNull() {
        Map<String, Double> stats = Calculator.calculateStats(null);

        assertEquals(0.0, stats.get("sum"));
        assertEquals(0.0, stats.get("average"));
    }

    @Test
    void testNormalizeArray() {
        double[] result = Calculator.normalizeArray(new double[]{0, 50, 100});

        assertEquals(3, result.length);
        assertEquals(0.0, result[0], 0.0001);
        assertEquals(0.5, result[1], 0.0001);
        assertEquals(1.0, result[2], 0.0001);
    }

    @Test
    void testNormalizeArraySameValues() {
        double[] result = Calculator.normalizeArray(new double[]{5, 5, 5});

        assertEquals(3, result.length);
        assertEquals(0.5, result[0], 0.0001);
        assertEquals(0.5, result[1], 0.0001);
        assertEquals(0.5, result[2], 0.0001);
    }

    @Test
    void testNormalizeArrayEmpty() {
        double[] result = Calculator.normalizeArray(new double[]{});
        assertEquals(0, result.length);
    }

    @Test
    void testWeightedAverage() {
        assertEquals(2.5, Calculator.weightedAverage(
                new double[]{1, 2, 3, 4},
                new double[]{1, 1, 1, 1}), 0.0001);

        assertEquals(4.0, Calculator.weightedAverage(
                new double[]{1, 2, 3, 4},
                new double[]{0, 0, 0, 1}), 0.0001);

        assertEquals(2.0, Calculator.weightedAverage(
                new double[]{1, 3},
                new double[]{1, 1}), 0.0001);
    }

    @Test
    void testWeightedAverageEmpty() {
        assertEquals(0.0, Calculator.weightedAverage(new double[]{}, new double[]{}));
        assertEquals(0.0, Calculator.weightedAverage(null, null));
    }

    @Test
    void testWeightedAverageMismatchedArrays() {
        assertEquals(0.0, Calculator.weightedAverage(
                new double[]{1, 2, 3},
                new double[]{1, 1}));
    }

    @Test
    void testVariance() {
        assertEquals(2.0, Calculator.variance(new double[]{1, 2, 3, 4, 5}), 0.0001);
        assertEquals(0.0, Calculator.variance(new double[]{5, 5, 5}), 0.0001);
        assertEquals(0.0, Calculator.variance(new double[]{}));
    }

    @Test
    void testStandardDeviation() {
        assertEquals(Math.sqrt(2.0), Calculator.standardDeviation(new double[]{1, 2, 3, 4, 5}), 0.0001);
        assertEquals(0.0, Calculator.standardDeviation(new double[]{5, 5, 5}), 0.0001);
    }

    @Test
    void testMedian() {
        assertEquals(3.0, Calculator.median(new double[]{1, 2, 3, 4, 5}), 0.0001);
        assertEquals(2.5, Calculator.median(new double[]{1, 2, 3, 4}), 0.0001);
        assertEquals(5.0, Calculator.median(new double[]{5}), 0.0001);
        assertEquals(0.0, Calculator.median(new double[]{}));
    }

    @Test
    void testPercentile() {
        double[] data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

        assertEquals(1, Calculator.percentile(data, 0), 0.0001);
        assertEquals(5, Calculator.percentile(data, 50), 0.0001);
        assertEquals(10, Calculator.percentile(data, 100), 0.0001);
    }

    @Test
    void testPercentileInvalidRange() {
        assertThrows(IllegalArgumentException.class, () ->
                Calculator.percentile(new double[]{1, 2, 3}, -1));
        assertThrows(IllegalArgumentException.class, () ->
                Calculator.percentile(new double[]{1, 2, 3}, 101));
    }
}
