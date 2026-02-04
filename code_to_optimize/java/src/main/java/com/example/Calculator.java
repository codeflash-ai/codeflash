package com.example;

import java.util.HashMap;
import java.util.Map;

/**
 * Calculator for statistics.
 */
public class Calculator {

    /**
     * Calculate statistics for an array of numbers.
     *
     * @param numbers Array of numbers to analyze
     * @return Map containing sum, average, min, max, and range
     */
    public static Map<String, Double> calculateStats(double[] numbers) {
        Map<String, Double> stats = new HashMap<>();

        if (numbers == null || numbers.length == 0) {
            stats.put("sum", 0.0);
            stats.put("average", 0.0);
            stats.put("min", 0.0);
            stats.put("max", 0.0);
            stats.put("range", 0.0);
            return stats;
        }

        double sum = MathHelpers.sumArray(numbers);
        double avg = MathHelpers.average(numbers);
        double min = MathHelpers.findMin(numbers);
        double max = MathHelpers.findMax(numbers);
        double range = max - min;

        stats.put("sum", sum);
        stats.put("average", avg);
        stats.put("min", min);
        stats.put("max", max);
        stats.put("range", range);

        return stats;
    }

    /**
     * Normalize an array of numbers to a 0-1 range.
     *
     * @param numbers Array of numbers to normalize
     * @return Normalized array
     */
    public static double[] normalizeArray(double[] numbers) {
        if (numbers == null || numbers.length == 0) {
            return new double[0];
        }

        double min = MathHelpers.findMin(numbers);
        double max = MathHelpers.findMax(numbers);
        double range = max - min;

        double[] result = new double[numbers.length];

        if (range == 0) {
            for (int i = 0; i < numbers.length; i++) {
                result[i] = 0.5;
            }
            return result;
        }

        for (int i = 0; i < numbers.length; i++) {
            result[i] = (numbers[i] - min) / range;
        }

        return result;
    }

    /**
     * Calculate the weighted average of values with corresponding weights.
     *
     * @param values Array of values
     * @param weights Array of weights (same length as values)
     * @return The weighted average
     */
    public static double weightedAverage(double[] values, double[] weights) {
        if (values == null || weights == null) {
            return 0;
        }

        if (values.length == 0 || values.length != weights.length) {
            return 0;
        }

        double weightedSum = 0;
        for (int i = 0; i < values.length; i++) {
            weightedSum = weightedSum + values[i] * weights[i];
        }

        double totalWeight = MathHelpers.sumArray(weights);
        if (totalWeight == 0) {
            return 0;
        }

        return weightedSum / totalWeight;
    }

    /**
     * Calculate the variance of an array.
     *
     * @param numbers Array of numbers
     * @return Variance
     */
    public static double variance(double[] numbers) {
        if (numbers == null || numbers.length == 0) {
            return 0;
        }

        double mean = MathHelpers.average(numbers);

        double sumSquaredDiff = 0;
        for (int i = 0; i < numbers.length; i++) {
            double diff = numbers[i] - mean;
            sumSquaredDiff = sumSquaredDiff + diff * diff;
        }

        return sumSquaredDiff / numbers.length;
    }

    /**
     * Calculate the standard deviation of an array.
     *
     * @param numbers Array of numbers
     * @return Standard deviation
     */
    public static double standardDeviation(double[] numbers) {
        return Math.sqrt(variance(numbers));
    }

    /**
     * Calculate the median of an array.
     *
     * @param numbers Array of numbers
     * @return Median value
     */
    public static double median(double[] numbers) {
        if (numbers == null || numbers.length == 0) {
            return 0;
        }

        int[] intArray = new int[numbers.length];
        for (int i = 0; i < numbers.length; i++) {
            intArray[i] = (int) numbers[i];
        }

        int[] sorted = BubbleSort.bubbleSort(intArray);

        int mid = sorted.length / 2;
        if (sorted.length % 2 == 0) {
            return (sorted[mid - 1] + sorted[mid]) / 2.0;
        } else {
            return sorted[mid];
        }
    }

    /**
     * Calculate percentile value.
     *
     * @param numbers Array of numbers
     * @param percentile Percentile to calculate (0-100)
     * @return Value at the specified percentile
     */
    public static double percentile(double[] numbers, int percentile) {
        if (numbers == null || numbers.length == 0) {
            return 0;
        }

        if (percentile < 0 || percentile > 100) {
            throw new IllegalArgumentException("Percentile must be between 0 and 100");
        }

        int[] intArray = new int[numbers.length];
        for (int i = 0; i < numbers.length; i++) {
            intArray[i] = (int) numbers[i];
        }

        int[] sorted = BubbleSort.bubbleSort(intArray);

        int index = (int) Math.ceil((percentile / 100.0) * sorted.length) - 1;
        index = Math.max(0, Math.min(index, sorted.length - 1));

        return sorted[index];
    }
}
