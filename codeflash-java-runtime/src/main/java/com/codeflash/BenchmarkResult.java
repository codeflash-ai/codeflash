package com.codeflash;

import java.util.Arrays;

/**
 * Result of a benchmark run with statistical analysis.
 *
 * Provides JMH-style statistics including mean, standard deviation,
 * and percentiles (p50, p90, p99).
 */
public final class BenchmarkResult {

    private final String methodId;
    private final long[] measurements;
    private final long mean;
    private final long stdDev;
    private final long min;
    private final long max;
    private final long p50;
    private final long p90;
    private final long p99;

    /**
     * Create a benchmark result from raw measurements.
     *
     * @param methodId Method that was benchmarked
     * @param measurements Array of timing measurements in nanoseconds
     */
    public BenchmarkResult(String methodId, long[] measurements) {
        this.methodId = methodId;
        this.measurements = measurements.clone();

        // Sort for percentile calculations
        long[] sorted = measurements.clone();
        Arrays.sort(sorted);

        this.min = sorted[0];
        this.max = sorted[sorted.length - 1];
        this.mean = calculateMean(sorted);
        this.stdDev = calculateStdDev(sorted, this.mean);
        this.p50 = percentile(sorted, 50);
        this.p90 = percentile(sorted, 90);
        this.p99 = percentile(sorted, 99);
    }

    private static long calculateMean(long[] values) {
        long sum = 0;
        for (long v : values) {
            sum += v;
        }
        return sum / values.length;
    }

    private static long calculateStdDev(long[] values, long mean) {
        if (values.length < 2) {
            return 0;
        }
        long sumSquaredDiff = 0;
        for (long v : values) {
            long diff = v - mean;
            sumSquaredDiff += diff * diff;
        }
        return (long) Math.sqrt(sumSquaredDiff / (values.length - 1));
    }

    private static long percentile(long[] sorted, int percentile) {
        int index = (int) Math.ceil(percentile / 100.0 * sorted.length) - 1;
        return sorted[Math.max(0, Math.min(index, sorted.length - 1))];
    }

    // Getters

    public String getMethodId() {
        return methodId;
    }

    public long[] getMeasurements() {
        return measurements.clone();
    }

    public int getIterationCount() {
        return measurements.length;
    }

    public long getMean() {
        return mean;
    }

    public long getStdDev() {
        return stdDev;
    }

    public long getMin() {
        return min;
    }

    public long getMax() {
        return max;
    }

    public long getP50() {
        return p50;
    }

    public long getP90() {
        return p90;
    }

    public long getP99() {
        return p99;
    }

    /**
     * Get mean in milliseconds.
     */
    public double getMeanMs() {
        return mean / 1_000_000.0;
    }

    /**
     * Get standard deviation in milliseconds.
     */
    public double getStdDevMs() {
        return stdDev / 1_000_000.0;
    }

    /**
     * Calculate coefficient of variation (CV) as percentage.
     * CV = (stdDev / mean) * 100
     * Lower is better (more stable measurements).
     */
    public double getCoefficientOfVariation() {
        if (mean == 0) {
            return 0;
        }
        return (stdDev * 100.0) / mean;
    }

    /**
     * Check if measurements are stable (CV < 10%).
     */
    public boolean isStable() {
        return getCoefficientOfVariation() < 10.0;
    }

    @Override
    public String toString() {
        return String.format(
            "BenchmarkResult{method='%s', mean=%.3fms, stdDev=%.3fms, p50=%.3fms, p90=%.3fms, p99=%.3fms, cv=%.1f%%, iterations=%d}",
            methodId,
            getMeanMs(),
            getStdDevMs(),
            p50 / 1_000_000.0,
            p90 / 1_000_000.0,
            p99 / 1_000_000.0,
            getCoefficientOfVariation(),
            measurements.length
        );
    }
}
