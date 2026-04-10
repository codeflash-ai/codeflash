package com.example;

public class Workload {

    public static int computeSum(int n) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += i;
        }
        return sum;
    }

    public static void main(String[] args) {
        // Run with large inputs so JFR can capture CPU samples.
        // Small inputs finish too fast (<1ms) for JFR's 10ms sampling interval.
        for (int round = 0; round < 100; round++) {
            computeSum(100_000);
        }

        // Also call with small inputs for variety in traced args
        System.out.println("computeSum(100) = " + computeSum(100));

        System.out.println("Workload complete.");
    }
}
