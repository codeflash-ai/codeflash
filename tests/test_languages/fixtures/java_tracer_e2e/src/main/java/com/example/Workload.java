package com.example;

public class Workload {

    public static int computeSum(int n) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += i;
        }
        return sum;
    }

    public static String repeatString(String s, int count) {
        String result = "";
        for (int i = 0; i < count; i++) {
            result = result + s;
        }
        return result;
    }

    public static void main(String[] args) {
        // Run methods with large inputs so JFR can capture CPU samples.
        // Small inputs finish too fast (<1ms) for JFR's 10ms sampling interval.
        // 100 rounds is enough for JFR to collect ~10 samples per function.
        for (int round = 0; round < 100; round++) {
            computeSum(100_000);
            repeatString("hello world ", 1000);
        }

        // Also call with small inputs for variety in traced args
        System.out.println("computeSum(100) = " + computeSum(100));
        System.out.println("repeatString(\"ab\", 3) = " + repeatString("ab", 3));

        System.out.println("Workload complete.");
    }
}
