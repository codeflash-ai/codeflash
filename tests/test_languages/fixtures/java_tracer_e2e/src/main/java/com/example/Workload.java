package com.example;

public class Workload {

    public static String repeatString(String s, int count) {
        String result = "";
        for (int i = 0; i < count; i++) {
            result = result + s;
        }
        return result;
    }

    public static void main(String[] args) {
        // Run with large inputs so JFR can capture CPU samples.
        for (int round = 0; round < 1000; round++) {
            repeatString("hello world ", 1000);
        }

        // Also call with small inputs for variety in traced args
        System.out.println("repeatString(\"ab\", 3) = " + repeatString("ab", 3));

        System.out.println("Workload complete.");
    }
}
