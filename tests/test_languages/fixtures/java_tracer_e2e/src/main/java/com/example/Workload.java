package com.example;

import java.util.ArrayList;
import java.util.List;

public class Workload {

    public static int computeSum(int n) {
        if (n <= 0) {
            return 0;
        }
        long nl = n;
        long result = (nl * (nl - 1)) / 2;
        return (int) result;
    }

    public static String repeatString(String s, int count) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < count; i++) {
            sb.append(s);
        }
        return sb.toString();
    }

    public static List<Integer> filterEvens(List<Integer> numbers) {
        // Pre-size result to avoid repeated resizes. Keep same behavior for null input (will NPE).
        List<Integer> result = new ArrayList<>(numbers.size());

        // Use indexed loop for RandomAccess lists (e.g., ArrayList) to avoid iterator allocation,
        // but fall back to iterator-based (enhanced for) loop for non-random-access lists
        // to prevent O(n^2) behavior on LinkedList.
        if (numbers instanceof java.util.RandomAccess) {
            for (int i = 0, sz = numbers.size(); i < sz; i++) {
                Integer num = numbers.get(i);
                if ((num & 1) == 0) { // faster parity check than modulus
                    result.add(num);
                }
            }
        } else {
            for (Integer n : numbers) {
                if ((n & 1) == 0) {
                    result.add(n);
                }
            }
        }
        return result;
    }

    public int instanceMethod(int x, int y) {
        // Inline computeSum logic to avoid the static method call overhead on the hot path.
        int sum;
        if (x <= 0) {
            sum = 0;
        } else {
            long nl = x;
            long result = (nl * (nl - 1)) >> 1;
            sum = (int) result;
        }
        return x * y + sum;
    }

    public static void main(String[] args) {
        // Exercise the methods so the tracer can capture invocations
        System.out.println("computeSum(100) = " + computeSum(100));
        System.out.println("computeSum(50) = " + computeSum(50));

        System.out.println("repeatString(\"ab\", 3) = " + repeatString("ab", 3));
        System.out.println("repeatString(\"x\", 5) = " + repeatString("x", 5));

        List<Integer> nums = new ArrayList<>();
        for (int i = 1; i <= 10; i++) nums.add(i);
        System.out.println("filterEvens(1..10) = " + filterEvens(nums));

        Workload w = new Workload();
        System.out.println("instanceMethod(5, 3) = " + w.instanceMethod(5, 3));
        System.out.println("instanceMethod(10, 2) = " + w.instanceMethod(10, 2));

        System.out.println("Workload complete.");
    }
}
