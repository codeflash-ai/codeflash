package com.example;

import java.util.ArrayList;
import java.util.List;

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

    public static List<Integer> filterEvens(List<Integer> numbers) {
        int size = numbers.size();
        List<Integer> result = new ArrayList<>(size);
        // Use indexed access for RandomAccess lists to avoid iterator overhead,
        // fall back to enhanced-for for other list implementations.
        if (numbers instanceof java.util.RandomAccess) {
            for (int i = 0; i < size; i++) {
                Integer num = numbers.get(i);
                int n = num;
                if ((n & 1) == 0) {
                    result.add(num);
                }
            }
        } else {
            for (Integer num : numbers) {
                int n = num;
                if ((n & 1) == 0) {
                    result.add(num);
                }
            }
        }
        return result;
    }

    public int instanceMethod(int x, int y) {
        return x * y + computeSum(x);
    }

    public static void main(String[] args) {
        // Run methods with large inputs so JFR can capture CPU samples.
        // Small inputs finish too fast (<1ms) for JFR's 10ms sampling interval.
        for (int round = 0; round < 1000; round++) {
            computeSum(100_000);
            repeatString("hello world ", 1000);

            List<Integer> nums = new ArrayList<>();
            for (int i = 1; i <= 10_000; i++) nums.add(i);
            filterEvens(nums);

            Workload w = new Workload();
            w.instanceMethod(100_000, 42);
        }

        // Also call with small inputs for variety in traced args
        System.out.println("computeSum(100) = " + computeSum(100));
        System.out.println("repeatString(\"ab\", 3) = " + repeatString("ab", 3));

        List<Integer> small = new ArrayList<>();
        for (int i = 1; i <= 10; i++) small.add(i);
        System.out.println("filterEvens(1..10) = " + filterEvens(small));

        Workload w = new Workload();
        System.out.println("instanceMethod(5, 3) = " + w.instanceMethod(5, 3));

        System.out.println("Workload complete.");
    }
}
