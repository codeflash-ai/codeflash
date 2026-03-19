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
        List<Integer> result = new ArrayList<>();
        for (int n : numbers) {
            if (n % 2 == 0) {
                result.add(n);
            }
        }
        return result;
    }

    public int instanceMethod(int x, int y) {
        return x * y + computeSum(x);
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
