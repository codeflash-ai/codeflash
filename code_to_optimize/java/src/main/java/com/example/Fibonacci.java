package com.example;

import java.util.ArrayList;
import java.util.List;

/**
 * Fibonacci implementations.
 */
public class Fibonacci {

    /**
     * Calculate the nth Fibonacci number using recursion.
     *
     * @param n Position in Fibonacci sequence (0-indexed)
     * @return The nth Fibonacci number
     */
    public static long fibonacci(int n) {
        if (n < 0) {
            throw new IllegalArgumentException("Fibonacci not defined for negative numbers");
        }
        if (n <= 1) {
            return n;
        }
        return fibonacci(n - 1) + fibonacci(n - 2);
    }

    /**
     * Check if a number is a Fibonacci number.
     *
     * @param num Number to check
     * @return true if num is a Fibonacci number
     */
    public static boolean isFibonacci(long num) {
        if (num < 0) {
            return false;
        }
        long check1 = 5 * num * num + 4;
        long check2 = 5 * num * num - 4;

        return isPerfectSquare(check1) || isPerfectSquare(check2);
    }

    /**
     * Check if a number is a perfect square.
     *
     * @param n Number to check
     * @return true if n is a perfect square
     */
    public static boolean isPerfectSquare(long n) {
        if (n < 0) {
            return false;
        }
        long sqrt = (long) Math.sqrt(n);
        return sqrt * sqrt == n;
    }

    /**
     * Generate an array of the first n Fibonacci numbers.
     *
     * @param n Number of Fibonacci numbers to generate
     * @return Array of first n Fibonacci numbers
     */
    public static long[] fibonacciSequence(int n) {
        if (n < 0) {
            throw new IllegalArgumentException("n must be non-negative");
        }
        if (n == 0) {
            return new long[0];
        }

        long[] result = new long[n];
        for (int i = 0; i < n; i++) {
            result[i] = fibonacci(i);
        }
        return result;
    }

    /**
     * Find the index of a Fibonacci number.
     *
     * @param fibNum The Fibonacci number to find
     * @return Index of the number, or -1 if not a Fibonacci number
     */
    public static int fibonacciIndex(long fibNum) {
        if (fibNum < 0) {
            return -1;
        }
        if (fibNum == 0) {
            return 0;
        }
        if (fibNum == 1) {
            return 1;
        }

        int index = 2;
        while (true) {
            long fib = fibonacci(index);
            if (fib == fibNum) {
                return index;
            }
            if (fib > fibNum) {
                return -1;
            }
            index++;
            if (index > 50) {
                return -1;
            }
        }
    }

    /**
     * Calculate sum of first n Fibonacci numbers.
     *
     * @param n Number of Fibonacci numbers to sum
     * @return Sum of first n Fibonacci numbers
     */
    public static long sumFibonacci(int n) {
        if (n <= 0) {
            return 0;
        }

        long sum = 0;
        for (int i = 0; i < n; i++) {
            sum = sum + fibonacci(i);
        }
        return sum;
    }

    /**
     * Get all Fibonacci numbers less than a given limit.
     *
     * @param limit Upper bound (exclusive)
     * @return List of Fibonacci numbers less than limit
     */
    public static List<Long> fibonacciUpTo(long limit) {
        List<Long> result = new ArrayList<>();

        if (limit <= 0) {
            return result;
        }

        int index = 0;
        while (true) {
            long fib = fibonacci(index);
            if (fib >= limit) {
                break;
            }
            result.add(fib);
            index++;
            if (index > 50) {
                break;
            }
        }

        return result;
    }

    /**
     * Check if two numbers are consecutive Fibonacci numbers.
     *
     * @param a First number
     * @param b Second number
     * @return true if a and b are consecutive Fibonacci numbers
     */
    public static boolean areConsecutiveFibonacci(long a, long b) {
        if (!isFibonacci(a) || !isFibonacci(b)) {
            return false;
        }

        int indexA = fibonacciIndex(a);
        int indexB = fibonacciIndex(b);

        return Math.abs(indexA - indexB) == 1;
    }

    /**
     * Sort an array in-place using bubble sort.
     * Intentionally naive O(n^2) implementation for optimization testing.
     *
     * @param arr Array to sort (modified in-place)
     */
    public static void sortArray(long[] arr) {
        if (arr == null) {
            throw new IllegalArgumentException("Array must not be null");
        }
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr.length - 1 - i; j++) {
                if (arr[j] > arr[j + 1]) {
                    long temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
    }

    /**
     * Append Fibonacci numbers up to a limit into the provided list.
     * Clears the list first, then fills it with Fibonacci numbers less than limit.
     * Uses repeated naive recursion — intentionally slow for optimization testing.
     *
     * @param output List to populate (cleared first)
     * @param limit  Upper bound (exclusive)
     */
    public static void collectFibonacciInto(List<Long> output, long limit) {
        if (output == null) {
            throw new IllegalArgumentException("Output list must not be null");
        }
        output.clear();

        if (limit <= 0) {
            return;
        }

        int index = 0;
        while (true) {
            long fib = fibonacci(index);
            if (fib >= limit) {
                break;
            }
            output.add(fib);
            index++;
            if (index > 50) {
                break;
            }
        }
    }

    /**
     * Compute running Fibonacci sums in-place.
     * result[i] = sum of fibonacci(0) through fibonacci(i).
     * Uses repeated naive recursion — intentionally O(n * 2^n).
     *
     * @param result Array to fill with running sums (must be pre-allocated)
     */
    public static void fillFibonacciRunningSums(long[] result) {
        if (result == null) {
            throw new IllegalArgumentException("Array must not be null");
        }
        for (int i = 0; i < result.length; i++) {
            long sum = 0;
            for (int j = 0; j <= i; j++) {
                sum += fibonacci(j);
            }
            result[i] = sum;
        }
    }
}
