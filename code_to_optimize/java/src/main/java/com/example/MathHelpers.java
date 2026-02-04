package com.example;

/**
 * Math utility functions.
 */
public class MathHelpers {

    /**
     * Calculate the sum of all elements in an array.
     *
     * @param arr Array of doubles to sum
     * @return Sum of all elements
     */
    public static double sumArray(double[] arr) {
        if (arr == null || arr.length == 0) {
            return 0;
        }
        double sum = 0;
        for (int i = 0; i < arr.length; i++) {
            sum = sum + arr[i];
        }
        return sum;
    }

    /**
     * Calculate the average of all elements in an array.
     *
     * @param arr Array of doubles
     * @return Average value
     */
    public static double average(double[] arr) {
        if (arr == null || arr.length == 0) {
            return 0;
        }
        double sum = 0;
        for (int i = 0; i < arr.length; i++) {
            sum = sum + arr[i];
        }
        return sum / arr.length;
    }

    /**
     * Find the maximum value in an array.
     *
     * @param arr Array of doubles
     * @return Maximum value
     */
    public static double findMax(double[] arr) {
        if (arr == null || arr.length == 0) {
            return Double.MIN_VALUE;
        }
        double max = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i];
            }
        }
        return max;
    }

    /**
     * Find the minimum value in an array.
     *
     * @param arr Array of doubles
     * @return Minimum value
     */
    public static double findMin(double[] arr) {
        if (arr == null || arr.length == 0) {
            return Double.MAX_VALUE;
        }
        double min = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] < min) {
                min = arr[i];
            }
        }
        return min;
    }

    /**
     * Calculate factorial using recursion.
     *
     * @param n Non-negative integer
     * @return n factorial (n!)
     */
    public static long factorial(int n) {
        if (n < 0) {
            throw new IllegalArgumentException("Factorial not defined for negative numbers");
        }
        if (n <= 1) {
            return 1;
        }
        return n * factorial(n - 1);
    }

    /**
     * Calculate power using repeated multiplication.
     *
     * @param base The base number
     * @param exponent The exponent (non-negative)
     * @return base raised to the power of exponent
     */
    public static double power(double base, int exponent) {
        if (exponent < 0) {
            return 1.0 / power(base, -exponent);
        }
        if (exponent == 0) {
            return 1;
        }
        double result = 1;
        for (int i = 0; i < exponent; i++) {
            result = result * base;
        }
        return result;
    }

    /**
     * Check if a number is prime using trial division.
     *
     * @param n Number to check
     * @return true if n is prime
     */
    public static boolean isPrime(int n) {
        if (n < 2) {
            return false;
        }
        for (int i = 2; i < n; i++) {
            if (n % i == 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * Calculate greatest common divisor.
     *
     * @param a First number
     * @param b Second number
     * @return GCD of a and b
     */
    public static int gcd(int a, int b) {
        a = Math.abs(a);
        b = Math.abs(b);
        if (a == 0) return b;
        if (b == 0) return a;

        int smaller = Math.min(a, b);
        int gcd = 1;
        for (int i = 1; i <= smaller; i++) {
            if (a % i == 0 && b % i == 0) {
                gcd = i;
            }
        }
        return gcd;
    }
}
