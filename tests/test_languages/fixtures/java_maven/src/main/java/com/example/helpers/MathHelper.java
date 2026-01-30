package com.example.helpers;

/**
 * Math utility functions - basic arithmetic operations.
 */
public class MathHelper {

    /**
     * Add two numbers.
     *
     * @param a First number
     * @param b Second number
     * @return Sum of a and b
     */
    public static double add(double a, double b) {
        return a + b;
    }

    /**
     * Multiply two numbers.
     *
     * @param a First number
     * @param b Second number
     * @return Product of a and b
     */
    public static double multiply(double a, double b) {
        return a * b;
    }

    /**
     * Calculate factorial recursively.
     *
     * @param n Non-negative integer
     * @return Factorial of n
     * @throws IllegalArgumentException if n is negative
     */
    public static long factorial(int n) {
        if (n < 0) {
            throw new IllegalArgumentException("Factorial not defined for negative numbers");
        }
        // Intentionally inefficient recursive implementation
        if (n <= 1) {
            return 1;
        }
        return n * factorial(n - 1);
    }

    /**
     * Calculate power using repeated multiplication.
     *
     * @param base Base number
     * @param exp Exponent (non-negative)
     * @return base raised to exp
     */
    public static double power(double base, int exp) {
        // Inefficient: linear time instead of log time
        double result = 1;
        for (int i = 0; i < exp; i++) {
            result = multiply(result, base);
        }
        return result;
    }

    /**
     * Check if a number is prime.
     *
     * @param n Number to check
     * @return true if n is prime, false otherwise
     */
    public static boolean isPrime(int n) {
        if (n < 2) {
            return false;
        }
        // Inefficient: checks all numbers up to n-1
        for (int i = 2; i < n; i++) {
            if (n % i == 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * Calculate greatest common divisor using Euclidean algorithm.
     *
     * @param a First number
     * @param b Second number
     * @return GCD of a and b
     */
    public static int gcd(int a, int b) {
        // Inefficient recursive implementation
        if (b == 0) {
            return a;
        }
        return gcd(b, a % b);
    }

    /**
     * Calculate least common multiple.
     *
     * @param a First number
     * @param b Second number
     * @return LCM of a and b
     */
    public static int lcm(int a, int b) {
        return (a * b) / gcd(a, b);
    }
}
