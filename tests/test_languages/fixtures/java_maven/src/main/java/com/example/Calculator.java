package com.example;

import com.example.helpers.MathHelper;
import com.example.helpers.Formatter;

/**
 * Calculator class - demonstrates class method optimization scenarios.
 * Uses helper functions from MathHelper and Formatter.
 */
public class Calculator {

    private int precision;
    private java.util.List<String> history;

    /**
     * Creates a Calculator with specified precision.
     * @param precision number of decimal places for formatting
     */
    public Calculator(int precision) {
        this.precision = precision;
        this.history = new java.util.ArrayList<>();
    }

    /**
     * Creates a Calculator with default precision of 2.
     */
    public Calculator() {
        this(2);
    }

    /**
     * Calculate compound interest with multiple helper dependencies.
     *
     * @param principal Initial amount
     * @param rate Interest rate (as decimal)
     * @param time Time in years
     * @param n Compounding frequency per year
     * @return Compound interest result formatted as string
     */
    public String calculateCompoundInterest(double principal, double rate, int time, int n) {
        Formatter.validateInput(principal, "principal");
        Formatter.validateInput(rate, "rate");

        // Inefficient: recalculates power multiple times
        double result = principal;
        for (int i = 0; i < n * time; i++) {
            result = MathHelper.multiply(result, MathHelper.add(1.0, rate / n));
        }

        double interest = result - principal;
        history.add("compound:" + interest);
        return Formatter.formatNumber(interest, precision);
    }

    /**
     * Calculate permutation using factorial helper.
     *
     * @param n Total items
     * @param r Items to choose
     * @return Permutation result (n! / (n-r)!)
     */
    public long permutation(int n, int r) {
        if (n < r) {
            return 0;
        }
        // Inefficient: calculates factorial(n) fully even when not needed
        return MathHelper.factorial(n) / MathHelper.factorial(n - r);
    }

    /**
     * Calculate combination (n choose r).
     *
     * @param n Total items
     * @param r Items to choose
     * @return Combination result (n! / (r! * (n-r)!))
     */
    public long combination(int n, int r) {
        if (n < r) {
            return 0;
        }
        // Inefficient: calculates full factorials
        return MathHelper.factorial(n) / (MathHelper.factorial(r) * MathHelper.factorial(n - r));
    }

    /**
     * Calculate Fibonacci number at position n.
     *
     * @param n Position in Fibonacci sequence (0-indexed)
     * @return Fibonacci number at position n
     */
    public long fibonacci(int n) {
        // Inefficient recursive implementation without memoization
        if (n <= 1) {
            return n;
        }
        return fibonacci(n - 1) + fibonacci(n - 2);
    }

    /**
     * Static method for quick calculations.
     *
     * @param a First number
     * @param b Second number
     * @return Sum of a and b
     */
    public static double quickAdd(double a, double b) {
        return MathHelper.add(a, b);
    }

    /**
     * Get calculation history.
     *
     * @return List of past calculations
     */
    public java.util.List<String> getHistory() {
        return new java.util.ArrayList<>(history);
    }

    /**
     * Get current precision setting.
     *
     * @return precision value
     */
    public int getPrecision() {
        return precision;
    }
}
