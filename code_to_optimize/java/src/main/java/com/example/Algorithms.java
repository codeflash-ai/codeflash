package com.example;

import java.util.ArrayList;
import java.util.List;

/**
 * Collection of algorithms that can be optimized by Codeflash.
 */
public class Algorithms {

    public long fibonacci(int n) {
        if (n <= 1) {
            return n;
        }
        // Fast doubling iterative method (O(log n) time, O(1) space)
        long a = 0; // F(0)
        long b = 1; // F(1)

        int highestBit = 31 - Integer.numberOfLeadingZeros(n);
        for (int i = highestBit; i >= 0; i--) {
            // Compute:
            // c = F(2k) = F(k) * (2*F(k+1) - F(k))
            // d = F(2k+1) = F(k)^2 + F(k+1)^2
            long twoBminusA = (b << 1) - a;
            long c = a * twoBminusA;
            long d = a * a + b * b;

            int mask = 1 << i;
            if ((n & mask) == 0) {
                a = c;
                b = d;
            } else {
                a = d;
                b = c + d;
            }
        }
        return a;
    }

    /**
     * Find all prime numbers up to n using naive approach.
     * This can be optimized with Sieve of Eratosthenes.
     *
     * @param n Upper bound for finding primes
     * @return List of all prime numbers <= n
     */
    public List<Integer> findPrimes(int n) {
        List<Integer> primes = new ArrayList<>();
        for (int i = 2; i <= n; i++) {
            if (isPrime(i)) {
                primes.add(i);
            }
        }
        return primes;
    }

    /**
     * Check if a number is prime using naive trial division.
     *
     * @param num Number to check
     * @return true if num is prime
     */
    private boolean isPrime(int num) {
        if (num < 2) return false;
        for (int i = 2; i < num; i++) {
            if (num % i == 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * Find duplicates in an array using O(n^2) nested loops.
     * This can be optimized with HashSet to O(n).
     *
     * @param arr Input array
     * @return List of duplicate elements
     */
    public List<Integer> findDuplicates(int[] arr) {
        List<Integer> duplicates = new ArrayList<>();
        for (int i = 0; i < arr.length; i++) {
            for (int j = i + 1; j < arr.length; j++) {
                if (arr[i] == arr[j] && !duplicates.contains(arr[i])) {
                    duplicates.add(arr[i]);
                }
            }
        }
        return duplicates;
    }

    /**
     * Calculate factorial recursively without tail optimization.
     *
     * @param n Number to calculate factorial for
     * @return n!
     */
    public long factorial(int n) {
        if (n <= 1) {
            return 1;
        }
        return n * factorial(n - 1);
    }

    /**
     * Concatenate strings in a loop using String concatenation.
     * Should be optimized to use StringBuilder.
     *
     * @param items List of strings to concatenate
     * @return Concatenated result
     */
    public String concatenateStrings(List<String> items) {
        String result = "";
        for (String item : items) {
            result = result + item + ", ";
        }
        if (result.length() > 2) {
            result = result.substring(0, result.length() - 2);
        }
        return result;
    }

    /**
     * Calculate sum of squares using a loop.
     * This is already efficient but shows a simple example.
     *
     * @param n Upper bound
     * @return Sum of squares from 1 to n
     */
    public long sumOfSquares(int n) {
        long sum = 0;
        for (int i = 1; i <= n; i++) {
            sum += (long) i * i;
        }
        return sum;
    }
}
