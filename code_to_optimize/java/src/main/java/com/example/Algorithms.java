package com.example;

import java.util.ArrayList;
import java.util.List;

/**
 * Collection of algorithms.
 */
public class Algorithms {

    /**
     * Calculate Fibonacci number using recursive approach.
     *
     * @param n The position in Fibonacci sequence (0-indexed)
     * @return The nth Fibonacci number
     */
    public long fibonacci(int n) {
        if (n <= 1) {
            return n;
        }
        // Fast-doubling iterative approach:
        // Maintain (a, b) = (F(k), F(k+1)) and process bits of n from MSB to LSB.
        long a = 0L; // F(0)
        long b = 1L; // F(1)

        int highestBit = 31 - Integer.numberOfLeadingZeros(n); // position of highest set bit
        int mask = 1 << highestBit;
        for (; mask != 0; mask >>>= 1) {
            // Loop invariant: (a, b) = (F(k), F(k+1)) for current k
            long twoBminusA = (b << 1) - a; // 2*b - a
            long c = a * twoBminusA;        // F(2k) = F(k) * (2*F(k+1) − F(k))
            long d = a * a + b * b;         // F(2k+1) = F(k)^2 + F(k+1)^2

            if ((n & mask) == 0) {
                // bit is 0 -> (F(2k), F(2k+1))
                a = c;
                b = d;
            } else {
                // bit is 1 -> (F(2k+1), F(2k+2)) = (d, c + d)
                a = d;
                b = c + d;
            }
        }
        return a;
    }

    /**
     * Find all prime numbers up to n.
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
     * Check if a number is prime using trial division.
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
     * Find duplicates in an array using nested loops.
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
     * Calculate factorial recursively.
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
