package com.example;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for Algorithms class.
 */
class AlgorithmsTest {

    private Algorithms algorithms;

    @BeforeEach
    void setUp() {
        algorithms = new Algorithms();
    }

    @Test
    @DisplayName("Fibonacci of 0 should return 0")
    void testFibonacciZero() {
        assertEquals(0, algorithms.fibonacci(0));
    }

    @Test
    @DisplayName("Fibonacci of 1 should return 1")
    void testFibonacciOne() {
        assertEquals(1, algorithms.fibonacci(1));
    }

    @Test
    @DisplayName("Fibonacci of 10 should return 55")
    void testFibonacciTen() {
        assertEquals(55, algorithms.fibonacci(10));
    }

    @Test
    @DisplayName("Fibonacci of 20 should return 6765")
    void testFibonacciTwenty() {
        assertEquals(6765, algorithms.fibonacci(20));
    }

    @Test
    @DisplayName("Find primes up to 10")
    void testFindPrimesUpToTen() {
        List<Integer> primes = algorithms.findPrimes(10);
        assertEquals(Arrays.asList(2, 3, 5, 7), primes);
    }

    @Test
    @DisplayName("Find primes up to 20")
    void testFindPrimesUpToTwenty() {
        List<Integer> primes = algorithms.findPrimes(20);
        assertEquals(Arrays.asList(2, 3, 5, 7, 11, 13, 17, 19), primes);
    }

    @Test
    @DisplayName("Find duplicates in array with duplicates")
    void testFindDuplicatesWithDuplicates() {
        int[] arr = {1, 2, 3, 2, 4, 3, 5};
        List<Integer> duplicates = algorithms.findDuplicates(arr);
        assertTrue(duplicates.contains(2));
        assertTrue(duplicates.contains(3));
        assertEquals(2, duplicates.size());
    }

    @Test
    @DisplayName("Find duplicates in array without duplicates")
    void testFindDuplicatesNoDuplicates() {
        int[] arr = {1, 2, 3, 4, 5};
        List<Integer> duplicates = algorithms.findDuplicates(arr);
        assertTrue(duplicates.isEmpty());
    }

    @Test
    @DisplayName("Factorial of 0 should return 1")
    void testFactorialZero() {
        assertEquals(1, algorithms.factorial(0));
    }

    @Test
    @DisplayName("Factorial of 5 should return 120")
    void testFactorialFive() {
        assertEquals(120, algorithms.factorial(5));
    }

    @Test
    @DisplayName("Factorial of 10 should return 3628800")
    void testFactorialTen() {
        assertEquals(3628800, algorithms.factorial(10));
    }

    @Test
    @DisplayName("Concatenate empty list")
    void testConcatenateEmptyList() {
        assertEquals("", algorithms.concatenateStrings(List.of()));
    }

    @Test
    @DisplayName("Concatenate single item")
    void testConcatenateSingleItem() {
        assertEquals("hello", algorithms.concatenateStrings(List.of("hello")));
    }

    @Test
    @DisplayName("Concatenate multiple items")
    void testConcatenateMultipleItems() {
        assertEquals("a, b, c", algorithms.concatenateStrings(Arrays.asList("a", "b", "c")));
    }

    @Test
    @DisplayName("Sum of squares up to 5")
    void testSumOfSquaresFive() {
        // 1 + 4 + 9 + 16 + 25 = 55
        assertEquals(55, algorithms.sumOfSquares(5));
    }

    @Test
    @DisplayName("Sum of squares up to 10")
    void testSumOfSquaresTen() {
        // 1 + 4 + 9 + 16 + 25 + 36 + 49 + 64 + 81 + 100 = 385
        assertEquals(385, algorithms.sumOfSquares(10));
    }
}
