package com.example;

import org.junit.jupiter.api.Test;
import java.util.ArrayList;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Fibonacci functions.
 */
class FibonacciTest {

    @Test
    void testFibonacci() {
        assertEquals(0, Fibonacci.fibonacci(0));
        assertEquals(1, Fibonacci.fibonacci(1));
        assertEquals(1, Fibonacci.fibonacci(2));
        assertEquals(2, Fibonacci.fibonacci(3));
        assertEquals(3, Fibonacci.fibonacci(4));
        assertEquals(5, Fibonacci.fibonacci(5));
        assertEquals(8, Fibonacci.fibonacci(6));
        assertEquals(13, Fibonacci.fibonacci(7));
        assertEquals(21, Fibonacci.fibonacci(8));
        assertEquals(55, Fibonacci.fibonacci(10));
    }

    @Test
    void testFibonacciNegative() {
        assertThrows(IllegalArgumentException.class, () -> Fibonacci.fibonacci(-1));
    }

    @Test
    void testIsFibonacci() {
        assertTrue(Fibonacci.isFibonacci(0));
        assertTrue(Fibonacci.isFibonacci(1));
        assertTrue(Fibonacci.isFibonacci(2));
        assertTrue(Fibonacci.isFibonacci(3));
        assertTrue(Fibonacci.isFibonacci(5));
        assertTrue(Fibonacci.isFibonacci(8));
        assertTrue(Fibonacci.isFibonacci(13));
        assertTrue(Fibonacci.isFibonacci(21));

        assertFalse(Fibonacci.isFibonacci(4));
        assertFalse(Fibonacci.isFibonacci(6));
        assertFalse(Fibonacci.isFibonacci(7));
        assertFalse(Fibonacci.isFibonacci(9));
        assertFalse(Fibonacci.isFibonacci(-1));
    }

    @Test
    void testIsPerfectSquare() {
        assertTrue(Fibonacci.isPerfectSquare(0));
        assertTrue(Fibonacci.isPerfectSquare(1));
        assertTrue(Fibonacci.isPerfectSquare(4));
        assertTrue(Fibonacci.isPerfectSquare(9));
        assertTrue(Fibonacci.isPerfectSquare(16));
        assertTrue(Fibonacci.isPerfectSquare(25));
        assertTrue(Fibonacci.isPerfectSquare(100));

        assertFalse(Fibonacci.isPerfectSquare(2));
        assertFalse(Fibonacci.isPerfectSquare(3));
        assertFalse(Fibonacci.isPerfectSquare(5));
        assertFalse(Fibonacci.isPerfectSquare(-1));
    }

    @Test
    void testFibonacciSequence() {
        assertArrayEquals(new long[]{}, Fibonacci.fibonacciSequence(0));
        assertArrayEquals(new long[]{0}, Fibonacci.fibonacciSequence(1));
        assertArrayEquals(new long[]{0, 1}, Fibonacci.fibonacciSequence(2));
        assertArrayEquals(new long[]{0, 1, 1, 2, 3}, Fibonacci.fibonacciSequence(5));
        assertArrayEquals(new long[]{0, 1, 1, 2, 3, 5, 8, 13, 21, 34}, Fibonacci.fibonacciSequence(10));
    }

    @Test
    void testFibonacciSequenceNegative() {
        assertThrows(IllegalArgumentException.class, () -> Fibonacci.fibonacciSequence(-1));
    }

    @Test
    void testFibonacciIndex() {
        assertEquals(0, Fibonacci.fibonacciIndex(0));
        assertEquals(1, Fibonacci.fibonacciIndex(1));
        assertEquals(3, Fibonacci.fibonacciIndex(2));
        assertEquals(4, Fibonacci.fibonacciIndex(3));
        assertEquals(5, Fibonacci.fibonacciIndex(5));
        assertEquals(6, Fibonacci.fibonacciIndex(8));
        assertEquals(7, Fibonacci.fibonacciIndex(13));

        assertEquals(-1, Fibonacci.fibonacciIndex(4));
        assertEquals(-1, Fibonacci.fibonacciIndex(6));
        assertEquals(-1, Fibonacci.fibonacciIndex(-1));
    }

    @Test
    void testSumFibonacci() {
        assertEquals(0, Fibonacci.sumFibonacci(0));
        assertEquals(0, Fibonacci.sumFibonacci(1));
        assertEquals(1, Fibonacci.sumFibonacci(2));
        assertEquals(2, Fibonacci.sumFibonacci(3));
        assertEquals(4, Fibonacci.sumFibonacci(4));
        assertEquals(7, Fibonacci.sumFibonacci(5));
        assertEquals(12, Fibonacci.sumFibonacci(6));
    }

    @Test
    void testFibonacciUpTo() {
        List<Long> result = Fibonacci.fibonacciUpTo(10);
        assertEquals(7, result.size());
        assertEquals(0L, result.get(0));
        assertEquals(1L, result.get(1));
        assertEquals(1L, result.get(2));
        assertEquals(2L, result.get(3));
        assertEquals(3L, result.get(4));
        assertEquals(5L, result.get(5));
        assertEquals(8L, result.get(6));
    }

    @Test
    void testFibonacciUpToZero() {
        List<Long> result = Fibonacci.fibonacciUpTo(0);
        assertTrue(result.isEmpty());
    }

    @Test
    void testAreConsecutiveFibonacci() {
        // Test consecutive Fibonacci pairs (from index 3 onwards to avoid ambiguity with 1,1)
        assertTrue(Fibonacci.areConsecutiveFibonacci(2, 3));  // indices 3 and 4
        assertTrue(Fibonacci.areConsecutiveFibonacci(3, 5));  // indices 4 and 5
        assertTrue(Fibonacci.areConsecutiveFibonacci(5, 8));  // indices 5 and 6
        assertTrue(Fibonacci.areConsecutiveFibonacci(8, 13)); // indices 6 and 7

        // Non-consecutive Fibonacci pairs
        assertFalse(Fibonacci.areConsecutiveFibonacci(2, 5));  // indices 3 and 5
        assertFalse(Fibonacci.areConsecutiveFibonacci(3, 8));  // indices 4 and 6

        // Non-Fibonacci number
        assertFalse(Fibonacci.areConsecutiveFibonacci(4, 5));  // 4 is not Fibonacci
    }

    @Test
    void testSortArray() {
        long[] arr = {5, 3, 8, 1, 2, 7, 4, 6};
        Fibonacci.sortArray(arr);
        assertArrayEquals(new long[]{1, 2, 3, 4, 5, 6, 7, 8}, arr);
    }

    @Test
    void testSortArrayAlreadySorted() {
        long[] arr = {1, 2, 3, 4, 5};
        Fibonacci.sortArray(arr);
        assertArrayEquals(new long[]{1, 2, 3, 4, 5}, arr);
    }

    @Test
    void testSortArrayReversed() {
        long[] arr = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
        Fibonacci.sortArray(arr);
        assertArrayEquals(new long[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, arr);
    }

    @Test
    void testSortArrayDuplicates() {
        long[] arr = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3};
        Fibonacci.sortArray(arr);
        assertArrayEquals(new long[]{1, 1, 2, 3, 3, 4, 5, 5, 6, 9}, arr);
    }

    @Test
    void testSortArrayEmpty() {
        long[] arr = {};
        Fibonacci.sortArray(arr);
        assertArrayEquals(new long[]{}, arr);
    }

    @Test
    void testSortArraySingle() {
        long[] arr = {42};
        Fibonacci.sortArray(arr);
        assertArrayEquals(new long[]{42}, arr);
    }

    @Test
    void testSortArrayNegatives() {
        long[] arr = {-3, -1, -4, -1, -5};
        Fibonacci.sortArray(arr);
        assertArrayEquals(new long[]{-5, -4, -3, -1, -1}, arr);
    }

    @Test
    void testSortArrayNull() {
        assertThrows(IllegalArgumentException.class, () -> Fibonacci.sortArray(null));
    }

    @Test
    void testCollectFibonacciInto() {
        List<Long> output = new ArrayList<>();
        Fibonacci.collectFibonacciInto(output, 10);
        assertEquals(7, output.size());
        assertEquals(List.of(0L, 1L, 1L, 2L, 3L, 5L, 8L), output);
    }

    @Test
    void testCollectFibonacciIntoZeroLimit() {
        List<Long> output = new ArrayList<>();
        Fibonacci.collectFibonacciInto(output, 0);
        assertTrue(output.isEmpty());
    }

    @Test
    void testCollectFibonacciIntoClearsExisting() {
        List<Long> output = new ArrayList<>(List.of(99L, 100L));
        Fibonacci.collectFibonacciInto(output, 5);
        assertEquals(List.of(0L, 1L, 1L, 2L, 3L), output);
    }

    @Test
    void testCollectFibonacciIntoNull() {
        assertThrows(IllegalArgumentException.class, () -> Fibonacci.collectFibonacciInto(null, 10));
    }

    @Test
    void testFillFibonacciRunningSums() {
        long[] result = new long[6];
        Fibonacci.fillFibonacciRunningSums(result);
        // sums: fib(0)=0, 0+1=1, 0+1+1=2, 0+1+1+2=4, 0+1+1+2+3=7, 0+1+1+2+3+5=12
        assertArrayEquals(new long[]{0, 1, 2, 4, 7, 12}, result);
    }

    @Test
    void testFillFibonacciRunningSumsEmpty() {
        long[] result = new long[0];
        Fibonacci.fillFibonacciRunningSums(result);
        assertArrayEquals(new long[]{}, result);
    }

    @Test
    void testFillFibonacciRunningSumsNull() {
        assertThrows(IllegalArgumentException.class, () -> Fibonacci.fillFibonacciRunningSums(null));
    }
}
