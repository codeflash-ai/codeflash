package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for MathHelpers utility class.
 */
class MathHelpersTest {

    @Test
    void testSumArray() {
        assertEquals(10.0, MathHelpers.sumArray(new double[]{1, 2, 3, 4}));
        assertEquals(0.0, MathHelpers.sumArray(new double[]{}));
        assertEquals(0.0, MathHelpers.sumArray(null));
        assertEquals(5.5, MathHelpers.sumArray(new double[]{5.5}));
        assertEquals(-3.0, MathHelpers.sumArray(new double[]{-1, -2, 0}));
    }

    @Test
    void testAverage() {
        assertEquals(2.5, MathHelpers.average(new double[]{1, 2, 3, 4}));
        assertEquals(0.0, MathHelpers.average(new double[]{}));
        assertEquals(0.0, MathHelpers.average(null));
        assertEquals(10.0, MathHelpers.average(new double[]{10}));
    }

    @Test
    void testFindMax() {
        assertEquals(4.0, MathHelpers.findMax(new double[]{1, 2, 3, 4}));
        assertEquals(-1.0, MathHelpers.findMax(new double[]{-5, -1, -10}));
        assertEquals(5.0, MathHelpers.findMax(new double[]{5}));
    }

    @Test
    void testFindMin() {
        assertEquals(1.0, MathHelpers.findMin(new double[]{1, 2, 3, 4}));
        assertEquals(-10.0, MathHelpers.findMin(new double[]{-5, -1, -10}));
        assertEquals(5.0, MathHelpers.findMin(new double[]{5}));
    }

    @Test
    void testFactorial() {
        assertEquals(1, MathHelpers.factorial(0));
        assertEquals(1, MathHelpers.factorial(1));
        assertEquals(2, MathHelpers.factorial(2));
        assertEquals(6, MathHelpers.factorial(3));
        assertEquals(120, MathHelpers.factorial(5));
        assertEquals(3628800, MathHelpers.factorial(10));
    }

    @Test
    void testFactorialNegative() {
        assertThrows(IllegalArgumentException.class, () -> MathHelpers.factorial(-1));
    }

    @Test
    void testPower() {
        assertEquals(8.0, MathHelpers.power(2, 3));
        assertEquals(1.0, MathHelpers.power(5, 0));
        assertEquals(1.0, MathHelpers.power(0, 0));
        assertEquals(0.0, MathHelpers.power(0, 5));
        assertEquals(0.5, MathHelpers.power(2, -1), 0.0001);
        assertEquals(0.125, MathHelpers.power(2, -3), 0.0001);
    }

    @Test
    void testIsPrime() {
        assertFalse(MathHelpers.isPrime(0));
        assertFalse(MathHelpers.isPrime(1));
        assertTrue(MathHelpers.isPrime(2));
        assertTrue(MathHelpers.isPrime(3));
        assertFalse(MathHelpers.isPrime(4));
        assertTrue(MathHelpers.isPrime(5));
        assertTrue(MathHelpers.isPrime(7));
        assertFalse(MathHelpers.isPrime(9));
        assertTrue(MathHelpers.isPrime(11));
        assertTrue(MathHelpers.isPrime(13));
        assertFalse(MathHelpers.isPrime(15));
    }

    @Test
    void testGcd() {
        assertEquals(6, MathHelpers.gcd(12, 18));
        assertEquals(1, MathHelpers.gcd(7, 13));
        assertEquals(5, MathHelpers.gcd(0, 5));
        assertEquals(5, MathHelpers.gcd(5, 0));
        assertEquals(4, MathHelpers.gcd(8, 12));
        assertEquals(3, MathHelpers.gcd(-9, 12));
    }
}
