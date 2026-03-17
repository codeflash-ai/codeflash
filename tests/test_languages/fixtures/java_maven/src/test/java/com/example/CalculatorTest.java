package com.example;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the Calculator class.
 */
@DisplayName("Calculator Tests")
class CalculatorTest {

    private Calculator calculator;

    @BeforeEach
    void setUp() {
        calculator = new Calculator(2);
    }

    @Nested
    @DisplayName("Compound Interest Tests")
    class CompoundInterestTests {

        @Test
        @DisplayName("should calculate compound interest for basic case")
        void testBasicCompoundInterest() {
            String result = calculator.calculateCompoundInterest(1000.0, 0.05, 1, 12);
            assertNotNull(result);
            assertTrue(result.contains("."));
        }

        @Test
        @DisplayName("should handle zero principal")
        void testZeroPrincipal() {
            String result = calculator.calculateCompoundInterest(0.0, 0.05, 1, 12);
            assertEquals("0.00", result);
        }

        @Test
        @DisplayName("should throw on negative principal")
        void testNegativePrincipal() {
            assertThrows(IllegalArgumentException.class, () ->
                calculator.calculateCompoundInterest(-100.0, 0.05, 1, 12)
            );
        }

        @ParameterizedTest
        @CsvSource({
            "1000, 0.05, 1, 12",
            "5000, 0.08, 2, 4",
            "10000, 0.03, 5, 1"
        })
        @DisplayName("should calculate for various inputs")
        void testVariousInputs(double principal, double rate, int time, int n) {
            String result = calculator.calculateCompoundInterest(principal, rate, time, n);
            assertNotNull(result);
            assertFalse(result.isEmpty());
        }
    }

    @Nested
    @DisplayName("Permutation Tests")
    class PermutationTests {

        @Test
        @DisplayName("should calculate permutation correctly")
        void testBasicPermutation() {
            assertEquals(120, calculator.permutation(5, 5));
            assertEquals(60, calculator.permutation(5, 3));
            assertEquals(20, calculator.permutation(5, 2));
        }

        @Test
        @DisplayName("should return 0 when n < r")
        void testInvalidPermutation() {
            assertEquals(0, calculator.permutation(3, 5));
        }

        @Test
        @DisplayName("should handle edge cases")
        void testEdgeCases() {
            assertEquals(1, calculator.permutation(5, 0));
            assertEquals(1, calculator.permutation(0, 0));
        }
    }

    @Nested
    @DisplayName("Combination Tests")
    class CombinationTests {

        @Test
        @DisplayName("should calculate combination correctly")
        void testBasicCombination() {
            assertEquals(10, calculator.combination(5, 3));
            assertEquals(10, calculator.combination(5, 2));
            assertEquals(1, calculator.combination(5, 5));
        }

        @Test
        @DisplayName("should return 0 when n < r")
        void testInvalidCombination() {
            assertEquals(0, calculator.combination(3, 5));
        }
    }

    @Nested
    @DisplayName("Fibonacci Tests")
    class FibonacciTests {

        @Test
        @DisplayName("should calculate fibonacci correctly")
        void testFibonacci() {
            assertEquals(0, calculator.fibonacci(0));
            assertEquals(1, calculator.fibonacci(1));
            assertEquals(1, calculator.fibonacci(2));
            assertEquals(2, calculator.fibonacci(3));
            assertEquals(5, calculator.fibonacci(5));
            assertEquals(55, calculator.fibonacci(10));
        }

        @ParameterizedTest
        @CsvSource({
            "0, 0",
            "1, 1",
            "2, 1",
            "3, 2",
            "4, 3",
            "5, 5",
            "6, 8",
            "7, 13"
        })
        @DisplayName("should match expected sequence")
        void testFibonacciSequence(int n, long expected) {
            assertEquals(expected, calculator.fibonacci(n));
        }
    }

    @Test
    @DisplayName("static quickAdd should work correctly")
    void testQuickAdd() {
        assertEquals(15.0, Calculator.quickAdd(10.0, 5.0));
        assertEquals(0.0, Calculator.quickAdd(-5.0, 5.0));
        assertEquals(-10.0, Calculator.quickAdd(-5.0, -5.0));
    }

    @Test
    @DisplayName("should track calculation history")
    void testHistory() {
        calculator.calculateCompoundInterest(1000.0, 0.05, 1, 12);
        calculator.calculateCompoundInterest(2000.0, 0.03, 2, 4);

        var history = calculator.getHistory();
        assertEquals(2, history.size());
        assertTrue(history.get(0).startsWith("compound:"));
    }

    @Test
    @DisplayName("should return correct precision")
    void testPrecision() {
        assertEquals(2, calculator.getPrecision());

        Calculator customCalc = new Calculator(4);
        assertEquals(4, customCalc.getPrecision());
    }
}
