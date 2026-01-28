/**
 * Tests for Fibonacci functions - CommonJS module
 */
const { fibonacci, isFibonacci, isPerfectSquare, fibonacciSequence } = require('../fibonacci');

describe('fibonacci', () => {
    test('returns 0 for n=0', () => {
        expect(fibonacci(0)).toBe(0);
    });

    test('returns 1 for n=1', () => {
        expect(fibonacci(1)).toBe(1);
    });

    test('returns 1 for n=2', () => {
        expect(fibonacci(2)).toBe(1);
    });

    test('returns 5 for n=5', () => {
        expect(fibonacci(5)).toBe(5);
    });

    test('returns 55 for n=10', () => {
        expect(fibonacci(10)).toBe(55);
    });

    test('returns 233 for n=13', () => {
        expect(fibonacci(13)).toBe(233);
    });
});

describe('isFibonacci', () => {
    test('returns true for Fibonacci numbers', () => {
        expect(isFibonacci(0)).toBe(true);
        expect(isFibonacci(1)).toBe(true);
        expect(isFibonacci(5)).toBe(true);
        expect(isFibonacci(8)).toBe(true);
        expect(isFibonacci(13)).toBe(true);
    });

    test('returns false for non-Fibonacci numbers', () => {
        expect(isFibonacci(4)).toBe(false);
        expect(isFibonacci(6)).toBe(false);
        expect(isFibonacci(7)).toBe(false);
    });
});

describe('isPerfectSquare', () => {
    test('returns true for perfect squares', () => {
        expect(isPerfectSquare(0)).toBe(true);
        expect(isPerfectSquare(1)).toBe(true);
        expect(isPerfectSquare(4)).toBe(true);
        expect(isPerfectSquare(9)).toBe(true);
        expect(isPerfectSquare(16)).toBe(true);
    });

    test('returns false for non-perfect squares', () => {
        expect(isPerfectSquare(2)).toBe(false);
        expect(isPerfectSquare(3)).toBe(false);
        expect(isPerfectSquare(5)).toBe(false);
    });
});

describe('fibonacciSequence', () => {
    test('returns empty array for n=0', () => {
        expect(fibonacciSequence(0)).toEqual([]);
    });

    test('returns first 5 Fibonacci numbers', () => {
        expect(fibonacciSequence(5)).toEqual([0, 1, 1, 2, 3]);
    });

    test('returns first 10 Fibonacci numbers', () => {
        expect(fibonacciSequence(10)).toEqual([0, 1, 1, 2, 3, 5, 8, 13, 21, 34]);
    });
});
