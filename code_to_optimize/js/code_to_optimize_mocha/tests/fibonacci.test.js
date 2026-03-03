/**
 * Tests for Fibonacci functions - Mocha + node:assert/strict
 */
const assert = require('node:assert/strict');
const { fibonacci, isFibonacci, isPerfectSquare, fibonacciSequence } = require('../fibonacci');

describe('fibonacci', () => {
    it('returns 0 for n=0', () => {
        assert.strictEqual(fibonacci(0), 0);
    });

    it('returns 1 for n=1', () => {
        assert.strictEqual(fibonacci(1), 1);
    });

    it('returns 1 for n=2', () => {
        assert.strictEqual(fibonacci(2), 1);
    });

    it('returns 5 for n=5', () => {
        assert.strictEqual(fibonacci(5), 5);
    });

    it('returns 55 for n=10', () => {
        assert.strictEqual(fibonacci(10), 55);
    });

    it('returns 233 for n=13', () => {
        assert.strictEqual(fibonacci(13), 233);
    });
});

describe('isFibonacci', () => {
    it('returns true for Fibonacci numbers', () => {
        assert.strictEqual(isFibonacci(0), true);
        assert.strictEqual(isFibonacci(1), true);
        assert.strictEqual(isFibonacci(5), true);
        assert.strictEqual(isFibonacci(8), true);
        assert.strictEqual(isFibonacci(13), true);
    });

    it('returns false for non-Fibonacci numbers', () => {
        assert.strictEqual(isFibonacci(4), false);
        assert.strictEqual(isFibonacci(6), false);
        assert.strictEqual(isFibonacci(7), false);
    });
});

describe('isPerfectSquare', () => {
    it('returns true for perfect squares', () => {
        assert.strictEqual(isPerfectSquare(0), true);
        assert.strictEqual(isPerfectSquare(1), true);
        assert.strictEqual(isPerfectSquare(4), true);
        assert.strictEqual(isPerfectSquare(9), true);
        assert.strictEqual(isPerfectSquare(16), true);
    });

    it('returns false for non-perfect squares', () => {
        assert.strictEqual(isPerfectSquare(2), false);
        assert.strictEqual(isPerfectSquare(3), false);
        assert.strictEqual(isPerfectSquare(5), false);
    });
});

describe('fibonacciSequence', () => {
    it('returns empty array for n=0', () => {
        assert.deepStrictEqual(fibonacciSequence(0), []);
    });

    it('returns first 5 Fibonacci numbers', () => {
        assert.deepStrictEqual(fibonacciSequence(5), [0, 1, 1, 2, 3]);
    });

    it('returns first 10 Fibonacci numbers', () => {
        assert.deepStrictEqual(fibonacciSequence(10), [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]);
    });
});
