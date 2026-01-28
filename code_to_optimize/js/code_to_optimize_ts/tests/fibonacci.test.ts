import { fibonacci, isFibonacci, isPerfectSquare, fibonacciSequence } from '../fibonacci';

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
    test('returns true for 0', () => {
        expect(isFibonacci(0)).toBe(true);
    });

    test('returns true for 1', () => {
        expect(isFibonacci(1)).toBe(true);
    });

    test('returns true for 8', () => {
        expect(isFibonacci(8)).toBe(true);
    });

    test('returns true for 13', () => {
        expect(isFibonacci(13)).toBe(true);
    });

    test('returns false for 4', () => {
        expect(isFibonacci(4)).toBe(false);
    });

    test('returns false for 6', () => {
        expect(isFibonacci(6)).toBe(false);
    });
});

describe('isPerfectSquare', () => {
    test('returns true for 0', () => {
        expect(isPerfectSquare(0)).toBe(true);
    });

    test('returns true for 1', () => {
        expect(isPerfectSquare(1)).toBe(true);
    });

    test('returns true for 4', () => {
        expect(isPerfectSquare(4)).toBe(true);
    });

    test('returns true for 16', () => {
        expect(isPerfectSquare(16)).toBe(true);
    });

    test('returns false for 2', () => {
        expect(isPerfectSquare(2)).toBe(false);
    });

    test('returns false for 3', () => {
        expect(isPerfectSquare(3)).toBe(false);
    });
});

describe('fibonacciSequence', () => {
    test('returns empty array for n=0', () => {
        expect(fibonacciSequence(0)).toEqual([]);
    });

    test('returns [0] for n=1', () => {
        expect(fibonacciSequence(1)).toEqual([0]);
    });

    test('returns first 5 Fibonacci numbers', () => {
        expect(fibonacciSequence(5)).toEqual([0, 1, 1, 2, 3]);
    });

    test('returns first 10 Fibonacci numbers', () => {
        expect(fibonacciSequence(10)).toEqual([0, 1, 1, 2, 3, 5, 8, 13, 21, 34]);
    });
});
