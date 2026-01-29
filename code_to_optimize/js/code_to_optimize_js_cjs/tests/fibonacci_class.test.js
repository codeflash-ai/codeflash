const { FibonacciCalculator } = require('../fibonacci_class');

describe('FibonacciCalculator', () => {
    let calc;

    beforeEach(() => {
        calc = new FibonacciCalculator();
    });

    describe('fibonacci', () => {
        test('returns 0 for n=0', () => {
            expect(calc.fibonacci(0)).toBe(0);
        });

        test('returns 1 for n=1', () => {
            expect(calc.fibonacci(1)).toBe(1);
        });

        test('returns 1 for n=2', () => {
            expect(calc.fibonacci(2)).toBe(1);
        });

        test('returns 5 for n=5', () => {
            expect(calc.fibonacci(5)).toBe(5);
        });

        test('returns 55 for n=10', () => {
            expect(calc.fibonacci(10)).toBe(55);
        });

        test('returns 233 for n=13', () => {
            expect(calc.fibonacci(13)).toBe(233);
        });
    });

    describe('isFibonacci', () => {
        test('returns true for 0', () => {
            expect(calc.isFibonacci(0)).toBe(true);
        });

        test('returns true for 1', () => {
            expect(calc.isFibonacci(1)).toBe(true);
        });

        test('returns true for 8', () => {
            expect(calc.isFibonacci(8)).toBe(true);
        });

        test('returns true for 13', () => {
            expect(calc.isFibonacci(13)).toBe(true);
        });

        test('returns false for 4', () => {
            expect(calc.isFibonacci(4)).toBe(false);
        });

        test('returns false for 6', () => {
            expect(calc.isFibonacci(6)).toBe(false);
        });
    });

    describe('isPerfectSquare', () => {
        test('returns true for 0', () => {
            expect(calc.isPerfectSquare(0)).toBe(true);
        });

        test('returns true for 1', () => {
            expect(calc.isPerfectSquare(1)).toBe(true);
        });

        test('returns true for 4', () => {
            expect(calc.isPerfectSquare(4)).toBe(true);
        });

        test('returns true for 16', () => {
            expect(calc.isPerfectSquare(16)).toBe(true);
        });

        test('returns false for 2', () => {
            expect(calc.isPerfectSquare(2)).toBe(false);
        });

        test('returns false for 3', () => {
            expect(calc.isPerfectSquare(3)).toBe(false);
        });
    });

    describe('fibonacciSequence', () => {
        test('returns empty array for n=0', () => {
            expect(calc.fibonacciSequence(0)).toEqual([]);
        });

        test('returns [0] for n=1', () => {
            expect(calc.fibonacciSequence(1)).toEqual([0]);
        });

        test('returns first 5 Fibonacci numbers', () => {
            expect(calc.fibonacciSequence(5)).toEqual([0, 1, 1, 2, 3]);
        });

        test('returns first 10 Fibonacci numbers', () => {
            expect(calc.fibonacciSequence(10)).toEqual([0, 1, 1, 2, 3, 5, 8, 13, 21, 34]);
        });
    });
});
