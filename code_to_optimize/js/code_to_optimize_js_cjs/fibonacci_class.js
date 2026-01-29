/**
 * Fibonacci Calculator Class - CommonJS module
 * Intentionally inefficient for optimization testing.
 */

class FibonacciCalculator {
    constructor() {
        // No initialization needed
    }

    /**
     * Calculate the nth Fibonacci number using naive recursion.
     * This is intentionally slow to demonstrate optimization potential.
     * @param {number} n - The index of the Fibonacci number to calculate
     * @returns {number} The nth Fibonacci number
     */
    fibonacci(n) {
        if (n <= 1) {
            return n;
        }
        return this.fibonacci(n - 1) + this.fibonacci(n - 2);
    }

    /**
     * Check if a number is a Fibonacci number.
     * @param {number} num - The number to check
     * @returns {boolean} True if num is a Fibonacci number
     */
    isFibonacci(num) {
        // A number is Fibonacci if one of (5*n*n + 4) or (5*n*n - 4) is a perfect square
        const check1 = 5 * num * num + 4;
        const check2 = 5 * num * num - 4;
        return this.isPerfectSquare(check1) || this.isPerfectSquare(check2);
    }

    /**
     * Check if a number is a perfect square.
     * @param {number} n - The number to check
     * @returns {boolean} True if n is a perfect square
     */
    isPerfectSquare(n) {
        const sqrt = Math.sqrt(n);
        return sqrt === Math.floor(sqrt);
    }

    /**
     * Generate an array of Fibonacci numbers up to n.
     * @param {number} n - The number of Fibonacci numbers to generate
     * @returns {number[]} Array of Fibonacci numbers
     */
    fibonacciSequence(n) {
        const result = [];
        for (let i = 0; i < n; i++) {
            result.push(this.fibonacci(i));
        }
        return result;
    }
}

// CommonJS exports
module.exports = { FibonacciCalculator };
