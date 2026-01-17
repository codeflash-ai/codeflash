/**
 * Fibonacci implementations - intentionally inefficient for optimization testing.
 */

/**
 * Calculate the nth Fibonacci number using naive recursion.
 * This is intentionally slow to demonstrate optimization potential.
 * @param {number} n - The index of the Fibonacci number to calculate
 * @returns {number} - The nth Fibonacci number
 */
function fibonacci(n) {
    const cache = new Map();
    function fib(x) {
        if (x <= 1) {
            return x;
        }
        const cached = cache.get(x);
        if (cached !== undefined) {
            return cached;
        }
        const result = fib(x - 1) + fib(x - 2);
        cache.set(x, result);
        return result;
    }
    return fib(n);
}

/**
 * Check if a number is a Fibonacci number.
 * @param {number} num - The number to check
 * @returns {boolean} - True if num is a Fibonacci number
 */
function isFibonacci(num) {
    // A number is Fibonacci if one of (5*n*n + 4) or (5*n*n - 4) is a perfect square
    const check1 = 5 * num * num + 4;
    const check2 = 5 * num * num - 4;

    return isPerfectSquare(check1) || isPerfectSquare(check2);
}

/**
 * Check if a number is a perfect square.
 * @param {number} n - The number to check
 * @returns {boolean} - True if n is a perfect square
 */
function isPerfectSquare(n) {
    const sqrt = Math.sqrt(n);
    return sqrt === Math.floor(sqrt);
}

/**
 * Generate an array of Fibonacci numbers up to n.
 * @param {number} n - The number of Fibonacci numbers to generate
 * @returns {number[]} - Array of Fibonacci numbers
 */
function fibonacciSequence(n) {
    const result = [];
    for (let i = 0; i < n; i++) {
        result.push(fibonacci(i));
    }
    return result;
}

module.exports = { fibonacci, isFibonacci, isPerfectSquare, fibonacciSequence };
