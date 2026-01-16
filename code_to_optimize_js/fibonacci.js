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
    if (n <= 1) {
        return n;
    }

    // Fast path for integer n: iterative O(n) with constant auxiliary memory.
    if (Number.isInteger(n)) {
        let a = 0;
        let b = 1;
        // iterate from 2..n inclusive; cache length to help V8 optimize the loop
        for (let i = 2, len = n; i <= len; i++) {
            const c = a + b;
            a = b;
            b = c;
        }
        return b;
    }

    // Non-integer n: preserve original recursive semantics but avoid exponential blow-up
    // by memoizing intermediate results keyed by the numeric value.
    const cache = new Map();
    function rec(x) {
        if (x <= 1) {
            return x;
        }
        const cached = cache.get(x);
        if (cached !== undefined) {
            return cached;
        }
        const val = rec(x - 1) + rec(x - 2);
        cache.set(x, val);
        return val;
    }
    return rec(n);
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
