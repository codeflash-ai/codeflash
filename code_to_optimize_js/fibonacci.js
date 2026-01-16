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

    // For integer inputs use an iterative O(n) loop with constant memory
    // This preserves exact integer Fibonacci values and avoids recursion/branch explosion.
    if (Number.isInteger(n)) {
        let a = 0;
        let b = 1;
        // iterate from 2..n inclusive
        for (let i = 2; i <= n; i++) {
            const tmp = a + b;
            a = b;
            b = tmp;
        }
        return b;
    }

    // For non-integer inputs preserve original recursive semantics but memoize
    // to eliminate exponential recomputation while returning identical results.
    const cache = new Map();
    function fib(k) {
        if (k <= 1) return k;
        if (cache.has(k)) return cache.get(k);
        const val = fib(k - 1) + fib(k - 2);
        cache.set(k, val);
        return val;
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
