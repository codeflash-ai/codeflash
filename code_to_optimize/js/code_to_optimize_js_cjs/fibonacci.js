const _fibCache = new Map([[0, 0], [1, 1]]);

/**
 * Fibonacci implementations - CommonJS module
 * Intentionally inefficient for optimization testing.
 */

function fibonacci(n) {
    if (n <= 1) {
        return n;
    }
    const cached = _fibCache.get(n);
    if (cached !== undefined) {
        return cached;
    }
    const result = fibonacci(n - 1) + fibonacci(n - 2);
    _fibCache.set(n, result);
    return result;
}

/**
 * Check if a number is a Fibonacci number.
 * @param {number} num - The number to check
 * @returns {boolean} True if num is a Fibonacci number
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
 * @returns {boolean} True if n is a perfect square
 */
function isPerfectSquare(n) {
    const sqrt = Math.sqrt(n);
    return sqrt === Math.floor(sqrt);
}

/**
 * Generate an array of Fibonacci numbers up to n.
 * @param {number} n - The number of Fibonacci numbers to generate
 * @returns {number[]} Array of Fibonacci numbers
 */
function fibonacciSequence(n) {
    const result = [];
    for (let i = 0; i < n; i++) {
        result.push(fibonacci(i));
    }
    return result;
}

// CommonJS exports
module.exports = {
    fibonacci,
    isFibonacci,
    isPerfectSquare,
    fibonacciSequence,
};
