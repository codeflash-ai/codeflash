/**
 * Fibonacci implementations - CommonJS module
 * Intentionally inefficient for optimization testing.
 */

/**
 * Calculate the nth Fibonacci number using naive recursion.
 * This is intentionally slow to demonstrate optimization potential.
 * @param {number} n - The index of the Fibonacci number to calculate
 * @returns {number} The nth Fibonacci number
 */
function fibonacci(n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
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
    if (n < 0) return false;
    if (n === 0) return true;
    
    // Integer square root using Newton's method (faster than Math.sqrt for this check)
    let x = n;
    let y = (x + 1) >> 1;
    while (y < x) {
        x = y;
        y = (x + n / x) >> 1;
    }
    return x * x === n;
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
