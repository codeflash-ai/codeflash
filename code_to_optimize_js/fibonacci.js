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

    // Number of 1-step decrements needed to reach a value <= 1
    const steps = Math.ceil(n - 1);
    const lowest = n - steps; // This is <= 1

    // Base values for indices lowest - 1 and lowest (both <= 1)
    let prevPrev = lowest - 1; // f(lowest - 1) == lowest - 1
    let prev = lowest;         // f(lowest) == lowest

    // Build values up from lowest to n using f(k) = f(k-1) + f(k-2)
    for (let i = 1; i <= steps; i++) {
        const current = prev + prevPrev;
        prevPrev = prev;
        prev = current;
    }

    return prev; // f(n)
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
