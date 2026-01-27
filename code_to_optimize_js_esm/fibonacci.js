/**
 * Fibonacci implementations - ES Module
 * Intentionally inefficient for optimization testing.
 */

/**
 * Calculate the nth Fibonacci number using naive recursion.
 * This is intentionally slow to demonstrate optimization potential.
 * @param {number} n - The index of the Fibonacci number to calculate
 * @returns {number} The nth Fibonacci number
 */
export function fibonacci(n) {
    if (n <= 1) {
        return n;
    }

    // For very special numeric values (NaN, Infinity) the original implementation
    // would recurse indefinitely. To preserve that behavior exactly, fall back
    // to a direct recursion in those cases.
    const m = Math.ceil(n - 1);
    if (!Number.isFinite(m)) {
        function _rec(x) {
            if (x <= 1) {
                return x;
            }
            return _rec(x - 1) + _rec(x - 2);
        }
        return _rec(n);
    }

    // start is the smallest value <= 1 encountered when stepping down by 1's from n
    const start = n - m;

    // Use the recurrence in the forward direction:
    // Let F_k = fibonacci(start + k). We need F_{-1} and F_0 as seeds:
    // F_{-1} = start - 1 (<= 1) and F_0 = start (<= 1)
    let a = start - 1; // F_{-1}
    let b = start;     // F_0

    // Advance m steps to reach F_m = fibonacci(n)
    for (let i = 1; i <= m; i++) {
        const next = b + a;
        a = b;
        b = next;
    }

    return b;
}

/**
 * Check if a number is a Fibonacci number.
 * @param {number} num - The number to check
 * @returns {boolean} True if num is a Fibonacci number
 */
export function isFibonacci(num) {
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
export function isPerfectSquare(n) {
    const sqrt = Math.sqrt(n);
    return sqrt === Math.floor(sqrt);
}

/**
 * Generate an array of Fibonacci numbers up to n.
 * @param {number} n - The number of Fibonacci numbers to generate
 * @returns {number[]} Array of Fibonacci numbers
 */
export function fibonacciSequence(n) {
    const result = [];
    for (let i = 0; i < n; i++) {
        result.push(fibonacci(i));
    }
    return result;
}
